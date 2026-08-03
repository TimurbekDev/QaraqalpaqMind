/**
 * Server-side proxy to the model gateway.
 *
 * The browser talks to THIS route, never to the gateway. That is the whole
 * point of the file: the API key lives in the server process only.
 *
 * Calling the gateway directly from client code would put the key in the
 * JavaScript bundle, where it is readable by anyone who opens devtools - and
 * `NEXT_PUBLIC_` variables are inlined at build time, so even the name of the
 * variable cannot make that safe. A key that reaches a browser is a published
 * key, and this one buys GPU time.
 *
 * The response is piped through untouched so streaming survives the extra hop.
 */

import { NextRequest } from "next/server";

// Node runtime, not edge: this reads process.env at request time and streams a
// long-lived response, and the edge runtime's limits buy nothing here.
export const runtime = "nodejs";
// Never cache a completion.
export const dynamic = "force-dynamic";

const GATEWAY_URL = process.env.QM_API_URL ?? "http://localhost:8000";
const API_KEY = process.env.QM_API_KEY ?? "";
const MODEL = process.env.QM_MODEL ?? "qaraqalpaqmind";

/** What the browser is allowed to send. Anything else is dropped. */
interface ClientRequest {
  messages: { role: "system" | "user" | "assistant"; content: string }[];
  temperature?: number;
}

const MAX_MESSAGES = 50;
const MAX_CHARS = 32_000;

function badRequest(message: string): Response {
  return Response.json({ error: { message, type: "invalid_request_error" } }, { status: 400 });
}

export async function POST(request: NextRequest): Promise<Response> {
  let body: ClientRequest;
  try {
    body = await request.json();
  } catch {
    return badRequest("Body is not valid JSON.");
  }

  if (!Array.isArray(body.messages) || body.messages.length === 0) {
    return badRequest("`messages` must be a non-empty array.");
  }
  if (body.messages.length > MAX_MESSAGES) {
    return badRequest(`Too many messages (limit ${MAX_MESSAGES}).`);
  }

  let total = 0;
  for (const message of body.messages) {
    if (typeof message?.content !== "string" || !["system", "user", "assistant"].includes(message?.role)) {
      return badRequest("Each message needs a string `content` and a valid `role`.");
    }
    total += message.content.length;
  }
  if (total > MAX_CHARS) {
    return badRequest(`Conversation too long (${total} characters, limit ${MAX_CHARS}).`);
  }

  // Rebuilt rather than forwarded. Passing the client's object through would
  // let a caller set `model`, `max_tokens` or any other field the gateway
  // accepts, using this server's key to do it.
  const upstream = {
    model: MODEL,
    messages: body.messages.map((m) => ({ role: m.role, content: m.content })),
    temperature: clamp(body.temperature ?? 0.7, 0, 2),
    stream: true,
  };

  let response: Response;
  try {
    response = await fetch(`${GATEWAY_URL}/v1/chat/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(API_KEY ? { Authorization: `Bearer ${API_KEY}` } : {}),
      },
      body: JSON.stringify(upstream),
      // Let the user's cancel propagate all the way to the GPU instead of
      // leaving a generation running for a browser that has gone away.
      signal: request.signal,
    });
  } catch (error) {
    // The gateway being down is an operational detail. Say that it is
    // unreachable without echoing an internal hostname to the browser.
    console.error("Gateway request failed:", error);
    return Response.json(
      { error: { message: "Model service is unreachable.", type: "upstream_error" } },
      { status: 502 },
    );
  }

  if (!response.ok || !response.body) {
    const detail = await response.text().catch(() => "");
    console.error(`Gateway returned ${response.status}: ${detail}`);
    // 401 here means THIS server's key is wrong - a deployment problem, not
    // something the visitor can fix, so it is not passed through as 401.
    return Response.json(
      { error: { message: "Model service returned an error.", type: "upstream_error" } },
      { status: response.status === 429 ? 429 : 502 },
    );
  }

  return new Response(response.body, {
    headers: {
      "Content-Type": "text/event-stream; charset=utf-8",
      "Cache-Control": "no-cache, no-transform",
      Connection: "keep-alive",
      // Tells nginx not to buffer this response even if proxy_buffering is on
      // somewhere in front of Next.js. Buffered SSE still returns the right
      // answer, all at once, which looks exactly like broken streaming.
      "X-Accel-Buffering": "no",
    },
  });
}

function clamp(value: number, low: number, high: number): number {
  return Number.isFinite(value) ? Math.min(high, Math.max(low, value)) : low;
}
