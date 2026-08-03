import type { StreamChunk } from "./types";

/**
 * Read an SSE body and yield the text deltas.
 *
 * The buffering is not incidental. A network chunk has no relationship to an
 * SSE event: one read can deliver half an event, or three and a half. Parsing
 * each chunk as if it were a complete frame works on a fast local connection
 * and corrupts output over a real network, which makes it a bug that passes
 * every test on a laptop.
 *
 * Events are separated by a blank line, so anything after the last "\n\n" is
 * an incomplete frame and stays in the buffer for the next read.
 */
export async function* readDeltas(
  body: ReadableStream<Uint8Array>,
  signal?: AbortSignal,
): AsyncGenerator<string> {
  const reader = body.getReader();
  // stream: true keeps a multi-byte character split across a chunk boundary
  // intact. Karakalpak text is full of two-byte characters - á ó ú ǵ ń ı - so
  // decoding each chunk independently would produce replacement characters at
  // random positions.
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      if (signal?.aborted) return;

      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      const events = buffer.split("\n\n");
      // The tail is either an incomplete event or an empty string. Either way
      // it is not ready to parse.
      buffer = events.pop() ?? "";

      for (const event of events) {
        const delta = parseEvent(event);
        if (delta === DONE) return;
        if (delta) yield delta;
      }
    }
  } finally {
    // Releasing the lock lets the connection close when the caller stops
    // early - otherwise a cancelled generation keeps the socket open.
    reader.releaseLock();
  }
}

const DONE = Symbol("done");

function parseEvent(event: string): string | typeof DONE | null {
  for (const line of event.split("\n")) {
    if (!line.startsWith("data:")) continue;

    const payload = line.slice(5).trim();
    if (payload === "[DONE]") return DONE;
    if (!payload) continue;

    try {
      const chunk = JSON.parse(payload) as StreamChunk;
      const content = chunk.choices?.[0]?.delta?.content;
      if (content) return content;
    } catch {
      // A malformed frame is not worth ending the stream over; the next one
      // is usually fine.
      console.warn("Skipping unparseable SSE frame");
    }
  }
  return null;
}
