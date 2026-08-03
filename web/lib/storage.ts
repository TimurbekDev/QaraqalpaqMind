import type { Message } from "./types";

/**
 * Conversation persistence in localStorage.
 *
 * A reload losing the whole conversation is the single worst thing about a
 * chat UI, and it happens constantly on mobile where the browser evicts
 * background tabs.
 *
 * Deliberately localStorage and not a server: the conversation stays on the
 * user's device. Sending it anywhere would need a retention policy and a
 * privacy notice, which is a decision to make explicitly rather than as a side
 * effect of adding a feature.
 */

const KEY = "qm.conversation.v1";
/** Roughly a long session. Past this, writes start failing on a 5 MB quota. */
const MAX_MESSAGES = 200;

export function loadConversation(): Message[] {
  if (typeof window === "undefined") return [];

  try {
    const raw = window.localStorage.getItem(KEY);
    if (!raw) return [];

    const parsed: unknown = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];

    // Validate rather than trust. This data survives across deploys, so an
    // older or hand-edited shape would otherwise crash the render with no way
    // for the user to recover except clearing site data.
    return parsed.filter(isMessage).slice(-MAX_MESSAGES);
  } catch {
    return [];
  }
}

export function saveConversation(messages: Message[]): void {
  if (typeof window === "undefined") return;

  try {
    // Errors are dropped, not thrown: private browsing and a full quota both
    // make this fail, and neither is a reason to break the chat.
    window.localStorage.setItem(KEY, JSON.stringify(messages.slice(-MAX_MESSAGES)));
  } catch {
    /* storage unavailable or full */
  }
}

export function clearConversation(): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.removeItem(KEY);
  } catch {
    /* nothing to do */
  }
}

function isMessage(value: unknown): value is Message {
  if (typeof value !== "object" || value === null) return false;
  const m = value as Record<string, unknown>;
  return (
    typeof m.id === "string" &&
    typeof m.content === "string" &&
    (m.role === "user" || m.role === "assistant" || m.role === "system")
  );
}
