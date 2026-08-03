export type Role = "system" | "user" | "assistant";

export interface Message {
  id: string;
  role: Role;
  content: string;
  /** Set when the request failed, so the turn can render as a retryable error. */
  error?: string;
  /** Epoch ms. Used for grouping and for "stopped early" affordances. */
  at?: number;
  /** True when the user pressed stop, so "continue" can be offered. */
  truncated?: boolean;
}

export interface Conversation {
  id: string;
  /** Derived from the first user message; editable later. */
  title: string;
  messages: Message[];
  createdAt: number;
  updatedAt: number;
}

export interface Settings {
  temperature: number;
  systemPrompt: string;
  sendOnEnter: boolean;
}

export const DEFAULT_SETTINGS: Settings = {
  temperature: 0.7,
  systemPrompt: "",
  // Enter sends by default, as in every comparable product. Anyone who writes
  // long multi-paragraph prompts can invert it.
  sendOnEnter: true,
};

/** One SSE frame from an OpenAI-compatible endpoint. */
export interface StreamChunk {
  choices?: {
    delta?: { role?: Role; content?: string };
    finish_reason?: string | null;
  }[];
}

export function newId(): string {
  // randomUUID needs a secure context. Plain http from another machine is
  // exactly how this gets tested first, and a crash there would look like the
  // whole app is broken.
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) return crypto.randomUUID();
  return `id-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
}

/** First line of the first user turn, trimmed to something a sidebar can show. */
export function deriveTitle(messages: Message[], fallback: string): string {
  const first = messages.find((m) => m.role === "user" && m.content.trim());
  if (!first) return fallback;
  const line = first.content.trim().split("\n")[0] ?? "";
  return line.length > 48 ? `${line.slice(0, 48).trimEnd()}…` : line;
}
