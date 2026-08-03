export type Role = "system" | "user" | "assistant";

export interface Message {
  id: string;
  role: Role;
  content: string;
  /** Set when the request failed, so the bubble can render as an error. */
  error?: string;
}

/** One SSE frame from an OpenAI-compatible endpoint. */
export interface StreamChunk {
  choices?: {
    delta?: { role?: Role; content?: string };
    finish_reason?: string | null;
  }[];
}

export function newId(): string {
  return crypto.randomUUID();
}
