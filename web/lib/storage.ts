import { DEFAULT_SETTINGS, type Conversation, type Message, type Settings } from "./types";

/**
 * Local persistence: conversations and settings.
 *
 * localStorage rather than a server. The conversation stays on the user's
 * device; sending it anywhere needs a retention policy and a privacy notice,
 * which should be a deliberate decision rather than a side effect of adding a
 * feature.
 *
 * Everything read back is validated rather than trusted. This data outlives
 * deploys, so an older or hand-edited shape would otherwise crash the first
 * render with no recovery available to the user except clearing site data.
 */

const CONVERSATIONS_KEY = "qm.conversations.v2";
const ACTIVE_KEY = "qm.active.v2";
const SETTINGS_KEY = "qm.settings.v1";
/** The single-conversation store shipped before the sidebar existed. */
const LEGACY_KEY = "qm.conversation.v1";

/** Past this, a 5 MB quota starts rejecting writes. */
const MAX_CONVERSATIONS = 100;
const MAX_MESSAGES_EACH = 300;

function read<T>(key: string, fallback: T): T {
  if (typeof window === "undefined") return fallback;
  try {
    const raw = window.localStorage.getItem(key);
    return raw ? (JSON.parse(raw) as T) : fallback;
  } catch {
    return fallback;
  }
}

function write(key: string, value: unknown): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // Private browsing and a full quota both land here, and neither is a
    // reason to break the chat.
  }
}

export function loadConversations(): Conversation[] {
  const stored = read<unknown>(CONVERSATIONS_KEY, null);

  if (Array.isArray(stored)) {
    return stored.filter(isConversation).slice(0, MAX_CONVERSATIONS);
  }

  // Nothing at v2: promote the old single conversation rather than dropping
  // it. Someone mid-conversation when this deploys should not lose it.
  const legacy = read<unknown>(LEGACY_KEY, null);
  if (Array.isArray(legacy)) {
    const messages = legacy.filter(isMessage);
    if (messages.length > 0) {
      const now = Date.now();
      return [{ id: `migrated-${now}`, title: "", messages, createdAt: now, updatedAt: now }];
    }
  }

  return [];
}

export function saveConversations(conversations: Conversation[]): void {
  write(
    CONVERSATIONS_KEY,
    conversations.slice(0, MAX_CONVERSATIONS).map((c) => ({
      ...c,
      messages: c.messages.slice(-MAX_MESSAGES_EACH),
    })),
  );
}

export function loadActiveId(): string | null {
  const value = read<unknown>(ACTIVE_KEY, null);
  return typeof value === "string" ? value : null;
}

export function saveActiveId(id: string | null): void {
  write(ACTIVE_KEY, id);
}

export function loadSettings(): Settings {
  const stored = read<Partial<Settings>>(SETTINGS_KEY, {});
  return {
    // Clamped on read, not only on write: this value is sent to the model, and
    // a hand-edited localStorage entry should not produce an invalid request.
    temperature: clamp(numberOr(stored.temperature, DEFAULT_SETTINGS.temperature), 0, 2),
    systemPrompt:
      typeof stored.systemPrompt === "string"
        ? stored.systemPrompt.slice(0, 4000)
        : DEFAULT_SETTINGS.systemPrompt,
    sendOnEnter:
      typeof stored.sendOnEnter === "boolean" ? stored.sendOnEnter : DEFAULT_SETTINGS.sendOnEnter,
  };
}

export function saveSettings(settings: Settings): void {
  write(SETTINGS_KEY, settings);
}

export function clearEverything(): void {
  if (typeof window === "undefined") return;
  for (const key of [CONVERSATIONS_KEY, ACTIVE_KEY, LEGACY_KEY]) {
    try {
      window.localStorage.removeItem(key);
    } catch {
      /* nothing to do */
    }
  }
}

function numberOr(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function clamp(value: number, low: number, high: number): number {
  return Math.min(high, Math.max(low, value));
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

function isConversation(value: unknown): value is Conversation {
  if (typeof value !== "object" || value === null) return false;
  const c = value as Record<string, unknown>;
  return (
    typeof c.id === "string" &&
    typeof c.title === "string" &&
    Array.isArray(c.messages) &&
    c.messages.every(isMessage)
  );
}
