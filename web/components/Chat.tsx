"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { clearConversation, loadConversation, saveConversation } from "@/lib/storage";
import { readDeltas } from "@/lib/stream";
import { kaa } from "@/lib/strings";
import { newId, type Message } from "@/lib/types";
import Composer from "./Composer";
import MessageList from "./MessageList";
import ThemeToggle from "./ThemeToggle";

export default function Chat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [streaming, setStreaming] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  // Restored after mount, not during render: localStorage does not exist on
  // the server, and seeding state from it would make the two markups disagree.
  useEffect(() => {
    const saved = loadConversation();
    if (saved.length > 0) setMessages(saved);
  }, []);

  // Not while streaming. Writing on every token would serialise the whole
  // conversation dozens of times a second for no benefit - the final state is
  // what matters, and an interrupted reply is saved by the effect that runs
  // when `streaming` flips back to false.
  useEffect(() => {
    if (!streaming) saveConversation(messages);
  }, [messages, streaming]);

  const stop = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    setStreaming(false);
  }, []);

  const reset = useCallback(() => {
    stop();
    setMessages([]);
    clearConversation();
  }, [stop]);

  /** Stream a reply for `history`, appending into `replyId`. */
  const run = useCallback(async (history: Message[], replyId: string) => {
    setStreaming(true);
    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          messages: history.map(({ role, content }) => ({ role, content })),
        }),
        signal: controller.signal,
      });

      if (!response.ok || !response.body) {
        setMessages((prev) => patch(prev, replyId, { error: errorFor(response.status) }));
        return;
      }

      for await (const delta of readDeltas(response.body, controller.signal)) {
        // Append rather than replace: each frame is one fragment, and the
        // functional form avoids dropping frames batched into one render.
        setMessages((prev) =>
          prev.map((m) => (m.id === replyId ? { ...m, content: m.content + delta } : m)),
        );
      }
    } catch (error) {
      // An abort is the user pressing stop, not a failure. Whatever streamed
      // before it stays on screen.
      if ((error as Error).name !== "AbortError") {
        console.error(error);
        setMessages((prev) => patch(prev, replyId, { error: kaa.errorUnreachable }));
      }
    } finally {
      abortRef.current = null;
      setStreaming(false);
    }
  }, []);

  const send = useCallback(
    (text: string) => {
      const trimmed = text.trim();
      if (!trimmed || streaming) return;

      const user: Message = { id: newId(), role: "user", content: trimmed };
      const reply: Message = { id: newId(), role: "assistant", content: "" };

      // Captured before the state update: setMessages is async, and the request
      // must carry the user turn that was just added.
      const history = [...messages, user];
      setMessages([...history, reply]);
      void run(history, reply.id);
    },
    [messages, streaming, run],
  );

  /** Discard the last assistant turn and ask again from the same history. */
  const regenerate = useCallback(() => {
    if (streaming) return;

    const lastAssistant = findLastIndex(messages, (m) => m.role === "assistant");
    if (lastAssistant < 0) return;

    const history = messages.slice(0, lastAssistant);
    const reply: Message = { id: newId(), role: "assistant", content: "" };
    setMessages([...history, reply]);
    void run(history, reply.id);
  }, [messages, streaming, run]);

  const canRegenerate =
    !streaming && messages.length > 0 && messages[messages.length - 1]?.role === "assistant";

  return (
    <div className="flex h-dvh flex-col bg-[var(--color-surface)]">
      <header
        className="flex shrink-0 items-center justify-between gap-3 border-b
                   border-[var(--color-line)] px-4 py-2.5"
      >
        <div className="min-w-0">
          <h1 className="truncate text-sm font-semibold">{kaa.appName}</h1>
          <p className="truncate text-xs text-[var(--color-ink-muted)]">{kaa.tagline}</p>
        </div>

        <div className="flex shrink-0 items-center gap-2">
          <ThemeToggle />
          <button
            type="button"
            onClick={reset}
            disabled={messages.length === 0}
            className="rounded-lg border border-[var(--color-line)] px-3 py-1.5 text-xs font-medium
                       transition-colors hover:bg-[var(--color-surface-muted)]
                       disabled:cursor-not-allowed disabled:opacity-40"
          >
            {kaa.newChat}
          </button>
        </div>
      </header>

      <MessageList
        messages={messages}
        streaming={streaming}
        onSuggestion={send}
        onRegenerate={regenerate}
        canRegenerate={canRegenerate}
      />

      <Composer onSend={send} onStop={stop} streaming={streaming} />
    </div>
  );
}

function patch(messages: Message[], id: string, fields: Partial<Message>): Message[] {
  return messages.map((m) => (m.id === id ? { ...m, ...fields } : m));
}

function findLastIndex<T>(items: T[], predicate: (item: T) => boolean): number {
  for (let i = items.length - 1; i >= 0; i--) {
    if (predicate(items[i] as T)) return i;
  }
  return -1;
}

function errorFor(status: number): string {
  if (status === 429) return kaa.errorRateLimited;
  if (status === 502 || status === 503) return kaa.errorUnreachable;
  return kaa.errorGeneric;
}
