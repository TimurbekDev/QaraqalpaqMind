"use client";

import { useCallback, useRef, useState } from "react";
import { readDeltas } from "@/lib/stream";
import { kaa } from "@/lib/strings";
import { newId, type Message } from "@/lib/types";
import Composer from "./Composer";
import MessageList from "./MessageList";

export default function Chat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [streaming, setStreaming] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  const stop = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    setStreaming(false);
  }, []);

  const reset = useCallback(() => {
    stop();
    setMessages([]);
  }, [stop]);

  const send = useCallback(
    async (text: string) => {
      const trimmed = text.trim();
      if (!trimmed || streaming) return;

      const user: Message = { id: newId(), role: "user", content: trimmed };
      const reply: Message = { id: newId(), role: "assistant", content: "" };

      // Captured before the state update, because setMessages is async and the
      // request body must contain the user turn we just added.
      const history = [...messages, user];

      setMessages([...history, reply]);
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
          setMessages((prev) => patch(prev, reply.id, { error: errorFor(response.status) }));
          return;
        }

        for await (const delta of readDeltas(response.body, controller.signal)) {
          // Append rather than replace: each frame is one fragment, and the
          // functional form avoids dropping frames that arrive in the same
          // React batch.
          setMessages((prev) =>
            prev.map((m) => (m.id === reply.id ? { ...m, content: m.content + delta } : m)),
          );
        }
      } catch (error) {
        // An abort is the user pressing stop, not a failure. Whatever streamed
        // before it stays on screen.
        if ((error as Error).name !== "AbortError") {
          console.error(error);
          setMessages((prev) => patch(prev, reply.id, { error: kaa.errorUnreachable }));
        }
      } finally {
        abortRef.current = null;
        setStreaming(false);
      }
    },
    [messages, streaming],
  );

  return (
    <div className="flex h-dvh flex-col">
      <header className="flex items-center justify-between border-b border-[var(--color-line)] px-4 py-3">
        <div>
          <h1 className="text-sm font-semibold">{kaa.appName}</h1>
          <p className="text-xs text-[var(--color-ink-muted)]">{kaa.tagline}</p>
        </div>
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
      </header>

      <MessageList messages={messages} streaming={streaming} onSuggestion={send} />

      <Composer onSend={send} onStop={stop} streaming={streaming} />
    </div>
  );
}

function patch(messages: Message[], id: string, fields: Partial<Message>): Message[] {
  return messages.map((m) => (m.id === id ? { ...m, ...fields } : m));
}

function errorFor(status: number): string {
  if (status === 429) return kaa.errorRateLimited;
  if (status === 502 || status === 503) return kaa.errorUnreachable;
  return kaa.errorGeneric;
}
