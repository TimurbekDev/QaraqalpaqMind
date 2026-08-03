"use client";

import { useEffect, useRef } from "react";
import { kaa, suggestions } from "@/lib/strings";
import type { Message } from "@/lib/types";

interface Props {
  messages: Message[];
  streaming: boolean;
  onSuggestion: (text: string) => void;
}

export default function MessageList({ messages, streaming, onSuggestion }: Props) {
  const endRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  // Whether the user is reading history. Scrolling them back to the bottom
  // while they scroll up is one of the most irritating things a chat UI does.
  const pinnedRef = useRef(true);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const onScroll = () => {
      const distance = container.scrollHeight - container.scrollTop - container.clientHeight;
      pinnedRef.current = distance < 80;
    };
    container.addEventListener("scroll", onScroll, { passive: true });
    return () => container.removeEventListener("scroll", onScroll);
  }, []);

  useEffect(() => {
    if (pinnedRef.current) {
      // "auto", not "smooth": a smooth scroll queued on every token never
      // finishes, so the view lags further behind the faster the model streams.
      endRef.current?.scrollIntoView({ behavior: "auto" });
    }
  }, [messages]);

  if (messages.length === 0) {
    return (
      <div className="flex flex-1 flex-col items-center justify-center gap-6 px-4 text-center">
        <div>
          <h2 className="text-2xl font-semibold">{kaa.emptyTitle}</h2>
          <p className="mt-1 text-sm text-[var(--color-ink-muted)]">{kaa.emptyBody}</p>
        </div>
        <div className="flex w-full max-w-lg flex-col gap-2">
          {suggestions.map((suggestion) => (
            <button
              key={suggestion}
              type="button"
              onClick={() => onSuggestion(suggestion)}
              className="rounded-xl border border-[var(--color-line)] px-4 py-3 text-start text-sm
                         transition-colors hover:bg-[var(--color-surface-muted)]"
            >
              {suggestion}
            </button>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div ref={containerRef} className="flex-1 overflow-y-auto">
      <div className="mx-auto flex max-w-3xl flex-col gap-5 px-4 py-6">
        {messages.map((message, index) => (
          <Bubble
            key={message.id}
            message={message}
            // Only the last assistant message can still be growing.
            streaming={streaming && index === messages.length - 1}
          />
        ))}
        <div ref={endRef} />
      </div>
    </div>
  );
}

function Bubble({ message, streaming }: { message: Message; streaming: boolean }) {
  const isUser = message.role === "user";

  if (message.error) {
    return (
      <div role="alert" className="rounded-xl border border-red-500/40 bg-red-500/5 px-4 py-3 text-sm text-red-600 dark:text-red-400">
        {message.error}
      </div>
    );
  }

  return (
    <div className={isUser ? "flex justify-end" : "flex justify-start"}>
      <div className={isUser ? "max-w-[85%]" : "w-full"}>
        <div className="mb-1 text-xs font-medium text-[var(--color-ink-muted)]">
          {isUser ? kaa.you : kaa.assistant}
        </div>
        <div
          className={[
            "whitespace-pre-wrap break-words text-[15px] leading-relaxed",
            isUser
              ? "rounded-2xl bg-[var(--color-surface-muted)] px-4 py-2.5"
              : "",
            // The caret marks "still generating". Without it an empty bubble
            // during time-to-first-token looks like nothing happened.
            streaming ? "streaming-caret" : "",
          ].join(" ")}
          // Announce assistant output to screen readers as it arrives, but
          // politely - assertive would interrupt on every token.
          aria-live={!isUser && streaming ? "polite" : undefined}
        >
          {message.content}
        </div>
      </div>
    </div>
  );
}
