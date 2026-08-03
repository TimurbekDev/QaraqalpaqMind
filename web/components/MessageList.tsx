"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { kaa, suggestions } from "@/lib/strings";
import type { Message } from "@/lib/types";
import CopyButton from "./CopyButton";
import Markdown from "./Markdown";

interface Props {
  messages: Message[];
  streaming: boolean;
  onSuggestion: (text: string) => void;
  onRegenerate: () => void;
  canRegenerate: boolean;
}

/** How close to the bottom still counts as "following along". */
const PINNED_PX = 80;

export default function MessageList({
  messages,
  streaming,
  onSuggestion,
  onRegenerate,
  canRegenerate,
}: Props) {
  const endRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const pinnedRef = useRef(true);
  const [showJump, setShowJump] = useState(false);

  const updatePinned = useCallback(() => {
    const container = containerRef.current;
    if (!container) return;
    const distance = container.scrollHeight - container.scrollTop - container.clientHeight;
    pinnedRef.current = distance < PINNED_PX;
    setShowJump(!pinnedRef.current);
  }, []);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    container.addEventListener("scroll", updatePinned, { passive: true });
    return () => container.removeEventListener("scroll", updatePinned);
  }, [updatePinned]);

  useEffect(() => {
    // Follow the stream only while the user is at the bottom. Yanking them back
    // down while they scroll up to reread is the most irritating thing a chat
    // UI can do.
    if (pinnedRef.current) endRef.current?.scrollIntoView({ behavior: "auto" });
  }, [messages]);

  const jumpToLatest = () => {
    pinnedRef.current = true;
    setShowJump(false);
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  if (messages.length === 0) {
    return (
      <div className="flex flex-1 flex-col items-center justify-center gap-7 overflow-y-auto px-4 py-8">
        <div className="text-center">
          <h2 className="text-2xl font-semibold">{kaa.emptyTitle}</h2>
          <p className="mt-1.5 text-sm text-[var(--color-ink-muted)]">{kaa.emptyBody}</p>
        </div>

        <div className="grid w-full max-w-xl grid-cols-1 gap-2 sm:grid-cols-2">
          {suggestions.map((suggestion) => (
            <button
              key={suggestion.prompt}
              type="button"
              onClick={() => onSuggestion(suggestion.prompt)}
              className="group rounded-xl border border-[var(--color-line)] p-3 text-start
                         transition-colors hover:border-[var(--color-accent)]
                         hover:bg-[var(--color-surface-muted)]"
            >
              <div className="text-xs font-medium text-[var(--color-ink-muted)]">
                {suggestion.title}
              </div>
              <div className="mt-0.5 text-sm">{suggestion.prompt}</div>
            </button>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="relative flex-1 overflow-hidden">
      <div ref={containerRef} className="h-full overflow-y-auto">
        <div className="mx-auto flex max-w-3xl flex-col gap-6 px-4 py-6">
          {messages.map((message, index) => (
            <Bubble
              key={message.id}
              message={message}
              // Only the last message can still be growing.
              streaming={streaming && index === messages.length - 1}
            />
          ))}

          {canRegenerate && (
            <div className="flex justify-center">
              <button
                type="button"
                onClick={onRegenerate}
                className="inline-flex items-center gap-1.5 rounded-lg border
                           border-[var(--color-line)] px-3 py-1.5 text-xs font-medium
                           text-[var(--color-ink-muted)] transition-colors
                           hover:bg-[var(--color-surface-muted)] hover:text-[var(--color-ink)]"
              >
                <RefreshIcon />
                {kaa.regenerate}
              </button>
            </div>
          )}

          <div ref={endRef} />
        </div>
      </div>

      {showJump && (
        <button
          type="button"
          onClick={jumpToLatest}
          aria-label={kaa.scrollToLatest}
          className="absolute bottom-4 left-1/2 -translate-x-1/2 rounded-full border
                     border-[var(--color-line)] bg-[var(--color-surface)] p-2 shadow-md
                     transition-colors hover:bg-[var(--color-surface-muted)]"
        >
          <ArrowDownIcon />
        </button>
      )}
    </div>
  );
}

function Bubble({ message, streaming }: { message: Message; streaming: boolean }) {
  const isUser = message.role === "user";

  if (message.error) {
    return (
      <div
        role="alert"
        className="rounded-xl border border-red-500/40 bg-red-500/5 px-4 py-3 text-sm
                   text-red-600 dark:text-red-400"
      >
        {message.error}
      </div>
    );
  }

  if (isUser) {
    return (
      <div className="flex justify-end">
        <div className="max-w-[85%] rounded-2xl bg-[var(--color-surface-muted)] px-4 py-2.5">
          <p className="whitespace-pre-wrap break-words text-[15px] leading-relaxed">
            {message.content}
          </p>
        </div>
      </div>
    );
  }

  // An empty assistant bubble means the request is out and no token has come
  // back. Without something here the UI looks frozen for the whole
  // time-to-first-token, which on a cold model is seconds.
  const waiting = streaming && message.content.length === 0;

  return (
    <div className="group/message flex flex-col gap-1.5">
      <div className="flex items-center gap-2">
        <span className="text-xs font-medium text-[var(--color-ink-muted)]">{kaa.assistant}</span>
        {!streaming && message.content.length > 0 && (
          <div className="opacity-0 transition-opacity group-hover/message:opacity-100 focus-within:opacity-100">
            <CopyButton value={message.content} />
          </div>
        )}
      </div>

      <div
        className={`flex flex-col gap-2.5 break-words text-[15px] leading-relaxed
                    ${streaming && !waiting ? "streaming-caret" : ""}`}
        // Announce output as it arrives, politely - assertive would interrupt
        // the screen reader on every token.
        aria-live={streaming ? "polite" : undefined}
        aria-busy={streaming || undefined}
      >
        {waiting ? <TypingDots /> : <Markdown text={message.content} />}
      </div>
    </div>
  );
}

function TypingDots() {
  return (
    <div className="flex items-center gap-1 py-1" aria-label={kaa.thinking}>
      {[0, 1, 2].map((i) => (
        <span
          key={i}
          className="typing-dot h-1.5 w-1.5 rounded-full bg-[var(--color-ink-muted)]"
          style={{ animationDelay: `${i * 0.15}s` }}
        />
      ))}
    </div>
  );
}

function RefreshIcon() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor"
         strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M21 12a9 9 0 1 1-3-6.7L21 8" />
      <path d="M21 3v5h-5" />
    </svg>
  );
}

function ArrowDownIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor"
         strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M12 5v14M19 12l-7 7-7-7" />
    </svg>
  );
}
