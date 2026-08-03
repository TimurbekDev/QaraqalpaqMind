"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import Icon from "@/components/ui/Icon";
import { kaa } from "@/lib/strings";
import type { Message } from "@/lib/types";
import type { ChatStatus } from "@/lib/useChat";
import EmptyState from "./EmptyState";
import MessageItem from "./MessageItem";

interface Props {
  messages: Message[];
  status: ChatStatus;
  onSuggestion: (text: string) => void;
  onRegenerate: () => void;
  onRetry: () => void;
  onContinue: () => void;
  onEdit: (id: string, content: string) => void;
}

/** How close to the bottom still counts as "following along". */
const PINNED_PX = 96;

export default function MessageList({
  messages, status, onSuggestion, onRegenerate, onRetry, onContinue, onEdit,
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const endRef = useRef<HTMLDivElement>(null);
  const pinnedRef = useRef(true);
  const [showJump, setShowJump] = useState(false);

  const busy = status !== "idle";

  const onScroll = useCallback(() => {
    const container = containerRef.current;
    if (!container) return;
    const distance = container.scrollHeight - container.scrollTop - container.clientHeight;
    pinnedRef.current = distance < PINNED_PX;
    setShowJump(!pinnedRef.current && container.scrollHeight > container.clientHeight * 1.2);
  }, []);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    container.addEventListener("scroll", onScroll, { passive: true });
    return () => container.removeEventListener("scroll", onScroll);
  }, [onScroll]);

  useEffect(() => {
    // Follow the stream only while the user is at the bottom. Pulling them back
    // down mid-read is the most irritating thing a chat UI does.
    //
    // "auto", not "smooth": a smooth scroll re-queued on every token never
    // completes, so the view falls further behind the faster the model streams.
    if (pinnedRef.current) endRef.current?.scrollIntoView({ behavior: "auto" });
  }, [messages]);

  const jump = () => {
    pinnedRef.current = true;
    setShowJump(false);
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  if (messages.length === 0) {
    return <EmptyState onSuggestion={onSuggestion} />;
  }

  const lastIndex = messages.length - 1;

  return (
    <div className="relative min-h-0 flex-1">
      <div ref={containerRef} className="h-full overflow-y-auto overscroll-contain">
        <div className="mx-auto flex w-full max-w-[var(--measure)] flex-col gap-7 px-4 py-6">
          {messages.map((message, index) => {
            const isLast = index === lastIndex;
            const streaming = busy && isLast;
            return (
              <MessageItem
                key={message.id}
                message={message}
                streaming={streaming}
                waiting={streaming && status === "waiting" && !message.content}
                isLast={isLast}
                busy={busy}
                onRegenerate={onRegenerate}
                onRetry={onRetry}
                onContinue={onContinue}
                onEdit={onEdit}
              />
            );
          })}
          {/* Space below the last turn so the newest message is not welded to
              the composer while reading. */}
          <div ref={endRef} className="h-4" />
        </div>
      </div>

      {showJump && (
        <button
          type="button"
          onClick={jump}
          aria-label={kaa.scrollToLatest}
          title={kaa.scrollToLatest}
          className="absolute bottom-4 left-1/2 flex h-9 w-9 -translate-x-1/2 items-center
                     justify-center rounded-full border border-[var(--line)]
                     bg-[var(--surface)] text-[var(--ink-muted)] shadow-lg
                     transition-colors hover:bg-[var(--surface-2)] hover:text-[var(--ink)]"
        >
          <Icon name="arrowDown" />
        </button>
      )}
    </div>
  );
}
