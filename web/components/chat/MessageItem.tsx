"use client";

import { memo, useEffect, useRef, useState } from "react";
import Markdown from "@/components/markdown/Markdown";
import CopyButton from "@/components/ui/CopyButton";
import Icon from "@/components/ui/Icon";
import IconButton from "@/components/ui/IconButton";
import { kaa } from "@/lib/strings";
import type { Message } from "@/lib/types";

interface Props {
  message: Message;
  /** True only for the last message while a generation is running. */
  streaming: boolean;
  /** True before the first token arrives. */
  waiting: boolean;
  isLast: boolean;
  busy: boolean;
  onRegenerate: () => void;
  onRetry: () => void;
  onContinue: () => void;
  onEdit: (id: string, content: string) => void;
}

/**
 * Memoised on identity plus the few flags that change its appearance.
 *
 * Every token appends to the last message, which re-renders the whole list.
 * Without this, a 40-message conversation reparses 40 markdown trees per token
 * and the stream visibly stutters. Only the streaming message actually changes.
 */
function MessageItem({
  message, streaming, waiting, isLast, busy,
  onRegenerate, onRetry, onContinue, onEdit,
}: Props) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(message.content);
  const editRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (!editing) return;
    const element = editRef.current;
    if (!element) return;
    element.focus();
    element.setSelectionRange(element.value.length, element.value.length);
    element.style.height = "auto";
    element.style.height = `${element.scrollHeight}px`;
  }, [editing]);

  // --- User turn ---------------------------------------------------------

  if (message.role === "user") {
    if (editing) {
      return (
        <div className="flex justify-end">
          <div className="w-full max-w-[46rem] rounded-2xl border border-[var(--accent)]
                          bg-[var(--surface)] p-2">
            <textarea
              ref={editRef}
              value={draft}
              onChange={(event) => {
                setDraft(event.target.value);
                event.target.style.height = "auto";
                event.target.style.height = `${event.target.scrollHeight}px`;
              }}
              onKeyDown={(event) => {
                if (event.key === "Escape") setEditing(false);
                if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
                  onEdit(message.id, draft);
                  setEditing(false);
                }
              }}
              rows={1}
              aria-label={kaa.edit}
              className="w-full resize-none bg-transparent px-2 py-1 text-[15px] outline-none"
            />
            <div className="flex justify-end gap-1.5 pt-1">
              <button
                type="button"
                onClick={() => { setDraft(message.content); setEditing(false); }}
                className="h-8 rounded-lg px-3 text-xs text-[var(--ink-muted)]
                           transition-colors hover:bg-[var(--surface-3)]"
              >
                {kaa.cancel}
              </button>
              <button
                type="button"
                onClick={() => { onEdit(message.id, draft); setEditing(false); }}
                disabled={!draft.trim()}
                className="h-8 rounded-lg bg-[var(--accent)] px-3 text-xs font-medium
                           text-[var(--accent-ink)] transition-opacity disabled:opacity-40"
              >
                {kaa.save}
              </button>
            </div>
          </div>
        </div>
      );
    }

    return (
      <div className="group/msg flex animate-in flex-col items-end gap-1">
        <div className="max-w-[85%] rounded-2xl rounded-ee-md bg-[var(--surface-3)] px-4 py-2.5">
          <p className="whitespace-pre-wrap break-words">{message.content}</p>
        </div>
        <div className="flex opacity-0 transition-opacity
                        group-hover/msg:opacity-100 focus-within:opacity-100">
          <IconButton
            label={kaa.edit}
            onClick={() => { setDraft(message.content); setEditing(true); }}
            disabled={busy}
            className="h-8 min-w-8"
          >
            <Icon name="edit" size={13} />
          </IconButton>
          <CopyButton value={message.content} compact />
        </div>
      </div>
    );
  }

  // --- Assistant turn ----------------------------------------------------

  if (message.error) {
    return (
      <div
        role="alert"
        className="flex animate-in items-start gap-3 rounded-xl border border-[var(--danger)]/35
                   bg-[var(--danger-soft)] px-4 py-3"
      >
        <span className="mt-0.5 shrink-0 text-[var(--danger)]">
          <Icon name="alert" size={16} />
        </span>
        <div className="min-w-0 flex-1">
          <p className="text-sm text-[var(--danger)]">{message.error}</p>
          {/* Partial output is kept, not discarded - it may be all the user
              needed, and throwing it away to show an error loses their answer. */}
          {message.content && (
            <div className="prose mt-2 break-words text-[var(--ink)]">
              <Markdown text={message.content} />
            </div>
          )}
          <button
            type="button"
            onClick={onRetry}
            disabled={busy}
            className="mt-2 inline-flex h-8 items-center gap-1.5 rounded-lg border
                       border-[var(--line)] bg-[var(--surface)] px-3 text-xs font-medium
                       transition-colors hover:bg-[var(--surface-2)] disabled:opacity-40"
          >
            <Icon name="refresh" size={13} />
            {kaa.retry}
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="group/msg flex animate-in flex-col gap-1.5">
      <div className="flex items-center gap-2">
        <span className="text-xs font-semibold text-[var(--ink-muted)]">{kaa.assistant}</span>
        {message.truncated && !streaming && (
          <span className="rounded-md bg-[var(--surface-3)] px-1.5 py-0.5 text-[11px]
                           text-[var(--ink-faint)]">
            {kaa.stopped}
          </span>
        )}
      </div>

      <div
        className={`prose min-w-0 break-words ${streaming && !waiting ? "streaming-caret" : ""}`}
        // Announced politely as it arrives; assertive would interrupt the
        // screen reader on every token.
        aria-live={streaming ? "polite" : undefined}
        aria-busy={streaming || undefined}
      >
        {waiting ? <TypingDots /> : <Markdown text={message.content} />}
      </div>

      {!streaming && message.content && (
        <div
          className={`flex items-center gap-0.5 pt-0.5 transition-opacity
                      ${isLast ? "" : "opacity-0 group-hover/msg:opacity-100 focus-within:opacity-100"}`}
          aria-label={kaa.srMessageActions}
        >
          <CopyButton value={message.content} />
          {isLast && (
            <>
              <IconButton
                label={kaa.regenerate}
                text={kaa.regenerate}
                onClick={onRegenerate}
                disabled={busy}
                className="text-xs"
              >
                <Icon name="refresh" size={13} />
              </IconButton>
              {message.truncated && (
                <IconButton
                  label={kaa.continueGen}
                  text={kaa.continueGen}
                  onClick={onContinue}
                  disabled={busy}
                  className="text-xs"
                >
                  <Icon name="play" size={12} />
                </IconButton>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
}

function TypingDots() {
  return (
    <div className="flex items-center gap-1.5 py-1.5" role="status" aria-label={kaa.thinking}>
      {[0, 1, 2].map((i) => (
        <span
          key={i}
          className="typing-dot h-2 w-2 rounded-full bg-[var(--ink-faint)]"
          style={{ animationDelay: `${i * 0.16}s` }}
        />
      ))}
    </div>
  );
}

export default memo(MessageItem);
