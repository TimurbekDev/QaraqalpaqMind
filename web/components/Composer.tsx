"use client";

import { useEffect, useRef, useState } from "react";
import { kaa } from "@/lib/strings";

interface Props {
  onSend: (text: string) => void;
  onStop: () => void;
  streaming: boolean;
}

const MAX_TEXTAREA_PX = 200;
/** Matches the server-side per-conversation cap in app/api/chat/route.ts. */
const MAX_CHARS = 32_000;
/** Only show the counter when it is close enough to matter. */
const COUNTER_FROM = MAX_CHARS * 0.8;

export default function Composer({ onSend, onStop, streaming }: Props) {
  const [text, setText] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Grow with the content up to a cap, then scroll inside. Reset to "auto"
  // first or the box can only ever get taller: scrollHeight includes the
  // height already set.
  useEffect(() => {
    const element = textareaRef.current;
    if (!element) return;
    element.style.height = "auto";
    element.style.height = `${Math.min(element.scrollHeight, MAX_TEXTAREA_PX)}px`;
  }, [text]);

  // Return focus once a reply finishes, so the next message can be typed
  // without reaching for the mouse.
  useEffect(() => {
    if (!streaming) textareaRef.current?.focus();
  }, [streaming]);

  const submit = () => {
    if (!text.trim() || streaming) return;
    onSend(text);
    setText("");
  };

  const overLimit = text.length > MAX_CHARS;

  return (
    <div className="shrink-0 border-t border-[var(--color-line)] bg-[var(--color-surface)] px-4 py-3">
      <form
        className="mx-auto flex max-w-3xl items-end gap-2"
        onSubmit={(event) => {
          event.preventDefault();
          submit();
        }}
      >
        <div className="relative flex-1">
          <textarea
            ref={textareaRef}
            value={text}
            onChange={(event) => setText(event.target.value)}
            onKeyDown={(event) => {
              // Enter sends, Shift+Enter is a newline. The isComposing check is
              // what keeps this from firing mid-word for anyone using an IME.
              if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
                event.preventDefault();
                submit();
              }
            }}
            rows={1}
            placeholder={kaa.placeholder}
            aria-label={kaa.placeholder}
            aria-invalid={overLimit || undefined}
            className={`w-full resize-none rounded-xl border bg-[var(--color-surface)]
                        px-3.5 py-2.5 text-[15px] outline-none
                        placeholder:text-[var(--color-ink-muted)]
                        ${overLimit
                          ? "border-red-500 focus:border-red-500"
                          : "border-[var(--color-line)] focus:border-[var(--color-accent)]"}`}
          />

          {text.length > COUNTER_FROM && (
            <span
              className={`pointer-events-none absolute bottom-1.5 end-2.5 text-[11px] tabular-nums
                          ${overLimit ? "text-red-500" : "text-[var(--color-ink-muted)]"}`}
            >
              {text.length.toLocaleString()} / {MAX_CHARS.toLocaleString()}
            </span>
          )}
        </div>

        {streaming ? (
          <button
            type="button"
            onClick={onStop}
            className="flex h-[46px] items-center gap-1.5 rounded-xl border
                       border-[var(--color-line)] px-4 text-sm font-medium transition-colors
                       hover:bg-[var(--color-surface-muted)]"
          >
            <StopIcon />
            <span className="hidden sm:inline">{kaa.stop}</span>
          </button>
        ) : (
          <button
            type="submit"
            disabled={!text.trim() || overLimit}
            aria-label={kaa.send}
            className="flex h-[46px] items-center gap-1.5 rounded-xl bg-[var(--color-accent)]
                       px-4 text-sm font-medium text-white transition-opacity
                       disabled:cursor-not-allowed disabled:opacity-40"
          >
            <SendIcon />
            <span className="hidden sm:inline">{kaa.send}</span>
          </button>
        )}
      </form>

      <p className="mx-auto mt-2 max-w-3xl text-center text-xs text-[var(--color-ink-muted)]">
        {kaa.disclaimer}
      </p>
    </div>
  );
}

function SendIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor"
         strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M12 19V5M5 12l7-7 7 7" />
    </svg>
  );
}

function StopIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
      <rect x="6" y="6" width="12" height="12" rx="2" />
    </svg>
  );
}
