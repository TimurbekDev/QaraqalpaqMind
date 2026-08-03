"use client";

import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from "react";
import Icon from "@/components/ui/Icon";
import { kaa } from "@/lib/strings";
import type { ChatStatus } from "@/lib/useChat";

interface Props {
  onSend: (text: string) => void;
  onStop: () => void;
  status: ChatStatus;
  sendOnEnter: boolean;
}

export interface ComposerHandle {
  focus: () => void;
}

const MAX_HEIGHT_PX = 220;
/** Matches the per-conversation cap enforced in app/api/chat/route.ts. */
const MAX_CHARS = 32_000;
const COUNTER_FROM = MAX_CHARS * 0.8;

const Composer = forwardRef<ComposerHandle, Props>(function Composer(
  { onSend, onStop, status, sendOnEnter },
  ref,
) {
  const [text, setText] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const busy = status !== "idle";

  useImperativeHandle(ref, () => ({ focus: () => textareaRef.current?.focus() }), []);

  // Grow with the content up to a cap, then scroll inside. Reset to "auto"
  // first, or the box only ever gets taller: scrollHeight includes the height
  // already set.
  useEffect(() => {
    const element = textareaRef.current;
    if (!element) return;
    element.style.height = "auto";
    element.style.height = `${Math.min(element.scrollHeight, MAX_HEIGHT_PX)}px`;
  }, [text]);

  // Return focus when a reply finishes, so the next message can be typed
  // without reaching for the mouse.
  useEffect(() => {
    if (!busy) textareaRef.current?.focus();
  }, [busy]);

  const overLimit = text.length > MAX_CHARS;
  const canSend = Boolean(text.trim()) && !overLimit && !busy;

  const submit = () => {
    if (!canSend) return;
    onSend(text);
    setText("");
  };

  return (
    <div className="shrink-0 bg-gradient-to-t from-[var(--surface)] via-[var(--surface)] to-transparent
                    px-4 pb-3 pt-2">
      <div className="mx-auto w-full max-w-[var(--measure)]">
        <form
          onSubmit={(event) => {
            event.preventDefault();
            submit();
          }}
          className={`flex items-end gap-2 rounded-2xl border bg-[var(--surface)] p-2
                      shadow-sm transition-colors
                      ${overLimit
                        ? "border-[var(--danger)]"
                        : "border-[var(--line)] focus-within:border-[var(--accent)]"}`}
        >
          <textarea
            ref={textareaRef}
            value={text}
            onChange={(event) => setText(event.target.value)}
            onKeyDown={(event) => {
              if (event.key !== "Enter") return;
              // isComposing is what keeps this from firing mid-word for anyone
              // using an IME - the keystroke that commits a candidate is also
              // an Enter.
              if (event.nativeEvent.isComposing) return;

              const wantsSend = sendOnEnter ? !event.shiftKey : event.metaKey || event.ctrlKey;
              if (wantsSend) {
                event.preventDefault();
                submit();
              }
            }}
            rows={1}
            placeholder={kaa.placeholder}
            aria-label={kaa.placeholder}
            aria-invalid={overLimit || undefined}
            // The textarea is the app's primary control; the browser should not
            // autocapitalise or autocorrect a language it does not know.
            autoCapitalize="sentences"
            spellCheck={false}
            className="max-h-[220px] min-h-[24px] flex-1 resize-none bg-transparent px-2 py-1.5
                       text-[15px] leading-relaxed outline-none
                       placeholder:text-[var(--ink-faint)]"
          />

          {busy ? (
            <button
              type="button"
              onClick={onStop}
              aria-label={kaa.stop}
              title={kaa.stop}
              className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl
                         border border-[var(--line)] text-[var(--ink)]
                         transition-colors hover:bg-[var(--surface-2)]"
            >
              <Icon name="stop" size={13} />
            </button>
          ) : (
            <button
              type="submit"
              disabled={!canSend}
              aria-label={kaa.send}
              title={kaa.send}
              className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl
                         bg-[var(--accent)] text-[var(--accent-ink)] transition-all
                         disabled:cursor-not-allowed disabled:bg-[var(--surface-4)]
                         disabled:text-[var(--ink-faint)]"
            >
              <Icon name="send" size={16} />
            </button>
          )}
        </form>

        <div className="flex items-center justify-between gap-3 px-1 pt-1.5">
          <p className="text-[11px] leading-tight text-[var(--ink-faint)]">{kaa.disclaimer}</p>
          {text.length > COUNTER_FROM && (
            <span
              className={`shrink-0 text-[11px] tabular-nums
                          ${overLimit ? "text-[var(--danger)]" : "text-[var(--ink-faint)]"}`}
              role={overLimit ? "alert" : undefined}
            >
              {text.length.toLocaleString()} / {MAX_CHARS.toLocaleString()}
            </span>
          )}
        </div>
      </div>
    </div>
  );
});

export default Composer;
