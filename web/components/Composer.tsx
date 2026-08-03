"use client";

import { useEffect, useRef, useState } from "react";
import { kaa } from "@/lib/strings";

interface Props {
  onSend: (text: string) => void;
  onStop: () => void;
  streaming: boolean;
}

const MAX_TEXTAREA_PX = 200;

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

  const submit = () => {
    if (!text.trim() || streaming) return;
    onSend(text);
    setText("");
  };

  return (
    <div className="border-t border-[var(--color-line)] bg-[var(--color-surface)] px-4 py-3">
      <form
        className="mx-auto flex max-w-3xl items-end gap-2"
        onSubmit={(event) => {
          event.preventDefault();
          submit();
        }}
      >
        <textarea
          ref={textareaRef}
          value={text}
          onChange={(event) => setText(event.target.value)}
          onKeyDown={(event) => {
            // Enter sends, Shift+Enter is a newline. Checking isComposing is
            // what keeps this from firing mid-word for anyone using an IME.
            if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
              event.preventDefault();
              submit();
            }
          }}
          rows={1}
          placeholder={kaa.placeholder}
          aria-label={kaa.placeholder}
          className="flex-1 resize-none rounded-xl border border-[var(--color-line)]
                     bg-[var(--color-surface)] px-3.5 py-2.5 text-[15px] outline-none
                     placeholder:text-[var(--color-ink-muted)]
                     focus:border-[var(--color-accent)]"
        />

        {streaming ? (
          <button
            type="button"
            onClick={onStop}
            className="rounded-xl border border-[var(--color-line)] px-4 py-2.5 text-sm font-medium
                       transition-colors hover:bg-[var(--color-surface-muted)]"
          >
            {kaa.stop}
          </button>
        ) : (
          <button
            type="submit"
            disabled={!text.trim()}
            className="rounded-xl bg-[var(--color-accent)] px-4 py-2.5 text-sm font-medium text-white
                       transition-opacity disabled:cursor-not-allowed disabled:opacity-40"
          >
            {kaa.send}
          </button>
        )}
      </form>

      <p className="mx-auto mt-2 max-w-3xl text-center text-xs text-[var(--color-ink-muted)]">
        {kaa.disclaimer}
      </p>
    </div>
  );
}
