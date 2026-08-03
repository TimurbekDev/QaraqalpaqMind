"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { kaa } from "@/lib/strings";
import Icon from "./Icon";

interface Props {
  value: string;
  /** Icon only, for dense toolbars like a code-block header. */
  compact?: boolean;
}

export default function CopyButton({ value, compact = false }: Props) {
  const [copied, setCopied] = useState(false);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Clearing on unmount stops React warning about a state update on a gone
  // component - which happens whenever someone copies and immediately switches
  // conversation.
  useEffect(() => () => { if (timer.current) clearTimeout(timer.current); }, []);

  const copy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(value);
    } catch {
      // The Clipboard API needs a secure context, so it fails on plain http
      // from another machine - which is exactly how this gets tested first.
      // execCommand is deprecated but still the only fallback that works there.
      const area = document.createElement("textarea");
      area.value = value;
      area.setAttribute("readonly", "");
      area.style.position = "fixed";
      area.style.opacity = "0";
      document.body.appendChild(area);
      area.select();
      try {
        document.execCommand("copy");
      } catch {
        return;
      } finally {
        document.body.removeChild(area);
      }
    }

    setCopied(true);
    if (timer.current) clearTimeout(timer.current);
    timer.current = setTimeout(() => setCopied(false), 1600);
  }, [value]);

  const label = copied ? kaa.copied : kaa.copy;

  return (
    <button
      type="button"
      onClick={copy}
      aria-label={label}
      title={label}
      className="inline-flex h-8 items-center gap-1.5 rounded-lg px-2 text-xs
                 text-[var(--ink-muted)] transition-colors duration-150
                 hover:bg-[var(--surface-3)] hover:text-[var(--ink)]"
    >
      <Icon name={copied ? "check" : "copy"} size={13} />
      {!compact && <span>{label}</span>}
      {/* Announced on change, so a screen-reader user gets confirmation the
          icon swap alone would not give them. */}
      <span className="sr-only" role="status" aria-live="polite">
        {copied ? kaa.copied : ""}
      </span>
    </button>
  );
}
