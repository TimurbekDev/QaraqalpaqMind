"use client";

import { useEffect, useRef, useState } from "react";
import { kaa } from "@/lib/strings";

export default function CopyButton({ value, label }: { value: string; label?: string }) {
  const [copied, setCopied] = useState(false);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Clearing the timeout on unmount stops React warning about a state update
  // on a component that has gone - which happens whenever someone copies a
  // message and immediately starts a new chat.
  useEffect(() => () => { if (timer.current) clearTimeout(timer.current); }, []);

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(value);
      setCopied(true);
      if (timer.current) clearTimeout(timer.current);
      timer.current = setTimeout(() => setCopied(false), 1500);
    } catch {
      // Clipboard access needs a secure context: it fails on plain http from
      // another machine, which is exactly how this gets tested first.
      console.warn("Clipboard unavailable (needs HTTPS or localhost)");
    }
  };

  return (
    <button
      type="button"
      onClick={copy}
      aria-label={label ?? kaa.copy}
      title={copied ? kaa.copied : (label ?? kaa.copy)}
      className="inline-flex items-center gap-1 rounded-md px-1.5 py-1 text-[11px]
                 text-[var(--color-ink-muted)] transition-colors
                 hover:bg-[var(--color-line)] hover:text-[var(--color-ink)]"
    >
      {copied ? <CheckIcon /> : <CopyIcon />}
      <span>{copied ? kaa.copied : kaa.copy}</span>
    </button>
  );
}

function CopyIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor"
         strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <rect x="9" y="9" width="13" height="13" rx="2" />
      <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
    </svg>
  );
}

function CheckIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor"
         strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M20 6 9 17l-5-5" />
    </svg>
  );
}
