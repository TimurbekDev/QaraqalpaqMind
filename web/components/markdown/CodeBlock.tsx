"use client";

import { memo, useMemo } from "react";
import { TOKEN_CLASS, tokenize } from "@/lib/highlight";
import CopyButton from "../ui/CopyButton";

/**
 * Memoised, because tokenizing runs on every render and a streaming message
 * re-renders on every token. Without this, a long code block makes the whole
 * stream stutter as it grows.
 */
function CodeBlock({ language, code }: { language: string; code: string }) {
  const tokens = useMemo(() => tokenize(code, language), [code, language]);

  return (
    <div className="my-3 overflow-hidden rounded-xl border border-[var(--line)] bg-[var(--surface-2)]">
      <div className="flex items-center justify-between border-b border-[var(--line)] py-1.5 pe-1.5 ps-3">
        <span className="font-mono text-[11px] uppercase tracking-wide text-[var(--ink-faint)]">
          {language || "text"}
        </span>
        <CopyButton value={code} compact />
      </div>
      {/* Long lines scroll inside the block. The page body must never scroll
          sideways, and min-w-0 is what lets this shrink inside a flex parent. */}
      <pre className="min-w-0 overflow-x-auto p-3.5 text-[13px] leading-[1.6]">
        <code className="font-mono">
          {tokens.map((token, i) => (
            <span key={i} className={TOKEN_CLASS[token.kind]}>
              {token.text}
            </span>
          ))}
        </code>
      </pre>
    </div>
  );
}

export default memo(CodeBlock);
