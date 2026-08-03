"use client";

import { kaa, suggestions } from "@/lib/strings";

/**
 * The empty state does two jobs: say what this is, and show what to type.
 *
 * The suggestions are the important half. A blank box in a language with almost
 * no existing AI tooling gives a first-time user nothing to go on, and "ask me
 * anything" is not an answer - concrete examples teach the range of the thing
 * in one glance.
 */
export default function EmptyState({ onSuggestion }: { onSuggestion: (text: string) => void }) {
  return (
    <div className="min-h-0 flex-1 overflow-y-auto">
      <div className="mx-auto flex min-h-full w-full max-w-[var(--measure)] flex-col
                      items-center justify-center gap-8 px-4 py-10">
        <div className="text-center">
          <h2 className="text-[1.75rem] font-semibold tracking-tight">{kaa.emptyTitle}</h2>
          <p className="mt-1.5 text-[var(--ink-muted)]">{kaa.emptyBody}</p>
        </div>

        <ul className="grid w-full grid-cols-1 gap-2 sm:grid-cols-2">
          {suggestions.map((suggestion) => (
            <li key={suggestion.prompt}>
              <button
                type="button"
                onClick={() => onSuggestion(suggestion.prompt)}
                className="h-full w-full rounded-xl border border-[var(--line)]
                           bg-[var(--surface)] p-3.5 text-start transition-all duration-150
                           hover:-translate-y-px hover:border-[var(--line-strong)]
                           hover:bg-[var(--surface-2)] hover:shadow-sm"
              >
                <span className="block text-[11px] font-semibold uppercase tracking-wide
                                 text-[var(--ink-faint)]">
                  {suggestion.title}
                </span>
                <span className="mt-1 block text-sm leading-snug">{suggestion.prompt}</span>
              </button>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}
