"use client";

import { useEffect, useState } from "react";
import { kaa } from "@/lib/strings";
import Dialog from "./ui/Dialog";

/** Modifier symbol, resolved on the client so a Mac shows ⌘ and Windows Ctrl. */
function useModifier(): string {
  const [mod, setMod] = useState("Ctrl");
  useEffect(() => {
    // navigator.platform is deprecated but the replacement is not universal;
    // this only picks a label, so a wrong guess is cosmetic.
    if (/Mac|iPhone|iPad/.test(navigator.platform)) setMod("⌘");
  }, []);
  return mod;
}

export default function ShortcutsDialog({
  open, onClose,
}: {
  open: boolean;
  onClose: () => void;
}) {
  const mod = useModifier();

  const rows: [string, string][] = [
    [`${mod} + K`, kaa.shortcutSearch],
    [`${mod} + Shift + O`, kaa.shortcutNew],
    [`${mod} + B`, kaa.shortcutSidebar],
    [`${mod} + ,`, kaa.shortcutSettings],
    ["/", kaa.shortcutFocus],
    ["Esc", kaa.shortcutStop],
    ["?", kaa.shortcutShortcuts],
  ];

  return (
    <Dialog open={open} onClose={onClose} title={kaa.shortcuts}>
      <dl className="flex flex-col">
        {rows.map(([keys, description]) => (
          <div
            key={keys}
            className="flex items-center justify-between gap-4 border-b border-[var(--line)]
                       py-2.5 last:border-0"
          >
            <dt className="text-sm text-[var(--ink-muted)]">{description}</dt>
            <dd className="flex shrink-0 gap-1">
              {keys.split(" + ").map((key) => (
                <kbd
                  key={key}
                  className="rounded-md border border-[var(--line)] bg-[var(--surface-2)]
                             px-1.5 py-0.5 text-[11px] font-medium text-[var(--ink)]"
                >
                  {key}
                </kbd>
              ))}
            </dd>
          </div>
        ))}
      </dl>
    </Dialog>
  );
}
