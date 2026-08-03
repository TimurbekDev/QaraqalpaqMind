"use client";

import { useEffect, useRef, type ReactNode } from "react";
import { kaa } from "@/lib/strings";
import Icon from "./Icon";
import IconButton from "./IconButton";

interface Props {
  open: boolean;
  onClose: () => void;
  title: string;
  children: ReactNode;
}

/**
 * Built on the native <dialog> element.
 *
 * `showModal()` gives focus trapping, Escape-to-close, inertness of the page
 * behind, and correct screen-reader semantics - all of which a div-based modal
 * has to reimplement, and usually reimplements incompletely. The tradeoff is
 * styling the ::backdrop instead of rendering one, which is a fair trade.
 */
export default function Dialog({ open, onClose, title, children }: Props) {
  const ref = useRef<HTMLDialogElement>(null);

  useEffect(() => {
    const dialog = ref.current;
    if (!dialog) return;

    if (open && !dialog.open) dialog.showModal();
    else if (!open && dialog.open) dialog.close();
  }, [open]);

  // Escape fires `cancel`/`close` on the element itself, not through React, so
  // the parent's state has to be told separately or it reopens on next render.
  useEffect(() => {
    const dialog = ref.current;
    if (!dialog) return;
    const handle = () => onClose();
    dialog.addEventListener("close", handle);
    return () => dialog.removeEventListener("close", handle);
  }, [onClose]);

  return (
    <dialog
      ref={ref}
      aria-labelledby="dialog-title"
      className="m-auto w-[min(32rem,calc(100vw-2rem))] rounded-2xl border border-[var(--line)]
                 bg-[var(--surface)] p-0 text-[var(--ink)] shadow-2xl
                 backdrop:bg-black/40 backdrop:backdrop-blur-[2px]"
      // A click on the backdrop lands on the dialog element itself, since the
      // content is a child. Comparing the target is what distinguishes them.
      onClick={(event) => {
        if (event.target === ref.current) onClose();
      }}
    >
      <div className="flex items-center justify-between border-b border-[var(--line)] px-5 py-3.5">
        <h2 id="dialog-title" className="text-sm font-semibold">
          {title}
        </h2>
        <IconButton label={kaa.close} onClick={onClose}>
          <Icon name="close" />
        </IconButton>
      </div>
      <div className="max-h-[70vh] overflow-y-auto px-5 py-4">{children}</div>
    </dialog>
  );
}
