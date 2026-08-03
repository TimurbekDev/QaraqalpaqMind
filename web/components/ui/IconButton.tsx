"use client";

import { forwardRef, type ButtonHTMLAttributes } from "react";

interface Props extends ButtonHTMLAttributes<HTMLButtonElement> {
  /** Required. These buttons are icon-only, so nothing else names them. */
  label: string;
  variant?: "ghost" | "outline";
  /** Visible text beside the icon, on wide screens only. */
  text?: string;
}

/**
 * The one icon-only button.
 *
 * `label` is not optional because an icon button with no accessible name is
 * unusable with a screen reader, and that is easy to forget per-call-site.
 * Minimum 36px so it clears the 24px WCAG 2.2 target-size floor comfortably
 * and stays usable with a thumb.
 */
const IconButton = forwardRef<HTMLButtonElement, Props>(function IconButton(
  { label, variant = "ghost", text, className = "", children, ...rest },
  ref,
) {
  const base =
    "inline-flex h-9 min-w-9 items-center justify-center gap-1.5 rounded-lg px-2 text-sm " +
    "transition-colors duration-150 disabled:cursor-not-allowed disabled:opacity-40";

  const variants = {
    ghost: "text-[var(--ink-muted)] hover:bg-[var(--surface-3)] hover:text-[var(--ink)]",
    outline:
      "border border-[var(--line)] text-[var(--ink)] hover:bg-[var(--surface-2)]",
  };

  return (
    <button
      ref={ref}
      type="button"
      aria-label={label}
      title={label}
      className={`${base} ${variants[variant]} ${className}`}
      {...rest}
    >
      {children}
      {text && <span className="hidden pe-0.5 sm:inline">{text}</span>}
    </button>
  );
});

export default IconButton;
