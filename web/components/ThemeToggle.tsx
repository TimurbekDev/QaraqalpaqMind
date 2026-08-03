"use client";

import { useEffect, useState } from "react";
import { kaa } from "@/lib/strings";
import { applyTheme, loadTheme, saveTheme, type Theme } from "@/lib/theme";

const ORDER: Theme[] = ["system", "light", "dark"];

export default function ThemeToggle() {
  // Always "system" on the first render. Reading localStorage during render
  // would make the server and client markup disagree and break hydration; the
  // inline script in <head> has already applied the real theme to <html>, so
  // there is no flash while this catches up.
  const [theme, setTheme] = useState<Theme>("system");
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setTheme(loadTheme());
    setMounted(true);
  }, []);

  const cycle = () => {
    const next = ORDER[(ORDER.indexOf(theme) + 1) % ORDER.length] ?? "system";
    setTheme(next);
    saveTheme(next);
    applyTheme(next);
  };

  const label = {
    system: kaa.themeSystem,
    light: kaa.themeLight,
    dark: kaa.themeDark,
  }[theme];

  return (
    <button
      type="button"
      onClick={cycle}
      aria-label={`${kaa.theme}: ${label}`}
      title={`${kaa.theme}: ${label}`}
      className="inline-flex h-8 w-8 items-center justify-center rounded-lg border
                 border-[var(--color-line)] text-[var(--color-ink-muted)] transition-colors
                 hover:bg-[var(--color-surface-muted)] hover:text-[var(--color-ink)]"
    >
      {/* Before mount the stored theme is unknown, so render the neutral icon
          rather than guessing and swapping it a frame later. */}
      {!mounted || theme === "system" ? <SystemIcon /> : theme === "light" ? <SunIcon /> : <MoonIcon />}
    </button>
  );
}

const svg = {
  width: 16,
  height: 16,
  viewBox: "0 0 24 24",
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 2,
  strokeLinecap: "round",
  strokeLinejoin: "round",
  "aria-hidden": true,
} as const;

function SystemIcon() {
  return (
    <svg {...svg}>
      <rect x="2" y="3" width="20" height="14" rx="2" />
      <path d="M8 21h8M12 17v4" />
    </svg>
  );
}

function SunIcon() {
  return (
    <svg {...svg}>
      <circle cx="12" cy="12" r="4" />
      <path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />
    </svg>
  );
}

function MoonIcon() {
  return (
    <svg {...svg}>
      <path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z" />
    </svg>
  );
}
