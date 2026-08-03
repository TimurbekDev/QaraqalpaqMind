"use client";

import { kaa } from "@/lib/strings";
import { applyTheme, saveTheme, type Theme } from "@/lib/theme";
import type { Settings } from "@/lib/types";
import Dialog from "./ui/Dialog";
import Icon from "./ui/Icon";

interface Props {
  open: boolean;
  onClose: () => void;
  settings: Settings;
  onChange: (settings: Settings) => void;
  theme: Theme;
  onThemeChange: (theme: Theme) => void;
  onClearAll: () => void;
}

const THEMES: { value: Theme; label: string; icon: "monitor" | "sun" | "moon" }[] = [
  { value: "system", label: kaa.themeSystem, icon: "monitor" },
  { value: "light", label: kaa.themeLight, icon: "sun" },
  { value: "dark", label: kaa.themeDark, icon: "moon" },
];

export default function SettingsDialog({
  open, onClose, settings, onChange, theme, onThemeChange, onClearAll,
}: Props) {
  const set = <K extends keyof Settings>(key: K, value: Settings[K]) =>
    onChange({ ...settings, [key]: value });

  return (
    <Dialog open={open} onClose={onClose} title={kaa.settings}>
      <div className="flex flex-col gap-6">
        <Field label={kaa.theme}>
          {/* A segmented control rather than a cycling button: three states are
              one too many to discover by clicking, and this shows all of them. */}
          <div role="radiogroup" aria-label={kaa.theme}
               className="flex gap-1 rounded-xl bg-[var(--surface-2)] p-1">
            {THEMES.map((option) => (
              <button
                key={option.value}
                type="button"
                role="radio"
                aria-checked={theme === option.value}
                onClick={() => {
                  onThemeChange(option.value);
                  saveTheme(option.value);
                  applyTheme(option.value);
                }}
                className={`flex h-9 flex-1 items-center justify-center gap-1.5 rounded-lg
                            text-sm transition-colors
                            ${theme === option.value
                              ? "bg-[var(--surface)] font-medium shadow-sm"
                              : "text-[var(--ink-muted)] hover:text-[var(--ink)]"}`}
              >
                <Icon name={option.icon} size={14} />
                {option.label}
              </button>
            ))}
          </div>
        </Field>

        <Field label={kaa.temperature} hint={kaa.temperatureHint}>
          <div className="flex items-center gap-3">
            <input
              type="range"
              min={0}
              max={2}
              step={0.1}
              value={settings.temperature}
              onChange={(event) => set("temperature", Number(event.target.value))}
              aria-label={kaa.temperature}
              aria-valuetext={settings.temperature.toFixed(1)}
              className="h-1.5 flex-1 cursor-pointer appearance-none rounded-full
                         bg-[var(--surface-4)] accent-[var(--accent)]"
            />
            <output className="w-8 text-end text-sm tabular-nums text-[var(--ink-muted)]">
              {settings.temperature.toFixed(1)}
            </output>
          </div>
        </Field>

        <Field label={kaa.systemPrompt} hint={kaa.systemPromptHint}>
          <textarea
            value={settings.systemPrompt}
            onChange={(event) => set("systemPrompt", event.target.value.slice(0, 4000))}
            rows={3}
            aria-label={kaa.systemPrompt}
            className="w-full resize-y rounded-xl border border-[var(--line)]
                       bg-[var(--surface)] px-3 py-2 text-sm outline-none
                       focus:border-[var(--accent)]"
          />
        </Field>

        <Field label={kaa.sendOnEnter} hint={kaa.sendOnEnterHint}>
          <button
            type="button"
            role="switch"
            aria-checked={settings.sendOnEnter}
            aria-label={kaa.sendOnEnter}
            onClick={() => set("sendOnEnter", !settings.sendOnEnter)}
            className={`relative h-6 w-11 rounded-full transition-colors
                        ${settings.sendOnEnter ? "bg-[var(--accent)]" : "bg-[var(--surface-4)]"}`}
          >
            <span
              className={`absolute top-0.5 h-5 w-5 rounded-full bg-white shadow transition-all
                          ${settings.sendOnEnter ? "start-[1.375rem]" : "start-0.5"}`}
            />
          </button>
        </Field>

        <div className="border-t border-[var(--line)] pt-5">
          <h3 className="text-xs font-semibold uppercase tracking-wide text-[var(--danger)]">
            {kaa.dangerZone}
          </h3>
          <button
            type="button"
            onClick={() => { if (confirm(kaa.clearAllConfirm)) { onClearAll(); onClose(); } }}
            className="mt-2.5 inline-flex h-9 items-center gap-2 rounded-lg border
                       border-[var(--danger)]/40 px-3 text-sm text-[var(--danger)]
                       transition-colors hover:bg-[var(--danger-soft)]"
          >
            <Icon name="trash" size={14} />
            {kaa.clearAll}
          </button>
        </div>
      </div>
    </Dialog>
  );
}

function Field({
  label, hint, children,
}: {
  label: string;
  hint?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="flex flex-col gap-2">
      <div>
        <p className="text-sm font-medium">{label}</p>
        {hint && <p className="mt-0.5 text-xs text-[var(--ink-muted)]">{hint}</p>}
      </div>
      {children}
    </div>
  );
}
