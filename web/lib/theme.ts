export type Theme = "system" | "light" | "dark";

export const THEME_KEY = "qm.theme";

/**
 * Applied to <html> as data-theme so CSS can override the media query.
 *
 * "system" writes no attribute at all rather than resolving to light or dark:
 * a resolved value would freeze at page load and stop following the OS when
 * the user switches at sunset.
 */
export function applyTheme(theme: Theme): void {
  const root = document.documentElement;
  if (theme === "system") {
    root.removeAttribute("data-theme");
  } else {
    root.setAttribute("data-theme", theme);
  }
}

export function loadTheme(): Theme {
  if (typeof window === "undefined") return "system";
  const stored = window.localStorage.getItem(THEME_KEY);
  return stored === "light" || stored === "dark" ? stored : "system";
}

export function saveTheme(theme: Theme): void {
  try {
    if (theme === "system") window.localStorage.removeItem(THEME_KEY);
    else window.localStorage.setItem(THEME_KEY, theme);
  } catch {
    /* private browsing */
  }
}

/**
 * Runs before first paint, inlined in <head>.
 *
 * Without it the page renders in the media-query theme and then swaps to the
 * stored one once React hydrates - a white flash on every load for anyone who
 * chose dark, which is exactly the users most bothered by a white flash.
 */
export const THEME_BOOTSTRAP = `
(function(){try{var t=localStorage.getItem("${THEME_KEY}");
if(t==="light"||t==="dark")document.documentElement.setAttribute("data-theme",t);}catch(e){}})();
`.trim();
