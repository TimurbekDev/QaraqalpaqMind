import type { Metadata, Viewport } from "next";
import { kaa } from "@/lib/strings";
import { THEME_BOOTSTRAP } from "@/lib/theme";
import "./globals.css";

export const metadata: Metadata = {
  title: kaa.appName,
  description: kaa.tagline,
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  // The composer is pinned to the bottom; without this the mobile keyboard
  // covers it.
  interactiveWidget: "resizes-content",
  // Matches the two surface colours, so the browser chrome and the overscroll
  // area follow the theme instead of staying white.
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#ffffff" },
    { media: "(prefers-color-scheme: dark)", color: "#0f1116" },
  ],
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  // `lang="kaa"` is the ISO 639-3 code for Karakalpak. It tells a screen reader
  // which language to pronounce and browsers which spellcheck rules to apply.
  return (
    <html lang="kaa" suppressHydrationWarning>
      <head>
        {/* Runs before first paint. Without it, anyone who chose dark gets a
            white flash on every load while React hydrates - and they are
            precisely the users most bothered by one. The content is a constant
            in lib/theme.ts, never user input. */}
        <script dangerouslySetInnerHTML={{ __html: THEME_BOOTSTRAP }} />
      </head>
      <body className="min-h-full">{children}</body>
    </html>
  );
}
