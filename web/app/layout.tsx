import type { Metadata, Viewport } from "next";
import { kaa } from "@/lib/strings";
import "./globals.css";

export const metadata: Metadata = {
  title: kaa.appName,
  description: kaa.tagline,
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  // The composer is fixed to the bottom; without this the mobile keyboard
  // covers it.
  interactiveWidget: "resizes-content",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  // `lang="kaa"` is the ISO 639-3 code for Karakalpak. It is what tells a
  // screen reader which language to pronounce, and browsers which
  // hyphenation and spellcheck rules to apply.
  return (
    <html lang="kaa">
      <body className="min-h-full">{children}</body>
    </html>
  );
}
