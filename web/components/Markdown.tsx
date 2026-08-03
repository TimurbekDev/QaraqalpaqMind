"use client";

import { Fragment, type ReactNode } from "react";
import CopyButton from "./CopyButton";

/**
 * A small markdown renderer for model output.
 *
 * Written rather than pulled in because the alternative is react-markdown plus
 * remark plus rehype - about 100 kB for a feature set an LLM chat bubble does
 * not use, on a page whose whole bundle is currently 105 kB.
 *
 * Everything here builds React elements. Nothing reaches
 * dangerouslySetInnerHTML, so model output cannot inject markup no matter what
 * it emits - which matters more than usual here, because the model's output is
 * partly determined by whatever a user typed into it.
 */
export default function Markdown({ text }: { text: string }) {
  return <>{renderBlocks(text)}</>;
}

function renderBlocks(text: string): ReactNode[] {
  const blocks: ReactNode[] = [];
  const lines = text.split("\n");
  let index = 0;
  let key = 0;

  while (index < lines.length) {
    const line = lines[index] ?? "";

    // Fenced code. The closing fence may be missing entirely - it usually is,
    // mid-stream - so an unterminated block runs to the end rather than
    // swallowing the render.
    const fence = /^```(\w*)\s*$/.exec(line);
    if (fence) {
      const language = fence[1] ?? "";
      const body: string[] = [];
      index++;
      while (index < lines.length && !/^```\s*$/.test(lines[index] ?? "")) {
        body.push(lines[index] ?? "");
        index++;
      }
      index++; // past the closing fence, if there was one
      blocks.push(<CodeBlock key={key++} language={language} code={body.join("\n")} />);
      continue;
    }

    if (!line.trim()) {
      index++;
      continue;
    }

    const heading = /^(#{1,3})\s+(.*)$/.exec(line);
    if (heading) {
      const level = (heading[1] ?? "#").length;
      const content = inline(heading[2] ?? "");
      const sizes = ["text-lg font-semibold", "text-base font-semibold", "text-sm font-semibold"];
      blocks.push(
        <p key={key++} className={`${sizes[level - 1] ?? sizes[2]} mt-3 first:mt-0`}>
          {content}
        </p>,
      );
      index++;
      continue;
    }

    if (/^>\s?/.test(line)) {
      const quoted: string[] = [];
      while (index < lines.length && /^>\s?/.test(lines[index] ?? "")) {
        quoted.push((lines[index] ?? "").replace(/^>\s?/, ""));
        index++;
      }
      blocks.push(
        <blockquote
          key={key++}
          className="border-s-2 border-[var(--color-line)] ps-3 text-[var(--color-ink-muted)]"
        >
          {inline(quoted.join(" "))}
        </blockquote>,
      );
      continue;
    }

    const bullet = /^\s*[-*]\s+/;
    const numbered = /^\s*\d+[.)]\s+/;
    if (bullet.test(line) || numbered.test(line)) {
      const ordered = numbered.test(line);
      const pattern = ordered ? numbered : bullet;
      const items: string[] = [];
      while (index < lines.length && pattern.test(lines[index] ?? "")) {
        items.push((lines[index] ?? "").replace(pattern, ""));
        index++;
      }
      const List = ordered ? "ol" : "ul";
      blocks.push(
        <List
          key={key++}
          className={`ms-5 flex flex-col gap-1 ${ordered ? "list-decimal" : "list-disc"}`}
        >
          {items.map((item, i) => (
            <li key={i}>{inline(item)}</li>
          ))}
        </List>,
      );
      continue;
    }

    // Paragraph: consecutive non-blank lines that start no other block.
    const paragraph: string[] = [];
    while (index < lines.length) {
      const current = lines[index] ?? "";
      if (
        !current.trim() ||
        /^```/.test(current) ||
        /^#{1,3}\s/.test(current) ||
        /^>\s?/.test(current) ||
        bullet.test(current) ||
        numbered.test(current)
      ) {
        break;
      }
      paragraph.push(current);
      index++;
    }
    blocks.push(<p key={key++}>{inline(paragraph.join("\n"))}</p>);
  }

  return blocks;
}

function CodeBlock({ language, code }: { language: string; code: string }) {
  return (
    <div className="group relative my-1 overflow-hidden rounded-lg border border-[var(--color-line)]">
      <div className="flex items-center justify-between bg-[var(--color-surface-muted)] px-3 py-1.5">
        <span className="font-mono text-[11px] text-[var(--color-ink-muted)]">
          {language || "code"}
        </span>
        <CopyButton value={code} />
      </div>
      {/* Long lines scroll inside the block. Without the min-w-0 the flex
          parent refuses to shrink and the whole page scrolls sideways. */}
      <pre className="min-w-0 overflow-x-auto p-3">
        <code className="font-mono text-[13px] leading-relaxed">{code}</code>
      </pre>
    </div>
  );
}

/** Inline spans: `code`, **bold**, *italic*, and http(s) links. */
function inline(text: string): ReactNode {
  const pattern =
    /(`[^`\n]+`)|(\*\*[^*\n]+\*\*)|(\*[^*\n]+\*)|(\[[^\]\n]+\]\([^)\s]+\))/g;

  const parts: ReactNode[] = [];
  let last = 0;
  let key = 0;
  let match: RegExpExecArray | null;

  while ((match = pattern.exec(text)) !== null) {
    if (match.index > last) parts.push(text.slice(last, match.index));
    const token = match[0];

    if (token.startsWith("`")) {
      parts.push(
        <code
          key={key++}
          className="rounded bg-[var(--color-surface-muted)] px-1 py-0.5 font-mono text-[13px]"
        >
          {token.slice(1, -1)}
        </code>,
      );
    } else if (token.startsWith("**")) {
      parts.push(<strong key={key++}>{token.slice(2, -2)}</strong>);
    } else if (token.startsWith("*")) {
      parts.push(<em key={key++}>{token.slice(1, -1)}</em>);
    } else {
      const link = /^\[([^\]]+)\]\(([^)\s]+)\)$/.exec(token);
      parts.push(link ? renderLink(link[1] ?? "", link[2] ?? "", key++) : token);
    }

    last = match.index + token.length;
  }

  if (last < text.length) parts.push(text.slice(last));
  return parts.map((part, i) => <Fragment key={i}>{part}</Fragment>);
}

function renderLink(label: string, href: string, key: number): ReactNode {
  // http and https only. A model can emit javascript: or data: URLs - either
  // because it learned them or because a user asked it to - and rendering one
  // as a clickable anchor turns model output into script execution.
  let safe = false;
  try {
    safe = ["http:", "https:"].includes(new URL(href).protocol);
  } catch {
    safe = false;
  }

  if (!safe) return <span key={key}>{label}</span>;

  return (
    <a
      key={key}
      href={href}
      target="_blank"
      // noreferrer alongside noopener: without it the opened page can read
      // where it came from, and older browsers need noopener spelled out.
      rel="noopener noreferrer"
      className="text-[var(--color-accent)] underline underline-offset-2"
    >
      {label}
    </a>
  );
}
