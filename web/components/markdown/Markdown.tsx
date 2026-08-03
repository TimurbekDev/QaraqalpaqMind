"use client";

import { Fragment, memo, type ReactNode } from "react";
import CodeBlock from "./CodeBlock";

/**
 * A small markdown renderer for model output.
 *
 * Written rather than pulled in: react-markdown plus remark plus rehype is
 * ~100 kB for a feature set a chat bubble does not use, on a page whose whole
 * bundle is ~110 kB.
 *
 * Everything here builds React elements. Nothing reaches
 * dangerouslySetInnerHTML, so model output cannot inject markup no matter what
 * it emits - which matters more than usual here, because that output is partly
 * determined by whatever a user typed in.
 *
 * Memoised on `text`: a streaming message re-renders on every token, and
 * reparsing a long answer each time is what makes streaming feel janky.
 */
function Markdown({ text }: { text: string }) {
  return <>{renderBlocks(text)}</>;
}

export default memo(Markdown);

function renderBlocks(text: string): ReactNode[] {
  const blocks: ReactNode[] = [];
  const lines = text.split("\n");
  let index = 0;
  let key = 0;

  while (index < lines.length) {
    const line = lines[index] ?? "";

    // Fenced code. The closing fence is usually missing mid-stream, so an
    // unterminated block runs to the end rather than swallowing the render.
    const fence = /^\s*```(\w*)\s*$/.exec(line);
    if (fence) {
      const language = fence[1] ?? "";
      const body: string[] = [];
      index++;
      while (index < lines.length && !/^\s*```\s*$/.test(lines[index] ?? "")) {
        body.push(lines[index] ?? "");
        index++;
      }
      index++;
      blocks.push(<CodeBlock key={key++} language={language} code={body.join("\n")} />);
      continue;
    }

    if (!line.trim()) {
      index++;
      continue;
    }

    // Table: a header row, a separator of dashes, then body rows.
    if (isTableRow(line) && isTableSeparator(lines[index + 1] ?? "")) {
      const header = splitRow(line);
      index += 2;
      const rows: string[][] = [];
      while (index < lines.length && isTableRow(lines[index] ?? "")) {
        rows.push(splitRow(lines[index] ?? ""));
        index++;
      }
      blocks.push(<Table key={key++} header={header} rows={rows} />);
      continue;
    }

    const heading = /^(#{1,4})\s+(.*)$/.exec(line);
    if (heading) {
      const level = (heading[1] ?? "#").length;
      const sizes = [
        "text-[1.35rem] font-semibold mt-5",
        "text-[1.15rem] font-semibold mt-5",
        "text-[1rem] font-semibold mt-4",
        "text-[0.95rem] font-semibold mt-4",
      ];
      blocks.push(
        <p key={key++} className={`${sizes[level - 1] ?? sizes[3]} first:mt-0`}>
          {inline(heading[2] ?? "")}
        </p>,
      );
      index++;
      continue;
    }

    // A horizontal rule, but only on its own line - otherwise "---" inside a
    // sentence disappears.
    if (/^\s*([-*_])\s*\1\s*\1[\s\-*_]*$/.test(line)) {
      blocks.push(<hr key={key++} className="my-4 border-[var(--line)]" />);
      index++;
      continue;
    }

    if (/^\s*>\s?/.test(line)) {
      const quoted: string[] = [];
      while (index < lines.length && /^\s*>\s?/.test(lines[index] ?? "")) {
        quoted.push((lines[index] ?? "").replace(/^\s*>\s?/, ""));
        index++;
      }
      blocks.push(
        <blockquote
          key={key++}
          className="my-3 border-s-[3px] border-[var(--accent)] bg-[var(--surface-2)]
                     py-2 pe-3 ps-4 text-[var(--ink-muted)]"
        >
          {inline(quoted.join(" "))}
        </blockquote>,
      );
      continue;
    }

    const bullet = /^\s*[-*+]\s+/;
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
          className={`my-2 flex flex-col gap-1.5 ps-5 ${ordered ? "list-decimal" : "list-disc"}`}
        >
          {items.map((item, i) => (
            <li key={i} className="ps-1 marker:text-[var(--ink-faint)]">
              {inline(item)}
            </li>
          ))}
        </List>,
      );
      continue;
    }

    const paragraph: string[] = [];
    while (index < lines.length) {
      const current = lines[index] ?? "";
      if (!current.trim() || startsBlock(current, lines[index + 1] ?? "")) break;
      paragraph.push(current);
      index++;
    }
    blocks.push(
      <p key={key++} className="whitespace-pre-wrap">
        {inline(paragraph.join("\n"))}
      </p>,
    );
  }

  return blocks;
}

function startsBlock(line: string, next: string): boolean {
  return (
    /^\s*```/.test(line) ||
    /^#{1,4}\s/.test(line) ||
    /^\s*>\s?/.test(line) ||
    /^\s*[-*+]\s+/.test(line) ||
    /^\s*\d+[.)]\s+/.test(line) ||
    (isTableRow(line) && isTableSeparator(next))
  );
}

function isTableRow(line: string): boolean {
  return /^\s*\|.*\|\s*$/.test(line);
}

function isTableSeparator(line: string): boolean {
  return /^\s*\|[\s:|-]+\|\s*$/.test(line) && line.includes("-");
}

function splitRow(line: string): string[] {
  return line.trim().replace(/^\||\|$/g, "").split("|").map((cell) => cell.trim());
}

function Table({ header, rows }: { header: string[]; rows: string[][] }) {
  return (
    // Wide tables scroll inside their own container. Letting the page scroll
    // sideways instead breaks every other block on the screen.
    <div className="my-3 overflow-x-auto rounded-xl border border-[var(--line)]">
      <table className="w-full border-collapse text-sm">
        <thead>
          <tr className="bg-[var(--surface-2)]">
            {header.map((cell, i) => (
              <th
                key={i}
                scope="col"
                className="border-b border-[var(--line)] px-3 py-2 text-start font-semibold"
              >
                {inline(cell)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, r) => (
            <tr key={r} className="border-b border-[var(--line)] last:border-0">
              {row.map((cell, c) => (
                <td key={c} className="px-3 py-2 align-top">
                  {inline(cell)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** Inline spans: `code`, **bold**, *italic*, ~~strike~~, and http(s) links. */
function inline(text: string): ReactNode {
  const pattern =
    /(`[^`\n]+`)|(\*\*[^*\n]+\*\*)|(~~[^~\n]+~~)|(\*[^*\n]+\*)|(\[[^\]\n]*\]\([^)\s]+\))/g;

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
          className="rounded-md border border-[var(--line)] bg-[var(--surface-2)]
                     px-[0.3em] py-[0.1em] font-mono text-[0.875em]"
        >
          {token.slice(1, -1)}
        </code>,
      );
    } else if (token.startsWith("**")) {
      parts.push(<strong key={key++} className="font-semibold">{token.slice(2, -2)}</strong>);
    } else if (token.startsWith("~~")) {
      parts.push(<s key={key++} className="text-[var(--ink-faint)]">{token.slice(2, -2)}</s>);
    } else if (token.startsWith("*")) {
      parts.push(<em key={key++}>{token.slice(1, -1)}</em>);
    } else {
      const link = /^\[([^\]]*)\]\(([^)\s]+)\)$/.exec(token);
      parts.push(link ? renderLink(link[1] ?? "", link[2] ?? "", key++) : token);
    }

    last = match.index + token.length;
  }

  if (last < text.length) parts.push(text.slice(last));
  return parts.map((part, i) => <Fragment key={i}>{part}</Fragment>);
}

function renderLink(label: string, href: string, key: number): ReactNode {
  // http and https only. A model can emit javascript: or data: URLs - because
  // it learned them, or because someone asked it to - and rendering one as a
  // clickable anchor turns model output into script execution.
  let safe = false;
  try {
    safe = ["http:", "https:"].includes(new URL(href).protocol);
  } catch {
    safe = false;
  }

  if (!safe) return <span key={key}>{label || href}</span>;

  return (
    <a
      key={key}
      href={href}
      target="_blank"
      // noreferrer alongside noopener: without it the opened page can read
      // where it came from, and older browsers need noopener spelled out.
      rel="noopener noreferrer"
      className="text-[var(--accent)] underline decoration-[var(--accent)]/40 underline-offset-2
                 transition-colors hover:decoration-[var(--accent)]"
    >
      {label || href}
    </a>
  );
}
