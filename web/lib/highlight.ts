/**
 * A minimal syntax tokenizer for code blocks.
 *
 * Shiki and Prism are the right answer for a documentation site: correct
 * grammars, hundreds of languages, and 200 kB-1 MB of JavaScript. This page's
 * entire bundle is around 110 kB, and code in a Karakalpak chat is incidental,
 * so paying ten times the page weight for it is the wrong trade.
 *
 * What this does instead: comments, strings, numbers and a keyword list, over
 * a handful of languages. It will not colour every token correctly. It is
 * deliberately conservative - an unrecognised token is left plain rather than
 * guessed, because miscoloured code is worse than uncoloured code.
 */

export type TokenKind = "plain" | "comment" | "string" | "number" | "keyword" | "function";

export interface Token {
  kind: TokenKind;
  text: string;
}

/** Space-separated for readability; split once per call. */
const KEYWORDS: Record<string, string> = {
  python: "def class return if elif else for while in not and or import from as try except finally raise with lambda None True False async await yield pass break continue global assert del is",
  javascript: "function class return if else for while in of not new import from as export default const let var try catch finally throw async await yield break continue typeof instanceof null undefined true false this extends super",
  typescript: "function class return if else for while in of new import from as export default const let var try catch finally throw async await yield break continue typeof instanceof null undefined true false this extends super interface type enum implements readonly public private protected",
  bash: "if then else elif fi for while do done case esac function return export local source echo cd exit set unset",
  json: "true false null",
  sql: "select from where insert update delete into values join left right inner outer on group by order having limit as and or not null create table drop alter index",
};

const ALIASES: Record<string, string> = {
  py: "python", js: "javascript", jsx: "javascript", ts: "typescript", tsx: "typescript",
  sh: "bash", shell: "bash", zsh: "bash", console: "bash", yml: "yaml",
};

const LINE_COMMENT: Record<string, string> = {
  python: "#", bash: "#", yaml: "#", javascript: "//", typescript: "//", sql: "--",
};

export function tokenize(code: string, language: string): Token[] {
  const lang = ALIASES[language.toLowerCase()] ?? language.toLowerCase();
  const keywords = new Set(KEYWORDS[lang]?.split(" ") ?? []);
  const commentMarker = LINE_COMMENT[lang];

  // Unknown language: return one plain token rather than applying another
  // language's rules, which is how `#` in a C file ends up grey.
  if (!commentMarker && keywords.size === 0) return [{ kind: "plain", text: code }];

  const tokens: Token[] = [];
  let buffer = "";
  let index = 0;

  const flush = () => {
    if (!buffer) return;
    tokens.push({ kind: keywords.has(buffer) ? "keyword" : "plain", text: buffer });
    buffer = "";
  };

  while (index < code.length) {
    const char = code[index]!;

    if (commentMarker && code.startsWith(commentMarker, index)) {
      flush();
      const end = code.indexOf("\n", index);
      const stop = end === -1 ? code.length : end;
      tokens.push({ kind: "comment", text: code.slice(index, stop) });
      index = stop;
      continue;
    }

    if (char === '"' || char === "'" || char === "`") {
      flush();
      let end = index + 1;
      // Walk past escaped quotes so "he said \"hi\"" stays one string.
      while (end < code.length && code[end] !== char) {
        if (code[end] === "\\") end++;
        end++;
      }
      tokens.push({ kind: "string", text: code.slice(index, Math.min(end + 1, code.length)) });
      index = end + 1;
      continue;
    }

    if (/\d/.test(char) && !/[\w]/.test(code[index - 1] ?? "")) {
      flush();
      let end = index;
      while (end < code.length && /[\d.xXa-fA-F_]/.test(code[end]!)) end++;
      tokens.push({ kind: "number", text: code.slice(index, end) });
      index = end;
      continue;
    }

    if (/[\w$]/.test(char)) {
      buffer += char;
      index++;
      continue;
    }

    flush();
    tokens.push({ kind: "plain", text: char });
    index++;
  }

  flush();
  return tokens;
}

export const TOKEN_CLASS: Record<TokenKind, string> = {
  plain: "",
  comment: "text-[var(--code-comment)]",
  string: "text-[var(--code-string)]",
  number: "text-[var(--code-number)]",
  keyword: "text-[var(--code-keyword)]",
  function: "text-[var(--code-function)]",
};
