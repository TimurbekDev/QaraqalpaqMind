# The chat interface

Phase 11: a Next.js 15 chat UI in `web/`, in Karakalpak.

```
browser ──> nginx ──> Next.js ──> gateway ──> vLLM
                       :3000       :8000       :8001
                       holds the
                       API key
```

## 1. The one architectural decision

**The browser never talks to the gateway.** It posts to `/api/chat`, a route
handler running in the Next.js server process, which attaches the API key and
pipes the response back.

The alternative — calling the gateway from `Chat.tsx` — needs the key in client
code, and there is no way to put it there safely. `NEXT_PUBLIC_*` variables are
inlined into the JavaScript bundle at build time, so the key would ship to every
visitor and be readable in devtools. This key buys GPU time.

Two tests enforce it, and both fail if the rule is broken:

- no `NEXT_PUBLIC_` anywhere in `web/`
- no `process.env` in any component; only `app/api/chat/route.ts` reads it

The route handler **rebuilds** the upstream request rather than forwarding the
client's body. Forwarding would let a caller set `model`, `max_tokens` or any
other field the gateway accepts, using the server's key to do it. Verified: a
request carrying `"model": "evil"` reaches the gateway as `"qaraqalpaqmind"`.

It also validates: at most 50 messages, 32,000 characters, and roles restricted
to `system`/`user`/`assistant`. Bad input gets a 400 with a reason.

## 2. Running it

```bash
cd web
cp .env.example .env.local     # set QM_API_KEY
npm install
npm run dev
```

With the gateway on `localhost:8000` (`qm serve` or the compose stack). The
**echo backend** is enough to develop the whole UI — it streams canned
Karakalpak replies with no GPU and no model.

In the stack:

```bash
cd deployment
docker compose up -d           # api + web + nginx
# open http://localhost
```

`QM_WEB_API_KEY` is separate from `QM_API_KEYS` on purpose. The latter is a
comma-separated list; passing it straight through would send
`Authorization: Bearer key-one,key-two` the moment a second key existed, and
every request would 401 with nothing in the logs pointing at the comma.

## 3. Streaming

`lib/stream.ts` reads the SSE body and yields text deltas. Two details there are
load-bearing:

**Buffer until a blank line.** A network chunk has no relationship to an SSE
event — one read can deliver half an event or three and a half. Parsing each
chunk as a complete frame works on localhost and corrupts output over a real
network, so it is a bug that passes every test on a laptop.

**`TextDecoder` with `stream: true`.** Karakalpak is full of two-byte
characters — á ó ú ǵ ń ı. A multi-byte character split across a chunk boundary
decodes to a replacement character unless the decoder carries state between
reads, and the corruption lands exactly on the letters that distinguish words.

Buffering must also be off at every hop. nginx buffers by default, and so would
the `/api/chat` route without `X-Accel-Buffering: no`. A buffered stream still
returns the correct answer, delivered in one lump when generation finishes —
which is indistinguishable from broken frontend code. Verified end to end
through both proxies: chunks arrive ~75 ms apart.

## 4. Design audit and what changed

The first version worked but was a single-conversation demo. Ranked by how much
each problem cost a real user:

| # | Problem | Why it mattered | Fix |
|---|---|---|---|
| 1 | One conversation only | Every new question destroyed the previous answer. No way back to anything. | Sidebar with conversation list, grouped Today / This week / Older |
| 2 | No way to find past work | With history, a flat list is unusable past ~20 items | Search over titles **and** message bodies, deferred so typing never blocks |
| 3 | Full-width text | Lines ran to the window edge; the eye loses its place returning to the next line | One `--measure` token, 46rem (~65 characters) |
| 4 | Nothing between "sent" and "first token" | On a cold model that gap is seconds and the page looks frozen | `status` distinguishes `waiting` from `streaming`: dots, then caret |
| 5 | A failed answer was a dead end | The error replaced the partial output and offered nothing | Error keeps partial text and adds Retry |
| 6 | Stop discarded the rest | Stopping mid-answer meant re-asking from scratch | Truncated turns are marked and offer **Continue** |
| 7 | No way to fix a typo'd question | Users retyped the whole prompt | Inline edit on user turns, re-asks from that point |
| 8 | Mouse-only | No shortcuts at all | ⌘K, ⌘⇧O, ⌘B, ⌘, `/`, `Esc`, `?` with a help dialog |
| 9 | No settings | Temperature and system prompt were hardcoded | Settings dialog, persisted |
| 10 | Theme was a cycling icon | Three states are one too many to discover by clicking | Segmented control showing all three |
| 11 | Code blocks unstyled | Unreadable for anything longer than a line | Header with language + copy, and highlighting |
| 12 | Tables not rendered | Markdown tables came out as pipes | Real `<table>`, scrolling in its own container |
| 13 | Icons drifted | Each component inlined its own SVGs at different stroke weights | One `Icon` component |
| 14 | Streaming re-rendered everything | Every token reparsed every markdown tree | `memo` on message, markdown and code block |

### Decisions worth stating

**User messages are bubbles; assistant messages are not.** The assistant writes
long-form prose, and a bubble around 400 words adds a box that constrains the
measure without adding information. The user's turns are short and benefit from
the visual anchor of "this is what I asked". This matches what ChatGPT and
Claude settled on, for the same reason.

**46rem measure, not full width.** Comfortable reading is roughly 60–75
characters per line. Wider looks like it uses the space better and is measurably
harder to read.

**No voice button, no file attachments, no citations.** There is no
speech-to-text endpoint, the model is text-only Qwen3, and retrieval is Phase 9.
A control that does nothing teaches users the product is broken — worse than an
absent feature. They go in when the backend exists.

**Hand-rolled syntax highlighting.** Shiki and Prism are correct and cost
200 kB–1 MB. The whole page is 116 kB. The tokenizer here handles comments,
strings, numbers and keywords for six languages and leaves unknown tokens plain
— miscoloured code is worse than uncoloured code.

**Native `<dialog>`.** `showModal()` gives focus trapping, Escape, page
inertness and correct semantics for free. A div-based modal reimplements all of
that, usually incompletely.

## 5. What the interface does

| Feature | Note |
|---|---|
| Conversation persistence | localStorage, validated on read. A reload losing everything is the worst thing a chat UI does, and mobile browsers evict background tabs constantly. Stays on the device — sending it anywhere would need a retention policy. |
| Markdown rendering | Written here rather than pulled in: react-markdown + remark + rehype is ~100 kB for a feature set a chat bubble does not use, on a page whose whole bundle is 108 kB. Cost of the hand-rolled one: 3 kB. |
| Copy button | On assistant messages and on every code block. |
| Regenerate | Drops the last assistant turn and re-asks from the same history. |
| Theme: system / light / dark | Stored, with an inline bootstrap script so there is no white flash before hydration for anyone who chose dark. |
| Typing indicator | An empty bubble during time-to-first-token looks frozen; on a cold model that is seconds. |
| Jump-to-latest | Appears only when scrolled away from the bottom. |
| Character counter | Shows from 80% of the server's 32,000 limit, so hitting it is not a surprise. |
| Focus return | The composer refocuses when a reply finishes. |
| Conversation search | Titles and message bodies, `useDeferredValue` so typing stays responsive. |
| Inline edit | On user turns; re-asks from that point and drops what followed. |
| Continue after stop | A stopped answer is marked and resumes into the same message. |
| Retry on error | Keeps whatever streamed before the failure. |
| Settings | Temperature, system prompt, Enter-to-send, theme, clear-all. |
| Keyboard | ⌘K search, ⌘⇧O new, ⌘B sidebar, ⌘, settings, `/` focus, `Esc` stop, `?` help. |

### Markdown safety

Everything builds React elements; nothing reaches `dangerouslySetInnerHTML`.
This matters more than usual because model output is partly determined by
whatever a user typed into it.

Links are restricted to `http:` and `https:`. A model can emit `javascript:` or
`data:` URLs — because it learned them, or because someone asked — and
rendering one as a clickable anchor turns model output into script execution.
Both render as inert text instead.

Verified by server-rendering a page of cases and inspecting the HTML: headings,
bold, italic, inline code, both list kinds, blockquote, fenced code with a
language label, an **unterminated** fence (the common case mid-stream), a
`https:` link with `rel="noopener noreferrer"`, and `javascript:`/`data:` links
producing no anchor. 16/16.

## 6. Interface details worth knowing

| Thing | Why |
|---|---|
| `lang="kaa"` on `<html>` | ISO 639-3 for Karakalpak. Tells screen readers what to pronounce and browsers which spellcheck rules apply. |
| No web font | A font loaded over the network that lacks `ǵ` or `ń` renders tofu for exactly the letters that carry meaning. System stacks cover Latin Extended-A. |
| Scroll only when pinned | Following the stream is right until the user scrolls up to read. Then it is not. |
| `scrollIntoView({behavior: "auto"})` | A smooth scroll queued per token never finishes, so the view falls further behind the faster the model streams. |
| `isComposing` check on Enter | Without it, Enter submits mid-word for anyone using an IME. |
| Abort is not an error | Pressing stop keeps whatever streamed so far; only real failures render as errors. |
| `AbortSignal` forwarded upstream | A cancelled generation stops occupying the GPU instead of running for a browser that has gone. |

## 7. The strings need a native speaker

Every user-visible string is in `web/lib/strings.ts`, collected there so a
native speaker can review the whole interface without reading React.

**They were written by a non-native speaker.** Treat them as a first draft, the
same as the SFT seed examples. The orthography is Latin 2016: `á ó ú ǵ ń` and
dotless `ı` are distinct letters, not decorated versions of `a o u g n i` — a
well-meant "fix" to plain ASCII changes the words. A test asserts these
characters are still present.

## 8. Verified

Run against the stack, not asserted:

| Check | Result |
|---|---|
| `npm run build` | clean, 116 kB first load |
| `npm run typecheck` | clean (`strict`, `noUncheckedIndexedAccess`) |
| UI through nginx | 200, `lang="kaa"`, correct 2016 orthography |
| CSS from the standalone image | 200 |
| API key in HTML or client bundle | **not present** |
| Streaming, nginx → Next.js → gateway | chunks ~75 ms apart |
| Invalid body / role / content type | 400 with a reason |
| `"model": "evil"` in the request | ignored; `qaraqalpaqmind` sent |
| Web image size | 78 MB |
| Accessibility in served HTML | skip link, `aria-expanded`/`aria-controls`, labelled search and composer |
| `temperature: 99` / `-5` | clamped server-side, 200 |
| 51 messages | 400 |
| Markdown renderer, SSR'd | 16/16, incl. `javascript:`/`data:` rendering inert |
| Theme, persistence, copy, regenerate | present in served HTML |

## 9. Not done yet

- No syntax highlighting inside code blocks.
- One conversation, not a list of them. Starting a new chat discards the old.
- No `/api/chat` rate limit of its own — it relies on nginx per-IP limits and
  the gateway per-key limits. Since the UI holds one key, all visitors share
  that key's budget. Give the UI its own key and its own quota before opening
  it to the public.
- The strings are unreviewed (§7).
