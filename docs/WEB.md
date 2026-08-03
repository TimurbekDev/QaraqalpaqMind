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

## 4. What the interface does

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

## 5. Interface details worth knowing

| Thing | Why |
|---|---|
| `lang="kaa"` on `<html>` | ISO 639-3 for Karakalpak. Tells screen readers what to pronounce and browsers which spellcheck rules apply. |
| No web font | A font loaded over the network that lacks `ǵ` or `ń` renders tofu for exactly the letters that carry meaning. System stacks cover Latin Extended-A. |
| Scroll only when pinned | Following the stream is right until the user scrolls up to read. Then it is not. |
| `scrollIntoView({behavior: "auto"})` | A smooth scroll queued per token never finishes, so the view falls further behind the faster the model streams. |
| `isComposing` check on Enter | Without it, Enter submits mid-word for anyone using an IME. |
| Abort is not an error | Pressing stop keeps whatever streamed so far; only real failures render as errors. |
| `AbortSignal` forwarded upstream | A cancelled generation stops occupying the GPU instead of running for a browser that has gone. |

## 6. The strings need a native speaker

Every user-visible string is in `web/lib/strings.ts`, collected there so a
native speaker can review the whole interface without reading React.

**They were written by a non-native speaker.** Treat them as a first draft, the
same as the SFT seed examples. The orthography is Latin 2016: `á ó ú ǵ ń` and
dotless `ı` are distinct letters, not decorated versions of `a o u g n i` — a
well-meant "fix" to plain ASCII changes the words. A test asserts these
characters are still present.

## 7. Verified

Run against the stack, not asserted:

| Check | Result |
|---|---|
| `npm run build` | clean, 108 kB first load |
| `npm run typecheck` | clean (`strict`, `noUncheckedIndexedAccess`) |
| UI through nginx | 200, `lang="kaa"`, correct 2016 orthography |
| CSS from the standalone image | 200 |
| API key in HTML or client bundle | **not present** |
| Streaming, nginx → Next.js → gateway | chunks ~75 ms apart |
| Invalid body / role / content type | 400 with a reason |
| `"model": "evil"` in the request | ignored; `qaraqalpaqmind` sent |
| Web image size | 78 MB |
| Markdown renderer, SSR'd | 16/16, incl. `javascript:`/`data:` rendering inert |
| Theme, persistence, copy, regenerate | present in served HTML |

## 8. Not done yet

- No syntax highlighting inside code blocks.
- One conversation, not a list of them. Starting a new chat discards the old.
- No `/api/chat` rate limit of its own — it relies on nginx per-IP limits and
  the gateway per-key limits. Since the UI holds one key, all visitors share
  that key's budget. Give the UI its own key and its own quota before opening
  it to the public.
- The strings are unreviewed (§6).
