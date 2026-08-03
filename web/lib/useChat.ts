"use client";

import { useCallback, useRef, useState } from "react";
import { readDeltas } from "./stream";
import { kaa } from "./strings";
import { newId, type Message, type Settings } from "./types";

export type ChatStatus = "idle" | "waiting" | "streaming";

interface Options {
  settings: Settings;
  /** Persist a new message array for the conversation being edited. */
  commit: (messages: Message[]) => void;
}

/**
 * Owns one in-flight generation.
 *
 * `status` distinguishes "request sent, nothing back yet" from "tokens
 * arriving". The UI needs both: the first shows a typing indicator, the second
 * a caret, and collapsing them into a boolean makes time-to-first-token look
 * like a frozen page.
 */
export function useChat({ settings, commit }: Options) {
  const [status, setStatus] = useState<ChatStatus>("idle");
  const abortRef = useRef<AbortController | null>(null);
  // The live message array. State updates are batched and a stream produces
  // many per second, so the generation reads and writes this instead.
  const workingRef = useRef<Message[]>([]);

  const busy = status !== "idle";

  const stop = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
  }, []);

  const run = useCallback(
    async (history: Message[], replyId: string, seed = "") => {
      setStatus("waiting");
      const controller = new AbortController();
      abortRef.current = controller;

      let text = seed;
      let received = false;

      const flush = (extra: Partial<Message> = {}) => {
        workingRef.current = workingRef.current.map((m) =>
          m.id === replyId ? { ...m, content: text, ...extra } : m,
        );
        commit(workingRef.current);
      };

      try {
        const response = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            messages: history.map(({ role, content }) => ({ role, content })),
            temperature: settings.temperature,
            system: settings.systemPrompt || undefined,
          }),
          signal: controller.signal,
        });

        if (!response.ok || !response.body) {
          flush({ error: errorFor(response.status) });
          return;
        }

        for await (const delta of readDeltas(response.body, controller.signal)) {
          if (!received) {
            received = true;
            setStatus("streaming");
          }
          text += delta;
          flush();
        }
        flush({ error: undefined, truncated: false });
      } catch (error) {
        if ((error as Error).name === "AbortError") {
          // The user pressed stop. Keep what arrived and mark it so the UI can
          // offer "continue" instead of treating it as a failure.
          flush({ truncated: true });
        } else {
          console.error(error);
          flush({ error: kaa.errorUnreachable });
        }
      } finally {
        abortRef.current = null;
        setStatus("idle");
      }
    },
    [settings.temperature, settings.systemPrompt, commit],
  );

  /** Append a user turn and generate a reply. */
  const send = useCallback(
    (current: Message[], text: string) => {
      const trimmed = text.trim();
      if (!trimmed || abortRef.current) return;

      const user: Message = { id: newId(), role: "user", content: trimmed, at: Date.now() };
      const reply: Message = { id: newId(), role: "assistant", content: "", at: Date.now() };
      const history = [...current, user];

      workingRef.current = [...history, reply];
      commit(workingRef.current);
      void run(history, reply.id);
    },
    [commit, run],
  );

  /** Discard the last assistant turn and ask again from the same history. */
  const regenerate = useCallback(
    (current: Message[]) => {
      if (abortRef.current) return;
      const index = lastIndexOf(current, (m) => m.role === "assistant");
      if (index < 0) return;

      const history = current.slice(0, index);
      const reply: Message = { id: newId(), role: "assistant", content: "", at: Date.now() };
      workingRef.current = [...history, reply];
      commit(workingRef.current);
      void run(history, reply.id);
    },
    [commit, run],
  );

  /** Keep generating into the same message after a stop. */
  const continueGeneration = useCallback(
    (current: Message[]) => {
      if (abortRef.current) return;
      const index = lastIndexOf(current, (m) => m.role === "assistant");
      const target = current[index];
      if (!target) return;

      workingRef.current = current.map((m) =>
        m.id === target.id ? { ...m, truncated: false } : m,
      );
      commit(workingRef.current);
      // The partial answer goes back as history so the model continues rather
      // than restarting, and `seed` keeps the text already on screen.
      void run(current.slice(0, index + 1), target.id, target.content);
    },
    [commit, run],
  );

  /** Re-send after an error, reusing the same assistant slot. */
  const retry = useCallback(
    (current: Message[]) => {
      if (abortRef.current) return;
      const index = lastIndexOf(current, (m) => m.role === "assistant");
      const target = current[index];
      if (!target) return;

      workingRef.current = current.map((m) =>
        m.id === target.id ? { ...m, error: undefined, content: "" } : m,
      );
      commit(workingRef.current);
      void run(current.slice(0, index), target.id);
    },
    [commit, run],
  );

  /** Rewrite a user turn and drop everything after it. */
  const editAndResend = useCallback(
    (current: Message[], messageId: string, content: string) => {
      if (abortRef.current) return;
      const index = current.findIndex((m) => m.id === messageId);
      if (index < 0) return;

      const history = [
        ...current.slice(0, index),
        { ...current[index]!, content: content.trim(), at: Date.now() },
      ];
      const reply: Message = { id: newId(), role: "assistant", content: "", at: Date.now() };
      workingRef.current = [...history, reply];
      commit(workingRef.current);
      void run(history, reply.id);
    },
    [commit, run],
  );

  return {
    status,
    busy,
    stop,
    send,
    regenerate,
    continueGeneration,
    retry,
    editAndResend,
    setWorking: (messages: Message[]) => {
      workingRef.current = messages;
    },
  };
}

function lastIndexOf<T>(items: T[], predicate: (item: T) => boolean): number {
  for (let i = items.length - 1; i >= 0; i--) if (predicate(items[i] as T)) return i;
  return -1;
}

function errorFor(status: number): string {
  if (status === 429) return kaa.errorRateLimited;
  if (status === 502 || status === 503) return kaa.errorUnreachable;
  return kaa.errorGeneric;
}
