"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  loadActiveId,
  loadConversations,
  saveActiveId,
  saveConversations,
} from "./storage";
import { deriveTitle, newId, type Conversation, type Message } from "./types";

/**
 * Owns the conversation list and which one is open.
 *
 * Kept out of the components so the chat surface stays presentational and the
 * persistence rules live in one place. A reducer would be the next step if this
 * grows another few operations; at eight it is still easier to read as
 * functions.
 */
export function useConversations(untitledLabel: string) {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [hydrated, setHydrated] = useState(false);

  // Restored after mount. localStorage does not exist on the server, and
  // seeding state from it during render would make the two markups disagree.
  useEffect(() => {
    const stored = loadConversations();
    setConversations(stored);

    const wanted = loadActiveId();
    setActiveId(stored.some((c) => c.id === wanted) ? wanted : (stored[0]?.id ?? null));
    setHydrated(true);
  }, []);

  // Never write before hydration: the first render has an empty list, and
  // persisting that would erase everything the user had.
  useEffect(() => {
    if (hydrated) saveConversations(conversations);
  }, [conversations, hydrated]);

  useEffect(() => {
    if (hydrated) saveActiveId(activeId);
  }, [activeId, hydrated]);

  const active = useMemo(
    () => conversations.find((c) => c.id === activeId) ?? null,
    [conversations, activeId],
  );

  const touch = useCallback((id: string, messages: Message[]) => {
    setConversations((prev) =>
      prev.map((c) =>
        c.id === id
          ? {
              ...c,
              messages,
              // Title is derived once, from the first user turn. Recomputing it
              // on every update would rename a conversation under the user
              // whenever they edited their opening message.
              title: c.title || deriveTitle(messages, ""),
              updatedAt: Date.now(),
            }
          : c,
      ),
    );
  }, []);

  const create = useCallback((): string => {
    const now = Date.now();
    const conversation: Conversation = {
      id: newId(),
      title: "",
      messages: [],
      createdAt: now,
      updatedAt: now,
    };
    setConversations((prev) => [conversation, ...prev]);
    setActiveId(conversation.id);
    return conversation.id;
  }, []);

  /** Reuse the open conversation if it is still empty, rather than stacking blanks. */
  const startNew = useCallback((): string => {
    if (active && active.messages.length === 0) return active.id;
    return create();
  }, [active, create]);

  const remove = useCallback((id: string) => {
    setConversations((prev) => {
      const next = prev.filter((c) => c.id !== id);
      setActiveId((current) => (current === id ? (next[0]?.id ?? null) : current));
      return next;
    });
  }, []);

  const rename = useCallback((id: string, title: string) => {
    setConversations((prev) =>
      prev.map((c) => (c.id === id ? { ...c, title: title.trim().slice(0, 80) } : c)),
    );
  }, []);

  const clearAll = useCallback(() => {
    setConversations([]);
    setActiveId(null);
  }, []);

  // Sorted for display without reordering the stored array on every render.
  const ordered = useMemo(
    () => [...conversations].sort((a, b) => b.updatedAt - a.updatedAt),
    [conversations],
  );

  const titleFor = useCallback(
    (c: Conversation) => c.title || deriveTitle(c.messages, untitledLabel),
    [untitledLabel],
  );

  return {
    conversations: ordered,
    active,
    activeId,
    hydrated,
    setActiveId,
    startNew,
    create,
    remove,
    rename,
    clearAll,
    touch,
    titleFor,
  };
}

/** Groups conversations into Today / This week / Older for the sidebar. */
export function useGrouped(
  conversations: Conversation[],
  labels: { today: string; week: string; older: string },
) {
  return useMemo(() => {
    const now = Date.now();
    const day = 86_400_000;
    const groups: { label: string; items: Conversation[] }[] = [
      { label: labels.today, items: [] },
      { label: labels.week, items: [] },
      { label: labels.older, items: [] },
    ];

    for (const c of conversations) {
      const age = now - c.updatedAt;
      const bucket = age < day ? 0 : age < day * 7 ? 1 : 2;
      groups[bucket]?.items.push(c);
    }

    return groups.filter((g) => g.items.length > 0);
  }, [conversations, labels]);
}

/** Stable ref to the newest value, for callbacks that must not re-create. */
export function useLatest<T>(value: T) {
  const ref = useRef(value);
  ref.current = value;
  return ref;
}
