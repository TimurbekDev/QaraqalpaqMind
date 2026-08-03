"use client";

import { useDeferredValue, useEffect, useMemo, useRef, useState } from "react";
import { kaa } from "@/lib/strings";
import type { Conversation } from "@/lib/types";
import { useGrouped } from "@/lib/useConversations";
import Icon from "../ui/Icon";
import IconButton from "../ui/IconButton";

interface Props {
  conversations: Conversation[];
  activeId: string | null;
  open: boolean;
  titleFor: (c: Conversation) => string;
  onSelect: (id: string) => void;
  onNew: () => void;
  onDelete: (id: string) => void;
  onRename: (id: string, title: string) => void;
  onClose: () => void;
  onOpenSettings: () => void;
}

export default function Sidebar({
  conversations, activeId, open, titleFor,
  onSelect, onNew, onDelete, onRename, onClose, onOpenSettings,
}: Props) {
  const [query, setQuery] = useState("");
  const searchRef = useRef<HTMLInputElement>(null);

  // Typing stays responsive while filtering a long list: the input updates at
  // once and the results catch up, rather than every keystroke blocking on the
  // filter.
  const deferredQuery = useDeferredValue(query);

  const filtered = useMemo(() => {
    const needle = deferredQuery.trim().toLocaleLowerCase("kaa");
    if (!needle) return conversations;
    return conversations.filter((c) => {
      if (titleFor(c).toLocaleLowerCase("kaa").includes(needle)) return true;
      // Search message bodies too. Searching only titles finds nothing for the
      // conversation you remember by something said in the middle of it.
      return c.messages.some((m) => m.content.toLocaleLowerCase("kaa").includes(needle));
    });
  }, [conversations, deferredQuery, titleFor]);

  const groups = useGrouped(filtered, {
    today: kaa.today, week: kaa.thisWeek, older: kaa.older,
  });

  // Focused when the panel opens on desktop, so ⌘K lands in the box. Not on
  // mobile: raising the keyboard over a list the user wanted to browse is
  // hostile.
  useEffect(() => {
    if (open && window.matchMedia("(min-width: 768px)").matches) {
      searchRef.current?.focus();
    }
  }, [open]);

  return (
    <aside
      id="sidebar"
      aria-label={kaa.conversations}
      // Off-canvas on mobile, in-flow on desktop. `hidden` would remove it from
      // the accessibility tree but also kill the slide transition, so it stays
      // translated instead - with inert so nothing inside is tabbable when shut.
      inert={!open ? true : undefined}
      className={`fixed inset-y-0 start-0 z-40 flex w-[17rem] flex-col border-e
                  border-[var(--line)] bg-[var(--surface-2)]
                  transition-transform duration-200 ease-out
                  md:static md:z-auto md:translate-x-0
                  ${open ? "translate-x-0" : "-translate-x-full md:hidden"}`}
    >
      <div className="flex items-center gap-1.5 p-2.5">
        <button
          type="button"
          onClick={onNew}
          className="flex h-9 flex-1 items-center gap-2 rounded-lg border border-[var(--line)]
                     bg-[var(--surface)] px-3 text-sm font-medium transition-colors
                     hover:bg-[var(--surface-3)]"
        >
          <Icon name="plus" size={15} />
          {kaa.newChat}
        </button>
        {/* Only reachable on mobile, where the sidebar overlays the chat. */}
        <IconButton label={kaa.closeSidebar} onClick={onClose} className="md:hidden">
          <Icon name="close" />
        </IconButton>
      </div>

      <div className="px-2.5 pb-2">
        <div className="relative">
          <span className="pointer-events-none absolute inset-y-0 start-2.5 flex items-center
                           text-[var(--ink-faint)]">
            <Icon name="search" size={14} />
          </span>
          <input
            ref={searchRef}
            type="search"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder={kaa.searchPlaceholder}
            aria-label={kaa.searchPlaceholder}
            className="h-9 w-full rounded-lg border border-[var(--line)] bg-[var(--surface)]
                       ps-8 pe-2.5 text-sm outline-none transition-colors
                       placeholder:text-[var(--ink-faint)] focus:border-[var(--accent)]"
          />
        </div>
      </div>

      <nav className="min-h-0 flex-1 overflow-y-auto px-2.5 pb-2">
        {filtered.length === 0 ? (
          <p className="px-1 py-6 text-center text-xs text-[var(--ink-faint)]">
            {query ? kaa.noResults : kaa.noConversations}
          </p>
        ) : (
          groups.map((group) => (
            <section key={group.label} className="mb-3">
              <h2 className="px-1.5 pb-1 pt-1.5 text-[11px] font-semibold uppercase
                             tracking-wide text-[var(--ink-faint)]">
                {group.label}
              </h2>
              <ul className="flex flex-col gap-0.5">
                {group.items.map((conversation) => (
                  <ConversationRow
                    key={conversation.id}
                    conversation={conversation}
                    title={titleFor(conversation)}
                    active={conversation.id === activeId}
                    onSelect={onSelect}
                    onDelete={onDelete}
                    onRename={onRename}
                  />
                ))}
              </ul>
            </section>
          ))
        )}
      </nav>

      <div className="border-t border-[var(--line)] p-2.5">
        <button
          type="button"
          onClick={onOpenSettings}
          className="flex h-9 w-full items-center gap-2 rounded-lg px-2.5 text-sm
                     text-[var(--ink-muted)] transition-colors
                     hover:bg-[var(--surface-3)] hover:text-[var(--ink)]"
        >
          <Icon name="settings" size={15} />
          {kaa.settings}
        </button>
      </div>
    </aside>
  );
}

function ConversationRow({
  conversation, title, active, onSelect, onDelete, onRename,
}: {
  conversation: Conversation;
  title: string;
  active: boolean;
  onSelect: (id: string) => void;
  onDelete: (id: string) => void;
  onRename: (id: string, title: string) => void;
}) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(title);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (editing) {
      inputRef.current?.focus();
      inputRef.current?.select();
    }
  }, [editing]);

  const commit = () => {
    const next = draft.trim();
    if (next && next !== title) onRename(conversation.id, next);
    setEditing(false);
  };

  if (editing) {
    return (
      <li>
        <input
          ref={inputRef}
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          onBlur={commit}
          onKeyDown={(event) => {
            if (event.key === "Enter") commit();
            if (event.key === "Escape") {
              setDraft(title);
              setEditing(false);
            }
          }}
          aria-label={kaa.rename}
          className="h-9 w-full rounded-lg border border-[var(--accent)] bg-[var(--surface)]
                     px-2.5 text-sm outline-none"
        />
      </li>
    );
  }

  return (
    <li className="group/row relative">
      <button
        type="button"
        onClick={() => onSelect(conversation.id)}
        aria-current={active ? "page" : undefined}
        // pe-16 leaves room for the two action buttons so a long title is
        // ellipsised rather than sliding underneath them.
        className={`flex h-9 w-full items-center rounded-lg ps-2.5 pe-16 text-start text-sm
                    transition-colors
                    ${active
                      ? "bg-[var(--surface-4)] font-medium text-[var(--ink)]"
                      : "text-[var(--ink-muted)] hover:bg-[var(--surface-3)] hover:text-[var(--ink)]"}`}
      >
        <span className="truncate">{title}</span>
      </button>

      {/* Shown on hover, and on keyboard focus - focus-within is what keeps
          these reachable by tab rather than mouse-only. */}
      <div
        className="absolute inset-y-0 end-1 flex items-center gap-0.5 opacity-0
                   transition-opacity group-hover/row:opacity-100 focus-within:opacity-100"
      >
        <IconButton
          label={kaa.rename}
          onClick={() => { setDraft(title); setEditing(true); }}
          className="h-7 min-w-7 px-1"
        >
          <Icon name="edit" size={13} />
        </IconButton>
        <IconButton
          label={kaa.delete}
          onClick={() => { if (confirm(kaa.deleteConfirm)) onDelete(conversation.id); }}
          className="h-7 min-w-7 px-1 hover:text-[var(--danger)]"
        >
          <Icon name="trash" size={13} />
        </IconButton>
      </div>
    </li>
  );
}
