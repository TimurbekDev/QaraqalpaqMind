"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { clearEverything, loadSettings, saveSettings } from "@/lib/storage";
import { kaa } from "@/lib/strings";
import { loadTheme, type Theme } from "@/lib/theme";
import { DEFAULT_SETTINGS, type Message, type Settings } from "@/lib/types";
import { useChat } from "@/lib/useChat";
import { useConversations } from "@/lib/useConversations";
import Composer, { type ComposerHandle } from "./chat/Composer";
import MessageList from "./chat/MessageList";
import Sidebar from "./chat/Sidebar";
import SettingsDialog from "./SettingsDialog";
import ShortcutsDialog from "./ShortcutsDialog";
import Icon from "./ui/Icon";
import IconButton from "./ui/IconButton";

/** Stable empty array, so "no conversation open" does not invalidate callbacks. */
const EMPTY: Message[] = [];

export default function ChatShell() {
  const store = useConversations(kaa.untitled);
  const [settings, setSettings] = useState<Settings>(DEFAULT_SETTINGS);
  const [theme, setTheme] = useState<Theme>("system");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [shortcutsOpen, setShortcutsOpen] = useState(false);
  const composerRef = useRef<ComposerHandle>(null);

  useEffect(() => {
    setSettings(loadSettings());
    setTheme(loadTheme());
    // Open on desktop, closed on mobile: the sidebar is an overlay there, and
    // landing on a panel that covers the chat hides the thing you came for.
    setSidebarOpen(window.matchMedia("(min-width: 768px)").matches);
  }, []);

  const activeId = store.activeId;

  // Memoised because `?? EMPTY` would otherwise mint a new array on every
  // render, and `messages` is a dependency of the callbacks below - which
  // would then be rebuilt on every keystroke of a stream.
  const messages = useMemo(() => store.active?.messages ?? EMPTY, [store.active]);

  // Depends on `touch`, not on the whole store object: the store is rebuilt
  // each render and would drag this identity with it.
  const { touch, create, startNew } = store;
  const commit = useCallback(
    (next: Message[]) => {
      if (activeId) touch(activeId, next);
    },
    [activeId, touch],
  );

  const chat = useChat({ settings, commit });

  /** Ensure a conversation exists, then run `action` against its messages. */
  const withConversation = useCallback(
    (action: (current: Message[]) => void) => {
      if (!activeId) {
        create();
        // The new conversation is empty, so the action runs against nothing.
        // Deferred a tick so `activeId` has landed before `commit` fires.
        setTimeout(() => action([]), 0);
        return;
      }
      action(messages);
    },
    [activeId, create, messages],
  );

  const send = useCallback(
    (text: string) => {
      chat.setWorking(messages);
      withConversation((current) => chat.send(current, text));
    },
    [chat, messages, withConversation],
  );

  const act = useCallback(
    (operation: (current: Message[]) => void) => {
      chat.setWorking(messages);
      operation(messages);
    },
    [chat, messages],
  );

  const newChat = useCallback(() => {
    startNew();
    composerRef.current?.focus();
    if (!window.matchMedia("(min-width: 768px)").matches) setSidebarOpen(false);
  }, [startNew]);

  const updateSettings = useCallback((next: Settings) => {
    setSettings(next);
    saveSettings(next);
  }, []);

  // --- Keyboard shortcuts --------------------------------------------------
  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      const typing =
        target?.tagName === "INPUT" ||
        target?.tagName === "TEXTAREA" ||
        target?.isContentEditable;

      const mod = event.metaKey || event.ctrlKey;

      if (mod && event.key.toLowerCase() === "k") {
        event.preventDefault();
        setSidebarOpen(true);
        // The sidebar focuses its search box when it opens.
        return;
      }
      if (mod && event.shiftKey && event.key.toLowerCase() === "o") {
        event.preventDefault();
        newChat();
        return;
      }
      if (mod && event.key.toLowerCase() === "b") {
        event.preventDefault();
        setSidebarOpen((open) => !open);
        return;
      }
      if (mod && event.key === ",") {
        event.preventDefault();
        setSettingsOpen(true);
        return;
      }

      if (event.key === "Escape" && chat.busy) {
        chat.stop();
        return;
      }

      // Single-key shortcuts must never fire while typing, or "/" and "?"
      // become impossible to write in a message.
      if (typing) return;

      if (event.key === "/") {
        event.preventDefault();
        composerRef.current?.focus();
      }
      if (event.key === "?") {
        event.preventDefault();
        setShortcutsOpen(true);
      }
    };

    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [chat, newChat]);

  return (
    <div className="flex h-dvh overflow-hidden bg-[var(--surface)]">
      <a
        href="#composer"
        className="sr-only focus:not-sr-only focus:absolute focus:start-3 focus:top-3 focus:z-50
                   focus:rounded-lg focus:bg-[var(--accent)] focus:px-3 focus:py-2
                   focus:text-sm focus:text-[var(--accent-ink)]"
      >
        {kaa.srSkipToInput}
      </a>

      {/* Scrim, mobile only. Clicking outside an overlay panel should close it. */}
      {sidebarOpen && (
        <div
          onClick={() => setSidebarOpen(false)}
          aria-hidden="true"
          className="fixed inset-0 z-30 bg-black/40 md:hidden"
        />
      )}

      <Sidebar
        conversations={store.conversations}
        activeId={store.activeId}
        open={sidebarOpen}
        titleFor={store.titleFor}
        onSelect={(id) => {
          store.setActiveId(id);
          if (!window.matchMedia("(min-width: 768px)").matches) setSidebarOpen(false);
        }}
        onNew={newChat}
        onDelete={store.remove}
        onRename={store.rename}
        onClose={() => setSidebarOpen(false)}
        onOpenSettings={() => setSettingsOpen(true)}
      />

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex h-12 shrink-0 items-center gap-1 border-b border-[var(--line)] px-2.5">
          <IconButton
            label={sidebarOpen ? kaa.closeSidebar : kaa.openSidebar}
            onClick={() => setSidebarOpen((open) => !open)}
            aria-expanded={sidebarOpen}
            aria-controls="sidebar"
          >
            <Icon name="sidebar" />
          </IconButton>

          <div className="min-w-0 flex-1 px-1">
            <h1 className="truncate text-sm font-semibold">
              {store.active ? store.titleFor(store.active) : kaa.appName}
            </h1>
          </div>

          <IconButton label={kaa.newChat} onClick={newChat}>
            <Icon name="plus" />
          </IconButton>
          <IconButton
            label={kaa.shortcuts}
            onClick={() => setShortcutsOpen(true)}
            className="hidden sm:inline-flex"
          >
            <Icon name="keyboard" />
          </IconButton>
          <IconButton label={kaa.settings} onClick={() => setSettingsOpen(true)}>
            <Icon name="settings" />
          </IconButton>
        </header>

        <MessageList
          messages={messages}
          status={chat.status}
          onSuggestion={send}
          onRegenerate={() => act((current) => chat.regenerate(current))}
          onRetry={() => act((current) => chat.retry(current))}
          onContinue={() => act((current) => chat.continueGeneration(current))}
          onEdit={(id, content) => act((current) => chat.editAndResend(current, id, content))}
        />

        <div id="composer">
          <Composer
            ref={composerRef}
            onSend={send}
            onStop={chat.stop}
            status={chat.status}
            sendOnEnter={settings.sendOnEnter}
          />
        </div>
      </div>

      <SettingsDialog
        open={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        settings={settings}
        onChange={updateSettings}
        theme={theme}
        onThemeChange={setTheme}
        onClearAll={() => {
          store.clearAll();
          clearEverything();
        }}
      />
      <ShortcutsDialog open={shortcutsOpen} onClose={() => setShortcutsOpen(false)} />
    </div>
  );
}
