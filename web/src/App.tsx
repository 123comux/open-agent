import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ChatClient } from "./api/client";
import { Chat } from "./components/Chat";
import { Sidebar } from "./components/Sidebar";
import { Settings } from "./components/Settings";
import { TraceDashboard } from "./components/TraceDashboard";
import type { AgentSettings, AgentSettingsUpdate, Message, Tool } from "./types";

function generateSessionId(): string {
  return `sess-${Math.random().toString(36).slice(2, 10)}`;
}

function newId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID();
  }
  return `id-${Math.random().toString(36).slice(2)}-${Date.now()}`;
}

export default function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const [tools, setTools] = useState<Tool[]>([]);
  const [toolsLoading, setToolsLoading] = useState(true);
  const [sessionId, setSessionId] = useState(generateSessionId);
  const [sessions, setSessions] = useState<string[]>([]);
  const [apiUrl, setApiUrl] = useState<string>(
    () => (import.meta.env.VITE_API_URL as string | undefined) ?? ""
  );
  const [healthy, setHealthy] = useState<boolean | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [settings, setSettings] = useState<AgentSettings | null>(null);
  const [settingsLoading, setSettingsLoading] = useState(false);
  const [view, setView] = useState<"chat" | "traces">("chat");
  const cancelRef = useRef<(() => void) | null>(null);

  const client = useMemo(() => new ChatClient(apiUrl), [apiUrl]);

  const refreshSessions = useCallback(() => {
    client
      .listSessions()
      .then((s) => setSessions(s))
      .catch(() => {
        /* sessions unavailable; sidebar will just show empty list */
      });
  }, [client]);

  useEffect(() => {
    let active = true;
    setToolsLoading(true);
    setHealthy(null);

    client
      .healthCheck()
      .then((ok) => {
        if (active) setHealthy(ok);
      });

    client
      .getTools()
      .then((t) => {
        if (active) setTools(t);
      })
      .catch(() => {
        /* tools unavailable; health flag conveys backend status */
      })
      .finally(() => {
        if (active) setToolsLoading(false);
      });

    return () => {
      active = false;
    };
  }, [client]);

  // Fetch sessions list on mount and whenever the session or client changes.
  useEffect(() => {
    refreshSessions();
  }, [refreshSessions, sessionId]);

  // Fetch runtime settings when the client changes.
  useEffect(() => {
    let active = true;
    setSettingsLoading(true);
    client
      .getSettings()
      .then((s) => {
        if (active) setSettings(s);
      })
      .catch(() => {
        /* settings unavailable; Settings panel will use defaults */
      })
      .finally(() => {
        if (active) setSettingsLoading(false);
      });
    return () => {
      active = false;
    };
  }, [client]);

  const handleSettingsSave = useCallback(
    async (update: AgentSettingsUpdate) => {
      await client.updateSettings(update);
      const refreshed = await client.getSettings();
      setSettings(refreshed);
    },
    [client]
  );

  // Cleanup any active stream on unmount
  useEffect(() => {
    return () => cancelRef.current?.();
  }, []);

  // Switch to an existing session, loading its history into the message list.
  const handleSelectSession = useCallback(
    (id: string) => {
      if (id === sessionId) return;
      cancelRef.current?.();
      cancelRef.current = null;
      setError(null);
      setSessionId(id);
      setMessages([]);
      client
        .getSessionHistory(id)
        .then((history) => {
          setMessages(
            history.map((m) => ({
              id: newId(),
              role: (m.role === "user" ? "user" : "assistant") as Message["role"],
              content: m.content,
              timestamp: Date.now(),
            }))
          );
        })
        .catch(() => {
          /* history unavailable; leave messages empty */
        });
    },
    [client, sessionId]
  );

  // Start a fresh session, clearing the current conversation.
  const handleNewSession = useCallback(() => {
    cancelRef.current?.();
    cancelRef.current = null;
    setError(null);
    setMessages([]);
    setSessionId(generateSessionId());
  }, []);

  // Delete a session from the backend and refresh the sidebar list.
  const handleDeleteSession = useCallback(
    (id: string) => {
      client
        .clearSession(id)
        .then(() => {
          setSessions((prev) => prev.filter((s) => s !== id));
          if (id === sessionId) {
            handleNewSession();
          }
        })
        .catch(() => {
          /* deletion failed; ignore */
        });
    },
    [client, sessionId, handleNewSession]
  );

  const handleRenameSession = useCallback(
    async (oldId: string, newId: string) => {
      await client.renameSession(oldId, newId);
      setSessions((prev) =>
        prev.map((s) => (s === oldId ? newId : s))
      );
      if (oldId === sessionId) {
        setSessionId(newId);
      }
    },
    [client, sessionId]
  );

  const handleExportSession = useCallback(
    async (id: string, fmt: "json" | "md") => {
      await client.exportSession(id, fmt);
    },
    [client]
  );

  const handleSend = useCallback(
    (text: string) => {
      const trimmed = text.trim();
      if (!trimmed || loading) return;
      setError(null);

      const userMsg: Message = {
        id: newId(),
        role: "user",
        content: trimmed,
        timestamp: Date.now(),
      };
      setMessages((prev) => [...prev, userMsg]);
      setLoading(true);

      // Use streaming via WebSocket
      const assistantId = newId();
      const toolCallsAccum: {
        step: number;
        name: string;
        arguments: Record<string, unknown>;
        observation: string;
        is_error: boolean;
      }[] = [];
      const thoughtsAccum: string[] = [];

      // Create placeholder assistant message that we update as tokens arrive
      setMessages((prev) => [
        ...prev,
        {
          id: assistantId,
          role: "assistant",
          content: "",
          streaming: true,
          timestamp: Date.now(),
        },
      ]);

      cancelRef.current = client.streamMessage(trimmed, sessionId, {
        onToken: (chunk) => {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantId
                ? { ...m, content: m.content + chunk }
                : m
            )
          );
        },
        onThought: (content) => {
          thoughtsAccum.push(content);
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantId
                ? { ...m, thoughts: [...thoughtsAccum] }
                : m
            )
          );
        },
        onToolStart: (name, args) => {
          toolCallsAccum.push({
            step: toolCallsAccum.length + 1,
            name,
            arguments: args,
            observation: "",
            is_error: false,
          });
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantId
                ? { ...m, toolCalls: [...toolCallsAccum] }
                : m
            )
          );
        },
        onToolEnd: (name, observation, isError) => {
          const tc = toolCallsAccum.find((t) => t.name === name && !t.observation);
          if (tc) {
            tc.observation = observation;
            tc.is_error = isError;
          }
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantId
                ? { ...m, toolCalls: [...toolCallsAccum] }
                : m
            )
          );
        },
        onDone: (response, steps, toolCalls) => {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantId
                ? {
                    ...m,
                    streaming: false,
                    content: response,
                    steps,
                    toolCalls: toolCalls.length > 0 ? toolCalls : toolCallsAccum,
                  }
                : m
            )
          );
          setLoading(false);
          cancelRef.current = null;
          refreshSessions();
        },
        onError: (err) => {
          setError(err);
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantId
                ? {
                    ...m,
                    streaming: false,
                    content: m.content || "Failed to get response.",
                  }
                : m
            )
          );
          setLoading(false);
          cancelRef.current = null;
        },
      });
    },
    [client, loading, sessionId, refreshSessions]
  );

  return (
    <div className="flex h-full w-full bg-[#0a0a0f] font-sans text-zinc-100">
      <Sidebar
        client={client}
        tools={tools}
        loading={toolsLoading}
        healthy={healthy}
        sessions={sessions}
        activeSessionId={sessionId}
        onSelectSession={handleSelectSession}
        onNewSession={handleNewSession}
        onDeleteSession={handleDeleteSession}
        onRenameSession={handleRenameSession}
        onExportSession={handleExportSession}
      >
        <Settings
          apiUrl={apiUrl}
          onApiUrlChange={setApiUrl}
          sessionId={sessionId}
          onRegenerateSession={() => setSessionId(generateSessionId())}
          healthy={healthy}
          tools={tools}
          settings={settings}
          settingsLoading={settingsLoading}
          onSettingsSave={handleSettingsSave}
        />
      </Sidebar>
      <main className="flex min-w-0 flex-1 flex-col">
        <div className="flex items-center justify-end gap-2 border-b border-white/5 px-4 py-2">
          {(["chat", "traces"] as const).map((v) => (
            <button
              key={v}
              type="button"
              onClick={() => setView(v)}
              className={`rounded-md px-3 py-1 text-xs font-medium transition-colors ${
                view === v
                  ? "bg-emerald-500/20 text-emerald-300"
                  : "text-zinc-500 hover:bg-white/5 hover:text-zinc-300"
              }`}
            >
              {v === "chat" ? "Chat" : "Traces"}
            </button>
          ))}
        </div>
        {view === "chat" ? (
          <Chat
            messages={messages}
            loading={loading}
            onSend={handleSend}
            onStop={() => {
              cancelRef.current?.();
              cancelRef.current = null;
              setLoading(false);
              setMessages((prev) =>
                prev.map((m) =>
                  m.streaming ? { ...m, streaming: false } : m
                )
              );
            }}
            error={error}
          />
        ) : (
          <TraceDashboard client={client} />
        )}
      </main>
    </div>
  );
}
