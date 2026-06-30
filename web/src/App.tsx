import { useCallback, useEffect, useMemo, useState } from "react";
import { ChatClient } from "./api/client";
import { Chat } from "./components/Chat";
import { Sidebar } from "./components/Sidebar";
import { Settings } from "./components/Settings";
import type { ChatResponse, Message, Tool } from "./types";

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
  const [apiUrl, setApiUrl] = useState<string>(
    () => (import.meta.env.VITE_API_URL as string | undefined) ?? ""
  );
  const [healthy, setHealthy] = useState<boolean | null>(null);
  const [error, setError] = useState<string | null>(null);

  const client = useMemo(() => new ChatClient(apiUrl), [apiUrl]);

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

  const handleSend = useCallback(
    async (text: string) => {
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

      try {
        const res: ChatResponse = await client.sendMessage(trimmed, sessionId);
        const assistantMsg: Message = {
          id: newId(),
          role: "assistant",
          content: res.response,
          steps: res.steps,
          toolCalls: res.tool_calls_made,
          timestamp: Date.now(),
        };
        setMessages((prev) => [...prev, assistantMsg]);
      } catch (e) {
        setError(e instanceof Error ? e.message : "Request failed");
      } finally {
        setLoading(false);
      }
    },
    [client, loading, sessionId]
  );

  return (
    <div className="flex h-full w-full bg-[#0a0a0f] font-sans text-zinc-100">
      <Sidebar tools={tools} loading={toolsLoading} healthy={healthy}>
        <Settings
          apiUrl={apiUrl}
          onApiUrlChange={setApiUrl}
          sessionId={sessionId}
          onRegenerateSession={() => setSessionId(generateSessionId())}
          healthy={healthy}
        />
      </Sidebar>
      <main className="flex min-w-0 flex-1 flex-col">
        <Chat
          messages={messages}
          loading={loading}
          onSend={handleSend}
          error={error}
        />
      </main>
    </div>
  );
}
