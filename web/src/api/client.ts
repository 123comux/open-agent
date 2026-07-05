import type {
  AgentSettings,
  AgentSettingsUpdate,
  ChatResponse,
  KnowledgeBaseDocument,
  StreamEvent,
  Tool,
  ToolCallInfo,
  Trace,
} from "../types";

const BASE_URL: string = import.meta.env.VITE_API_URL ?? "";

/**
 * Thin HTTP/WebSocket client for the Open Agent FastAPI backend.
 *
 * The base URL defaults to an empty string (same-origin requests), which works
 * with the Vite dev-server proxy in development. Override per-session by
 * instantiating with an explicit `baseUrl` (e.g. `http://localhost:8000`).
 */
export class ChatClient {
  private readonly baseUrl: string;

  constructor(baseUrl: string = BASE_URL) {
    this.baseUrl = baseUrl.replace(/\/+$/, "");
  }

  /** Send a single message and await the agent's full response. */
  async sendMessage(message: string, sessionId: string): Promise<ChatResponse> {
    const res = await fetch(`${this.baseUrl}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, session_id: sessionId }),
    });
    if (!res.ok) {
      throw new Error(`Chat request failed: ${res.status} ${res.statusText}`);
    }
    return (await res.json()) as ChatResponse;
  }

  /**
   * Stream a conversation over WebSocket, calling callbacks as events arrive.
   * Returns a function to cancel the stream.
   */
  streamMessage(
    message: string,
    sessionId: string,
    callbacks: {
      onToken?: (chunk: string) => void;
      onThought?: (content: string, step: number) => void;
      onToolStart?: (name: string, args: Record<string, unknown>) => void;
      onToolEnd?: (name: string, observation: string, isError: boolean) => void;
      onDone?: (response: string, steps: number, toolCalls: ToolCallInfo[]) => void;
      onError?: (error: string) => void;
    }
  ): () => void {
    const origin = this.baseUrl || window.location.origin;
    const wsUrl = `${origin.replace(/^http/, "ws")}/ws/chat`;
    const ws = new WebSocket(wsUrl);
    let cancelled = false;

    ws.onopen = () => {
      ws.send(JSON.stringify({ message, session_id: sessionId }));
    };

    ws.onmessage = (event) => {
      try {
        const data: StreamEvent = JSON.parse(event.data);
        switch (data.type) {
          case "token":
            callbacks.onToken?.(data.content);
            break;
          case "thought":
            callbacks.onThought?.(data.content, data.step);
            break;
          case "tool_start":
            callbacks.onToolStart?.(data.name, data.arguments);
            break;
          case "tool_end":
            callbacks.onToolEnd?.(data.name, data.observation, data.is_error);
            break;
          case "done":
            callbacks.onDone?.(data.response, data.steps, data.tool_calls_made);
            ws.close();
            break;
        }
      } catch {
        // ignore malformed frames
      }
    };

    ws.onerror = () => {
      if (!cancelled) {
        callbacks.onError?.("WebSocket connection error");
      }
    };

    return () => {
      cancelled = true;
      if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) {
        ws.close();
      }
    };
  }

  /** Fetch the list of available tools. */
  async getTools(): Promise<Tool[]> {
    const res = await fetch(`${this.baseUrl}/api/tools`);
    if (!res.ok) {
      throw new Error(`Tools request failed: ${res.status} ${res.statusText}`);
    }
    const data = (await res.json()) as { tools: Tool[] };
    return data.tools;
  }

  /** Return `true` when the backend reports a healthy status. */
  async healthCheck(): Promise<boolean> {
    try {
      const res = await fetch(`${this.baseUrl}/api/health`);
      if (!res.ok) return false;
      const data = (await res.json()) as { status: string };
      return data.status === "ok";
    } catch {
      return false;
    }
  }

  /** List all conversation sessions. */
  async listSessions(): Promise<string[]> {
    const res = await fetch(`${this.baseUrl}/api/sessions`);
    if (!res.ok) return [];
    const data = (await res.json()) as { sessions: string[] };
    return data.sessions;
  }

  /** Get conversation history for a session. */
  async getSessionHistory(sessionId: string): Promise<{ role: string; content: string }[]> {
    const res = await fetch(`${this.baseUrl}/api/sessions/${sessionId}/history`);
    if (!res.ok) return [];
    const data = (await res.json()) as { messages: { role: string; content: string }[] };
    return data.messages;
  }

  /** Clear a session's history. */
  async clearSession(sessionId: string): Promise<void> {
    await fetch(`${this.baseUrl}/api/sessions/${sessionId}`, { method: "DELETE" });
  }

  /** Rename a session. */
  async renameSession(sessionId: string, newSessionId: string): Promise<{ status: string; old_session_id: string; new_session_id: string }> {
    const res = await fetch(`${this.baseUrl}/api/sessions/${encodeURIComponent(sessionId)}/rename`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ new_session_id: newSessionId }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ message: res.statusText }));
      throw new Error(err.message || `Rename failed: ${res.status}`);
    }
    return res.json();
  }

  /** Search sessions by id and message content. */
  async searchSessions(query: string): Promise<{ session_id: string; matches: number }[]> {
    if (!query.trim()) return [];
    const res = await fetch(`${this.baseUrl}/api/sessions/search?q=${encodeURIComponent(query)}`);
    if (!res.ok) return [];
    const data = (await res.json()) as { results: { session_id: string; matches: number }[] };
    return data.results ?? [];
  }

  /** Export a session as JSON or Markdown. */
  async exportSession(sessionId: string, fmt: "json" | "md" = "md"): Promise<void> {
    const res = await fetch(
      `${this.baseUrl}/api/sessions/${encodeURIComponent(sessionId)}/export?fmt=${fmt}`
    );
    if (!res.ok) {
      const err = await res.json().catch(() => ({ message: res.statusText }));
      throw new Error(err.message || `Export failed: ${res.status}`);
    }
    const blob = await res.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${sessionId}.${fmt}`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
  }

  /** Upload a document file for RAG indexing. */
  async uploadFile(file: File, kbName: string = "default"): Promise<{ status: string; filename: string; kb_name: string; chunks: number }> {
    const formData = new FormData();
    formData.append("file", file);
    const res = await fetch(`${this.baseUrl}/api/upload?kb_name=${encodeURIComponent(kbName)}`, {
      method: "POST",
      body: formData,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ message: res.statusText }));
      throw new Error(err.message || `Upload failed: ${res.status}`);
    }
    return res.json();
  }

  /** Fetch current runtime settings from the backend. */
  async getSettings(): Promise<AgentSettings> {
    const res = await fetch(`${this.baseUrl}/api/settings`);
    if (!res.ok) {
      throw new Error(`Settings request failed: ${res.status} ${res.statusText}`);
    }
    return (await res.json()) as AgentSettings;
  }

  /** Update runtime settings on the backend. */
  async updateSettings(settings: AgentSettingsUpdate): Promise<{ status: string }> {
    const res = await fetch(`${this.baseUrl}/api/settings`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(settings),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ message: res.statusText }));
      throw new Error(err.message || `Settings update failed: ${res.status}`);
    }
    return (await res.json()) as { status: string };
  }

  /**
   * Open a WebSocket to `/ws/chat` for streaming conversation. The caller is
   * responsible for sending JSON frames of shape `{ message, session_id }` and
   * handling incoming response frames.
   */
  connectWebSocket(sessionId: string): WebSocket {
    const origin = this.baseUrl || window.location.origin;
    const wsUrl = `${origin.replace(/^http/, "ws")}/ws/chat?session_id=${encodeURIComponent(
      sessionId
    )}`;
    return new WebSocket(wsUrl);
  }

  /** List all registered knowledge base names. */
  async listKnowledgeBases(): Promise<string[]> {
    const res = await fetch(`${this.baseUrl}/api/knowledge-bases`);
    if (!res.ok) return [];
    const data = (await res.json()) as { knowledge_bases: string[] };
    return data.knowledge_bases ?? [];
  }

  /** List documents indexed in a knowledge base. */
  async listKbDocuments(kbName: string): Promise<KnowledgeBaseDocument[]> {
    const res = await fetch(
      `${this.baseUrl}/api/knowledge-bases/${encodeURIComponent(kbName)}/documents`
    );
    if (!res.ok) {
      const err = await res.json().catch(() => ({ message: res.statusText }));
      throw new Error(err.message || `Failed to list documents: ${res.status}`);
    }
    const data = (await res.json()) as { documents: KnowledgeBaseDocument[] };
    return data.documents ?? [];
  }

  /** Delete all chunks from ``source`` in ``kbName``. */
  async deleteKbDocument(kbName: string, source: string): Promise<{ removed: number }> {
    const res = await fetch(
      `${this.baseUrl}/api/knowledge-bases/${encodeURIComponent(kbName)}/documents?source=${encodeURIComponent(source)}`,
      { method: "DELETE" }
    );
    if (!res.ok) {
      const err = await res.json().catch(() => ({ message: res.statusText }));
      throw new Error(err.message || `Delete failed: ${res.status}`);
    }
    return (await res.json()) as { removed: number };
  }

  /** List recent traces from the local tracer. */
  async listTraces(limit: number = 100): Promise<Trace[]> {
    const res = await fetch(`${this.baseUrl}/api/traces?limit=${limit}`);
    if (!res.ok) return [];
    const data = (await res.json()) as { traces: Trace[] };
    return data.traces ?? [];
  }

  /** Fetch a single trace by id. */
  async getTrace(traceId: string): Promise<Trace | null> {
    const res = await fetch(`${this.baseUrl}/api/traces/${encodeURIComponent(traceId)}`);
    if (!res.ok) return null;
    return (await res.json()) as Trace;
  }
}
