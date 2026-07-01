import type { ChatResponse, Tool } from "../types";

const BASE_URL: string = import.meta.env.VITE_API_URL ?? "";

/** Events streamed over WebSocket during an agent run. */
export type StreamEvent =
  | { type: "token"; content: string }
  | { type: "tool_start"; name: string; arguments: Record<string, unknown> }
  | { type: "tool_end"; name: string; observation: string; is_error: boolean }
  | { type: "done"; response: string; steps: number; tool_calls_made: ToolCallInfo[] };

interface ToolCallInfo {
  step: number;
  name: string;
  arguments: Record<string, unknown>;
  observation: string;
  is_error: boolean;
}

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
}
