import type { ChatResponse, Tool } from "../types";

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
