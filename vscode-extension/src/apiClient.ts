import * as http from 'http';
import * as https from 'https';
import WebSocket from 'ws';

import type { ChatResponse, Tool, ToolCallInfo } from '../../shared/types';

export type { ChatResponse, Tool } from '../../shared/types';

/**
 * Perform an HTTP(S) request against the Open Agent backend using Node's
 * built-in `http`/`https` modules. Resolves with the parsed JSON response (or
 * the raw string when the body is not JSON), and rejects on transport errors
 * or non-2xx status codes. Used to proxy all webview requests so the webview
 * itself never needs network access.
 */
function requestJson(url: string, method: string, body?: unknown): Promise<unknown> {
  return new Promise((resolve, reject) => {
    let target: URL;
    try {
      target = new URL(url);
    } catch {
      reject(new Error(`Invalid URL: ${url}`));
      return;
    }

    const transport = target.protocol === 'https:' ? https : http;
    const payload = body !== undefined ? JSON.stringify(body) : undefined;
    const headers: http.OutgoingHttpHeaders = { Accept: 'application/json' };
    if (payload !== undefined) {
      headers['Content-Type'] = 'application/json';
      headers['Content-Length'] = Buffer.byteLength(payload).toString();
    }

    const reqOptions: http.RequestOptions = {
      method,
      hostname: target.hostname,
      port: target.port ? Number(target.port) : undefined,
      path: `${target.pathname}${target.search}`,
      headers,
    };

    const req = transport.request(reqOptions, (res) => {
      const parts: Buffer[] = [];
      res.on('data', (chunk: Buffer) => parts.push(chunk));
      res.on('end', () => {
        const text = Buffer.concat(parts).toString('utf8');
        const status = res.statusCode ?? 0;
        if (status < 200 || status >= 300) {
          reject(
            new Error(`HTTP ${status}: ${text || res.statusMessage || 'request failed'}`)
          );
          return;
        }
        if (!text) {
          resolve(undefined);
          return;
        }
        try {
          resolve(JSON.parse(text));
        } catch {
          resolve(text);
        }
      });
    });

    req.on('error', reject);
    if (payload !== undefined) {
      req.write(payload);
    }
    req.end();
  });
}

/** Send a chat message to the backend and return the agent's full response. */
export async function chatWithBackend(
  apiUrl: string,
  message: string,
  sessionId: string
): Promise<ChatResponse> {
  const base = apiUrl.replace(/\/+$/, '');
  const data = await requestJson(`${base}/api/chat`, 'POST', {
    message,
    session_id: sessionId,
  });
  return data as ChatResponse;
}

/**
 * Stream a conversation over the backend WebSocket, invoking callbacks as
 * events arrive. Returns a function that closes the connection.
 *
 * Falls back to the non-streaming `/api/chat` endpoint when the WebSocket
 * cannot be established.
 */
export function streamChatFromBackend(
  apiUrl: string,
  message: string,
  sessionId: string,
  callbacks: {
    onStart?: () => void;
    onToken?: (content: string) => void;
    onThought?: (content: string, step: number) => void;
    onToolStart?: (name: string, args: Record<string, unknown>) => void;
    onToolEnd?: (name: string, observation: string, isError: boolean) => void;
    onDone?: (response: string, steps: number, toolCalls: ToolCallInfo[]) => void;
    onError?: (error: string) => void;
  }
): () => void {
  const base = apiUrl.replace(/\/+$/, '');
  const wsUrl = `${base.replace(/^http/, 'ws')}/ws/chat`;
  let closed = false;
  let done = false;

  try {
    const ws = new WebSocket(wsUrl);

    ws.on('open', () => {
      if (closed) {
        ws.close();
        return;
      }
      callbacks.onStart?.();
      ws.send(JSON.stringify({ message, session_id: sessionId }));
    });

    ws.on('message', (data) => {
      if (closed) return;
      try {
        const payload = JSON.parse(data.toString()) as {
          type: string;
          content?: string;
          step?: number;
          name?: string;
          arguments?: Record<string, unknown>;
          observation?: string;
          is_error?: boolean;
          response?: string;
          steps?: number;
          tool_calls_made?: ToolCallInfo[];
        };
        switch (payload.type) {
          case 'token':
            callbacks.onToken?.(payload.content ?? '');
            break;
          case 'thought':
            callbacks.onThought?.(payload.content ?? '', payload.step ?? 0);
            break;
          case 'tool_start':
            callbacks.onToolStart?.(payload.name ?? '', payload.arguments ?? {});
            break;
          case 'tool_end':
            callbacks.onToolEnd?.(
              payload.name ?? '',
              payload.observation ?? '',
              payload.is_error ?? false
            );
            break;
          case 'done':
            done = true;
            callbacks.onDone?.(
              payload.response ?? '',
              payload.steps ?? 0,
              payload.tool_calls_made ?? []
            );
            ws.close();
            break;
        }
      } catch {
        // ignore malformed frames
      }
    });

    ws.on('error', (err) => {
      if (closed) return;
      callbacks.onError?.(err.message || 'WebSocket error');
    });

    ws.on('close', () => {
      if (!done && !closed) {
        callbacks.onError?.('Connection closed unexpectedly');
      }
    });

    return () => {
      if (closed) return;
      closed = true;
      if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) {
        ws.close();
      }
    };
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    callbacks.onError?.(message);
    return () => {
      // no-op
    };
  }
}

/** Fetch the list of available tools from the backend. */
export async function getToolsFromBackend(apiUrl: string): Promise<Tool[]> {
  const base = apiUrl.replace(/\/+$/, '');
  const data = await requestJson(`${base}/api/tools`, 'GET');
  const result = (data as { tools?: Tool[] } | undefined)?.tools;
  return result ?? [];
}

/** Probe the backend `/api/health` endpoint. Returns `false` on any failure. */
export async function healthCheckBackend(apiUrl: string): Promise<boolean> {
  try {
    const base = apiUrl.replace(/\/+$/, '');
    const data = await requestJson(`${base}/api/health`, 'GET');
    return (data as { status?: string } | undefined)?.status === 'ok';
  } catch {
    return false;
  }
}
