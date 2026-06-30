import * as http from 'http';
import * as https from 'https';
import * as vscode from 'vscode';
import { ChatPanelProvider } from './chatPanel';

/** A single tool invocation recorded during an agent run. */
export interface ToolCallInfo {
  step: number;
  name: string;
  arguments: Record<string, unknown>;
  observation: string;
  is_error: boolean;
}

/** Response body of `POST /api/chat`. */
export interface ChatResponse {
  response: string;
  steps: number;
  tool_calls_made: ToolCallInfo[];
}

/** A tool description returned by `GET /api/tools`. */
export interface Tool {
  name: string;
  description: string;
  parameters: Record<string, unknown>;
}

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

/**
 * Extension entry point. Registers the chat webview provider and the
 * `setApiUrl` / `sendPrompt` commands.
 */
export function activate(context: vscode.ExtensionContext): void {
  const provider = new ChatPanelProvider(context);

  context.subscriptions.push(
    vscode.window.registerWebviewViewProvider('openAgent.chat', provider, {
      webviewOptions: { retainContextWhenHidden: true },
    })
  );

  context.subscriptions.push(
    vscode.commands.registerCommand('open-agent.setApiUrl', async () => {
      const config = vscode.workspace.getConfiguration('open-agent');
      const current = config.get<string>('apiUrl', 'http://localhost:8000');
      const value = await vscode.window.showInputBox({
        prompt: 'Open Agent API server URL',
        value: current,
        placeHolder: 'http://localhost:8000',
        validateInput: (input: string) => {
          try {
            const parsed = new URL(input);
            if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
              return 'URL must use the http or https protocol';
            }
            return undefined;
          } catch {
            return 'Enter a valid URL (e.g. http://localhost:8000)';
          }
        },
      });
      if (value) {
        await config.update('apiUrl', value, vscode.ConfigurationTarget.Global);
        vscode.window.showInformationMessage(`Open Agent API URL set to ${value}`);
        provider.onApiUrlChanged();
      }
    })
  );

  context.subscriptions.push(
    vscode.commands.registerCommand('open-agent.sendPrompt', async () => {
      const text = await vscode.window.showInputBox({
        prompt: 'Send a prompt to Open Agent',
        placeHolder: 'Ask anything…',
      });
      if (text) {
        provider.sendPrompt(text);
      }
    })
  );
}

export function deactivate(): void {
  // Nothing to dispose; subscriptions are tracked on the extension context.
}
