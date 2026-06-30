import * as vscode from 'vscode';
import {
  chatWithBackend,
  getToolsFromBackend,
  healthCheckBackend,
  type ChatResponse,
  type Tool,
} from './extension';

/**
 * Provides the Open Agent chat webview view hosted in the activity bar sidebar.
 *
 * The webview is a self-contained single-page app (inline HTML/CSS/JS) that
 * posts messages to this provider; the provider proxies the requests to the
 * Open Agent backend over HTTP (via the helpers in `extension.ts`) and
 * forwards the results back to the webview.
 */
export class ChatPanelProvider implements vscode.WebviewViewProvider {
  private view?: vscode.WebviewView;
  private readonly sessionId: string;

  constructor(private readonly context: vscode.ExtensionContext) {
    this.sessionId = `vscode-${process.pid}-${Date.now()}`;
  }

  /** Called by VS Code when the chat view is first made visible. */
  resolveWebviewView(webviewView: vscode.WebviewView): void {
    this.view = webviewView;
    webviewView.webview.options = {
      enableScripts: true,
    };
    webviewView.webview.html = this.getHtml();

    webviewView.webview.onDidReceiveMessage(
      (msg: unknown) => {
        void this.handleMessage(msg);
      },
      undefined,
      this.context.subscriptions
    );

    // Initial health probe once the view is alive.
    void this.probeHealth();
  }

  /** Re-check backend connectivity after the API URL changes. */
  onApiUrlChanged(): void {
    this.post({ type: 'config', apiUrl: this.getApiUrl() });
    void this.probeHealth();
  }

  /** Programmatically send a prompt (used by the `sendPrompt` command). */
  sendPrompt(message: string): void {
    if (!this.view) {
      void vscode.window.showWarningMessage(
        'Open the Open Agent chat panel to send a prompt.'
      );
      return;
    }
    this.post({ type: 'sendPrompt', message });
  }

  private getApiUrl(): string {
    return vscode.workspace
      .getConfiguration('open-agent')
      .get<string>('apiUrl', 'http://localhost:8000');
  }

  private post(message: unknown): void {
    this.view?.webview.postMessage(message);
  }

  private async probeHealth(): Promise<void> {
    const ok = await healthCheckBackend(this.getApiUrl());
    this.post({ type: 'health', ok });
  }

  private async handleMessage(msg: unknown): Promise<void> {
    if (!msg || typeof msg !== 'object') return;
    const data = msg as { type?: string; message?: string };
    switch (data.type) {
      case 'ready':
        this.post({ type: 'config', apiUrl: this.getApiUrl() });
        void this.probeHealth();
        break;
      case 'chat':
        await this.handleChat(data.message ?? '');
        break;
      case 'getTools':
        await this.handleGetTools();
        break;
      case 'checkHealth':
        void this.probeHealth();
        break;
      default:
        break;
    }
  }

  private async handleChat(message: string): Promise<void> {
    if (!message.trim()) return;
    this.post({ type: 'loading', loading: true });
    try {
      const result: ChatResponse = await chatWithBackend(
        this.getApiUrl(),
        message,
        this.sessionId
      );
      this.post({ type: 'chatResponse', result });
    } catch (err) {
      const errorText = err instanceof Error ? err.message : String(err);
      this.post({ type: 'chatError', error: errorText });
    } finally {
      this.post({ type: 'loading', loading: false });
    }
  }

  private async handleGetTools(): Promise<void> {
    try {
      const tools: Tool[] = await getToolsFromBackend(this.getApiUrl());
      this.post({ type: 'tools', tools });
    } catch (err) {
      const errorText = err instanceof Error ? err.message : String(err);
      this.post({ type: 'toolsError', error: errorText });
    }
  }

  /**
   * Build the self-contained webview HTML. Inline CSS provides a dark theme
   * matching VS Code; inline JS handles rendering, message history, and
   * communication back to this provider via `postMessage`.
   */
  private getHtml(): string {
    return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline';" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Open Agent</title>
<style>
  :root {
    --bg: #1e1e1e;
    --bg-panel: #252526;
    --bg-input: #2d2d30;
    --bg-hover: #2a2d2e;
    --border: #3c3c3c;
    --fg: #cccccc;
    --fg-muted: #9d9d9d;
    --accent: #0e639c;
    --accent-hover: #1177bb;
    --user-bubble: #264f78;
    --assistant-bubble: #2d2d30;
    --error: #f48771;
    --error-bg: #5a1d1d;
    --success: #4ec9b0;
    --tool-bg: #181818;
    --tool-border: #3c3c3c;
  }
  * { box-sizing: border-box; }
  html, body {
    height: 100%;
    margin: 0;
    padding: 0;
    background: var(--bg);
    color: var(--fg);
    font-family: var(--vscode-font-family, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif);
    font-size: 13px;
  }
  #app { display: flex; flex-direction: column; height: 100vh; }
  header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 8px 12px;
    background: var(--bg-panel);
    border-bottom: 1px solid var(--border);
    flex-shrink: 0;
  }
  .title { font-weight: 600; font-size: 13px; display: flex; align-items: center; gap: 6px; }
  .header-actions { display: flex; gap: 6px; align-items: center; }
  .health {
    font-size: 11px;
    padding: 2px 8px;
    border-radius: 10px;
    border: 1px solid var(--border);
    color: var(--fg-muted);
  }
  .health.ok { color: var(--success); border-color: var(--success); }
  .health.err { color: var(--error); border-color: var(--error); }
  .icon-btn {
    background: transparent;
    border: 1px solid var(--border);
    color: var(--fg);
    padding: 3px 8px;
    border-radius: 4px;
    cursor: pointer;
    font-size: 11px;
  }
  .icon-btn:hover { background: var(--bg-hover); }
  #messages {
    flex: 1;
    overflow-y: auto;
    padding: 12px;
    display: flex;
    flex-direction: column;
    gap: 10px;
  }
  .msg { display: flex; flex-direction: column; max-width: 92%; }
  .msg.user { align-self: flex-end; align-items: flex-end; }
  .msg.assistant { align-self: flex-start; align-items: flex-start; }
  .bubble {
    padding: 8px 12px;
    border-radius: 8px;
    line-height: 1.45;
    white-space: pre-wrap;
    word-break: break-word;
  }
  .msg.user .bubble { background: var(--user-bubble); color: #fff; border-bottom-right-radius: 2px; }
  .msg.assistant .bubble { background: var(--assistant-bubble); color: var(--fg); border: 1px solid var(--border); border-bottom-left-radius: 2px; }
  .role {
    font-size: 10px;
    color: var(--fg-muted);
    margin: 0 4px 3px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }
  .meta { font-size: 10px; color: var(--fg-muted); margin: 4px 4px 0; }
  .tool-calls { margin-top: 6px; display: flex; flex-direction: column; gap: 4px; width: 100%; }
  details.tool {
    background: var(--tool-bg);
    border: 1px solid var(--tool-border);
    border-radius: 4px;
    padding: 4px 8px;
    font-size: 11px;
  }
  details.tool > summary {
    cursor: pointer;
    color: var(--fg);
    display: flex;
    align-items: center;
    gap: 6px;
    list-style: none;
  }
  details.tool > summary::-webkit-details-marker { display: none; }
  details.tool > summary::before { content: "\\25B6"; font-size: 8px; transition: transform .15s; color: var(--fg-muted); }
  details.tool[open] > summary::before { transform: rotate(90deg); }
  .tool-name { font-weight: 600; color: #4fc1ff; }
  .tool-step { color: var(--fg-muted); }
  .tool-error .tool-name { color: var(--error); }
  .tool-body { margin-top: 6px; padding-top: 6px; border-top: 1px solid var(--tool-border); }
  .tool-body pre {
    margin: 4px 0;
    padding: 6px;
    background: #111111;
    border-radius: 3px;
    overflow-x: auto;
    white-space: pre-wrap;
    word-break: break-word;
    font-family: var(--vscode-editor-font-family, Consolas, "Courier New", monospace);
    font-size: 11px;
    color: #d4d4d4;
  }
  .tool-label { color: var(--fg-muted); font-size: 10px; text-transform: uppercase; letter-spacing: .5px; margin-top: 4px; }
  #loading { display: none; align-self: flex-start; padding: 4px 12px; color: var(--fg-muted); font-size: 12px; }
  #loading.show { display: flex; align-items: center; gap: 8px; }
  .spinner {
    width: 12px; height: 12px;
    border: 2px solid var(--border);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  #error {
    display: none;
    margin: 0 12px 6px;
    padding: 8px 10px;
    background: var(--error-bg);
    border: 1px solid var(--error);
    border-radius: 4px;
    color: var(--error);
    font-size: 12px;
    white-space: pre-wrap;
    word-break: break-word;
  }
  #error.show { display: block; }
  #input-area {
    flex-shrink: 0;
    padding: 8px 12px;
    border-top: 1px solid var(--border);
    background: var(--bg-panel);
    display: flex;
    gap: 8px;
    align-items: flex-end;
  }
  #input {
    flex: 1;
    background: var(--bg-input);
    border: 1px solid var(--border);
    color: var(--fg);
    border-radius: 4px;
    padding: 8px 10px;
    font-family: inherit;
    font-size: 13px;
    resize: none;
    min-height: 36px;
    max-height: 140px;
    line-height: 1.4;
  }
  #input:focus { outline: 1px solid var(--accent); border-color: var(--accent); }
  #send {
    background: var(--accent);
    color: #fff;
    border: none;
    border-radius: 4px;
    padding: 8px 14px;
    font-size: 13px;
    cursor: pointer;
    font-weight: 600;
    height: 36px;
  }
  #send:hover { background: var(--accent-hover); }
  #send:disabled { background: #3a3a3a; color: var(--fg-muted); cursor: not-allowed; }
  .empty { color: var(--fg-muted); text-align: center; margin-top: 40px; font-size: 12px; padding: 0 20px; }
  .empty h2 { color: var(--fg); font-size: 14px; margin: 0 0 8px; }
  #messages::-webkit-scrollbar { width: 10px; }
  #messages::-webkit-scrollbar-thumb { background: #424242; border-radius: 5px; }
  #messages::-webkit-scrollbar-track { background: transparent; }
</style>
</head>
<body>
<div id="app">
  <header>
    <div class="title">Open Agent</div>
    <div class="header-actions">
      <span id="health" class="health">checking…</span>
      <button id="tools-btn" class="icon-btn" title="List available tools">Tools</button>
      <button id="clear-btn" class="icon-btn" title="Clear conversation">Clear</button>
    </div>
  </header>
  <div id="messages">
    <div class="empty" id="empty">
      <h2>Open Agent</h2>
      <p>Ask anything. The agent can reason, call tools, and retrieve context to answer.</p>
    </div>
  </div>
  <div id="loading"><span class="spinner"></span> Agent is thinking…</div>
  <div id="error"></div>
  <div id="input-area">
    <textarea id="input" rows="1" placeholder="Ask Open Agent… (Enter to send, Shift+Enter for newline)"></textarea>
    <button id="send">Send</button>
  </div>
</div>
<script>
(function () {
  const vscode = acquireVsCodeApi();
  const messagesEl = document.getElementById('messages');
  const inputEl = document.getElementById('input');
  const sendBtn = document.getElementById('send');
  const loadingEl = document.getElementById('loading');
  const errorEl = document.getElementById('error');
  const healthEl = document.getElementById('health');
  const toolsBtn = document.getElementById('tools-btn');
  const clearBtn = document.getElementById('clear-btn');
  let emptyEl = document.getElementById('empty');
  let busy = false;

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }
  function fmtJson(obj) {
    try { return JSON.stringify(obj, null, 2); } catch (e) { return String(obj); }
  }
  function scrollBottom() { messagesEl.scrollTop = messagesEl.scrollHeight; }
  function setBusy(b) {
    busy = b;
    sendBtn.disabled = b;
    loadingEl.classList.toggle('show', b);
    if (b) { errorEl.classList.remove('show'); }
  }
  function removeEmpty() {
    if (emptyEl && emptyEl.parentNode) { emptyEl.parentNode.removeChild(emptyEl); }
  }
  function showEmpty() {
    const e = document.createElement('div');
    e.className = 'empty';
    e.innerHTML = '<h2>Open Agent</h2><p>Ask anything. The agent can reason, call tools, and retrieve context to answer.</p>';
    messagesEl.innerHTML = '';
    messagesEl.appendChild(e);
    emptyEl = e;
  }
  function addMessage(role, content, opts) {
    opts = opts || {};
    removeEmpty();
    const wrap = document.createElement('div');
    wrap.className = 'msg ' + role;

    const roleLabel = document.createElement('div');
    roleLabel.className = 'role';
    roleLabel.textContent = role === 'user' ? 'You' : 'Assistant';
    wrap.appendChild(roleLabel);

    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    bubble.textContent = content;
    wrap.appendChild(bubble);

    if (role === 'assistant') {
      if (typeof opts.steps === 'number') {
        const meta = document.createElement('div');
        meta.className = 'meta';
        meta.textContent = opts.steps + ' step' + (opts.steps === 1 ? '' : 's');
        wrap.appendChild(meta);
      }
      if (Array.isArray(opts.toolCalls) && opts.toolCalls.length) {
        const tc = document.createElement('div');
        tc.className = 'tool-calls';
        opts.toolCalls.forEach(function (call) {
          const det = document.createElement('details');
          det.className = 'tool' + (call.is_error ? ' tool-error' : '');
          const sum = document.createElement('summary');
          const stepSpan = document.createElement('span');
          stepSpan.className = 'tool-step';
          stepSpan.textContent = '#' + (call.step != null ? call.step : '?');
          const nameSpan = document.createElement('span');
          nameSpan.className = 'tool-name';
          nameSpan.textContent = call.name || 'tool';
          const status = document.createElement('span');
          status.className = 'tool-step';
          status.textContent = call.is_error ? '· error' : '· ok';
          sum.appendChild(stepSpan);
          sum.appendChild(nameSpan);
          sum.appendChild(status);
          det.appendChild(sum);

          const body = document.createElement('div');
          body.className = 'tool-body';
          const argLabel = document.createElement('div');
          argLabel.className = 'tool-label';
          argLabel.textContent = 'Arguments';
          const argPre = document.createElement('pre');
          argPre.textContent = fmtJson(call.arguments);
          const obsLabel = document.createElement('div');
          obsLabel.className = 'tool-label';
          obsLabel.textContent = 'Observation';
          const obsPre = document.createElement('pre');
          obsPre.textContent = call.observation != null ? String(call.observation) : '';
          body.appendChild(argLabel); body.appendChild(argPre);
          body.appendChild(obsLabel); body.appendChild(obsPre);
          det.appendChild(body);
          tc.appendChild(det);
        });
        wrap.appendChild(tc);
      }
    }
    messagesEl.appendChild(wrap);
    scrollBottom();
  }
  function showError(msg) {
    errorEl.textContent = msg;
    errorEl.classList.add('show');
  }
  function send() {
    const text = inputEl.value.trim();
    if (!text || busy) return;
    addMessage('user', text);
    inputEl.value = '';
    autoGrow();
    setBusy(true);
    vscode.postMessage({ type: 'chat', message: text });
  }
  function autoGrow() {
    inputEl.style.height = 'auto';
    inputEl.style.height = Math.min(inputEl.scrollHeight, 140) + 'px';
  }
  function showTools(tools) {
    const lines = tools.map(function (t) {
      return '• ' + escapeHtml(t.name || 'unknown') + ' — ' + escapeHtml(t.description || '');
    });
    const body = lines.length ? lines.join('\\n') : 'No tools available.';
    addMessage('assistant', 'Available tools:\\n\\n' + body, {});
  }

  sendBtn.addEventListener('click', send);
  inputEl.addEventListener('keydown', function (e) {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
  });
  inputEl.addEventListener('input', autoGrow);
  toolsBtn.addEventListener('click', function () { vscode.postMessage({ type: 'getTools' }); });
  clearBtn.addEventListener('click', function () {
    showEmpty();
    errorEl.classList.remove('show');
  });

  window.addEventListener('message', function (event) {
    const msg = event.data;
    if (!msg || typeof msg.type !== 'string') return;
    switch (msg.type) {
      case 'config':
        break;
      case 'health':
        if (msg.ok) { healthEl.textContent = 'online'; healthEl.className = 'health ok'; }
        else { healthEl.textContent = 'offline'; healthEl.className = 'health err'; }
        break;
      case 'loading':
        setBusy(!!msg.loading);
        break;
      case 'chatResponse':
        setBusy(false);
        addMessage('assistant', (msg.result && msg.result.response) || '(empty response)', {
          steps: msg.result ? msg.result.steps : undefined,
          toolCalls: msg.result ? msg.result.tool_calls_made : []
        });
        break;
      case 'chatError':
        setBusy(false);
        showError(msg.error || 'Unknown error');
        break;
      case 'tools':
        showTools(msg.tools || []);
        break;
      case 'toolsError':
        showError(msg.error || 'Failed to load tools');
        break;
      case 'sendPrompt':
        if (msg.message) { inputEl.value = msg.message; autoGrow(); send(); }
        break;
      default:
        break;
    }
  });

  vscode.postMessage({ type: 'ready' });
  inputEl.focus();
})();
</script>
</body>
</html>`;
  }
}
