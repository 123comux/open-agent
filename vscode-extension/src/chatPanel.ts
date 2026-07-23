import * as vscode from 'vscode';
import type { Tool } from '../../shared/types';
import {
  chatWithBackend,
  getToolsFromBackend,
  healthCheckBackend,
  streamChatFromBackend,
} from './apiClient';

/**
 * Provides the Open Agent chat webview view hosted in the activity bar sidebar.
 *
 * The webview is a self-contained single-page app (inline HTML/CSS/JS) that
 * posts messages to this provider; the provider proxies the requests to the
 * Open Agent backend over HTTP/WebSocket (via the helpers in `extension.ts`)
 * and forwards the results back to the webview.
 */
export class ChatPanelProvider implements vscode.WebviewViewProvider {
  private view?: vscode.WebviewView;
  private readonly sessionId: string;
  private cancelStream?: () => void;

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
      case 'stopStream':
        this.cancelStream?.();
        this.cancelStream = undefined;
        this.post({ type: 'streamStopped' });
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
    this.post({ type: 'streamStart' });

    let fallback = true;
    this.cancelStream = streamChatFromBackend(
      this.getApiUrl(),
      message,
      this.sessionId,
      {
        onStart: () => {
          fallback = false;
        },
        onToken: (content) => {
          fallback = false;
          this.post({ type: 'token', content });
        },
        onThought: (content, step) => {
          fallback = false;
          this.post({ type: 'thought', content, step });
        },
        onToolStart: (name, args: Record<string, unknown>) => {
          fallback = false;
          this.post({ type: 'toolStart', name, arguments: args });
        },
        onToolEnd: (name, observation, is_error) => {
          fallback = false;
          this.post({ type: 'toolEnd', name, observation, is_error });
        },
        onDone: (response, steps, tool_calls_made) => {
          fallback = false;
          this.post({ type: 'streamDone', response, steps, tool_calls_made });
          this.cancelStream = undefined;
        },
        onError: async (error) => {
          this.cancelStream = undefined;
          if (fallback) {
            // WebSocket unavailable; fall back to plain HTTP chat.
            try {
              const result = await chatWithBackend(
                this.getApiUrl(),
                message,
                this.sessionId
              );
              this.post({ type: 'streamDone', ...result, tool_calls_made: result.tool_calls_made });
            } catch (err) {
              const errorText = err instanceof Error ? err.message : String(err);
              this.post({ type: 'streamError', error: errorText });
            }
          } else {
            this.post({ type: 'streamError', error });
          }
        },
      }
    );
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
  .thinking {
    width: 100%;
    margin-top: 6px;
    border: 1px solid var(--border);
    border-radius: 4px;
    background: var(--tool-bg);
  }
  .thinking summary {
    cursor: pointer;
    padding: 4px 8px;
    font-size: 11px;
    color: var(--fg-muted);
    list-style: none;
  }
  .thinking summary::-webkit-details-marker { display: none; }
  .thinking summary::before { content: "\\25B6"; font-size: 8px; margin-right: 6px; transition: transform .15s; }
  .thinking[open] summary::before { transform: rotate(90deg); }
  .thinking-list {
    padding: 0 8px 6px;
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  .thinking-item {
    font-size: 11px;
    color: var(--fg-muted);
    display: flex;
    gap: 6px;
  }
  .thinking-item::before { content: "\\2192"; color: var(--success); }
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
    min-width: 64px;
  }
  #send:hover { background: var(--accent-hover); }
  #send:disabled { background: #3a3a3a; color: var(--fg-muted); cursor: not-allowed; }
  #send.stop {
    background: var(--error);
    color: #fff;
  }
  #send.stop:hover { background: #d96c5a; }
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
  let currentAssistant = null;
  let currentToolCalls = [];
  let currentThoughts = [];

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
    loadingEl.classList.toggle('show', b);
    if (b) {
      errorEl.classList.remove('show');
      sendBtn.textContent = 'Stop';
      sendBtn.classList.add('stop');
      sendBtn.disabled = false;
    } else {
      sendBtn.textContent = 'Send';
      sendBtn.classList.remove('stop');
      sendBtn.disabled = !inputEl.value.trim();
    }
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
    currentAssistant = null;
    currentToolCalls = [];
    currentThoughts = [];
  }
  function addUserMessage(content) {
    removeEmpty();
    const wrap = document.createElement('div');
    wrap.className = 'msg user';
    const roleLabel = document.createElement('div');
    roleLabel.className = 'role';
    roleLabel.textContent = 'You';
    wrap.appendChild(roleLabel);
    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    bubble.textContent = content;
    wrap.appendChild(bubble);
    messagesEl.appendChild(wrap);
    scrollBottom();
  }
  function startAssistantMessage() {
    removeEmpty();
    currentAssistant = { wrap: document.createElement('div'), bubble: document.createElement('div'), meta: null, toolCallsEl: null, thinkingEl: null };
    currentAssistant.wrap.className = 'msg assistant';
    const roleLabel = document.createElement('div');
    roleLabel.className = 'role';
    roleLabel.textContent = 'Assistant';
    currentAssistant.wrap.appendChild(roleLabel);
    currentAssistant.bubble.className = 'bubble';
    currentAssistant.wrap.appendChild(currentAssistant.bubble);
    messagesEl.appendChild(currentAssistant.wrap);
    currentToolCalls = [];
    currentThoughts = [];
    scrollBottom();
  }
  function appendToken(content) {
    if (!currentAssistant) startAssistantMessage();
    currentAssistant.bubble.textContent += content;
    scrollBottom();
  }
  function renderThinking() {
    if (!currentAssistant || currentThoughts.length === 0) return;
    if (!currentAssistant.thinkingEl) {
      currentAssistant.thinkingEl = document.createElement('details');
      currentAssistant.thinkingEl.className = 'thinking';
      const sum = document.createElement('summary');
      sum.textContent = 'Thinking Chain';
      currentAssistant.thinkingEl.appendChild(sum);
      const list = document.createElement('div');
      list.className = 'thinking-list';
      currentAssistant.thinkingEl.appendChild(list);
      currentAssistant.wrap.appendChild(currentAssistant.thinkingEl);
    }
    const list = currentAssistant.thinkingEl.querySelector('.thinking-list');
    list.innerHTML = '';
    currentThoughts.forEach(function (t) {
      const item = document.createElement('div');
      item.className = 'thinking-item';
      item.textContent = t;
      list.appendChild(item);
    });
    currentAssistant.thinkingEl.open = true;
    scrollBottom();
  }
  function addThought(content, step) {
    if (!currentAssistant) startAssistantMessage();
    currentThoughts.push('[' + (step || currentThoughts.length + 1) + '] ' + content);
    renderThinking();
  }
  function ensureToolCalls() {
    if (!currentAssistant.toolCallsEl) {
      currentAssistant.toolCallsEl = document.createElement('div');
      currentAssistant.toolCallsEl.className = 'tool-calls';
      currentAssistant.wrap.appendChild(currentAssistant.toolCallsEl);
    }
    return currentAssistant.toolCallsEl;
  }
  function startTool(name, args) {
    if (!currentAssistant) startAssistantMessage();
    currentToolCalls.push({ name: name || 'tool', arguments: args || {}, observation: '', is_error: false, el: null });
    renderToolCalls();
  }
  function endTool(name, observation, isError) {
    const call = currentToolCalls.slice().reverse().find(function (c) { return c.name === name && c.observation === ''; });
    if (call) {
      call.observation = observation != null ? String(observation) : '';
      call.is_error = !!isError;
    }
    renderToolCalls();
  }
  function renderToolCalls() {
    const container = ensureToolCalls();
    container.innerHTML = '';
    currentToolCalls.forEach(function (call, idx) {
      const det = document.createElement('details');
      det.className = 'tool' + (call.is_error ? ' tool-error' : '');
      det.open = call.observation === '';
      const sum = document.createElement('summary');
      const stepSpan = document.createElement('span');
      stepSpan.className = 'tool-step';
      stepSpan.textContent = '#' + (idx + 1);
      const nameSpan = document.createElement('span');
      nameSpan.className = 'tool-name';
      nameSpan.textContent = call.name;
      const status = document.createElement('span');
      status.className = 'tool-step';
      status.textContent = call.observation === '' ? '· running' : (call.is_error ? '· error' : '· ok');
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
      body.appendChild(argLabel); body.appendChild(argPre);

      if (call.observation !== '') {
        const obsLabel = document.createElement('div');
        obsLabel.className = 'tool-label';
        obsLabel.textContent = 'Observation';
        const obsPre = document.createElement('pre');
        obsPre.textContent = call.observation;
        body.appendChild(obsLabel); body.appendChild(obsPre);
      }
      det.appendChild(body);
      container.appendChild(det);
    });
    scrollBottom();
  }
  function finalizeAssistant(response, steps, toolCalls) {
    if (!currentAssistant) startAssistantMessage();
    currentAssistant.bubble.textContent = response || '';
    if (typeof steps === 'number' && steps > 0) {
      const meta = document.createElement('div');
      meta.className = 'meta';
      meta.textContent = steps + ' step' + (steps === 1 ? '' : 's');
      currentAssistant.wrap.appendChild(meta);
    }
    if (Array.isArray(toolCalls) && toolCalls.length > 0) {
      currentToolCalls = toolCalls.map(function (c, i) {
        return {
          name: c.name || 'tool',
          arguments: c.arguments || {},
          observation: c.observation != null ? String(c.observation) : '',
          is_error: !!c.is_error
        };
      });
      renderToolCalls();
    }
    currentAssistant = null;
    currentToolCalls = [];
    currentThoughts = [];
    scrollBottom();
  }
  function showError(msg) {
    errorEl.textContent = msg;
    errorEl.classList.add('show');
  }
  function send() {
    const text = inputEl.value.trim();
    if (busy) {
      vscode.postMessage({ type: 'stopStream' });
      return;
    }
    if (!text) return;
    addUserMessage(text);
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
    addUserMessage('');
    startAssistantMessage();
    currentAssistant.bubble.textContent = 'Available tools:\\n\\n' + body;
    currentAssistant = null;
    scrollBottom();
  }

  sendBtn.addEventListener('click', send);
  inputEl.addEventListener('keydown', function (e) {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
  });
  inputEl.addEventListener('input', function () {
    autoGrow();
    if (!busy) { sendBtn.disabled = !inputEl.value.trim(); }
  });
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
      case 'streamStart':
        setBusy(true);
        break;
      case 'token':
        appendToken(msg.content || '');
        break;
      case 'thought':
        addThought(msg.content || '', msg.step);
        break;
      case 'toolStart':
        startTool(msg.name, msg.arguments);
        break;
      case 'toolEnd':
        endTool(msg.name, msg.observation, msg.is_error);
        break;
      case 'streamDone':
      case 'chatResponse':
        setBusy(false);
        finalizeAssistant(msg.response || (msg.result && msg.result.response), msg.steps || (msg.result && msg.result.steps), msg.tool_calls_made || (msg.result && msg.result.tool_calls_made));
        break;
      case 'streamStopped':
        setBusy(false);
        if (currentAssistant) {
          currentAssistant = null;
          currentToolCalls = [];
          currentThoughts = [];
        }
        break;
      case 'streamError':
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
