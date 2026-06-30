# Open Agent for VS Code

A sidebar chat panel for the [Open Agent](../) agentic RAG autonomous work
assistant. The extension adds an activity-bar entry that hosts a chat UI
communicating with the Open Agent Python backend over HTTP.

## Features

- **Sidebar chat panel** — a self-contained webview chat UI with a dark theme
  matching VS Code, message bubbles (user right-aligned, assistant
  left-aligned), and a bottom input box with a Send button.
- **Tool call details** — each assistant response can expand collapsible
  sections showing the tool name, arguments, and observation for every tool
  invocation the agent made.
- **Backend health indicator** — the header shows whether the Open Agent API
  server is reachable (`online` / `offline`).
- **Tool listing** — a `Tools` button fetches and displays the tools the agent
  has registered.
- **Loading & error states** — a spinner while the agent is thinking and an
  inline error banner when a request fails.
- **Configurable API URL** — point the extension at any running Open Agent
  server via a setting or command.

## Requirements

- VS Code 1.85 or newer.
- A running Open Agent API server (the `open_agent.server.api` FastAPI app).
  Start it with the `open-agent-server` console script (defaults to
  `http://localhost:8000`).

## Extension Settings

This extension contributes the following setting:

| Setting            | Default                  | Description                          |
| ------------------ | ------------------------ | ------------------------------------ |
| `open-agent.apiUrl`| `http://localhost:8000`  | URL of the Open Agent API server.    |

## Commands

- **Open Agent: Set API URL** (`open-agent.setApiUrl`) — prompts for the
  backend URL and saves it to `open-agent.apiUrl`.
- **Open Agent: Send Prompt** (`open-agent.sendPrompt`) — prompts for a message
  and sends it to the chat panel.

## Usage

1. Start the Open Agent backend server (e.g. `open-agent-server`).
2. Open the Open Agent icon in the activity bar.
3. The header should read `online`. If not, run
   **Open Agent: Set API URL** and point it at your server.
4. Type a message in the input box and press **Enter** (or click **Send**).
   Use **Shift+Enter** for a newline.
5. Expand the collapsible tool-call sections under an assistant reply to
   inspect the agent's tool usage. Click **Tools** to list available tools,
   or **Clear** to reset the conversation.

## Build

```bash
cd vscode-extension
npm install
npm run compile      # produce out/extension.js
npm run watch        # recompile on change during development
```

Press <kbd>F5</kbd> in VS Code to launch an Extension Development Host with the
extension loaded.

## Architecture

- `src/extension.ts` — entry point. Registers the webview view provider and
  commands, and proxies backend requests using Node's built-in `http`/`https`
  modules (`/api/chat`, `/api/tools`, `/api/health`).
- `src/chatPanel.ts` — `ChatPanelProvider` implementing
  `vscode.WebviewViewProvider`. Renders the inline HTML chat UI, forwards
  webview `postMessage` requests to the backend, and relays results back.
- `media/icon.svg` — activity-bar icon.
