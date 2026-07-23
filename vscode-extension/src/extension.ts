import * as vscode from 'vscode';

import { BackendManager } from './backendManager';
import { ChatPanelProvider } from './chatPanel';

export type { ChatResponse, Tool } from './apiClient';
export {
  chatWithBackend,
  getToolsFromBackend,
  healthCheckBackend,
  streamChatFromBackend,
} from './apiClient';

let backendManager: BackendManager | undefined;

/**
 * Extension entry point. Registers the chat webview provider and the
 * `setApiUrl` / `sendPrompt` commands, and optionally starts the Python
 * backend process locally.
 */
export function activate(context: vscode.ExtensionContext): void {
  backendManager = new BackendManager(context);
  const provider = new ChatPanelProvider(context);

  // Start the backend automatically unless disabled. Fire-and-forget; the
  // chat panel will probe connectivity independently.
  void backendManager.start();

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

  context.subscriptions.push(
    vscode.commands.registerCommand('open-agent.startBackend', async () => {
      const started = await backendManager?.start();
      if (started) {
        void vscode.window.showInformationMessage('Open Agent backend started');
      } else if (started === false) {
        void vscode.window.showErrorMessage('Open Agent backend failed to start');
      }
    })
  );

  context.subscriptions.push(
    vscode.commands.registerCommand('open-agent.stopBackend', () => {
      backendManager?.stop();
      void vscode.window.showInformationMessage('Open Agent backend stopped');
    })
  );

  context.subscriptions.push(
    vscode.commands.registerCommand('open-agent.restartBackend', async () => {
      backendManager?.stop();
      const started = await backendManager?.start();
      if (started) {
        void vscode.window.showInformationMessage('Open Agent backend restarted');
      } else {
        void vscode.window.showErrorMessage('Open Agent backend failed to restart');
      }
    })
  );
}

export function deactivate(): void {
  backendManager?.stop();
}
