import * as child_process from 'child_process';
import * as vscode from 'vscode';

import { healthCheckBackend } from './apiClient';

export type BackendStatus = 'stopped' | 'starting' | 'running' | 'failed';

/**
 * Manages the lifecycle of the Open Agent Python backend process.
 *
 * When the extension activates, this manager can automatically spawn the
 * backend as a child process. It monitors stdout/stderr through a VS Code
 * output channel, polls the health endpoint until the server is ready, and
 * kills the process when the extension deactivates.
 */
export class BackendManager {
  private process?: child_process.ChildProcess;
  private status: BackendStatus = 'stopped';
  private readonly outputChannel: vscode.OutputChannel;
  private readonly onStatusChangeEmitter =
    new vscode.EventEmitter<BackendStatus>();

  public readonly onStatusChange = this.onStatusChangeEmitter.event;

  constructor(context: vscode.ExtensionContext) {
    this.outputChannel = vscode.window.createOutputChannel('Open Agent Backend');
    context.subscriptions.push(this.outputChannel);
    context.subscriptions.push(this.onStatusChangeEmitter);
  }

  /** Current lifecycle status of the managed backend. */
  get currentStatus(): BackendStatus {
    return this.status;
  }

  /**
   * Start the backend if ``open-agent.autoStartBackend`` is enabled and no
   * backend is already running at ``open-agent.apiUrl``.
   *
   * Returns ``true`` once the health endpoint responds, or ``false`` if
   * auto-start is disabled or the backend could not be started.
   */
  async start(): Promise<boolean> {
    const config = vscode.workspace.getConfiguration('open-agent');
    const autoStart = config.get<boolean>('autoStartBackend', true);
    const apiUrl = config.get<string>('apiUrl', 'http://localhost:8000');

    if (!autoStart) {
      return false;
    }

    if (this.process) {
      return this.status === 'running';
    }

    // If something is already listening, just use it.
    if (await healthCheckBackend(apiUrl)) {
      this.setStatus('running');
      return true;
    }

    const python = config.get<string>('pythonExecutable', 'python');
    const cwd = config.get<string>('backendCwd', '') || this.getWorkspaceRoot();
    const { host, port } = parseHostPort(apiUrl);

    return new Promise((resolve) => {
      this.setStatus('starting');
      this.outputChannel.appendLine(
        `Starting Open Agent backend at ${apiUrl} (${python})…`
      );
      this.outputChannel.show(true);

      const args = [
        '-m',
        'open_agent.cli',
        'serve',
        '--host',
        host,
        '--port',
        String(port),
      ];

      const env = { ...process.env };
      env.OPEN_AGENT_SERVER_HOST = host;
      env.OPEN_AGENT_SERVER_PORT = String(port);

      try {
        this.process = child_process.spawn(python, args, {
          cwd: cwd || undefined,
          env,
          stdio: ['ignore', 'pipe', 'pipe'],
        });
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        this.outputChannel.appendLine(`Failed to spawn backend: ${message}`);
        this.setStatus('failed');
        void vscode.window.showErrorMessage(
          `Open Agent backend failed to start: ${message}`
        );
        resolve(false);
        return;
      }

      this.process.stdout?.on('data', (data: Buffer) => {
        this.outputChannel.append(data.toString());
      });

      this.process.stderr?.on('data', (data: Buffer) => {
        this.outputChannel.append(data.toString());
      });

      this.process.on('error', (err) => {
        const message = err instanceof Error ? err.message : String(err);
        this.outputChannel.appendLine(`Backend process error: ${message}`);
        if (this.status === 'starting') {
          this.setStatus('failed');
          void vscode.window.showErrorMessage(
            `Open Agent backend error: ${message}`
          );
          resolve(false);
        }
      });

      this.process.on('exit', (code, signal) => {
        const reason = signal ? `signal ${signal}` : `code ${code ?? '?'}`;
        this.outputChannel.appendLine(`Backend process exited (${reason})`);
        this.process = undefined;
        if (this.status === 'starting') {
          this.setStatus('failed');
          resolve(false);
        } else {
          this.setStatus('stopped');
        }
      });

      const startTime = Date.now();
      const timeoutMs = 60000;
      const intervalMs = 500;

      const timer = setInterval(async () => {
        if (this.status !== 'starting') {
          clearInterval(timer);
          return;
        }

        const elapsed = Date.now() - startTime;
        if (elapsed > timeoutMs) {
          clearInterval(timer);
          this.outputChannel.appendLine('Backend startup timed out');
          this.stop();
          this.setStatus('failed');
          void vscode.window.showErrorMessage(
            'Open Agent backend startup timed out. Check the output channel for details.'
          );
          resolve(false);
          return;
        }

        if (await healthCheckBackend(apiUrl)) {
          clearInterval(timer);
          this.setStatus('running');
          this.outputChannel.appendLine('Backend is ready');
          resolve(true);
        }
      }, intervalMs);
    });
  }

  /** Stop the managed backend process, if any. */
  stop(): void {
    if (!this.process) {
      return;
    }
    this.outputChannel.appendLine('Stopping Open Agent backend…');
    this.process.kill();
    this.process = undefined;
    this.setStatus('stopped');
  }

  private setStatus(status: BackendStatus): void {
    this.status = status;
    this.onStatusChangeEmitter.fire(status);
  }

  private getWorkspaceRoot(): string | undefined {
    if (vscode.workspace.workspaceFolders && vscode.workspace.workspaceFolders.length > 0) {
      return vscode.workspace.workspaceFolders[0].uri.fsPath;
    }
    return undefined;
  }
}

function parseHostPort(apiUrl: string): { host: string; port: number } {
  let host = '127.0.0.1';
  let port = 8000;
  try {
    const parsed = new URL(apiUrl);
    host = parsed.hostname || host;
    port = parsed.port ? Number(parsed.port) : port;
  } catch {
    // Keep defaults for malformed URLs.
  }
  return { host, port };
}
