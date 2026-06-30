import { useEffect, useState } from "react";

interface SettingsProps {
  apiUrl: string;
  onApiUrlChange: (url: string) => void;
  sessionId: string;
  onRegenerateSession: () => void;
  healthy: boolean | null;
}

export function Settings({
  apiUrl,
  onApiUrlChange,
  sessionId,
  onRegenerateSession,
  healthy,
}: SettingsProps) {
  const [open, setOpen] = useState(false);
  const [draft, setDraft] = useState(apiUrl);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    setDraft(apiUrl);
  }, [apiUrl]);

  const save = () => {
    onApiUrlChange(draft.trim());
  };

  const copySession = async () => {
    try {
      await navigator.clipboard.writeText(sessionId);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard unavailable */
    }
  };

  const statusLabel =
    healthy === null ? "Checking…" : healthy ? "Online" : "Offline";
  const statusColor =
    healthy === null
      ? "text-amber-300"
      : healthy
      ? "text-emerald-300"
      : "text-rose-300";

  return (
    <div className="rounded-lg border border-white/5 bg-white/[0.02]">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center gap-2 px-3 py-2 text-left transition-colors hover:bg-white/5"
      >
        <svg
          className="h-4 w-4 text-zinc-400"
          viewBox="0 0 20 20"
          fill="currentColor"
          aria-hidden="true"
        >
          <path
            fillRule="evenodd"
            d="M7.84 1.804A1 1 0 018.82 1h2.36a1 1 0 01.98.804l.331 1.652a6.993 6.993 0 011.929 1.115l1.598-.54a1 1 0 011.186.447l1.18 2.044a1 1 0 01-.205 1.251l-1.267 1.113a7.047 7.047 0 010 2.228l1.267 1.113a1 1 0 01.206 1.25l-1.18 2.045a1 1 0 01-1.187.447l-1.598-.54a6.993 6.993 0 01-1.929 1.115l-.33 1.652a1 1 0 01-.98.804H8.82a1 1 0 01-.98-.804l-.331-1.652a6.993 6.993 0 01-1.929-1.115l-1.598.54a1 1 0 01-1.186-.447l-1.18-2.044a1 1 0 01.205-1.251l1.267-1.114a7.05 7.05 0 010-2.227L1.821 7.773a1 1 0 01-.206-1.25l1.18-2.045a1 1 0 011.187-.447l1.598.54A6.993 6.993 0 017.51 3.456l.33-1.652zM10 13a3 3 0 100-6 3 3 0 000 6z"
            clipRule="evenodd"
          />
        </svg>
        <span className="text-xs font-semibold uppercase tracking-wider text-zinc-400">
          Settings
        </span>
        <svg
          className={`ml-auto h-4 w-4 text-zinc-500 transition-transform ${
            open ? "rotate-180" : ""
          }`}
          viewBox="0 0 20 20"
          fill="currentColor"
          aria-hidden="true"
        >
          <path
            fillRule="evenodd"
            d="M5.23 7.21a.75.75 0 011.06.02L10 11.06l3.71-3.83a.75.75 0 111.08 1.04l-4.25 4.39a.75.75 0 01-1.08 0L5.21 8.27a.75.75 0 01.02-1.06z"
            clipRule="evenodd"
          />
        </svg>
      </button>

      {open && (
        <div className="space-y-4 border-t border-white/5 px-3 py-3">
          <div>
            <label className="mb-1 block text-[10px] font-medium uppercase tracking-wider text-zinc-500">
              API URL
            </label>
            <div className="flex gap-2">
              <input
                value={draft}
                onChange={(e) => setDraft(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") save();
                }}
                placeholder="http://localhost:8000"
                spellCheck={false}
                className="min-w-0 flex-1 rounded-md border border-white/10 bg-black/30 px-2 py-1.5 font-mono text-xs text-zinc-200 outline-none focus:border-emerald-400/40"
              />
              <button
                type="button"
                onClick={save}
                className="rounded-md bg-emerald-500 px-2.5 py-1.5 text-xs font-semibold text-zinc-950 transition-colors hover:bg-emerald-400"
              >
                Save
              </button>
            </div>
            <p className="mt-1 text-[10px] text-zinc-600">
              Leave blank to use same-origin (dev proxy).
            </p>
          </div>

          <div>
            <div className="mb-1 flex items-center justify-between">
              <label className="text-[10px] font-medium uppercase tracking-wider text-zinc-500">
                Session
              </label>
              <button
                type="button"
                onClick={onRegenerateSession}
                className="text-[10px] font-medium text-emerald-300 hover:text-emerald-200"
              >
                Regenerate
              </button>
            </div>
            <div className="flex items-center gap-2 rounded-md border border-white/10 bg-black/30 px-2 py-1.5">
              <span className="truncate font-mono text-xs text-zinc-300">
                {sessionId}
              </span>
              <button
                type="button"
                onClick={copySession}
                className="ml-auto shrink-0 text-zinc-500 transition-colors hover:text-zinc-300"
                title="Copy session ID"
                aria-label="Copy session ID"
              >
                {copied ? (
                  <svg
                    className="h-3.5 w-3.5 text-emerald-400"
                    viewBox="0 0 20 20"
                    fill="currentColor"
                  >
                    <path
                      fillRule="evenodd"
                      d="M16.704 4.153a.75.75 0 01.143 1.052l-8 10.5a.75.75 0 01-1.127.075l-4.5-4.5a.75.75 0 011.06-1.06l3.894 3.893 7.48-9.817a.75.75 0 011.05-.143z"
                      clipRule="evenodd"
                    />
                  </svg>
                ) : (
                  <svg
                    className="h-3.5 w-3.5"
                    viewBox="0 0 20 20"
                    fill="currentColor"
                  >
                    <path d="M7 3.5A1.5 1.5 0 018.5 2h3.879a1.5 1.5 0 011.06.44l3.122 3.12A1.5 1.5 0 0117 6.622V12.5a1.5 1.5 0 01-1.5 1.5h-1v-3.379a3 3 0 00-.879-2.121L10.5 5.379A3 3 0 008.379 4.5H7v-1z" />
                    <path d="M4.5 6A1.5 1.5 0 003 7.5v9A1.5 1.5 0 004.5 18h7a1.5 1.5 0 001.5-1.5v-5.879a1.5 1.5 0 00-.44-1.06L9.44 6.439A1.5 1.5 0 008.378 6H4.5z" />
                  </svg>
                )}
              </button>
            </div>
          </div>

          <div className="space-y-1.5">
            <div className="flex items-center justify-between">
              <span className="text-[10px] font-medium uppercase tracking-wider text-zinc-500">
                Model
              </span>
              <span className="font-mono text-xs text-zinc-300">Open Agent</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-[10px] font-medium uppercase tracking-wider text-zinc-500">
                Backend
              </span>
              <span className={`text-xs font-medium ${statusColor}`}>
                {statusLabel}
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
