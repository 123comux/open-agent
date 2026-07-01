import { useRef, useState, type ReactNode } from "react";
import type { Tool } from "../types";

interface SidebarProps {
  tools: Tool[];
  loading: boolean;
  healthy: boolean | null;
  sessions: string[];
  activeSessionId: string;
  onSelectSession: (id: string) => void;
  onNewSession: () => void;
  onDeleteSession: (id: string) => void;
  onUploadFile: (file: File) => Promise<void>;
  /** Settings panel rendered in the sidebar footer. */
  children: ReactNode;
}

export function Sidebar({
  tools,
  loading,
  healthy,
  sessions,
  activeSessionId,
  onSelectSession,
  onNewSession,
  onDeleteSession,
  onUploadFile,
  children,
}: SidebarProps) {
  const [toolsCollapsed, setToolsCollapsed] = useState(false);
  const [sessionsCollapsed, setSessionsCollapsed] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadMsg, setUploadMsg] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  return (
    <aside className="flex w-72 shrink-0 flex-col border-r border-white/5 bg-zinc-950/40">
      <div className="flex items-center gap-3 px-4 py-4">
        <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-emerald-400 to-teal-600 text-sm font-bold text-zinc-950">
          OA
        </div>
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <h1 className="truncate text-sm font-semibold text-zinc-100">
              Open Agent
            </h1>
            <StatusDot healthy={healthy} />
          </div>
          <p className="truncate text-xs text-zinc-500">v0.1.0</p>
        </div>
      </div>

      <div className="scrollbar-thin flex-1 overflow-y-auto px-3 pb-4">
        <div className="mb-3">
          <div className="flex items-center gap-2 px-2 py-2">
            <span className="text-xs font-semibold uppercase tracking-wider text-zinc-400">
              Sessions
            </span>
            <button
              type="button"
              onClick={onNewSession}
              className="ml-auto flex items-center gap-1 rounded-md border border-white/5 bg-white/5 px-2 py-1 text-[11px] font-medium text-zinc-300 transition-colors hover:border-emerald-400/30 hover:bg-emerald-400/10 hover:text-emerald-300"
            >
              <svg
                className="h-3 w-3"
                viewBox="0 0 20 20"
                fill="currentColor"
                aria-hidden="true"
              >
                <path d="M10 3a.75.75 0 01.75.75v5.5h5.5a.75.75 0 010 1.5h-5.5v5.5a.75.75 0 01-1.5 0v-5.5h-5.5a.75.75 0 010-1.5h5.5v-5.5A.75.75 0 0110 3z" />
              </svg>
              New
            </button>
          </div>
          <button
            type="button"
            onClick={() => setSessionsCollapsed((v) => !v)}
            className="flex w-full items-center gap-2 rounded-lg px-2 py-1 text-left text-[11px] text-zinc-500 transition-colors hover:bg-white/5"
          >
            <span>
              {sessions.length} session{sessions.length === 1 ? "" : "s"}
            </span>
            <svg
              className={`ml-auto h-3.5 w-3.5 text-zinc-600 transition-transform ${
                sessionsCollapsed ? "" : "rotate-180"
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
          {!sessionsCollapsed && (
            <div className="mt-1 space-y-1">
              <button
                type="button"
                onClick={onNewSession}
                className="flex w-full items-center gap-2 rounded-lg border border-emerald-400/30 bg-emerald-400/10 px-2.5 py-2 text-left text-xs text-zinc-100 transition-colors hover:bg-emerald-400/15"
              >
                <span className="h-1.5 w-1.5 shrink-0 rounded-full bg-emerald-400" />
                <span className="truncate font-mono">{activeSessionId}</span>
                {!sessions.includes(activeSessionId) && (
                  <span className="ml-auto text-[10px] uppercase tracking-wider text-emerald-400">
                    current
                  </span>
                )}
              </button>
              {sessions
                .filter((s) => s !== activeSessionId)
                .map((s) => (
                  <SessionItem
                    key={s}
                    id={s}
                    active={s === activeSessionId}
                    onSelect={onSelectSession}
                    onDelete={onDeleteSession}
                  />
                ))}
              {sessions.length === 0 && (
                <p className="px-2 py-1.5 text-[11px] text-zinc-600">
                  No saved sessions yet.
                </p>
              )}
            </div>
          )}
        </div>

        {/* File Upload Section */}
        <div className="mb-3">
          <div className="px-2 py-2">
            <span className="text-xs font-semibold uppercase tracking-wider text-zinc-400">
              Knowledge Base
            </span>
          </div>
          <div
            onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragOver(false);
              const f = e.dataTransfer.files[0];
              if (f) {
                setUploading(true);
                setUploadMsg(null);
                onUploadFile(f)
                  .then(() => setUploadMsg(`Indexed: ${f.name}`))
                  .catch((err) => setUploadMsg(`Error: ${err.message}`))
                  .finally(() => setUploading(false));
              }
            }}
            onClick={() => fileInputRef.current?.click()}
            className={`cursor-pointer rounded-lg border border-dashed px-3 py-4 text-center transition-colors ${
              dragOver
                ? "border-emerald-400/50 bg-emerald-400/10"
                : "border-white/10 bg-white/[0.02] hover:border-white/20 hover:bg-white/5"
            }`}
          >
            {uploading ? (
              <p className="text-xs text-emerald-400">Indexing…</p>
            ) : (
              <>
                <svg className="mx-auto mb-1 h-5 w-5 text-zinc-500" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
                  <path d="M5.5 17a4.5 4.5 0 01-1.44-8.77 5.5 5.5 0 0110.43-2.49 4.5 4.5 0 011.51 8.76H5.5zM8.5 7.25a.75.75 0 00-1.5 0v4.69L5.78 10.72a.75.75 0 00-1.06 1.06l2.75 2.75a.75.75 0 001.06 0l2.75-2.75a.75.75 0 10-1.06-1.06L8.5 11.94V7.25z" />
                </svg>
                <p className="text-[11px] text-zinc-500">Drop file or click to upload</p>
                <p className="mt-0.5 text-[10px] text-zinc-600">PDF, DOCX, TXT, MD, CSV, JSON</p>
              </>
            )}
            <input
              ref={fileInputRef}
              type="file"
              className="hidden"
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) {
                  setUploading(true);
                  setUploadMsg(null);
                  onUploadFile(f)
                    .then(() => setUploadMsg(`Indexed: ${f.name}`))
                    .catch((err) => setUploadMsg(`Error: ${err.message}`))
                    .finally(() => setUploading(false));
                }
              }}
            />
          </div>
          {uploadMsg && (
            <p className={`mt-1.5 px-2 text-[11px] ${uploadMsg.startsWith("Error") ? "text-rose-400" : "text-emerald-400"}`}>
              {uploadMsg}
            </p>
          )}
        </div>

        <button
          type="button"
          onClick={() => setToolsCollapsed((v) => !v)}
          className="flex w-full items-center gap-2 rounded-lg px-2 py-2 text-left transition-colors hover:bg-white/5"
        >
          <span className="text-xs font-semibold uppercase tracking-wider text-zinc-400">
            Tools{tools.length > 0 ? ` (${tools.length})` : ""}
          </span>
          <svg
            className={`ml-auto h-4 w-4 text-zinc-500 transition-transform ${
              toolsCollapsed ? "" : "rotate-180"
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

        {!toolsCollapsed && (
          <div className="space-y-1.5">
            {loading ? (
              <div className="space-y-2 px-1 py-2">
                {[0, 1, 2].map((i) => (
                  <div
                    key={i}
                    className="h-10 animate-pulse rounded-lg bg-white/5"
                  />
                ))}
              </div>
            ) : tools.length === 0 ? (
              <p className="px-2 py-3 text-xs text-zinc-600">
                No tools available.
              </p>
            ) : (
              tools.map((t) => <ToolItem key={t.name} tool={t} />)
            )}
          </div>
        )}
      </div>

      <div className="border-t border-white/5 p-3">{children}</div>
    </aside>
  );
}

function ToolItem({ tool }: { tool: Tool }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="overflow-hidden rounded-lg border border-white/5 bg-white/[0.02]">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center gap-2 px-3 py-2 text-left transition-colors hover:bg-white/5"
      >
        <span className="h-1.5 w-1.5 shrink-0 rounded-full bg-emerald-400/70" />
        <span className="truncate font-mono text-xs font-medium text-zinc-200">
          {tool.name}
        </span>
        <svg
          className={`ml-auto h-3.5 w-3.5 shrink-0 text-zinc-600 transition-transform ${
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
        <div className="border-t border-white/5 px-3 py-2.5">
          <p className="text-xs leading-relaxed text-zinc-400">
            {tool.description}
          </p>
          <div className="mt-2">
            <div className="mb-1 text-[10px] uppercase tracking-wider text-zinc-600">
              Parameters
            </div>
            <pre className="scrollbar-thin overflow-x-auto rounded-md bg-black/30 p-2 font-mono text-[11px] leading-relaxed text-zinc-400">
              {JSON.stringify(tool.parameters, null, 2)}
            </pre>
          </div>
        </div>
      )}
    </div>
  );
}

function SessionItem({
  id,
  active,
  onSelect,
  onDelete,
}: {
  id: string;
  active: boolean;
  onSelect: (id: string) => void;
  onDelete: (id: string) => void;
}) {
  return (
    <div
      className={`group flex w-full items-center gap-2 rounded-lg border px-2.5 py-2 text-xs transition-colors ${
        active
          ? "border-emerald-400/30 bg-emerald-400/10 text-zinc-100"
          : "border-white/5 bg-white/[0.02] text-zinc-400 hover:bg-white/5"
      }`}
    >
      <button
        type="button"
        onClick={() => onSelect(id)}
        className="flex min-w-0 flex-1 items-center gap-2 text-left"
      >
        <span
          className={`h-1.5 w-1.5 shrink-0 rounded-full ${
            active ? "bg-emerald-400" : "bg-zinc-600"
          }`}
        />
        <span className="truncate font-mono">{id}</span>
      </button>
      <button
        type="button"
        onClick={(e) => {
          e.stopPropagation();
          onDelete(id);
        }}
        aria-label={`Delete session ${id}`}
        className="shrink-0 rounded p-1 text-zinc-600 opacity-0 transition-colors hover:bg-rose-500/10 hover:text-rose-400 focus:opacity-100 group-hover:opacity-100"
      >
        <svg
          className="h-3.5 w-3.5"
          viewBox="0 0 20 20"
          fill="currentColor"
          aria-hidden="true"
        >
          <path
            fillRule="evenodd"
            d="M8.75 1A2.75 2.75 0 006 3.75v.5H3.75a.75.75 0 000 1.5h.5l.6 9.4A2.75 2.75 0 007.59 18h4.82a2.75 2.75 0 002.74-2.85l.6-9.4h.5a.75.75 0 000-1.5H14v-.5A2.75 2.75 0 0011.25 1h-2.5zM7.5 3.75c0-.69.56-1.25 1.25-1.25h2.5c.69 0 1.25.56 1.25 1.25v.5h-5v-.5zm-1.7 2.15a.75.75 0 01.8.7l.55 8.4a1.25 1.25 0 002.49 0l.55-8.4a.75.75 0 011.5.1l-.55 8.4a2.75 2.75 0 01-5.49 0l-.55-8.4a.75.75 0 01.7-.8z"
            clipRule="evenodd"
          />
        </svg>
      </button>
    </div>
  );
}

function StatusDot({ healthy }: { healthy: boolean | null }) {
  if (healthy === null) {
    return (
      <span
        className="h-2 w-2 animate-pulse rounded-full bg-amber-400"
        title="Checking…"
      />
    );
  }
  if (healthy) {
    return <span className="h-2 w-2 rounded-full bg-emerald-400" title="Online" />;
  }
  return <span className="h-2 w-2 rounded-full bg-rose-500" title="Offline" />;
}
