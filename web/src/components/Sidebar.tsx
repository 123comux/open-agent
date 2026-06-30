import { useState, type ReactNode } from "react";
import type { Tool } from "../types";

interface SidebarProps {
  tools: Tool[];
  loading: boolean;
  healthy: boolean | null;
  /** Settings panel rendered in the sidebar footer. */
  children: ReactNode;
}

export function Sidebar({ tools, loading, healthy, children }: SidebarProps) {
  const [toolsCollapsed, setToolsCollapsed] = useState(false);

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
