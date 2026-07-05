import { useCallback, useEffect, useState } from "react";
import { ChatClient } from "../api/client";
import type { Trace, TraceSpan } from "../types";

interface TraceDashboardProps {
  client: ChatClient;
}

export function TraceDashboard({ client }: TraceDashboardProps) {
  const [traces, setTraces] = useState<Trace[]>([]);
  const [loading, setLoading] = useState(false);
  const [selected, setSelected] = useState<Trace | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const list = await client.listTraces(50);
      setTraces(list);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }, [client]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const selectTrace = useCallback(
    async (trace: Trace) => {
      const full = await client.getTrace(trace.id);
      setSelected(full ?? trace);
    },
    [client]
  );

  return (
    <div className="flex h-full w-full flex-col bg-[#0a0a0f]">
      <header className="flex items-center justify-between border-b border-white/5 px-6 py-4">
        <div>
          <h2 className="text-sm font-semibold text-zinc-100">Observability</h2>
          <p className="text-xs text-zinc-500">Local trace viewer</p>
        </div>
        <button
          type="button"
          onClick={refresh}
          disabled={loading}
          className="rounded-md border border-white/10 bg-white/5 px-3 py-1.5 text-xs font-medium text-zinc-300 transition-colors hover:bg-white/10 disabled:opacity-50"
        >
          {loading ? "Loading…" : "Refresh"}
        </button>
      </header>

      <div className="flex min-h-0 flex-1">
        <div className="w-80 shrink-0 overflow-y-auto border-r border-white/5 p-4">
          {error && (
            <div className="mb-3 rounded-md border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-xs text-rose-300">
              {error}
            </div>
          )}
          {traces.length === 0 ? (
            <p className="text-xs text-zinc-600">
              No traces yet. Enable local observability and run a chat to generate traces.
            </p>
          ) : (
            <div className="space-y-2">
              {traces.map((trace) => (
                <button
                  key={trace.id}
                  type="button"
                  onClick={() => selectTrace(trace)}
                  className={`w-full rounded-lg border px-3 py-2 text-left transition-colors ${
                    selected?.id === trace.id
                      ? "border-emerald-400/30 bg-emerald-400/10"
                      : "border-white/5 bg-white/[0.02] hover:bg-white/5"
                  }`}
                >
                  <div className="flex items-center gap-2">
                    <StatusBadge status={trace.status} />
                    <span className="truncate font-mono text-xs text-zinc-300">{trace.id}</span>
                  </div>
                  <p className="mt-1 truncate text-[11px] text-zinc-500">
                    {trace.name} · {formatTime(trace.start_time)}
                  </p>
                </button>
              ))}
            </div>
          )}
        </div>

        <div className="min-w-0 flex-1 overflow-y-auto p-6">
          {selected ? (
            <TraceDetail trace={selected} />
          ) : (
            <div className="flex h-full items-center justify-center text-xs text-zinc-600">
              Select a trace to view details.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function TraceDetail({ trace }: { trace: Trace }) {
  const duration =
    trace.end_time && trace.start_time
      ? Math.round(
          (new Date(trace.end_time).getTime() - new Date(trace.start_time).getTime()) / 10
        ) / 100
      : null;

  return (
    <div className="space-y-4">
      <div className="rounded-lg border border-white/5 bg-white/[0.02] p-4">
        <div className="flex items-center gap-2">
          <StatusBadge status={trace.status} />
          <h3 className="text-sm font-semibold text-zinc-100">{trace.name}</h3>
        </div>
        <div className="mt-2 grid grid-cols-2 gap-2 text-[11px] text-zinc-400">
          <div>
            <span className="text-zinc-600">ID:</span> {trace.id}
          </div>
          <div>
            <span className="text-zinc-600">Duration:</span>{" "}
            {duration !== null ? `${duration}s` : "running"}
          </div>
          <div>
            <span className="text-zinc-600">Start:</span> {formatTime(trace.start_time)}
          </div>
          <div>
            <span className="text-zinc-600">Status:</span> {trace.status}
          </div>
        </div>
        {trace.input && Object.keys(trace.input).length > 0 && (
          <div className="mt-3">
            <div className="mb-1 text-[10px] uppercase tracking-wider text-zinc-600">Input</div>
            <pre className="scrollbar-thin overflow-x-auto rounded-md bg-black/30 p-2 font-mono text-[11px] text-zinc-400">
              {JSON.stringify(trace.input, null, 2)}
            </pre>
          </div>
        )}
      </div>

      {trace.root_span && (
        <div>
          <h4 className="mb-2 text-xs font-semibold uppercase tracking-wider text-zinc-500">
            Span Tree
          </h4>
          <SpanNode span={trace.root_span} depth={0} />
        </div>
      )}
    </div>
  );
}

function SpanNode({ span, depth }: { span: TraceSpan; depth: number }) {
  const [open, setOpen] = useState(true);
  const duration =
    span.end_time && span.start_time
      ? Math.round(
          (new Date(span.end_time).getTime() - new Date(span.start_time).getTime()) / 10
        ) / 100
      : null;

  return (
    <div className="select-text">
      <div
        className="flex items-start gap-2 rounded-lg border border-white/5 bg-white/[0.02] px-3 py-2"
        style={{ marginLeft: depth * 16 }}
      >
        {span.children.length > 0 ? (
          <button
            type="button"
            onClick={() => setOpen((v) => !v)}
            className="mt-0.5 text-zinc-500 hover:text-zinc-300"
          >
            <svg
              className={`h-3.5 w-3.5 transition-transform ${open ? "" : "-rotate-90"}`}
              viewBox="0 0 20 20"
              fill="currentColor"
            >
              <path
                fillRule="evenodd"
                d="M5.23 7.21a.75.75 0 011.06.02L10 11.06l3.71-3.83a.75.75 0 111.08 1.04l-4.25 4.39a.75.75 0 01-1.08 0L5.21 8.27a.75.75 0 01.02-1.06z"
                clipRule="evenodd"
              />
            </svg>
          </button>
        ) : (
          <span className="mt-0.5 h-3.5 w-3.5" />
        )}
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <StatusBadge status={span.status} />
            <span className="truncate text-xs font-medium text-zinc-200">{span.name}</span>
            <span className="rounded bg-white/5 px-1.5 py-0.5 text-[10px] text-zinc-500">
              {span.type}
            </span>
            {duration !== null && (
              <span className="text-[10px] text-zinc-600">{duration}s</span>
            )}
          </div>
          {(span.input && Object.keys(span.input).length > 0) ||
          (span.output && Object.keys(span.output).length > 0) ? (
            <div className="mt-2 space-y-2">
              {span.input && Object.keys(span.input).length > 0 && (
                <details className="group">
                  <summary className="cursor-pointer text-[10px] uppercase tracking-wider text-zinc-600">
                    Input
                  </summary>
                  <pre className="scrollbar-thin mt-1 overflow-x-auto rounded-md bg-black/30 p-2 font-mono text-[11px] text-zinc-400">
                    {JSON.stringify(span.input, null, 2)}
                  </pre>
                </details>
              )}
              {span.output && Object.keys(span.output).length > 0 && (
                <details className="group">
                  <summary className="cursor-pointer text-[10px] uppercase tracking-wider text-zinc-600">
                    Output
                  </summary>
                  <pre className="scrollbar-thin mt-1 overflow-x-auto rounded-md bg-black/30 p-2 font-mono text-[11px] text-zinc-400">
                    {JSON.stringify(span.output, null, 2)}
                  </pre>
                </details>
              )}
            </div>
          ) : null}
        </div>
      </div>
      {open &&
        span.children.map((child) => <SpanNode key={child.id} span={child} depth={depth + 1} />)}
    </div>
  );
}

function StatusBadge({ status }: { status: string }) {
  const color =
    status === "ok" || status === "success"
      ? "bg-emerald-400"
      : status === "error"
      ? "bg-rose-500"
      : "bg-amber-400";
  return <span className={`h-2 w-2 rounded-full ${color}`} title={status} />;
}

function formatTime(iso: string): string {
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return iso;
  }
}
