import { useState } from "react";
import type { ToolCallInfo, ToolCallStatus } from "../types";

interface ToolCallProps {
  call: ToolCallInfo;
  /** Override the derived status (e.g. "running" while streaming). */
  status?: ToolCallStatus;
}

function deriveStatus(call: ToolCallInfo): ToolCallStatus {
  return call.is_error ? "error" : "done";
}

const STATUS_STYLES: Record<
  ToolCallStatus,
  { dot: string; ring: string; label: string; text: string }
> = {
  running: {
    dot: "bg-amber-400",
    ring: "border-amber-400/30 bg-amber-400/[0.06]",
    label: "Running",
    text: "text-amber-300",
  },
  done: {
    dot: "bg-emerald-400",
    ring: "border-emerald-400/20 bg-emerald-400/[0.04]",
    label: "Done",
    text: "text-emerald-300",
  },
  error: {
    dot: "bg-rose-500",
    ring: "border-rose-500/30 bg-rose-500/[0.06]",
    label: "Error",
    text: "text-rose-300",
  },
};

export function ToolCall({ call, status }: ToolCallProps) {
  const [open, setOpen] = useState(false);
  const current = status ?? deriveStatus(call);
  const s = STATUS_STYLES[current];

  return (
    <div className={`overflow-hidden rounded-lg border ${s.ring}`}>
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center gap-3 px-3 py-2 text-left transition-colors hover:bg-white/5"
      >
        <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-md bg-white/5 font-mono text-xs font-semibold text-zinc-400">
          {call.step}
        </span>
        <span
          className={`h-2 w-2 shrink-0 rounded-full ${s.dot} ${
            current === "running" ? "animate-pulse" : ""
          }`}
        />
        <span className="truncate font-mono text-sm font-medium text-zinc-100">
          {call.name}
        </span>
        <span
          className={`ml-auto text-[10px] font-semibold uppercase tracking-wider ${s.text}`}
        >
          {s.label}
        </span>
        <svg
          className={`h-4 w-4 shrink-0 text-zinc-500 transition-transform ${
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
        <div className="space-y-3 border-t border-white/5 px-3 py-3 font-mono text-xs">
          <div>
            <div className="mb-1 text-[10px] uppercase tracking-wider text-zinc-500">
              Arguments
            </div>
            <pre className="whitespace-pre-wrap break-words text-zinc-300">
              {JSON.stringify(call.arguments, null, 2)}
            </pre>
          </div>
          <div>
            <div className="mb-1 text-[10px] uppercase tracking-wider text-zinc-500">
              Observation
            </div>
            <pre
              className={`whitespace-pre-wrap break-words ${
                call.is_error ? "text-rose-300" : "text-zinc-300"
              }`}
            >
              {call.observation}
            </pre>
          </div>
        </div>
      )}
    </div>
  );
}
