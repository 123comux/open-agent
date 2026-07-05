import { useEffect, useMemo, useState } from "react";

import type { AgentSettings, AgentSettingsUpdate, Tool } from "../types";

interface SettingsProps {
  apiUrl: string;
  onApiUrlChange: (url: string) => void;
  sessionId: string;
  onRegenerateSession: () => void;
  healthy: boolean | null;
  tools: Tool[];
  settings: AgentSettings | null;
  settingsLoading: boolean;
  onSettingsSave: (update: AgentSettingsUpdate) => Promise<void>;
}

const DEFAULT_SETTINGS: AgentSettings = {
  model_provider: "openai",
  base_url: "https://api.openai.com/v1",
  model_name: "gpt-4o-mini",
  max_steps: 10,
  request_timeout: 60,
  embedding_model: "BAAI/bge-small-zh-v1.5",
  chunk_size: 500,
  chunk_overlap: 50,
  split_unit: "char",
  rag_top_k: 5,
  reranker_model: "BAAI/bge-reranker-v2-m3",
  rerank_k: 20,
  enabled_tools: [],
  enable_long_term_memory: false,
  long_term_memory_top_k: 3,
};

export function Settings({
  apiUrl,
  onApiUrlChange,
  sessionId,
  onRegenerateSession,
  healthy,
  tools,
  settings,
  settingsLoading,
  onSettingsSave,
}: SettingsProps) {
  const [open, setOpen] = useState(false);
  const [draftUrl, setDraftUrl] = useState(apiUrl);
  const [copied, setCopied] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<"model" | "rag" | "tools">("model");

  const base = useMemo(() => settings ?? DEFAULT_SETTINGS, [settings]);

  // Editable form state cloned from current settings.
  const [form, setForm] = useState<AgentSettings & { api_key: string }>({
    ...base,
    api_key: "",
  });

  useEffect(() => {
    setDraftUrl(apiUrl);
  }, [apiUrl]);

  useEffect(() => {
    setForm((prev) => ({ ...base, api_key: prev.api_key }));
  }, [base]);

  const saveApiUrl = () => {
    onApiUrlChange(draftUrl.trim());
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

  const updateField = <K extends keyof typeof form>(key: K, value: typeof form[K]) => {
    setForm((prev) => ({ ...prev, [key]: value } as typeof prev));
  };

  const toggleTool = (name: string) => {
    setForm((prev) => {
      const enabled = new Set(prev.enabled_tools);
      if (enabled.has(name)) {
        enabled.delete(name);
      } else {
        enabled.add(name);
      }
      return { ...prev, enabled_tools: Array.from(enabled) };
    });
  };

  const handleSave = async () => {
    setSaving(true);
    setError(null);
    try {
      const update: AgentSettingsUpdate = {
        model_provider: form.model_provider,
        base_url: form.base_url,
        model_name: form.model_name,
        max_steps: Number(form.max_steps),
        request_timeout: Number(form.request_timeout),
        embedding_model: form.embedding_model,
        chunk_size: Number(form.chunk_size),
        chunk_overlap: Number(form.chunk_overlap),
        split_unit: form.split_unit,
        rag_top_k: Number(form.rag_top_k),
        reranker_model: form.reranker_model,
        rerank_k: Number(form.rerank_k),
        enabled_tools: form.enabled_tools,
        enable_long_term_memory: form.enable_long_term_memory,
        long_term_memory_top_k: Number(form.long_term_memory_top_k),
      };
      if (form.api_key.trim()) {
        update.api_key = form.api_key.trim();
      }
      await onSettingsSave(update);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSaving(false);
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

  const inputClass =
    "w-full rounded-md border border-white/10 bg-black/30 px-2 py-1.5 font-mono text-xs text-zinc-200 outline-none focus:border-emerald-400/40";
  const selectClass =
    "w-full rounded-md border border-white/10 bg-black/30 px-2 py-1.5 text-xs text-zinc-200 outline-none focus:border-emerald-400/40";
  const labelClass =
    "mb-1 block text-[10px] font-medium uppercase tracking-wider text-zinc-500";

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
            <label className={labelClass}>API URL</label>
            <div className="flex gap-2">
              <input
                value={draftUrl}
                onChange={(e) => setDraftUrl(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") saveApiUrl();
                }}
                placeholder="http://localhost:8000"
                spellCheck={false}
                className="min-w-0 flex-1 rounded-md border border-white/10 bg-black/30 px-2 py-1.5 font-mono text-xs text-zinc-200 outline-none focus:border-emerald-400/40"
              />
              <button
                type="button"
                onClick={saveApiUrl}
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
              <label className={labelClass}>Session</label>
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
              <span className={labelClass}>Backend</span>
              <span className={`text-xs font-medium ${statusColor}`}>
                {statusLabel}
              </span>
            </div>
          </div>

          <div className="border-t border-white/5 pt-3">
            <div className="mb-2 flex items-center gap-2">
              {(["model", "rag", "tools"] as const).map((tab) => (
                <button
                  key={tab}
                  type="button"
                  onClick={() => setActiveTab(tab)}
                  className={`rounded-md px-2 py-1 text-[10px] font-semibold uppercase tracking-wider transition-colors ${
                    activeTab === tab
                      ? "bg-emerald-500/20 text-emerald-300"
                      : "text-zinc-500 hover:bg-white/5 hover:text-zinc-300"
                  }`}
                >
                  {tab}
                </button>
              ))}
            </div>

            {settingsLoading && (
              <div className="py-2 text-[10px] text-zinc-500">Loading settings…</div>
            )}

            {error && (
              <div className="mb-2 rounded-md border border-rose-500/30 bg-rose-500/10 px-2 py-1.5 text-[10px] text-rose-300">
                {error}
              </div>
            )}

            {activeTab === "model" && (
              <div className="space-y-3">
                <div>
                  <label className={labelClass}>Provider</label>
                  <select
                    value={form.model_provider}
                    onChange={(e) =>
                      updateField("model_provider", e.target.value as AgentSettings["model_provider"])
                    }
                    className={selectClass}
                  >
                    <option value="openai">OpenAI</option>
                    <option value="anthropic">Anthropic</option>
                    <option value="ollama">Ollama</option>
                  </select>
                </div>
                <div>
                  <label className={labelClass}>Base URL</label>
                  <input
                    value={form.base_url}
                    onChange={(e) => updateField("base_url", e.target.value)}
                    className={inputClass}
                  />
                </div>
                <div>
                  <label className={labelClass}>Model</label>
                  <input
                    value={form.model_name}
                    onChange={(e) => updateField("model_name", e.target.value)}
                    className={inputClass}
                  />
                </div>
                <div>
                  <label className={labelClass}>API Key</label>
                  <input
                    type="password"
                    value={form.api_key}
                    onChange={(e) => updateField("api_key", e.target.value)}
                    placeholder="Leave blank to keep current"
                    className={inputClass}
                  />
                </div>
                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <label className={labelClass}>Max Steps</label>
                    <input
                      type="number"
                      min={1}
                      max={50}
                      value={form.max_steps}
                      onChange={(e) => updateField("max_steps", Number(e.target.value))}
                      className={inputClass}
                    />
                  </div>
                  <div>
                    <label className={labelClass}>Timeout (s)</label>
                    <input
                      type="number"
                      min={5}
                      value={form.request_timeout}
                      onChange={(e) => updateField("request_timeout", Number(e.target.value))}
                      className={inputClass}
                    />
                  </div>
                </div>
              </div>
            )}

            {activeTab === "rag" && (
              <div className="space-y-3">
                <div>
                  <label className={labelClass}>Embedding Model</label>
                  <input
                    value={form.embedding_model}
                    onChange={(e) => updateField("embedding_model", e.target.value)}
                    className={inputClass}
                  />
                </div>
                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <label className={labelClass}>Chunk Size</label>
                    <input
                      type="number"
                      min={100}
                      value={form.chunk_size}
                      onChange={(e) => updateField("chunk_size", Number(e.target.value))}
                      className={inputClass}
                    />
                  </div>
                  <div>
                    <label className={labelClass}>Chunk Overlap</label>
                    <input
                      type="number"
                      min={0}
                      value={form.chunk_overlap}
                      onChange={(e) => updateField("chunk_overlap", Number(e.target.value))}
                      className={inputClass}
                    />
                  </div>
                </div>
                <div>
                  <label className={labelClass}>Split Unit</label>
                  <select
                    value={form.split_unit}
                    onChange={(e) => updateField("split_unit", e.target.value)}
                    className={selectClass}
                  >
                    <option value="char">char</option>
                    <option value="token">token</option>
                    <option value="sentence">sentence</option>
                  </select>
                </div>
                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <label className={labelClass}>RAG Top K</label>
                    <input
                      type="number"
                      min={1}
                      value={form.rag_top_k}
                      onChange={(e) => updateField("rag_top_k", Number(e.target.value))}
                      className={inputClass}
                    />
                  </div>
                  <div>
                    <label className={labelClass}>Rerank K</label>
                    <input
                      type="number"
                      min={1}
                      value={form.rerank_k}
                      onChange={(e) => updateField("rerank_k", Number(e.target.value))}
                      className={inputClass}
                    />
                  </div>
                </div>
                <div>
                  <label className={labelClass}>Reranker Model</label>
                  <input
                    value={form.reranker_model}
                    onChange={(e) => updateField("reranker_model", e.target.value)}
                    className={inputClass}
                  />
                </div>
                <div className="grid grid-cols-2 gap-2">
                  <div className="flex items-center gap-2">
                    <input
                      id="ltm"
                      type="checkbox"
                      checked={form.enable_long_term_memory}
                      onChange={(e) => updateField("enable_long_term_memory", e.target.checked)}
                      className="h-3.5 w-3.5 rounded border-white/10 bg-black/30 text-emerald-500"
                    />
                    <label htmlFor="ltm" className={labelClass}>
                      Long-term Memory
                    </label>
                  </div>
                  <div>
                    <label className={labelClass}>LTM Top K</label>
                    <input
                      type="number"
                      min={1}
                      value={form.long_term_memory_top_k}
                      onChange={(e) => updateField("long_term_memory_top_k", Number(e.target.value))}
                      className={inputClass}
                    />
                  </div>
                </div>
              </div>
            )}

            {activeTab === "tools" && (
              <div className="space-y-2">
                <p className="text-[10px] text-zinc-500">
                  Leave all unchecked to enable every tool.
                </p>
                {tools.map((tool) => {
                  const checked =
                    form.enabled_tools.length === 0 ||
                    form.enabled_tools.includes(tool.name);
                  return (
                    <label
                      key={tool.name}
                      className="flex items-start gap-2 rounded-md border border-white/5 bg-black/20 px-2 py-1.5"
                    >
                      <input
                        type="checkbox"
                        checked={checked}
                        onChange={() => toggleTool(tool.name)}
                        className="mt-0.5 h-3.5 w-3.5 rounded border-white/10 bg-black/30 text-emerald-500"
                      />
                      <div>
                        <div className="text-xs font-medium text-zinc-300">
                          {tool.name}
                        </div>
                        <div className="text-[10px] text-zinc-500">
                          {tool.description}
                        </div>
                      </div>
                    </label>
                  );
                })}
              </div>
            )}

            <button
              type="button"
              onClick={handleSave}
              disabled={saving || settingsLoading}
              className="mt-3 w-full rounded-md bg-emerald-500 px-2.5 py-1.5 text-xs font-semibold text-zinc-950 transition-colors hover:bg-emerald-400 disabled:bg-zinc-600"
            >
              {saving ? "Saving…" : "Apply Settings"}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
