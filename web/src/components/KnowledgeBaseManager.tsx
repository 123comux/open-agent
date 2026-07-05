import { useCallback, useEffect, useRef, useState } from "react";
import { ChatClient } from "../api/client";
import type { KnowledgeBaseDocument } from "../types";

interface KnowledgeBaseManagerProps {
  client: ChatClient;
}

export function KnowledgeBaseManager({ client }: KnowledgeBaseManagerProps) {
  const [kbs, setKbs] = useState<string[]>([]);
  const [selectedKb, setSelectedKb] = useState<string>("default");
  const [newKbName, setNewKbName] = useState<string>("");
  const [documents, setDocuments] = useState<KnowledgeBaseDocument[]>([]);
  const [loadingKbs, setLoadingKbs] = useState(false);
  const [loadingDocs, setLoadingDocs] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const refreshKbs = useCallback(async () => {
    setLoadingKbs(true);
    try {
      const list = await client.listKnowledgeBases();
      setKbs(list);
      if (!list.includes(selectedKb) && list.length > 0) {
        setSelectedKb(list[0]);
      }
    } catch {
      /* ignore */
    } finally {
      setLoadingKbs(false);
    }
  }, [client, selectedKb]);

  const refreshDocuments = useCallback(async () => {
    if (!selectedKb) return;
    setLoadingDocs(true);
    try {
      const docs = await client.listKbDocuments(selectedKb);
      setDocuments(docs);
    } catch (err) {
      setDocuments([]);
      setMessage(`Error: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setLoadingDocs(false);
    }
  }, [client, selectedKb]);

  useEffect(() => {
    refreshKbs();
  }, [refreshKbs]);

  useEffect(() => {
    refreshDocuments();
  }, [refreshDocuments]);

  const handleUpload = useCallback(
    async (file: File) => {
      const kbName = newKbName.trim() || selectedKb;
      setUploading(true);
      setMessage(null);
      try {
        await client.uploadFile(file, kbName);
        setMessage(`Indexed: ${file.name} → ${kbName}`);
        await refreshKbs();
        if (kbName !== selectedKb) {
          setSelectedKb(kbName);
          setNewKbName("");
        } else {
          await refreshDocuments();
        }
      } catch (err) {
        setMessage(`Error: ${err instanceof Error ? err.message : String(err)}`);
      } finally {
        setUploading(false);
      }
    },
    [client, selectedKb, newKbName, refreshKbs, refreshDocuments]
  );

  const handleDelete = useCallback(
    async (source: string) => {
      if (!confirm(`Delete all chunks from "${source}"?`)) return;
      try {
        await client.deleteKbDocument(selectedKb, source);
        await refreshDocuments();
        setMessage(`Deleted: ${source}`);
      } catch (err) {
        setMessage(`Error: ${err instanceof Error ? err.message : String(err)}`);
      }
    },
    [client, selectedKb, refreshDocuments]
  );

  const targetKbName = newKbName.trim() || selectedKb;

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between px-2">
        <span className="text-xs font-semibold uppercase tracking-wider text-zinc-400">
          Knowledge Base
        </span>
        <button
          type="button"
          onClick={() => {
            refreshKbs();
            refreshDocuments();
          }}
          disabled={loadingKbs || loadingDocs}
          className="rounded p-1 text-zinc-500 transition-colors hover:bg-white/5 hover:text-zinc-300 disabled:opacity-50"
          title="Refresh"
        >
          <svg className="h-3.5 w-3.5" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
            <path
              fillRule="evenodd"
              d="M15.312 11.424a5.5 5.5 0 01-9.201 2.466l-.312-.95a.75.75 0 111.426-.47l.312.95a4 4 0 006.69-1.793l.602-1.833a.75.75 0 011.426.468l-.602 1.833a5.5 5.5 0 01-2.34 3.295zm-7.876-2.848a5.5 5.5 0 019.201-2.466l.312.95a.75.75 0 11-1.426.47l-.312-.95a4 4 0 00-6.69 1.793l-.602 1.833a.75.75 0 11-1.426-.468l.602-1.833a5.5 5.5 0 012.34-3.295z"
              clipRule="evenodd"
            />
          </svg>
        </button>
      </div>

      <div className="space-y-2 px-2">
        <div className="flex items-center gap-2">
          <select
            value={selectedKb}
            onChange={(e) => setSelectedKb(e.target.value)}
            className="flex-1 rounded-lg border border-white/10 bg-zinc-900/50 px-2.5 py-1.5 text-xs text-zinc-200 outline-none focus:border-emerald-400/50"
          >
            {kbs.length === 0 && <option value="default">default</option>}
            {kbs.map((kb) => (
              <option key={kb} value={kb}>
                {kb}
              </option>
            ))}
          </select>
          <span className="text-[11px] text-zinc-500">or</span>
          <input
            type="text"
            value={newKbName}
            onChange={(e) => setNewKbName(e.target.value)}
            placeholder="new KB"
            className="w-20 rounded-lg border border-white/10 bg-zinc-900/50 px-2.5 py-1.5 text-xs text-zinc-200 placeholder-zinc-600 outline-none focus:border-emerald-400/50"
          />
        </div>

        <div
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragOver(false);
            const f = e.dataTransfer.files?.[0];
            if (f) void handleUpload(f);
          }}
          onClick={() => fileInputRef.current?.click()}
          className={`cursor-pointer rounded-lg border border-dashed px-3 py-3 text-center transition-colors ${
            dragOver
              ? "border-emerald-400/50 bg-emerald-400/10"
              : "border-white/10 bg-white/[0.02] hover:border-white/20 hover:bg-white/5"
          }`}
        >
          {uploading ? (
            <p className="text-xs text-emerald-400">Indexing…</p>
          ) : (
            <>
              <svg className="mx-auto mb-1 h-4 w-4 text-zinc-500" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
                <path d="M5.5 17a4.5 4.5 0 01-1.44-8.77 5.5 5.5 0 0110.43-2.49 4.5 4.5 0 011.51 8.76H5.5zM8.5 7.25a.75.75 0 00-1.5 0v4.69L5.78 10.72a.75.75 0 00-1.06 1.06l2.75 2.75a.75.75 0 001.06 0l2.75-2.75a.75.75 0 10-1.06-1.06L8.5 11.94V7.25z" />
              </svg>
              <p className="text-[11px] text-zinc-500">Drop file or click to upload</p>
              <p className="mt-0.5 text-[10px] text-zinc-600">Target: {targetKbName}</p>
            </>
          )}
          <input
            ref={fileInputRef}
            type="file"
            className="hidden"
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) void handleUpload(f);
              e.currentTarget.value = "";
            }}
          />
        </div>
      </div>

      {message && (
        <p className={`px-2 text-[11px] ${message.startsWith("Error") ? "text-rose-400" : "text-emerald-400"}`}>
          {message}
        </p>
      )}

      <div className="px-2">
        <div className="mb-1.5 flex items-center justify-between">
          <span className="text-[11px] font-medium uppercase tracking-wider text-zinc-500">
            Documents
          </span>
          <span className="text-[10px] text-zinc-600">{documents.length}</span>
        </div>
        {loadingDocs ? (
          <div className="space-y-1.5">
            {[0, 1].map((i) => (
              <div key={i} className="h-8 animate-pulse rounded-lg bg-white/5" />
            ))}
          </div>
        ) : documents.length === 0 ? (
          <p className="rounded-lg border border-white/5 bg-white/[0.02] px-2.5 py-2 text-[11px] text-zinc-600">
            No documents in this knowledge base.
          </p>
        ) : (
          <div className="max-h-48 space-y-1 overflow-y-auto pr-0.5 scrollbar-thin">
            {documents.map((doc) => (
              <div
                key={doc.source}
                className="group flex items-center justify-between gap-2 rounded-lg border border-white/5 bg-white/[0.02] px-2.5 py-2"
              >
                <div className="min-w-0 flex-1">
                  <p className="truncate text-[11px] text-zinc-300" title={doc.source}>
                    {doc.source.split("/").pop() || doc.source}
                  </p>
                  <p className="text-[10px] text-zinc-600">{doc.chunks} chunk{doc.chunks === 1 ? "" : "s"}</p>
                </div>
                <button
                  type="button"
                  onClick={() => handleDelete(doc.source)}
                  className="shrink-0 rounded p-1 text-zinc-600 opacity-0 transition-colors hover:bg-rose-500/10 hover:text-rose-400 group-hover:opacity-100"
                  title="Delete document"
                >
                  <svg className="h-3.5 w-3.5" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
                    <path
                      fillRule="evenodd"
                      d="M8.75 1A2.75 2.75 0 006 3.75v.5H3.75a.75.75 0 000 1.5h.5l.6 9.4A2.75 2.75 0 007.59 18h4.82a2.75 2.75 0 002.74-2.85l.6-9.4h.5a.75.75 0 000-1.5H14v-.5A2.75 2.75 0 0011.25 1h-2.5zM7.5 3.75c0-.69.56-1.25 1.25-1.25h2.5c.69 0 1.25.56 1.25 1.25v.5h-5v-.5z"
                      clipRule="evenodd"
                    />
                  </svg>
                </button>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
