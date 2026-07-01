import {
  useEffect,
  useRef,
  useState,
  type ChangeEvent,
  type FormEvent,
  type KeyboardEvent,
} from "react";
import ReactMarkdown from "react-markdown";
import { ToolCall } from "./ToolCall";
import type { Message, MessageRole } from "../types";

interface ChatProps {
  messages: Message[];
  loading: boolean;
  onSend: (text: string) => void;
  error: string | null;
}

const EXAMPLES = [
  "List the files in the current directory",
  "Write a Python function to reverse a string",
  "Search the web for the latest news on AI agents",
];

export function Chat({ messages, loading, onSend, error }: ChatProps) {
  const [input, setInput] = useState("");
  const scrollRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages, loading]);

  const resize = () => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
  };

  const onChange = (e: ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value);
    resize();
  };

  const submit = (e?: FormEvent) => {
    e?.preventDefault();
    const text = input.trim();
    if (!text || loading) return;
    onSend(text);
    setInput("");
    if (textareaRef.current) textareaRef.current.style.height = "auto";
  };

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  };

  return (
    <div className="flex h-full flex-col">
      <header className="flex items-center gap-3 border-b border-white/5 px-6 py-3">
        <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-br from-emerald-400 to-teal-600 text-xs font-bold text-zinc-950">
          OA
        </div>
        <div>
          <h2 className="text-sm font-semibold text-zinc-100">Open Agent</h2>
          <p className="text-xs text-zinc-500">Autonomous work assistant</p>
        </div>
      </header>

      <div
        ref={scrollRef}
        className="scrollbar-thin flex-1 overflow-y-auto px-4 py-6 sm:px-6"
      >
        <div className="mx-auto flex max-w-3xl flex-col gap-6">
          {messages.length === 0 && !loading && (
            <EmptyState onPick={onSend} />
          )}
          {messages.map((m) => (
            <MessageBubble key={m.id} message={m} />
          ))}
          {loading && <TypingIndicator />}
        </div>
      </div>

      {error && (
        <div className="mx-auto w-full max-w-3xl px-4 sm:px-6">
          <div className="mb-2 rounded-lg border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-sm text-rose-300">
            {error}
          </div>
        </div>
      )}

      <form onSubmit={submit} className="px-4 pb-4 sm:px-6">
        <div className="mx-auto flex max-w-3xl items-end gap-2 rounded-2xl border border-white/10 bg-white/5 p-2 transition-colors focus-within:border-emerald-400/40 focus-within:bg-white/[0.07]">
          <textarea
            ref={textareaRef}
            value={input}
            onChange={onChange}
            onKeyDown={onKeyDown}
            rows={1}
            placeholder="Ask Open Agent to do something…"
            className="scrollbar-thin max-h-[200px] flex-1 resize-none bg-transparent px-2 py-2 text-sm text-zinc-100 outline-none placeholder:text-zinc-500"
          />
          <button
            type="submit"
            disabled={!input.trim() || loading}
            aria-label="Send message"
            className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-emerald-500 text-zinc-950 transition-colors hover:bg-emerald-400 disabled:cursor-not-allowed disabled:bg-zinc-700 disabled:text-zinc-500"
          >
            <svg
              className="h-5 w-5"
              viewBox="0 0 20 20"
              fill="currentColor"
              aria-hidden="true"
            >
              <path d="M3.105 3.105a.75.75 0 01.815-.165l13 5.5a.75.75 0 010 1.378l-13 5.5a.75.75 0 01-1.03-.94l1.6-4.003H9a.75.75 0 000-1.5H4.49l-1.6-4.003a.75.75 0 01.215-.767z" />
            </svg>
          </button>
        </div>
        <p className="mx-auto mt-2 max-w-3xl px-2 text-center text-xs text-zinc-600">
          Enter to send · Shift+Enter for a new line
        </p>
      </form>
    </div>
  );
}

function EmptyState({ onPick }: { onPick?: (text: string) => void }) {
  return (
    <div className="flex flex-col items-center justify-center py-16 text-center">
      <div className="mb-4 flex h-14 w-14 items-center justify-center rounded-2xl bg-gradient-to-br from-emerald-400 to-teal-600 text-lg font-bold text-zinc-950">
        OA
      </div>
      <h2 className="text-lg font-semibold text-zinc-100">How can I help today?</h2>
      <p className="mt-1 text-sm text-zinc-500">
        Open Agent can reason, use tools, and complete tasks for you.
      </p>
      <div className="mt-6 flex w-full max-w-md flex-col gap-2">
        {EXAMPLES.map((ex) => (
          <button
            key={ex}
            type="button"
            onClick={() => onPick?.(ex)}
            className="rounded-lg border border-white/5 bg-white/5 px-4 py-2.5 text-left text-sm text-zinc-400 transition-colors hover:border-emerald-400/20 hover:bg-white/10 hover:text-zinc-200"
          >
            {ex}
          </button>
        ))}
      </div>
    </div>
  );
}

function TypingIndicator() {
  return (
    <div className="flex items-center gap-3">
      <Avatar role="assistant" />
      <div className="flex items-center gap-1.5 rounded-2xl rounded-tl-sm border border-white/10 bg-white/5 px-4 py-3">
        <span className="h-2 w-2 animate-bounce rounded-full bg-zinc-500 [animation-delay:-0.3s]" />
        <span className="h-2 w-2 animate-bounce rounded-full bg-zinc-500 [animation-delay:-0.15s]" />
        <span className="h-2 w-2 animate-bounce rounded-full bg-zinc-500" />
      </div>
    </div>
  );
}

function MessageBubble({ message }: { message: Message }) {
  const isUser = message.role === "user";
  return (
    <div className={`flex gap-3 ${isUser ? "flex-row-reverse" : ""}`}>
      <Avatar role={message.role} />
      <div
        className={`flex min-w-0 max-w-[85%] flex-col gap-2 ${
          isUser ? "items-end" : "items-start"
        }`}
      >
        {!isUser && message.thoughts && message.thoughts.length > 0 && (
          <details className="w-full rounded-lg border border-white/5 bg-white/[0.02] px-3 py-2">
            <summary className="cursor-pointer text-xs font-semibold uppercase tracking-wider text-zinc-500">
              Thinking Chain ({message.thoughts.length} steps)
            </summary>
            <div className="mt-2 space-y-1">
              {message.thoughts.map((t, i) => (
                <div key={i} className="flex gap-2 text-xs text-zinc-400">
                  <span className="text-emerald-400">→</span>
                  <span>{t}</span>
                </div>
              ))}
            </div>
          </details>
        )}
        <div
          className={`rounded-2xl px-4 py-3 text-sm leading-relaxed ${
            isUser
              ? "rounded-tr-sm bg-emerald-500 text-zinc-950"
              : "rounded-tl-sm border border-white/10 bg-white/5 text-zinc-200"
          }`}
        >
          {isUser ? (
            <p className="whitespace-pre-wrap break-words">{message.content}</p>
          ) : message.streaming && !message.content ? (
            <div className="flex items-center gap-1.5 py-1">
              <span className="h-2 w-2 animate-bounce rounded-full bg-zinc-500 [animation-delay:-0.3s]" />
              <span className="h-2 w-2 animate-bounce rounded-full bg-zinc-500 [animation-delay:-0.15s]" />
              <span className="h-2 w-2 animate-bounce rounded-full bg-zinc-500" />
            </div>
          ) : (
            <div className="prose-agent break-words">
              <ReactMarkdown>{message.content}</ReactMarkdown>
              {message.streaming && (
                <span className="ml-0.5 inline-block h-4 w-1.5 animate-pulse rounded-sm bg-emerald-400 align-text-bottom" />
              )}
            </div>
          )}
        </div>
        {!isUser && message.toolCalls && message.toolCalls.length > 0 && (
          <div className="w-full space-y-2">
            <div className="flex items-center gap-2 px-1 text-xs text-zinc-500">
              <span className="font-semibold uppercase tracking-wider">
                Tool calls
              </span>
              <span className="text-zinc-600">·</span>
              <span>{message.toolCalls.length}</span>
              {typeof message.steps === "number" && (
                <>
                  <span className="text-zinc-600">·</span>
                  <span>{message.steps} steps</span>
                </>
              )}
            </div>
            {message.toolCalls.map((c, i) => (
              <ToolCall key={`${c.name}-${c.step}-${i}`} call={c} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function Avatar({ role }: { role: MessageRole }) {
  if (role === "user") {
    return (
      <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-zinc-700 text-[10px] font-semibold text-zinc-300">
        YOU
      </div>
    );
  }
  return (
    <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-gradient-to-br from-emerald-400 to-teal-600 text-xs font-bold text-zinc-950">
      OA
    </div>
  );
}
