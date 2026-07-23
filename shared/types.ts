/** Shared TypeScript types for Open Agent frontends (Web UI and VS Code extension). */

/** A single tool invocation recorded during an agent run. */
export interface ToolCallInfo {
  step: number;
  name: string;
  arguments: Record<string, unknown>;
  observation: string;
  is_error: boolean;
}

export type MessageRole = "user" | "assistant";

/** A chat message in the conversation. */
export interface Message {
  id: string;
  role: MessageRole;
  content: string;
  /** Number of reasoning steps the agent took (assistant messages only). */
  steps?: number;
  /** Tool calls made while producing this assistant message. */
  toolCalls?: ToolCallInfo[];
  /** Thinking chain entries from the agent. */
  thoughts?: string[];
  /** True while the assistant message is actively streaming tokens. */
  streaming?: boolean;
  timestamp: number;
}

/** Response body of `POST /api/chat`. */
export interface ChatResponse {
  response: string;
  steps: number;
  tool_calls_made: ToolCallInfo[];
}

/** A tool description returned by `GET /api/tools`. */
export interface Tool {
  name: string;
  description: string;
  parameters: Record<string, unknown>;
}

/** Visual status of a tool call. */
export type ToolCallStatus = "running" | "done" | "error";

/** Events streamed over WebSocket during an agent run. */
export type StreamEvent =
  | { type: "token"; content: string }
  | { type: "thought"; content: string; step: number }
  | { type: "tool_start"; name: string; arguments: Record<string, unknown> }
  | { type: "tool_end"; name: string; observation: string; is_error: boolean }
  | { type: "done"; response: string; steps: number; tool_calls_made: ToolCallInfo[] };

/** Editable runtime settings returned by `GET /api/settings`. */
export interface AgentSettings {
  model_provider: "openai" | "anthropic" | "ollama" | "zhipu";
  base_url: string;
  model_name: string;
  max_steps: number;
  request_timeout: number;
  embedding_model: string;
  chunk_size: number;
  chunk_overlap: number;
  split_unit: string;
  rag_top_k: number;
  reranker_model: string;
  rerank_k: number;
  enabled_tools: string[];
  enable_long_term_memory: boolean;
  long_term_memory_top_k: number;
}

/** Partial updates accepted by `POST /api/settings`. */
export type AgentSettingsUpdate = Partial<AgentSettings> & {
  api_key?: string;
};

/** A single source indexed in a knowledge base. */
export interface KnowledgeBaseDocument {
  source: string;
  chunks: number;
}

/** Knowledge base summary returned by `GET /api/knowledge-bases`. */
export interface KnowledgeBaseInfo {
  name: string;
  documents: number;
  chunks: number;
}

/** Response body of `GET /api/knowledge-bases/{name}/documents`. */
export interface KnowledgeBaseDocumentsResponse {
  kb_name: string;
  documents: KnowledgeBaseDocument[];
}

/** A single span inside a trace. */
export interface TraceSpan {
  id: string;
  parent_id: string | null;
  trace_id: string;
  type: string;
  name: string;
  start_time: string;
  end_time: string | null;
  status: string;
  input: Record<string, unknown>;
  output: Record<string, unknown>;
  metadata: Record<string, unknown>;
  metrics: Record<string, unknown>;
  children: TraceSpan[];
}

/** A top-level trace returned by `GET /api/traces`. */
export interface Trace {
  id: string;
  name: string;
  start_time: string;
  end_time: string | null;
  status: string;
  input: Record<string, unknown>;
  metadata: Record<string, unknown>;
  root_span: TraceSpan | null;
}
