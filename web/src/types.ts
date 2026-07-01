export type MessageRole = "user" | "assistant";

/** A single tool invocation recorded during an agent run. */
export interface ToolCallInfo {
  step: number;
  name: string;
  arguments: Record<string, unknown>;
  observation: string;
  is_error: boolean;
}

/** A chat message in the conversation. */
export interface Message {
  id: string;
  role: MessageRole;
  content: string;
  /** Number of reasoning steps the agent took (assistant messages only). */
  steps?: number;
  /** Tool calls made while producing this assistant message. */
  toolCalls?: ToolCallInfo[];
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
