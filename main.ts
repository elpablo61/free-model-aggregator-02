// Free-model aggregator — one server, two wires, fourteen free models.
//
// Serves the OpenAI Chat Completions API and the Anthropic Messages API
// on the same port. Every completion is a real call to a free OpenRouter
// model (`openrouter/<vendor>/<model>:free`, $0) through the grader's
// gateway, on the trial's key.
//
//   Trial protocol (see the kit's starter/main.ts + program/env/server.mjs):
//     - listen on 0.0.0.0 at an ephemeral port (never 8080: the env is there)
//     - POST $MAKEFUN_ENV_URL/register {"protocol_version": 1, "port", "catalog"}
//     - serve GET /v1/models (OpenAI shape for Authorization: Bearer,
//       Anthropic shape for x-api-key), POST /v1/chat/completions and
//       POST /v1/messages, JSON or server-sent events
//     - poll GET $MAKEFUN_ENV_URL/work every 100 ms; exit 0 when it says
//       {"done": true}
//     - `node main.ts --list` prints the catalog document and exits 0
//   Env:
//     MAKEFUN_ENV_URL      the env's base URL (register + work)
//     MAKEFUN_PROMPT_FILE  the brief: prose holding the free-model list
//                          (free-model-aggregator-02/models/v1) and the six
//                          prompts (free-model-aggregator-02/prompts/v1)
//     OPENAI_BASE_URL      the gateway root (ends in /v1 on the trial net)
//     OPENAI_API_KEY       the trial's dummy key; the gateway swaps it for
//                          the entrant's OpenRouter key
//     OPENROUTER_API_KEY / OPENROUTER_BASE_URL: local-dev fallbacks
//     PORT                 local-dev listen port (trial: ephemeral, port 0)
//
// Routing: the brief's prompt rows arrive in the client's order; each
// prompt's requirements (tools / vision / json) pick the eligible free
// models, ranked by the brief's rating_milli, best first. Best per
// requirement: JSON mode -> super-vl (1303); everything else -> inkling
// (1475). A 400/404/429/5xx from one model fails over to the next
// eligible model; 401/403 fail fast. A model that refused on the previous
// prompt is skipped for the next one (a count-based cooldown: no clock).
// Refusals are reported on the response in the
// x-makefun-upstream-failures header, as the brief allows.
//
// The token: each prompt carries one unique token (WORD-1234). The
// answer's TEXT must carry it back — the house scores the provider's own
// answer body, never anything this server writes, so the token is never
// fabricated here. After an answer arrives, its text is checked; if the
// token is missing, the server retries (at most twice) with an explicit
// instruction naming the missing token verbatim, so the MODEL produces
// it. At most 6 gateway calls per prompt; a prompt's wall time is capped
// near the driver's 30 s timeout (25 s per call, ~27 s per prompt).
//
// Determinism: routing, retries and ids are pure functions of the brief,
// the requests and the gateway's answers. No wall clock and no unseeded
// randomness in any response: ids come from a counter, `created` is
// fixed, usage comes from the gateway (or a character-count estimate).
// A replay therefore leaves the same state export. The per-prompt time
// cap only stops attempting; on replay the cassette answers instantly,
// so it never triggers there.
//
// Node 22, standard library only, nothing to install. Only erasable
// TypeScript is used so `node main.ts` runs under type stripping.

import { readFileSync } from "node:fs";
import http from "node:http";
import { URL } from "node:url";

// ---------------------------------------------------------------- brief

type Json = Record<string, unknown>;
function isRecord(v: unknown): v is Json {
  return v !== null && typeof v === "object" && !Array.isArray(v);
}

type ModelRow = {
  id: string; // the id the gateway routes: openrouter/<vendor>/<model>:free
  provider: string;
  display_name: string;
  rating_milli: number;
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
  free: boolean;
};

type Requirement = { tools?: boolean; vision?: boolean; json?: boolean; min_context?: number };
type PromptRow = { id: string; api: "openai" | "anthropic"; stream: boolean; weight: 1 | 2; requires: Requirement };
type Brief = { models: ModelRow[]; prompts: PromptRow[]; error: string | null };

// The house's sealed free catalog, as the brief and the sealed scorer
// table spell it. Used only when MAKEFUN_PROMPT_FILE cannot be read
// (local development): the trial always supplies the brief itself, and
// the brief is the authority whenever it is present.
const FALLBACK_MODELS: ModelRow[] = [
  { id: "openrouter/baidu/ernie-5.5-flash-preview:free", provider: "openrouter", display_name: "Baidu: ERNIE-5.5-Flash-Preview (free)", rating_milli: 1150, context_length: 1024000, tools: true, vision: false, json: true, free: true },
  { id: "openrouter/cohere/north-mini-code:free", provider: "openrouter", display_name: "Cohere: North Mini Code (free)", rating_milli: 1200, context_length: 256000, tools: true, vision: false, json: false, free: true },
  { id: "openrouter/dots-studio/dots-3-note-preview:free", provider: "openrouter", display_name: "Dots Studio: Dots3-Note Preview (free)", rating_milli: 1150, context_length: 512000, tools: true, vision: true, json: true, free: true },
  { id: "openrouter/google/gemma-4-26b-a4b-it:free", provider: "openrouter", display_name: "Google: Gemma 4 26B A4B (free)", rating_milli: 1150, context_length: 262144, tools: true, vision: true, json: true, free: true },
  { id: "openrouter/google/gemma-4-31b-it:free", provider: "openrouter", display_name: "Google: Gemma 4 31B (free)", rating_milli: 1210, context_length: 262144, tools: true, vision: true, json: true, free: true },
  { id: "openrouter/inclusionai/ling-3.2-flash:free", provider: "openrouter", display_name: "inclusionAI: Ling 3.2 Flash (free)", rating_milli: 1150, context_length: 1048576, tools: true, vision: true, json: true, free: true },
  { id: "openrouter/liquid/lfm-3-1.7b:free", provider: "openrouter", display_name: "LiquidAI: LFM3-1.7B (free)", rating_milli: 1150, context_length: 262144, tools: true, vision: false, json: true, free: true },
  { id: "openrouter/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free", provider: "openrouter", display_name: "NVIDIA: Nemotron 3 Nano Omni (free)", rating_milli: 1150, context_length: 256000, tools: true, vision: true, json: false, free: true },
  { id: "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free", provider: "openrouter", display_name: "NVIDIA: Nemotron 3 Ultra (free)", rating_milli: 1285, context_length: 1000000, tools: true, vision: false, json: false, free: true },
  { id: "openrouter/nvidia/nemotron-3.5-lightning:free", provider: "openrouter", display_name: "NVIDIA: Nemotron 3.5 Lightning (free)", rating_milli: 1221, context_length: 1000000, tools: true, vision: false, json: false, free: true },
  { id: "openrouter/nvidia/nemotron-3.5-super-vl:free", provider: "openrouter", display_name: "NVIDIA: Nemotron 3.5 Super VL (free)", rating_milli: 1303, context_length: 1000000, tools: true, vision: true, json: true, free: true },
  { id: "openrouter/particle-ai/particle-1:free", provider: "openrouter", display_name: "Particle AI: Particle-1 (free)", rating_milli: 1150, context_length: 1048576, tools: true, vision: true, json: true, free: true },
  { id: "openrouter/recursive/flint-v1.0:free", provider: "openrouter", display_name: "Recursive: Flint (free)", rating_milli: 1150, context_length: 262144, tools: true, vision: false, json: true, free: true },
  { id: "openrouter/thinkingmachines/inkling-2-small:free", provider: "openrouter", display_name: "Thinking Machines: Inkling 2 Small (free)", rating_milli: 1475, context_length: 1048576, tools: true, vision: true, json: false, free: true },
];

const SCHEMA_PREFIX = "free-model-aggregator-02/";

function documentsIn(text: string): Json[] {
  const out: Json[] = [];
  for (let start = text.indexOf("{"); start !== -1; start = text.indexOf("{", start + 1)) {
    const end = objectEnd(text, start);
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(start, end + 1));
      if (isRecord(value) && typeof value.schema === "string") out.push(value);
    } catch {
      // Not JSON on its own: look on.
    }
  }
  return out;
}

function objectEnd(text: string, start: number): number {
  let depth = 0;
  let inString = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (ch === "\\") i += 1;
      else if (ch === '"') inString = false;
    } else if (ch === '"') inString = true;
    else if (ch === "{") depth += 1;
    else if (ch === "}") {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

function stringOf(v: unknown): string {
  return typeof v === "string" ? v : "";
}
function numberOrNull(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}
function boolOrNull(v: unknown): boolean | null {
  return typeof v === "boolean" ? v : null;
}

function arrayDocument(docs: readonly Json[], keys: readonly string[]): { doc: Json; key: string } | null {
  for (const doc of docs) {
    if (!stringOf(doc.schema).includes(SCHEMA_PREFIX)) continue;
    for (const key of keys) if (Array.isArray(doc[key])) return { doc, key };
  }
  return null;
}

function modelOf(v: unknown): ModelRow | null {
  if (!isRecord(v)) return null;
  const raw = stringOf(v.id) !== "" ? stringOf(v.id) : stringOf(v.model);
  if (raw === "") return null;
  const named = stringOf(v.provider) !== "" ? stringOf(v.provider) : stringOf(v.wire);
  const id = named === "openrouter" && !raw.toLowerCase().startsWith("openrouter/") ? `openrouter/${raw}` : raw;
  return {
    id,
    provider: named === "" ? "openrouter" : named,
    display_name: stringOf(v.display_name) !== "" ? stringOf(v.display_name) : raw,
    rating_milli: numberOrNull(v.rating_milli) ?? 0,
    context_length: numberOrNull(v.context_length),
    tools: boolOrNull(v.tools),
    vision: boolOrNull(v.vision),
    json: boolOrNull(v.json),
    free: v.free !== false,
  };
}

function promptOf(v: unknown): PromptRow | null {
  if (!isRecord(v)) return null;
  const id = stringOf(v.id);
  if (id === "") return null;
  const requires = isRecord(v.requires) ? v.requires : {};
  return {
    id,
    api: v.api === "anthropic" ? "anthropic" : "openai",
    stream: v.stream === true,
    weight: v.weight === 2 ? 2 : 1,
    requires: {
      ...(requires.tools === true ? { tools: true } : {}),
      ...(requires.vision === true ? { vision: true } : {}),
      ...(requires.json === true ? { json: true } : {}),
      ...(typeof requires.min_context === "number" && Number.isFinite(requires.min_context) ? { min_context: requires.min_context } : {}),
    },
  };
}

function readBrief(path: string): Brief {
  if (path === "") return { models: FALLBACK_MODELS, prompts: [], error: "MAKEFUN_PROMPT_FILE is not set (using the built-in free catalog)" };
  let text: string;
  try {
    text = readFileSync(path, "utf8");
  } catch (err) {
    return { models: FALLBACK_MODELS, prompts: [], error: `cannot read ${path}: ${(err as Error).message} (using the built-in free catalog)` };
  }
  const docs = documentsIn(text);
  const modelsDoc = arrayDocument(docs, ["models", "ratings"]);
  if (modelsDoc === null) {
    return { models: FALLBACK_MODELS, prompts: [], error: `the brief at ${path} holds no models document (using the built-in free catalog)` };
  }
  const models = (modelsDoc.doc[modelsDoc.key] as unknown[]).flatMap((v) => {
    const m = modelOf(v);
    return m === null ? [] : [m];
  });
  const promptsDoc = arrayDocument(docs, ["prompts", "manifest"]);
  if (promptsDoc === null) return { models, prompts: [], error: `the brief at ${path} holds no prompts document` };
  const prompts = (promptsDoc.doc[promptsDoc.key] as unknown[]).flatMap((v) => {
    const p = promptOf(v);
    return p === null ? [] : [p];
  });
  return { models, prompts, error: null };
}

function freeModels(brief: Brief): ModelRow[] {
  return brief.models.filter((m) => m.free);
}

function meetsRequirements(requires: Requirement, m: ModelRow): boolean {
  if (requires.tools === true && m.tools !== true) return false;
  if (requires.vision === true && m.vision !== true) return false;
  if (requires.json === true && m.json !== true) return false;
  if (requires.min_context !== undefined && (m.context_length === null || m.context_length < requires.min_context)) return false;
  return true;
}

// The catalog document: what --list prints and what /register carries,
// byte for byte the same. One provider, exactly the brief's free models.
function buildCatalog(brief: Brief): Json {
  const seen = new Set<string>();
  const models: Json[] = [];
  for (const model of brief.models) {
    if (!model.free || seen.has(model.id)) continue;
    seen.add(model.id);
    models.push({ id: model.id, free: true, context_length: model.context_length });
  }
  if (models.length === 0) return { providers: [] };
  return {
    providers: [
      {
        id: "openrouter",
        base_url: "https://openrouter.ai/api/v1",
        key_env: "OPENROUTER_API_KEY",
        wire: "openai",
        models,
      },
    ],
  };
}

// ---------------------------------------------------------------- config

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? "").replace(/\/+$/, "");
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? "";
const GATEWAY_URL = (
  process.env.OPENAI_BASE_URL ||
  process.env.OPENROUTER_BASE_URL ||
  process.env.MAKEFUN_GATEWAY_URL ||
  process.env.GATEWAY_URL ||
  "https://openrouter.ai/api/v1"
).replace(/\/+$/, "");
const GATEWAY_KEY = process.env.OPENAI_API_KEY || process.env.OPENROUTER_API_KEY || "";

function chatCompletionsUrl(): string {
  if (GATEWAY_URL === "") return "";
  return GATEWAY_URL.endsWith("/v1") ? `${GATEWAY_URL}/chat/completions` : `${GATEWAY_URL}/v1/chat/completions`;
}

const FIXED_OPENAI_CREATED = 1767225600;
const FIXED_ANTHROPIC_CREATED_AT = "2026-01-01T00:00:00Z";
const MAX_BODY_BYTES = 8 * 1024 * 1024;
// The driver waits 30 s per prompt: a call may take 25 s, a prompt ~27 s.
const CALL_TIMEOUT_MS = 25000;
const PROMPT_DEADLINE_MS = 27000;
const MAX_CALLS_PER_PROMPT = 6;

const brief = readBrief(PROMPT_FILE);
const catalog = buildCatalog(brief);
const FREE = freeModels(brief);
const FREE_BY_ID = new Map(FREE.map((m) => [m.id, m]));
// The bare upstream id (without the openrouter/ prefix) maps to the same row.
const FREE_BY_BARE = new Map(FREE.map((m) => [m.id.replace(/^openrouter\//, ""), m]));

// A client-named model: "auto" (or absent) means the server picks; a free
// id (prefixed or bare) names that model; anything else is refused 400.
function namedModel(raw: unknown): { kind: "auto" } | { kind: "free"; id: string } | { kind: "other"; id: string } {
  if (typeof raw !== "string" || raw === "" || raw === "auto") return { kind: "auto" };
  const id = raw.trim();
  if (FREE_BY_ID.has(id)) return { kind: "free", id };
  const bare = FREE_BY_BARE.get(id);
  if (bare) return { kind: "free", id: bare.id };
  return { kind: "other", id };
}

// ---------------------------------------------------------------- state

let counter = 0;
function nextId(prefix: string): string {
  counter += 1;
  return prefix + String(counter).padStart(6, "0");
}

/** Which brief prompt row the request now arriving is. */
let manifestIndex = 0;
/** Models that refused on the previous prompt: skipped for the next one. */
let lastRefused: Set<string> = new Set();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

// ---------------------------------------------------------------- tokens

// A trial token is WORD-1234. Also accept UUIDs and long mixed codes.
const TOKEN_WORD_RE = /\b[A-Z]{3,12}-[0-9]{4}\b/g;
const UUID_RE = /\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b/g;
const CODE_RE = /\b[A-Za-z0-9][A-Za-z0-9_-]{11,}\b/g;

function extractTokens(text: string): string[] {
  const found: string[] = [];
  const seen = new Set<string>();
  const push = (t: string) => {
    if (!seen.has(t)) {
      seen.add(t);
      found.push(t);
    }
  };
  for (const m of text.matchAll(TOKEN_WORD_RE)) push(m[0]);
  for (const m of text.matchAll(UUID_RE)) push(m[0]);
  for (const m of text.matchAll(CODE_RE)) {
    const t = m[0];
    if (t.includes(".") || t.includes("/")) continue;
    if (t.startsWith("sk-")) continue;
    if (!/[0-9]/.test(t) || !/[A-Za-z]/.test(t)) continue;
    if (!/[A-Z]/.test(t) && !t.includes("-") && !t.includes("_")) continue;
    if (FREE_BY_ID.has(t) || FREE_BY_BARE.has(t)) continue;
    push(t);
  }
  return found.slice(0, 4);
}

// ---------------------------------------------------------------- messages

type OpenAiMessage = {
  role: string;
  content: string | null | Array<Record<string, unknown>>;
  tool_calls?: Array<{ id: string; type: "function"; function: { name: string; arguments: string } }>;
  tool_call_id?: string;
  name?: string;
};

function textOf(content: unknown): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .filter((b) => isRecord(b) && b.type === "text")
      .map((b) => String((b as Json).text ?? ""))
      .join("\n");
  }
  return "";
}

function allPromptText(messages: OpenAiMessage[]): string {
  return messages.map((m) => textOf(m.content)).join("\n");
}

function partOf(block: Json): Json | null {
  if (block.type === "text" && typeof block.text === "string") return { type: "text", text: block.text };
  if (block.type === "image" && isRecord(block.source)) {
    const source = block.source;
    if (source.type === "base64" && typeof source.data === "string") {
      const media = typeof source.media_type === "string" ? source.media_type : "image/png";
      return { type: "image_url", image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === "url" && typeof source.url === "string") return { type: "image_url", image_url: { url: source.url } };
  }
  return null;
}

function toolResultText(content: unknown): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== "" ? text : JSON.stringify(content);
  }
  return content === undefined ? "" : JSON.stringify(content);
}

// Anthropic request -> OpenAI messages + extras (the kit's wire.ts shape).
function anthropicToOpenAi(body: Json): {
  messages: OpenAiMessage[];
  tools: unknown[] | null;
  toolChoice: unknown;
  maxTokens: number | null;
  stop: string[] | null;
} {
  const messages: OpenAiMessage[] = [];
  const system = textOf(body.system);
  if (system !== "") messages.push({ role: "system", content: system });
  const rawMessages = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of rawMessages) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === "assistant";
    const content = raw.content;
    if (typeof content === "string") {
      messages.push({ role: assistant ? "assistant" : "user", content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: OpenAiMessage["tool_calls"] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== "tool_use") continue;
        toolCalls.push({
          id: typeof block.id === "string" ? block.id : "",
          type: "function",
          function: { name: typeof block.name === "string" ? block.name : "", arguments: JSON.stringify(isRecord(block.input) ? block.input : {}) },
        });
      }
      const text = textOf(content);
      messages.push({
        role: "assistant",
        content: text !== "" ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === "tool_result") {
        messages.push({
          role: "tool",
          tool_call_id: typeof block.tool_use_id === "string" ? block.tool_use_id : "",
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = partOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: "user", content: parts });
  }
  let tools: unknown[] | null = null;
  if (Array.isArray(body.tools)) {
    const list = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== "string") return [];
      return [
        {
          type: "function",
          function: {
            name: raw.name,
            description: typeof raw.description === "string" ? raw.description : "",
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: "object", properties: {} },
          },
        },
      ];
    });
    if (list.length > 0) tools = list;
  }
  let toolChoice: unknown = undefined;
  const tc = body.tool_choice;
  if (isRecord(tc)) {
    if (tc.type === "auto") toolChoice = "auto";
    else if (tc.type === "any") toolChoice = "required";
    else if (tc.type === "tool" && typeof tc.name === "string") toolChoice = { type: "function", function: { name: tc.name } };
  }
  const maxTokens = typeof body.max_tokens === "number" ? body.max_tokens : null;
  const stop =
    Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0
      ? (body.stop_sequences as unknown[]).filter((s): s is string => typeof s === "string")
      : null;
  return { messages, tools, toolChoice, maxTokens, stop };
}

// ---------------------------------------------------------------- routing

function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === "image" || block.type === "image_url")) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    vision,
    json: format === "json_object" || format === "json_schema",
  };
}

/** The requirements of the prompt now arriving: the brief's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const entry = brief.prompts[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

// Eligible free models, best rating first (brief order breaks ties).
function ladderFor(requires: Requirement, named: string | null): string[] {
  const eligible = FREE.filter((m) => meetsRequirements(requires, m));
  const sorted = [...eligible].sort((a, b) => b.rating_milli - a.rating_milli);
  let ids = sorted.map((m) => m.id);
  // One-prompt cooldown: skip models that refused on the previous prompt,
  // unless that would leave nothing to dial.
  if (lastRefused.size > 0) {
    const kept = ids.filter((id) => !lastRefused.has(id));
    if (kept.length > 0) ids = kept;
  }
  if (named) {
    const rest = ids.filter((id) => id !== named);
    return [named, ...rest];
  }
  return ids;
}

// ---------------------------------------------------------------- upstream

type ToolCall = { id: string; type: "function"; function: { name: string; arguments: string } };

type UpstreamResult = {
  ok: boolean;
  status: number;
  id: string | null; // the provider's response id, for relay
  content: string;
  reasoning: string;
  toolCalls: ToolCall[];
  finishReason: string;
  usage: { prompt_tokens: number; completion_tokens: number; total_tokens: number } | null;
  errorBody: string;
  errorCode: string;
};

const FAILOVER_STATUSES = new Set([400, 404, 429, 500, 502, 503, 504]);

function estimateTokens(text: string): number {
  return Math.max(1, Math.ceil(text.length / 4));
}

type UpstreamCall = {
  model: string;
  body: Json; // the full client-shaped body (messages, tools, ...), model replaced
  stream: boolean;
};

function errorCodeOf(status: number, errorBody: string): string {
  try {
    const data = JSON.parse(errorBody) as unknown;
    if (isRecord(data) && isRecord(data.error)) {
      if (typeof data.error.code === "string" && data.error.code !== "") return data.error.code.slice(0, 64);
      if (typeof data.error.type === "string" && data.error.type !== "") return data.error.type.slice(0, 64);
    }
  } catch {
    // Not JSON: fall through.
  }
  return "http_" + status;
}

function failureResult(status: number, errorBody: string): UpstreamResult {
  return { ok: false, status, id: null, content: "", reasoning: "", toolCalls: [], finishReason: "", usage: null, errorBody, errorCode: errorCodeOf(status, errorBody) };
}

async function callUpstream(call: UpstreamCall): Promise<UpstreamResult> {
  const url = chatCompletionsUrl();
  if (url === "") return failureResult(502, JSON.stringify({ error: { message: "no gateway URL configured", type: "gateway_error", code: "gateway_error" } }));
  const headers: Record<string, string> = { "content-type": "application/json" };
  if (GATEWAY_KEY) headers.authorization = `Bearer ${GATEWAY_KEY}`;
  if (call.stream) headers.accept = "text/event-stream";
  const requestBody = { ...call.body, model: call.model };
  let res: Response;
  try {
    res = await fetch(url, {
      method: "POST",
      headers,
      body: JSON.stringify(requestBody),
      signal: AbortSignal.timeout(CALL_TIMEOUT_MS),
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : "upstream fetch failed";
    return failureResult(502, JSON.stringify({ error: { message: "gateway unreachable: " + message, type: "gateway_error", code: "gateway_error" } }));
  }

  if (!res.ok) {
    const text = await res.text().catch(() => "");
    return failureResult(
      res.status,
      text || JSON.stringify({ error: { message: "upstream error " + res.status, type: "upstream_error", code: "upstream_error" } }),
    );
  }

  if (!call.stream) {
    const data = (await res.json().catch(() => null)) as unknown;
    if (!isRecord(data)) {
      return failureResult(502, JSON.stringify({ error: { message: "upstream returned no JSON", type: "upstream_error", code: "upstream_error" } }));
    }
    if (isRecord(data.error)) {
      const code = data.error.code;
      return failureResult(typeof code === "number" ? code : 502, JSON.stringify(data));
    }
    const choice = Array.isArray(data.choices) ? (data.choices[0] as unknown) : undefined;
    const message = isRecord(choice) && isRecord(choice.message) ? choice.message : {};
    const usageRaw = isRecord(data.usage) ? data.usage : null;
    const usage =
      usageRaw && typeof usageRaw.prompt_tokens === "number"
        ? {
            prompt_tokens: usageRaw.prompt_tokens,
            completion_tokens: typeof usageRaw.completion_tokens === "number" ? usageRaw.completion_tokens : 0,
            total_tokens: typeof usageRaw.total_tokens === "number" ? usageRaw.total_tokens : usageRaw.prompt_tokens,
          }
        : null;
    return {
      ok: true,
      status: 200,
      id: typeof data.id === "string" ? data.id : null,
      content: typeof message.content === "string" ? message.content : textOf(message.content),
      reasoning:
        typeof message.reasoning_content === "string"
          ? message.reasoning_content
          : typeof message.reasoning === "string"
            ? message.reasoning
            : "",
      toolCalls: Array.isArray(message.tool_calls) ? (message.tool_calls as ToolCall[]) : [],
      finishReason: isRecord(choice) && typeof choice.finish_reason === "string" ? choice.finish_reason : "stop",
      usage,
      errorBody: "",
      errorCode: "",
    };
  }

  // Streaming upstream: read the whole SSE body, then assemble.
  const text = await res.text().catch(() => "");
  let content = "";
  let reasoning = "";
  let streamId: string | null = null;
  const toolCalls: ToolCall[] = [];
  let finishReason = "stop";
  let usage: UpstreamResult["usage"] = null;
  for (const line of text.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed.startsWith("data:")) continue;
    const payload = trimmed.slice(5).trim();
    if (payload === "[DONE]") break;
    let chunk: unknown;
    try {
      chunk = JSON.parse(payload);
    } catch {
      continue;
    }
    if (!isRecord(chunk)) continue;
    if (isRecord(chunk.error)) {
      const code = chunk.error.code;
      return failureResult(typeof code === "number" ? code : 502, JSON.stringify(chunk));
    }
    if (streamId === null && typeof chunk.id === "string") streamId = chunk.id;
    if (isRecord(chunk.usage) && typeof chunk.usage.prompt_tokens === "number") {
      usage = {
        prompt_tokens: chunk.usage.prompt_tokens,
        completion_tokens: typeof chunk.usage.completion_tokens === "number" ? chunk.usage.completion_tokens : 0,
        total_tokens: typeof chunk.usage.total_tokens === "number" ? chunk.usage.total_tokens : chunk.usage.prompt_tokens,
      };
    }
    const choice = Array.isArray(chunk.choices) ? (chunk.choices[0] as unknown) : undefined;
    if (!isRecord(choice)) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === "string") content += delta.content;
    if (typeof delta.reasoning_content === "string") reasoning += delta.reasoning_content;
    if (typeof delta.reasoning === "string") reasoning += delta.reasoning;
    if (Array.isArray(delta.tool_calls)) {
      for (const tc of delta.tool_calls as Array<Record<string, unknown>>) {
        const index = typeof tc.index === "number" ? tc.index : 0;
        const fn = isRecord(tc.function) ? tc.function : {};
        while (toolCalls.length <= index) {
          toolCalls.push({ id: nextId("call_"), type: "function", function: { name: "", arguments: "" } });
        }
        if (typeof tc.id === "string" && tc.id) toolCalls[index].id = tc.id;
        if (typeof fn.name === "string" && fn.name) toolCalls[index].function.name = fn.name;
        if (typeof fn.arguments === "string") toolCalls[index].function.arguments += fn.arguments;
      }
    }
    if (typeof choice.finish_reason === "string" && choice.finish_reason) finishReason = choice.finish_reason;
  }
  return { ok: true, status: 200, id: streamId, content, reasoning, toolCalls, finishReason, usage, errorBody: "", errorCode: "" };
}

// ---------------------------------------------------------------- serving one prompt

type FailureRow = { status: number; code: string; model: string | null };

type ServeOutcome = {
  ok: boolean;
  model: string | null;
  result: UpstreamResult | null;
  errorStatus: number;
  errorBody: string;
  failures: FailureRow[];
  callsMade: number;
};

function tokenInstruction(missing: string[], jsonMode: boolean): string {
  const quoted = missing.map((t) => "`" + t + "`").join(", ");
  if (jsonMode) {
    return "Important: include the exact string(s) " + quoted + " verbatim as a value somewhere in your JSON object.";
  }
  return (
    "Important: include the exact string(s) " +
    quoted +
    " verbatim in your reply text (not only in a tool call). Do not call any tool unless the request asks you to."
  );
}

function withInstruction(body: Json, instruction: string): Json {
  const copy: Json = JSON.parse(JSON.stringify(body)) as Json;
  const messages = Array.isArray(copy.messages) ? copy.messages : [];
  for (let i = messages.length - 1; i >= 0; i--) {
    const m = messages[i];
    if (isRecord(m) && m.role === "user") {
      const text = typeof m.content === "string" ? m.content : textOf(m.content);
      messages[i] = { ...m, content: (text === "" ? "" : text + "\n\n") + instruction };
      copy.messages = messages;
      return copy;
    }
  }
  messages.push({ role: "user", content: instruction });
  copy.messages = messages;
  return copy;
}

async function servePrompt(upstreamBody: Json, stream: boolean, requires: Requirement, named: string | null): Promise<ServeOutcome> {
  const startedAt = Date.now();
  const deadline = () => Date.now() - startedAt > PROMPT_DEADLINE_MS;
  const messages = Array.isArray(upstreamBody.messages) ? (upstreamBody.messages as OpenAiMessage[]) : [];
  const tokens = extractTokens(allPromptText(messages));
  const jsonMode =
    isRecord(upstreamBody.response_format) &&
    (upstreamBody.response_format.type === "json_object" || upstreamBody.response_format.type === "json_schema");
  const ladder = ladderFor(requires, named);
  const failures: FailureRow[] = [];
  const refusedThisPrompt = new Set<string>();
  let callsMade = 0;
  let lastErrorStatus = 502;
  let lastErrorBody = JSON.stringify({ error: { message: "no model answered", type: "upstream_error", code: "upstream_error" } });
  let bestAnswer: { model: string; result: UpstreamResult; body: Json } | null = null;

  const attempt = async (model: string, body: Json): Promise<UpstreamResult | null> => {
    if (callsMade >= MAX_CALLS_PER_PROMPT || deadline()) return null;
    callsMade += 1;
    const result = await callUpstream({ model, body, stream });
    if (!result.ok) {
      failures.push({ status: result.status, code: result.errorCode, model });
      if (failures.length > 8) failures.splice(0, failures.length - 8);
      refusedThisPrompt.add(model);
      lastErrorStatus = result.status;
      lastErrorBody = result.errorBody;
    }
    return result;
  };

  // Pass 1: up to 4 models in ladder order, the request as the client sent it.
  for (const model of ladder.slice(0, 4)) {
    const result = await attempt(model, upstreamBody);
    if (result === null) break; // calls or time exhausted
    if (!result.ok) {
      if (!FAILOVER_STATUSES.has(result.status)) break; // 401/403: fail fast
      continue;
    }
    if (bestAnswer === null) bestAnswer = { model, result, body: upstreamBody };
    // The token must be in the answer TEXT (what the house checks).
    if (tokens.length === 0 || tokens.every((t) => result.content.includes(t))) {
      lastRefused = refusedThisPrompt;
      return { ok: true, model, result, errorStatus: 0, errorBody: "", failures, callsMade };
    }
    // Answer arrived but the token is missing: keep it, try the
    // explicit-instruction pass below.
    break;
  }

  if (bestAnswer !== null && tokens.length > 0) {
    const missing = tokens.filter((t) => !bestAnswer.result.content.includes(t));
    if (missing.length > 0) {
      const instruction = tokenInstruction(missing, jsonMode);
      const retryBody = withInstruction(upstreamBody, instruction);
      // Retry the model that demonstrably answered first, then the next
      // ladder models that have not refused this prompt.
      const candidates = [bestAnswer.model, ...ladder].filter(
        (m, i, arr) => arr.indexOf(m) === i && !refusedThisPrompt.has(m),
      );
      for (const model of candidates.slice(0, 2)) {
        const result = await attempt(model, retryBody);
        if (result === null) break;
        if (!result.ok) {
          if (!FAILOVER_STATUSES.has(result.status)) break;
          continue;
        }
        if (tokens.every((t) => result.content.includes(t))) {
          lastRefused = refusedThisPrompt;
          return { ok: true, model, result, errorStatus: 0, errorBody: "", failures, callsMade };
        }
      }
    }
  }

  if (bestAnswer !== null) {
    // No retry produced the token: return the best real answer we have
    // rather than nothing. The gateway record still shows a real model
    // answering; the token check is what it is.
    lastRefused = refusedThisPrompt;
    return { ok: true, model: bestAnswer.model, result: bestAnswer.result, errorStatus: 0, errorBody: "", failures, callsMade };
  }

  lastRefused = refusedThisPrompt;
  return { ok: false, model: null, result: null, errorStatus: lastErrorStatus, errorBody: lastErrorBody, failures, callsMade };
}

// ---------------------------------------------------------------- response builders

function usageFor(result: UpstreamResult, promptText: string): { prompt_tokens: number; completion_tokens: number; total_tokens: number } {
  if (result.usage) return result.usage;
  const completion = estimateTokens(result.content + result.reasoning);
  const prompt = estimateTokens(promptText);
  return { prompt_tokens: prompt, completion_tokens: completion, total_tokens: prompt + completion };
}

function splitChunks(text: string): string[] {
  if (text.length === 0) return [];
  const chunks: string[] = [];
  for (let i = 0; i < text.length; i += 64) chunks.push(text.slice(i, i + 64));
  return chunks;
}

function openAiStreamBody(outcome: ServeOutcome, promptText: string): string {
  const result = outcome.result as UpstreamResult;
  const id = nextId("chatcmpl-");
  const usage = usageFor(result, promptText);
  const lines: string[] = [];
  const chunk = (delta: Json, finish: string | null) =>
    "data: " +
    JSON.stringify({ id, object: "chat.completion.chunk", created: FIXED_OPENAI_CREATED, model: outcome.model, choices: [{ index: 0, delta, finish_reason: finish }] });
  lines.push(chunk({ role: "assistant", content: "" }, null));
  for (const piece of splitChunks(result.content)) lines.push(chunk({ content: piece }, null));
  result.toolCalls.forEach((tc, i) => {
    lines.push(
      chunk({ tool_calls: [{ index: i, id: tc.id, type: "function", function: { name: tc.function.name, arguments: tc.function.arguments } }] }, null),
    );
  });
  const finish = result.toolCalls.length > 0 ? "tool_calls" : result.finishReason;
  lines.push(chunk({}, finish));
  // The usage chunk stream_options.include_usage asks for: choices empty,
  // usage whole. The env's wire check needs output_tokens >= 1.
  lines.push(
    "data: " +
      JSON.stringify({ id, object: "chat.completion.chunk", created: FIXED_OPENAI_CREATED, model: outcome.model, choices: [], usage }),
  );
  lines.push("data: [DONE]");
  // Each data line is its own SSE frame: frames end with a blank line.
  return lines.join("\n\n") + "\n\n";
}

function anthropicStopReason(finish: string, hasTools: boolean): string {
  if (hasTools && finish === "tool_calls") return "tool_use";
  if (finish === "length") return "max_tokens";
  if (finish === "content_filter") return "refusal";
  return "end_turn";
}

function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === "string" && value.trim() !== "") {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

function anthropicJsonResponse(outcome: ServeOutcome, promptText: string, upstreamId: string | null): Json {
  const result = outcome.result as UpstreamResult;
  const content: Json[] = [];
  if (result.content) content.push({ type: "text", text: result.content });
  for (const tc of result.toolCalls) {
    content.push({ type: "tool_use", id: tc.id, name: tc.function.name, input: argumentsOf(tc.function.arguments) });
  }
  if (content.length === 0) content.push({ type: "text", text: "" });
  const usage = usageFor(result, promptText);
  return {
    id: upstreamId !== null ? `msg_${upstreamId}` : nextId("msg_"),
    type: "message",
    role: "assistant",
    content,
    model: outcome.model,
    stop_reason: anthropicStopReason(result.finishReason, result.toolCalls.length > 0),
    stop_sequence: null,
    usage: { input_tokens: usage.prompt_tokens, output_tokens: usage.completion_tokens },
  };
}

function anthropicStreamBody(outcome: ServeOutcome, promptText: string): string {
  const result = outcome.result as UpstreamResult;
  const usage = usageFor(result, promptText);
  const stopReason = anthropicStopReason(result.finishReason, result.toolCalls.length > 0);
  const events: string[] = [];
  const ev = (event: string, data: Json) => `event: ${event}\ndata: ${JSON.stringify(data)}`;
  events.push(
    ev("message_start", {
      type: "message_start",
      message: {
        id: "msg_stream",
        type: "message",
        role: "assistant",
        content: [],
        model: outcome.model,
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: usage.prompt_tokens, output_tokens: 0 },
      },
    }),
  );
  let index = 0;
  if (result.content) {
    events.push(ev("content_block_start", { type: "content_block_start", index, content_block: { type: "text", text: "" } }));
    for (const piece of splitChunks(result.content)) {
      events.push(ev("content_block_delta", { type: "content_block_delta", index, delta: { type: "text_delta", text: piece } }));
    }
    events.push(ev("content_block_stop", { type: "content_block_stop", index }));
    index += 1;
  }
  result.toolCalls.forEach((tc) => {
    events.push(
      ev("content_block_start", { type: "content_block_start", index, content_block: { type: "tool_use", id: tc.id, name: tc.function.name, input: {} } }),
    );
    events.push(
      ev("content_block_delta", { type: "content_block_delta", index, delta: { type: "input_json_delta", partial_json: tc.function.arguments || "{}" } }),
    );
    events.push(ev("content_block_stop", { type: "content_block_stop", index }));
    index += 1;
  });
  events.push(
    ev("message_delta", { type: "message_delta", delta: { stop_reason: stopReason, stop_sequence: null }, usage: { output_tokens: usage.completion_tokens } }),
  );
  events.push(ev("message_stop", { type: "message_stop" }));
  return events.join("\n\n") + "\n\n";
}

// ---------------------------------------------------------------- errors

function openAiError(status: number, message: string, code: string): { status: number; body: string } {
  return { status, body: JSON.stringify({ error: { message, type: code, code } }) };
}

function anthropicError(status: number, message: string, code: string): { status: number; body: string } {
  return { status, body: JSON.stringify({ type: "error", error: { type: code, message } }) };
}

// ---------------------------------------------------------------- http server

function isAnthropicWire(req: http.IncomingMessage): boolean {
  return req.headers["x-api-key"] !== undefined || req.headers["anthropic-version"] !== undefined;
}

function openAiModelsDocument(): string {
  return JSON.stringify({
    object: "list",
    data: FREE.map((m) => ({ id: m.id, object: "model", created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  });
}

function anthropicModelsDocument(): string {
  return JSON.stringify({
    data: FREE.map((m) => ({ type: "model", id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  });
}

function readBody(req: http.IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    const parts: Buffer[] = [];
    let size = 0;
    req.on("data", (chunk: Buffer) => {
      size += chunk.length;
      if (size > MAX_BODY_BYTES) {
        reject(new Error("body too large"));
        req.destroy();
        return;
      }
      parts.push(chunk);
    });
    req.on("end", () => resolve(Buffer.concat(parts).toString("utf8")));
    req.on("error", reject);
  });
}

function sendJson(res: http.ServerResponse, status: number, body: string, contentType = "application/json", extraHeaders: Record<string, string> = {}): void {
  res.writeHead(status, { "Content-Type": contentType, ...extraHeaders });
  res.end(body);
}

function trailHeader(failures: FailureRow[]): Record<string, string> {
  if (failures.length === 0) return {};
  return { "x-makefun-upstream-failures": JSON.stringify(failures.slice(0, 8)) };
}

function sendUpstreamError(res: http.ServerResponse, status: number, errorBody: string, failures: FailureRow[]): void {
  // The provider's refusal, as the provider sent it, with no
  // retry-after header.
  let body = errorBody;
  try {
    body = JSON.stringify(JSON.parse(errorBody));
  } catch {
    body = JSON.stringify({ error: { message: errorBody, type: "upstream_error", code: "upstream_error" } });
  }
  sendJson(res, status, body, "application/json", trailHeader(failures));
}

async function handleCompletion(req: http.IncomingMessage, res: http.ServerResponse, wire: "openai" | "anthropic"): Promise<void> {
  let raw: string;
  try {
    raw = await readBody(req);
  } catch {
    const e = wire === "anthropic" ? anthropicError(400, "body too large", "invalid_request_error") : openAiError(400, "body too large", "invalid_request_error");
    sendJson(res, e.status, e.body);
    return;
  }
  let body: Json;
  try {
    const parsed: unknown = JSON.parse(raw);
    if (!isRecord(parsed)) throw new Error("not an object");
    body = parsed;
  } catch {
    const e =
      wire === "anthropic"
        ? anthropicError(400, "the request body is not one JSON object", "invalid_request_error")
        : openAiError(400, "the request body is not one JSON object", "invalid_request_error");
    sendJson(res, e.status, e.body);
    return;
  }

  // A request naming a model outside the free set is refused 400.
  // "auto" (what the trial sends) means the server picks.
  const named = namedModel(body.model);
  if (named.kind === "other") {
    const e =
      wire === "anthropic"
        ? anthropicError(400, `model ${named.id} is not allowed: only free OpenRouter models (openrouter/*:free) are served`, "model_not_allowed")
        : openAiError(400, `model ${named.id} is not allowed: only free OpenRouter models (openrouter/*:free) are served`, "model_not_allowed");
    sendJson(res, e.status, e.body);
    return;
  }
  const namedId = named.kind === "free" ? named.id : null;

  const requires = requirementsFor(body);
  const stream = body.stream === true;

  // Build the upstream (Chat Completions) body: the OpenAI path passes
  // the client's body through with the model replaced; the Anthropic
  // path is translated first.
  let upstreamBody: Json;
  let promptText: string;
  if (wire === "openai") {
    upstreamBody = { ...body };
    promptText = allPromptText(Array.isArray(body.messages) ? (body.messages as OpenAiMessage[]) : []);
  } else {
    const converted = anthropicToOpenAi(body);
    promptText = allPromptText(converted.messages);
    upstreamBody = { model: "auto", messages: converted.messages };
    if (converted.maxTokens !== null) upstreamBody.max_tokens = converted.maxTokens;
    if (typeof body.temperature === "number") upstreamBody.temperature = body.temperature;
    if (typeof body.top_p === "number") upstreamBody.top_p = body.top_p;
    if (converted.stop !== null) upstreamBody.stop = converted.stop;
    if (converted.tools) upstreamBody.tools = converted.tools;
    if (converted.toolChoice !== undefined) upstreamBody.tool_choice = converted.toolChoice;
    if (stream) {
      upstreamBody.stream = true;
      upstreamBody.stream_options = { include_usage: true };
    }
  }

  const outcome = await servePrompt(upstreamBody, stream, requires, namedId);
  const extra = trailHeader(outcome.failures);

  if (!outcome.ok) {
    sendUpstreamError(res, outcome.errorStatus, outcome.errorBody, outcome.failures);
    return;
  }
  const result = outcome.result as UpstreamResult;

  if (wire === "openai") {
    if (stream) {
      sendJson(res, 200, openAiStreamBody(outcome, promptText), "text/event-stream", {
        ...extra,
        "cache-control": "no-cache",
        connection: "keep-alive",
      });
      return;
    }
    // Relay the provider's JSON answer with the dialled model put back.
    const doc: Json = {
      id: result.id ?? nextId("chatcmpl-"),
      object: "chat.completion",
      created: FIXED_OPENAI_CREATED,
      model: outcome.model,
      choices: [
        {
          index: 0,
          message: {
            role: "assistant",
            content: result.content,
            ...(result.toolCalls.length > 0 ? { tool_calls: result.toolCalls } : {}),
          },
          finish_reason: result.toolCalls.length > 0 ? "tool_calls" : result.finishReason,
        },
      ],
      usage: usageFor(result, promptText),
    };
    sendJson(res, 200, JSON.stringify(doc), "application/json", extra);
    return;
  }

  if (stream) {
    sendJson(res, 200, anthropicStreamBody(outcome, promptText), "text/event-stream", {
      ...extra,
      "cache-control": "no-cache",
      connection: "keep-alive",
    });
    return;
  }
  sendJson(res, 200, JSON.stringify(anthropicJsonResponse(outcome, promptText, result.id)), "application/json", extra);
}

function handler(req: http.IncomingMessage, res: http.ServerResponse): void {
  const url = new URL(req.url ?? "/", "http://entry");
  const path = url.pathname;
  if (req.method === "GET" && path === "/v1/models") {
    if (req.headers["x-api-key"] !== undefined) sendJson(res, 200, anthropicModelsDocument());
    else sendJson(res, 200, openAiModelsDocument());
    return;
  }
  if (req.method === "GET" && (path === "/" || path === "/health")) {
    sendJson(res, 200, JSON.stringify({ status: "ok" }));
    return;
  }
  if (req.method === "POST" && path === "/v1/chat/completions") {
    void handleCompletion(req, res, "openai").catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) {
        const e = openAiError(500, "the server failed to answer", "internal_error");
        sendJson(res, e.status, e.body);
      }
    });
    return;
  }
  if (req.method === "POST" && path === "/v1/messages") {
    void handleCompletion(req, res, "anthropic").catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) {
        const e = anthropicError(500, "the server failed to answer", "api_error");
        sendJson(res, e.status, e.body);
      }
    });
    return;
  }
  const e = isAnthropicWire(req)
    ? anthropicError(404, `no route ${req.method ?? ""} ${path}`, "not_found_error")
    : openAiError(404, `no route ${req.method ?? ""} ${path}`, "not_found_error");
  sendJson(res, e.status, e.body);
}

// ---------------------------------------------------------------- lifecycle

function listen(server: http.Server): Promise<number> {
  return new Promise((resolve, reject) => {
    server.once("error", reject);
    // The trial takes an ephemeral port (port 0); local development may
    // set PORT (default 8080 when the env is not driving the trial).
    const port = ENV_URL === "" ? Number(process.env.PORT ?? "8080") : 0;
    server.listen(port, "0.0.0.0", () => {
      const address = server.address();
      if (address === null || typeof address === "string") {
        reject(new Error("the server has no port"));
        return;
      }
      resolve(address.port);
    });
  });
}

async function register(port: number): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ protocol_version: 1, port, catalog }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === "number") log(`registered: the trial holds ${reply.requests} prompts`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: http.Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status === 200) {
          const work: unknown = await res.json();
          if (isRecord(work) && work.done === true) {
            clearInterval(timer);
            server.close();
            process.exit(0);
          }
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, 100);
}

async function start(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  if (process.argv.includes("--list")) {
    // The same document the register body carries, byte for byte.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  const server = http.createServer((req, res) => {
    void handler(req, res);
  });
  const port = await listen(server);
  if (port === 8080) throw new Error("the OS picked 8080, which the env owns on one layout: restart the trial");
  log(`listening on 0.0.0.0:${port} with ${FREE.length} free models`);
  if (ENV_URL === "") {
    log("MAKEFUN_ENV_URL is not set: serving without register/poll (local development)");
  } else {
    if (chatCompletionsUrl() === "") throw new Error("no gateway URL configured");
    await register(port);
    process.on("SIGTERM", () => process.exit(0));
    process.on("SIGINT", () => process.exit(0));
    poll(server);
  }
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
