import {
  completeSimple as completeSimpleImport,
  getModel as getModelImport,
} from "@mariozechner/pi-ai";
import {
  PROVIDER_ID,
  applyFailureStats,
  applySuccessStats,
  classifyFailure,
  isFailoverAllowed,
  resolveProfileAccessSnapshot,
} from "./runtime.mjs";
import { updateStore } from "./store.mjs";

let completeSimpleImpl = completeSimpleImport;
let getModelImpl = getModelImport;
const DEFAULT_CODEX_INSTRUCTIONS = "You are a helpful assistant.";
const DEFAULT_CODEX_BASE_URL = "https://chatgpt.com/backend-api";
const CODEX_FORWARD_COMPAT_MODEL_IDS = new Set([
  "gpt-5.2",
  "gpt-5.4",
  "gpt-5.4-mini",
  "gpt-5.5",
  "gpt-5.6-sol",
]);
const CODEX_THINKING_LEVELS = new Set(["off", "minimal", "low", "medium", "high", "xhigh"]);
const CODEX_WEB_SEARCH_CONTEXT_SIZES = new Set(["low", "medium", "high"]);

function isOpenAICodexForwardCompatModel(modelId) {
  const selected = String(modelId ?? "").trim().toLowerCase();
  return CODEX_FORWARD_COMPAT_MODEL_IDS.has(selected);
}

function resolveCodexModel(modelId) {
  try {
    const resolved = getModelImpl(PROVIDER_ID, modelId);
    if (resolved) {
      return resolved;
    }
  } catch (error) {
    if (!isOpenAICodexForwardCompatModel(modelId)) {
      throw error;
    }
  }
  if (!isOpenAICodexForwardCompatModel(modelId)) {
    throw new Error(`Unknown model: ${PROVIDER_ID}/${modelId}`);
  }
  const trimmedModelId = String(modelId ?? "").trim();
  return {
    id: trimmedModelId,
    name: trimmedModelId,
    api: "openai-codex-responses",
    provider: PROVIDER_ID,
    baseUrl: DEFAULT_CODEX_BASE_URL,
    reasoning: true,
    input: ["text", "image"],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
    contextWindow: 200000,
    maxTokens: 200000,
  };
}

function normalizeThinkingLevel(value) {
  const selected = String(value ?? "").trim().toLowerCase();
  if (!selected) {
    return "low";
  }
  if (!CODEX_THINKING_LEVELS.has(selected)) {
    throw new Error("Invalid Codex thinking level. Expected off, minimal, low, medium, high, or xhigh.");
  }
  return selected;
}

function normalizeWebSearchMode(value) {
  const selected = String(value ?? "").trim().toLowerCase();
  return selected === "live" ? "live" : "cached";
}

function normalizeWebSearchContextSize(value) {
  const selected = String(value ?? "").trim().toLowerCase();
  if (CODEX_WEB_SEARCH_CONTEXT_SIZES.has(selected)) {
    return selected;
  }
  return "medium";
}

function normalizeAllowedDomains(domains) {
  if (!Array.isArray(domains)) {
    return [];
  }
  const deduped = [];
  const seen = new Set();
  for (const domain of domains) {
    let value = String(domain ?? "").trim().toLowerCase();
    if (!value) {
      continue;
    }
    value = value.replace(/^https?:\/\//, "").replace(/^www\./, "").replace(/\/.*$/, "");
    if (!value || seen.has(value)) {
      continue;
    }
    seen.add(value);
    deduped.push(value);
  }
  return deduped;
}

function toPiUserContent(content) {
  if (typeof content === "string") {
    return content;
  }
  if (!Array.isArray(content)) {
    return "";
  }
  const parts = [];
  for (const item of content) {
    if (!item || typeof item !== "object") continue;
    if (item.type === "text" && typeof item.text === "string") {
      parts.push({ type: "text", text: item.text });
    } else if (
      item.type === "image" &&
      typeof item.data === "string" &&
      typeof item.mimeType === "string"
    ) {
      parts.push({ type: "image", data: item.data, mimeType: item.mimeType });
    }
  }
  return parts;
}

function toPiMessages(messages, modelId) {
  const out = [];
  for (const message of messages ?? []) {
    if (!message || typeof message !== "object") continue;
    if (message.role === "user") {
      out.push({
        role: "user",
        content: toPiUserContent(message.content),
        timestamp: Date.now(),
      });
      continue;
    }
    if (message.role === "assistant") {
      const content = [];
      if (typeof message.content === "string" && message.content) {
        content.push({ type: "text", text: message.content });
      }
      for (const toolCall of message.tool_calls ?? []) {
        if (!toolCall?.name) continue;
        content.push({
          type: "toolCall",
          id: String(toolCall.id ?? ""),
          name: String(toolCall.name),
          arguments: toolCall.args && typeof toolCall.args === "object" ? toolCall.args : {},
        });
      }
      out.push({
        role: "assistant",
        content,
        api: "openai-codex-responses",
        provider: PROVIDER_ID,
        model: modelId,
        usage: {
          input: 0,
          output: 0,
          cacheRead: 0,
          cacheWrite: 0,
          totalTokens: 0,
          cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
        },
        stopReason: "stop",
        timestamp: Date.now(),
      });
      continue;
    }
    if (message.role === "toolResult") {
      const content = [];
      if (typeof message.content === "string" && message.content) {
        content.push({ type: "text", text: message.content });
      } else if (Array.isArray(message.content)) {
        for (const item of message.content) {
          if (item?.type === "text" && typeof item.text === "string") {
            content.push({ type: "text", text: item.text });
          }
        }
      }
      out.push({
        role: "toolResult",
        toolCallId: String(message.tool_call_id ?? ""),
        toolName: String(message.tool_name ?? ""),
        content,
        isError: message.is_error === true,
        timestamp: Date.now(),
      });
    }
  }
  return out;
}

function toPiTools(tools) {
  return (tools ?? [])
    .filter((tool) => tool?.name && tool?.parameters)
    .map((tool) => ({
      name: String(tool.name),
      description: String(tool.description ?? ""),
      parameters: tool.parameters,
    }));
}

function usageToResponse(usage) {
  return {
    input_tokens: usage?.input ?? 0,
    output_tokens: usage?.output ?? 0,
    total_tokens: usage?.totalTokens ?? 0,
    token_usage: {
      prompt_tokens: usage?.input ?? 0,
      completion_tokens: usage?.output ?? 0,
      total_tokens: usage?.totalTokens ?? 0,
    },
  };
}

function buildNativeWebSearchTool(request) {
  const allowedDomains = normalizeAllowedDomains(request.allowed_domains);
  const tool = {
    type: "web_search",
    external_web_access: normalizeWebSearchMode(request.mode) === "live",
    search_context_size: normalizeWebSearchContextSize(request.context_size),
  };
  if (allowedDomains.length > 0) {
    tool.filters = { allowed_domains: allowedDomains };
  }
  return tool;
}

function assistantTextContent(message) {
  return (message?.content ?? [])
    .filter((item) => item?.type === "text" && typeof item.text === "string" && item.text.trim())
    .map((item) => item.text.trim())
    .join("\n\n")
    .trim();
}

function extractSourcesFromText(text, allowedDomains) {
  const results = [];
  const seen = new Set();
  const matches = String(text ?? "").matchAll(/https?:\/\/[^\s)<>"']+/g);
  for (const match of matches) {
    const url = String(match[0] ?? "").replace(/[.,;:]+$/, "").trim();
    if (!url) {
      continue;
    }
    if (allowedDomains.length > 0) {
      try {
        const hostname = new URL(url).hostname.replace(/^www\./, "").toLowerCase();
        if (!allowedDomains.some((allowed) => hostname === allowed || hostname.endsWith(`.${allowed}`))) {
          continue;
        }
      } catch {
        continue;
      }
    }
    if (seen.has(url)) {
      continue;
    }
    seen.add(url);
    results.push({ title: "", url });
  }
  return results;
}

function convertAssistantMessage(message, profileId) {
  const text = [];
  const toolCalls = [];
  for (const item of message.content ?? []) {
    if (item?.type === "text" && typeof item.text === "string") {
      text.push(item.text);
    } else if (item?.type === "toolCall") {
      toolCalls.push({
        id: item.id,
        name: item.name,
        args: item.arguments ?? {},
      });
    }
  }
  return {
    profile_id: profileId,
    provider: PROVIDER_ID,
    model: message.model,
    content: text.join(""),
    tool_calls: toolCalls,
    stop_reason: message.stopReason,
    error_message: message.errorMessage ?? null,
    usage: usageToResponse(message.usage),
  };
}

async function runCompletionWithProfile(profile, accessToken, request) {
  const model = resolveCodexModel(request.model);
  const context = {
    systemPrompt:
      (typeof request.system_prompt === "string" && request.system_prompt.trim()) ||
      DEFAULT_CODEX_INSTRUCTIONS,
    messages: toPiMessages(request.messages, request.model),
    tools: toPiTools(request.tools),
  };
  const reasoning = normalizeThinkingLevel(request.reasoning);
  const options = {
    apiKey: accessToken,
    transport: request.transport ?? "auto",
    sessionId: request.session_id,
    maxTokens: request.max_tokens,
  };
  if (reasoning !== "off") {
    options.reasoning = reasoning;
  }
  const response = await completeSimpleImpl(model, context, options);
  return convertAssistantMessage(response, profile.id);
}

async function runWebSearchWithProfile(profile, accessToken, request) {
  const model = resolveCodexModel(request.model);
  const query = String(request.query ?? "").trim();
  if (!query) {
    throw new Error("query must not be empty");
  }
  const allowedDomains = normalizeAllowedDomains(request.allowed_domains);
  const context = {
    systemPrompt:
      "You are a concise research assistant. Search the web and answer with a brief grounded summary. " +
      "End with a `Sources:` section listing the source URLs you actually used, one per line.",
    messages: [
      {
        role: "user",
        timestamp: Date.now(),
        content:
          `Search query: ${query}\n` +
          (allowedDomains.length > 0
            ? `Restrict sources to these domains when possible: ${allowedDomains.join(", ")}`
            : "Use public web sources."),
      },
    ],
    tools: [],
  };
  const options = {
    apiKey: accessToken,
    transport: request.transport ?? "auto",
    sessionId: request.session_id,
    maxTokens: Number(request.max_tokens ?? 1200) || 1200,
    reasoning: normalizeThinkingLevel(request.reasoning ?? "low"),
    onPayload: (payload) => {
      if (!payload || typeof payload !== "object") {
        return;
      }
      const tools = Array.isArray(payload.tools) ? [...payload.tools] : [];
      tools.push(buildNativeWebSearchTool(request));
      payload.tools = tools;
    },
  };
  const response = await completeSimpleImpl(model, context, options);
  const content = assistantTextContent(response);
  const sources = extractSourcesFromText(content, allowedDomains);
  return {
    profile_id: profile.id,
    provider: PROVIDER_ID,
    model: response?.model ?? model.id,
    backend: "openai_codex_native",
    query,
    content,
    sources,
    usage: usageToResponse(response?.usage),
    response_id: null,
    stop_reason: response?.stopReason ?? null,
    mode: normalizeWebSearchMode(request.mode),
  };
}

async function resolveChatProfileSnapshot(preferredProfileId) {
  const resolved = await resolveProfileAccessSnapshot(preferredProfileId);
  return {
    profile: {
      id: resolved.profile_id,
      account_id: resolved.account_id ?? null,
    },
    access_token: resolved.access_token,
    expires: resolved.expires,
  };
}

async function markProfileSuccess(profileId) {
  await updateStore(async (store) => {
    if (!store?.profiles?.[profileId]) {
      return;
    }
    applySuccessStats(store, profileId, PROVIDER_ID);
  });
}

async function markProfileFailure(profileId, failureClass, message) {
  await updateStore(async (store) => {
    if (!store?.profiles?.[profileId]) {
      return;
    }
    applyFailureStats(store, profileId, failureClass, message);
  });
}

export async function chatComplete(request = {}) {
  const preferredProfileId = request.preferred_profile_id
    ? String(request.preferred_profile_id).trim()
    : undefined;
  const failures = [];
  while (true) {
    const resolved = await resolveChatProfileSnapshot(preferredProfileId);
    try {
      const finalResult = await runCompletionWithProfile(
        resolved.profile,
        resolved.access_token,
        request,
      );
      await markProfileSuccess(resolved.profile.id);
      return finalResult;
    } catch (error) {
      const failureClass = classifyFailure(error);
      if (failureClass === "network" || failureClass === "server") {
        try {
          const finalResult = await runCompletionWithProfile(
            resolved.profile,
            resolved.access_token,
            request,
          );
          await markProfileSuccess(resolved.profile.id);
          return finalResult;
        } catch (retryError) {
          const retryFailureClass = classifyFailure(retryError);
          await markProfileFailure(
            resolved.profile.id,
            retryFailureClass,
            retryError?.message ?? String(retryError),
          );
          failures.push(`${resolved.profile.id}:${retryFailureClass}`);
          if (!isFailoverAllowed(retryFailureClass)) {
            throw retryError;
          }
          continue;
        }
      }
      await markProfileFailure(
        resolved.profile.id,
        failureClass,
        error?.message ?? String(error),
      );
      failures.push(`${resolved.profile.id}:${failureClass}`);
      if (!isFailoverAllowed(failureClass)) {
        throw error;
      }
    }
  }
}

export async function webSearch(request = {}) {
  const preferredProfileId = request.preferred_profile_id
    ? String(request.preferred_profile_id).trim()
    : undefined;
  const failures = [];
  while (true) {
    const resolved = await resolveChatProfileSnapshot(preferredProfileId);
    try {
      const result = await runWebSearchWithProfile(
        resolved.profile,
        resolved.access_token,
        request,
      );
      await markProfileSuccess(resolved.profile.id);
      return result;
    } catch (error) {
      const failureClass = classifyFailure(error);
      await markProfileFailure(
        resolved.profile.id,
        failureClass,
        error?.message ?? String(error),
      );
      failures.push(`${resolved.profile.id}:${failureClass}`);
      if (!isFailoverAllowed(failureClass)) {
        throw error;
      }
      continue;
    }
  }
}

export function setChatDepsForTests(deps = {}) {
  completeSimpleImpl = deps.completeSimple ?? deps.complete ?? completeSimpleImport;
  getModelImpl = deps.getModel ?? getModelImport;
}
