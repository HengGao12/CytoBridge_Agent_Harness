import fs from "node:fs";
import {
  loginOpenAICodex as loginOpenAICodexImport,
  refreshOpenAICodexToken as refreshOpenAICodexTokenImport,
} from "@mariozechner/pi-ai";
import {
  getUsageStats,
  listProfiles,
  loadStore,
  saveStore,
  updateStore,
  upsertProfile,
} from "./store.mjs";

const PROVIDER_ID = "openai-codex";
const AUTH_RETRY_WINDOW_MS = 5 * 60 * 1000;
const RATE_LIMIT_COOLDOWN_MS = 15 * 60 * 1000;
const QUOTA_COOLDOWN_MS = 12 * 60 * 60 * 1000;
const TRANSIENT_COOLDOWN_MS = 2 * 60 * 1000;
const CODEX_USAGE_URL = "https://chatgpt.com/backend-api/wham/usage";
let loginOpenAICodexImpl = loginOpenAICodexImport;
let refreshOpenAICodexTokenImpl = refreshOpenAICodexTokenImport;
let fetchImpl = globalThis.fetch?.bind(globalThis) ?? null;

async function ensureFetchImpl() {
  if (typeof fetchImpl === "function") {
    return fetchImpl;
  }
  try {
    const mod = await import("node-fetch");
    const resolved = mod?.default ?? mod?.fetch ?? null;
    if (typeof resolved === "function") {
      fetchImpl = resolved;
      return fetchImpl;
    }
  } catch (error) {
    throw new Error(`Failed to initialize fetch fallback: ${error?.message ?? String(error)}`);
  }
  throw new Error("No fetch implementation is available in the current Node runtime.");
}

function profileIdFromCredentials(credentials, preferredProfileId) {
  if (preferredProfileId) {
    return preferredProfileId.trim();
  }
  const email = typeof credentials.email === "string" && credentials.email.trim()
    ? credentials.email.trim()
    : "";
  if (email) {
    return `${PROVIDER_ID}:${email}`;
  }
  if (typeof credentials.accountId === "string" && credentials.accountId.trim()) {
    return `${PROVIDER_ID}:${credentials.accountId.trim()}`;
  }
  return `${PROVIDER_ID}:default`;
}

function summarizeProfile(store, profile) {
  const stats = getUsageStats(store, profile.id);
  return {
    profile_id: profile.id,
    provider: profile.provider,
    enabled: profile.enabled !== false,
    email: profile.email ?? null,
    account_id: profile.account_id ?? null,
    expires: profile.expires,
    cooldown_until_ms: stats.cooldown_until_ms || 0,
    reauth_required: stats.reauth_required === true,
    last_error_code: stats.last_error_code ?? null,
    last_error_message: stats.last_error_message ?? null,
    last_success_ms: stats.last_success_ms || 0,
    last_used_ms: stats.last_used_ms || 0,
  };
}

function decodeJwtPayload(token) {
  const parts = String(token ?? "").split(".");
  if (parts.length !== 3) {
    return null;
  }
  try {
    const payload = parts[1].replace(/-/g, "+").replace(/_/g, "/");
    const padded = payload.padEnd(Math.ceil(payload.length / 4) * 4, "=");
    return JSON.parse(Buffer.from(padded, "base64").toString("utf8"));
  } catch {
    return null;
  }
}

function extractAccountIdFromToken(token) {
  const payload = decodeJwtPayload(token);
  return payload?.["https://api.openai.com/auth"]?.chatgpt_account_id ?? null;
}

function formatUsagePlan(data) {
  let plan = typeof data?.plan_type === "string" ? data.plan_type : null;
  const balanceRaw = data?.credits?.balance;
  if (balanceRaw === undefined || balanceRaw === null) {
    return plan;
  }
  const balance =
    typeof balanceRaw === "number" ? balanceRaw : Number.parseFloat(String(balanceRaw));
  if (!Number.isFinite(balance)) {
    return plan;
  }
  const balanceLabel = `$${balance.toFixed(2)}`;
  return plan ? `${plan} (${balanceLabel})` : balanceLabel;
}

function normalizeUsageWindows(data) {
  const windows = [];
  const primary = data?.rate_limit?.primary_window;
  if (primary && typeof primary === "object") {
    const hours = Math.round((Number(primary.limit_window_seconds ?? 10800) || 10800) / 3600);
    windows.push({
      label: `${hours}h`,
      used_percent: Number(primary.used_percent ?? 0) || 0,
      reset_at_ms: primary.reset_at ? Number(primary.reset_at) * 1000 : null,
    });
  }
  const secondary = data?.rate_limit?.secondary_window;
  if (secondary && typeof secondary === "object") {
    const hours = Math.round((Number(secondary.limit_window_seconds ?? 86400) || 86400) / 3600);
    windows.push({
      label: hours >= 24 ? "Day" : `${hours}h`,
      used_percent: Number(secondary.used_percent ?? 0) || 0,
      reset_at_ms: secondary.reset_at ? Number(secondary.reset_at) * 1000 : null,
    });
  }
  return windows;
}

function orderedProfilesForProvider(store, provider) {
  const profiles = listProfiles(store, provider);
  const idToProfile = new Map(profiles.map((profile) => [profile.id, profile]));
  const orderedIds = [
    ...(store.order[provider] ?? []),
    ...profiles.map((profile) => profile.id),
  ].filter((id) => idToProfile.has(id));
  return [...new Set(orderedIds)].map((id) => idToProfile.get(id)).filter(Boolean);
}

async function fetchCodexUsageSnapshot({ accessToken, accountId, timeoutMs = 5000 }) {
  const fetchFn = await ensureFetchImpl();
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const headers = {
      Authorization: `Bearer ${accessToken}`,
      Accept: "application/json",
      "User-Agent": "CytoBridge Codex Usage",
    };
    if (accountId) {
      headers["ChatGPT-Account-Id"] = accountId;
    }
    const response = await fetchFn(CODEX_USAGE_URL, {
      method: "GET",
      headers,
      signal: controller.signal,
    });
    const raw = await response.text();
    let data = null;
    try {
      data = raw ? JSON.parse(raw) : null;
    } catch {
      data = null;
    }
    if (!response.ok) {
      const detail =
        data?.error?.message ||
        data?.detail ||
        raw ||
        `HTTP ${response.status}`;
      throw new Error(`Codex usage query failed (${response.status}): ${detail}`);
    }
    return {
      plan: formatUsagePlan(data),
      windows: normalizeUsageWindows(data),
    };
  } finally {
    clearTimeout(timeout);
  }
}

export function classifyFailure(error) {
  const message = String(error?.message ?? error ?? "").toLowerCase();
  if (
    /401|unauthorized|invalid[_ -]?grant|invalid[_ -]?token|token refresh failed|state mismatch/.test(
      message,
    )
  ) {
    return "auth";
  }
  if (/429|rate.?limit|too many requests/.test(message)) {
    return "rate_limit";
  }
  if (/quota|billing|credit|usage limit|insufficient/.test(message)) {
    return "quota";
  }
  if (/timeout|timed out|econn|enotfound|socket|network|fetch failed|connection/.test(message)) {
    return "network";
  }
  if (/500|502|503|504|service unavailable|overloaded|upstream/.test(message)) {
    return "server";
  }
  return "fatal_request";
}

export function isFailoverAllowed(failureClass) {
  return ["auth", "rate_limit", "quota", "network", "server"].includes(failureClass);
}

function applyFailureStats(store, profileId, failureClass, message) {
  const now = Date.now();
  const stats = getUsageStats(store, profileId);
  stats.last_failure_ms = now;
  stats.last_used_ms = now;
  stats.last_error_code = failureClass;
  stats.last_error_message = message ? String(message) : undefined;
  stats.consecutive_failures = (stats.consecutive_failures || 0) + 1;

  if (failureClass === "auth") {
    stats.reauth_required = true;
    stats.cooldown_until_ms = 0;
    return;
  }
  if (failureClass === "rate_limit") {
    stats.cooldown_until_ms = now + RATE_LIMIT_COOLDOWN_MS;
    return;
  }
  if (failureClass === "quota") {
    stats.cooldown_until_ms = now + QUOTA_COOLDOWN_MS;
    return;
  }
  if (failureClass === "network" || failureClass === "server") {
    stats.cooldown_until_ms = now + TRANSIENT_COOLDOWN_MS;
    return;
  }
}

function applySuccessStats(store, profileId, provider = PROVIDER_ID) {
  const now = Date.now();
  const stats = getUsageStats(store, profileId);
  stats.last_used_ms = now;
  stats.last_success_ms = now;
  stats.last_failure_ms = 0;
  stats.cooldown_until_ms = 0;
  stats.consecutive_failures = 0;
  stats.last_error_code = undefined;
  stats.last_error_message = undefined;
  stats.reauth_required = false;
  store.last_good[provider] = profileId;
}

function preferredFirst(ids, preferredProfileId) {
  const deduped = [...new Set(ids)];
  if (!preferredProfileId || !deduped.includes(preferredProfileId)) {
    return deduped;
  }
  return [preferredProfileId, ...deduped.filter((id) => id !== preferredProfileId)];
}

export function resolveCandidateProfiles(store, provider, preferredProfileId) {
  const profiles = listProfiles(store, provider);
  const idToProfile = new Map(profiles.map((profile) => [profile.id, profile]));
  const orderedIds = [
    ...(store.last_good[provider] ? [store.last_good[provider]] : []),
    ...(store.order[provider] ?? []),
    ...profiles.map((profile) => profile.id),
  ].filter((id) => idToProfile.has(id));
  const ids = preferredFirst(orderedIds, preferredProfileId);
  const now = Date.now();
  return ids
    .map((id) => idToProfile.get(id))
    .filter(Boolean)
    .filter((profile) => {
      if (profile.enabled === false) return false;
      const stats = getUsageStats(store, profile.id);
      if (stats.reauth_required) return false;
      if ((stats.cooldown_until_ms || 0) > now) return false;
      return true;
    });
}

export async function authCodexLogin(params = {}) {
  const credentials = await loginOpenAICodexImpl({
    onAuth: ({ url, instructions }) => {
      process.stderr.write(`Open this URL to authenticate:\n${url}\n`);
      if (instructions) {
        process.stderr.write(`${instructions}\n`);
      }
    },
    onPrompt: async ({ message }) => {
      const { createInterface } = await import("node:readline/promises");
      const input = fs.createReadStream("/dev/tty");
      const output = fs.createWriteStream("/dev/tty");
      const rl = createInterface({ input, output });
      try {
        const answer = await rl.question(`${message} `);
        return answer;
      } finally {
        rl.close();
        input.close?.();
        output.close?.();
      }
    },
    onProgress: (message) => {
      process.stderr.write(`${message}\n`);
    },
  });

  let summary;
  await updateStore(async (store) => {
    const profileId = profileIdFromCredentials(credentials, params.profile_id);
    const profile = upsertProfile(store, {
      id: profileId,
      provider: PROVIDER_ID,
      type: "oauth",
      enabled: true,
      access: credentials.access,
      refresh: credentials.refresh,
      expires: credentials.expires,
      email: credentials.email,
      account_id: credentials.accountId,
      metadata: { source: "native_oauth" },
    });
    if (!Array.isArray(store.order[PROVIDER_ID])) {
      store.order[PROVIDER_ID] = [];
    }
    if (!store.order[PROVIDER_ID].includes(profile.id)) {
      store.order[PROVIDER_ID].push(profile.id);
    }
    applySuccessStats(store, profile.id, PROVIDER_ID);
    summary = summarizeProfile(store, profile);
  });
  return summary;
}

async function ensureProfileToken(store, profile) {
  const now = Date.now();
  const stats = getUsageStats(store, profile.id);
  stats.last_used_ms = now;
  const expires = Number(profile.expires || 0);
  if (expires > now + AUTH_RETRY_WINDOW_MS && profile.access) {
    return { access_token: profile.access, expires: profile.expires };
  }
  const refreshed = await refreshOpenAICodexTokenImpl(profile.refresh);
  profile.access = refreshed.access;
  profile.refresh = refreshed.refresh;
  profile.expires = refreshed.expires;
  profile.account_id = refreshed.accountId ?? profile.account_id;
  stats.reauth_required = false;
  stats.cooldown_until_ms = 0;
  return { access_token: profile.access, expires: profile.expires };
}

export async function resolveProfileAccess(store, preferredProfileId) {
  const candidates = resolveCandidateProfiles(store, PROVIDER_ID, preferredProfileId);
  const failures = [];
  for (const profile of candidates) {
    try {
      const token = await ensureProfileToken(store, profile);
      return {
        profile,
        access_token: token.access_token,
        expires: token.expires,
      };
    } catch (error) {
      const failureClass = classifyFailure(error);
      applyFailureStats(store, profile.id, failureClass, error?.message ?? String(error));
      failures.push(`${profile.id}:${failureClass}`);
    }
  }
  throw new Error(
    failures.length > 0
      ? `No usable ${PROVIDER_ID} profile. Failures: ${failures.join(", ")}`
      : `No usable ${PROVIDER_ID} profile.`,
  );
}

export async function resolveProfileAccessSnapshot(preferredProfileId) {
  const store = await loadStore();
  try {
    const picked = await resolveProfileAccess(store, preferredProfileId);
    await saveStore(store);
    return {
      profile_id: picked.profile.id,
      account_id: picked.profile.account_id ?? null,
      access_token: picked.access_token,
      expires: picked.expires,
    };
  } catch (error) {
    await saveStore(store).catch(() => {});
    throw error;
  }
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

export async function resolveToken(params = {}) {
  const preferredProfileId = params.preferred_profile_id
    ? String(params.preferred_profile_id).trim()
    : undefined;
  const resolvedProfile = await resolveProfileAccessSnapshot(preferredProfileId);
  return {
    profile_id: resolvedProfile.profile_id,
    provider: PROVIDER_ID,
    access_token: resolvedProfile.access_token,
    expires: resolvedProfile.expires,
  };
}

export async function usage(params = {}) {
  const provider = params.provider ? String(params.provider).trim() : PROVIDER_ID;
  if (provider !== PROVIDER_ID) {
    throw new Error(`Unsupported provider for usage query: ${provider}`);
  }
  const preferredProfileId = params.profile_id ? String(params.profile_id).trim() : undefined;
  const allProfiles = params.all_profiles === true;
  const timeoutMs = Number(params.timeout_ms ?? 5000) || 5000;
  const store = await loadStore();
  const profileIds = orderedProfilesForProvider(store, PROVIDER_ID).filter((profile) => {
    if (profile.enabled === false) {
      return false;
    }
    if (preferredProfileId) {
      return profile.id === preferredProfileId;
    }
    return true;
  }).map((profile) => profile.id);

  const profiles = [];
  for (const profileId of profileIds) {
    try {
      const resolved = await resolveProfileAccessSnapshot(profileId);
      const accountId = resolved.account_id || extractAccountIdFromToken(resolved.access_token);
      const snapshot = await fetchCodexUsageSnapshot({
        accessToken: resolved.access_token,
        accountId,
        timeoutMs,
      });
      await markProfileSuccess(resolved.profile_id);
      profiles.push({
        profile_id: resolved.profile_id,
        account_id: accountId,
        plan: snapshot.plan,
        windows: snapshot.windows,
        error: null,
      });
    } catch (error) {
      const failureClass = classifyFailure(error);
      try {
        await markProfileFailure(profileId, failureClass, error?.message ?? String(error));
      } catch {
        // ignore secondary store-update failure in usage reporting
      }
      profiles.push({
        profile_id: profileId,
        account_id: null,
        plan: null,
        windows: [],
        error: error?.message ?? String(error),
      });
    }
    if (!allProfiles) {
      break;
    }
  }
  return {
    provider,
    fetched_at_ms: Date.now(),
    profiles,
  };
}

export async function reportSuccess(params = {}) {
  await updateStore(async (store) => {
    const profileId = String(params.profile_id ?? "").trim();
    if (!profileId || !store.profiles[profileId]) {
      throw new Error(`Unknown profile: ${profileId}`);
    }
    applySuccessStats(store, profileId, PROVIDER_ID);
  });
  return { ok: true };
}

export async function reportFailure(params = {}) {
  await updateStore(async (store) => {
    const profileId = String(params.profile_id ?? "").trim();
    if (!profileId || !store.profiles[profileId]) {
      throw new Error(`Unknown profile: ${profileId}`);
    }
    const failureClass = String(params.failure_class ?? "").trim() || "fatal_request";
    applyFailureStats(store, profileId, failureClass, params.message);
  });
  return { ok: true };
}

export async function listProfileSummaries(params = {}) {
  const provider = params.provider ? String(params.provider).trim() : PROVIDER_ID;
  return await updateStore(async (store) =>
    orderedProfilesForProvider(store, provider).map((profile) => summarizeProfile(store, profile)),
  );
}

export async function enableProfile(params = {}) {
  const profileId = String(params.profile_id ?? "").trim();
  let summary = null;
  await updateStore(async (store) => {
    const profile = store.profiles[profileId];
    if (!profile) {
      throw new Error(`Unknown profile: ${profileId}`);
    }
    profile.enabled = true;
    summary = summarizeProfile(store, profile);
  });
  return summary;
}

export async function disableProfile(params = {}) {
  const profileId = String(params.profile_id ?? "").trim();
  let summary = null;
  await updateStore(async (store) => {
    const profile = store.profiles[profileId];
    if (!profile) {
      throw new Error(`Unknown profile: ${profileId}`);
    }
    profile.enabled = false;
    summary = summarizeProfile(store, profile);
  });
  return summary;
}

export async function setProfileOrder(params = {}) {
  const provider = String(params.provider ?? PROVIDER_ID).trim();
  const ids = Array.isArray(params.ids)
    ? params.ids.map((id) => String(id ?? "").trim()).filter(Boolean)
    : [];
  await updateStore(async (store) => {
    const validIds = ids.filter((id) => store.profiles[id]?.provider === provider);
    const remaining = listProfiles(store, provider)
      .map((profile) => profile.id)
      .filter((id) => !validIds.includes(id));
    store.order[provider] = [...validIds, ...remaining];
  });
  return { provider, ids };
}

export async function status(params = {}) {
  const provider = params.provider ? String(params.provider).trim() : PROVIDER_ID;
  return await updateStore(async (store) => {
    const profiles = orderedProfilesForProvider(store, provider).map((profile) =>
      summarizeProfile(store, profile),
    );
    return {
      provider,
      available_count: profiles.filter(
        (profile) =>
          profile.enabled &&
          !profile.reauth_required &&
          (profile.cooldown_until_ms || 0) <= Date.now(),
      ).length,
      last_good_profile_id: store.last_good[provider] ?? null,
      profiles,
    };
  });
}

export {
  AUTH_RETRY_WINDOW_MS,
  PROVIDER_ID,
  QUOTA_COOLDOWN_MS,
  RATE_LIMIT_COOLDOWN_MS,
  TRANSIENT_COOLDOWN_MS,
  applyFailureStats,
  applySuccessStats,
  ensureProfileToken,
};

export function setRuntimeDepsForTests(deps = {}) {
  loginOpenAICodexImpl = deps.loginOpenAICodex ?? loginOpenAICodexImport;
  refreshOpenAICodexTokenImpl = deps.refreshOpenAICodexToken ?? refreshOpenAICodexTokenImport;
  fetchImpl = deps.fetch ?? globalThis.fetch?.bind(globalThis) ?? null;
}
