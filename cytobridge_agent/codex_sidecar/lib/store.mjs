import fs from "node:fs";
import os from "node:os";
import path from "node:path";

const STORE_VERSION = 1;
const LOCK_RETRY_MS = 50;
const LOCK_MAX_ATTEMPTS = 200;
const LOCK_STALE_MS = 60_000;

function defaultStore() {
  return {
    version: STORE_VERSION,
    profiles: {},
    order: {},
    last_good: {},
    usage_stats: {},
  };
}

export function resolveStorePath() {
  const explicit = process.env.CYTOBRIDGE_AUTH_PROFILES_PATH?.trim();
  if (explicit) {
    return explicit;
  }
  return path.join(os.homedir(), ".cellcompass", "auth_profiles.json");
}

function resolveLockPath(storePath) {
  return `${storePath}.lock`;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function buildLockMetadata() {
  return {
    pid: process.pid,
    created_at_ms: Date.now(),
    hostname: os.hostname(),
  };
}

async function writeLockMetadata(handle) {
  const payload = `${JSON.stringify(buildLockMetadata())}\n`;
  await handle.writeFile(payload, "utf8");
}

function isProcessAlive(pid) {
  if (!Number.isInteger(pid) || pid <= 0) {
    return false;
  }
  try {
    process.kill(pid, 0);
    return true;
  } catch (error) {
    if (error?.code === "ESRCH") {
      return false;
    }
    return true;
  }
}

async function readLockMetadata(lockPath) {
  let stats = null;
  try {
    stats = await fs.promises.stat(lockPath);
  } catch (error) {
    if (error?.code === "ENOENT") {
      return null;
    }
    throw error;
  }
  let raw = "";
  try {
    raw = await fs.promises.readFile(lockPath, "utf8");
  } catch (error) {
    if (error?.code === "ENOENT") {
      return null;
    }
    throw error;
  }
  let parsed = {};
  if (raw.trim()) {
    try {
      parsed = JSON.parse(raw);
    } catch {
      parsed = {};
    }
  }
  return {
    pid: Number(parsed.pid ?? 0) || 0,
    created_at_ms: Number(parsed.created_at_ms ?? 0) || 0,
    mtime_ms: Number(stats.mtimeMs ?? 0) || 0,
  };
}

async function maybeRecoverStaleLock(lockPath) {
  const metadata = await readLockMetadata(lockPath);
  if (!metadata) {
    return false;
  }
  const ageMs = Date.now() - (metadata.created_at_ms || metadata.mtime_ms || 0);
  const deadOwner = metadata.pid > 0 && !isProcessAlive(metadata.pid);
  const legacyEmptyLock = metadata.pid <= 0 && ageMs > LOCK_STALE_MS;
  const longHeldLock = ageMs > LOCK_STALE_MS;
  if (!deadOwner && !legacyEmptyLock && !longHeldLock) {
    return false;
  }
  try {
    await fs.promises.unlink(lockPath);
    return true;
  } catch (error) {
    if (error?.code === "ENOENT") {
      return true;
    }
    return false;
  }
}

async function withFileLock(storePath, fn) {
  const lockPath = resolveLockPath(storePath);
  await fs.promises.mkdir(path.dirname(storePath), { recursive: true });
  for (let attempt = 0; attempt < LOCK_MAX_ATTEMPTS; attempt += 1) {
    try {
      const handle = await fs.promises.open(lockPath, "wx");
      try {
        await writeLockMetadata(handle);
        return await fn();
      } finally {
        await handle.close().catch(() => {});
        await fs.promises.unlink(lockPath).catch(() => {});
      }
    } catch (error) {
      if (error?.code !== "EEXIST") {
        throw error;
      }
      if (await maybeRecoverStaleLock(lockPath)) {
        continue;
      }
      await sleep(LOCK_RETRY_MS);
    }
  }
  throw new Error(`Timed out waiting for auth store lock: ${lockPath}`);
}

function normalizeProfile(rawId, rawProfile) {
  if (!rawProfile || typeof rawProfile !== "object") {
    return null;
  }
  const profile = { ...rawProfile };
  profile.id = String(rawProfile.id ?? rawId ?? "").trim();
  profile.provider = String(rawProfile.provider ?? "").trim();
  profile.type = String(rawProfile.type ?? "oauth").trim();
  profile.enabled = rawProfile.enabled !== false;
  profile.access = typeof rawProfile.access === "string" ? rawProfile.access : "";
  profile.refresh = typeof rawProfile.refresh === "string" ? rawProfile.refresh : "";
  profile.expires = Number(rawProfile.expires ?? 0);
  profile.email = typeof rawProfile.email === "string" ? rawProfile.email : undefined;
  profile.account_id =
    typeof rawProfile.account_id === "string" ? rawProfile.account_id : undefined;
  profile.metadata = rawProfile.metadata && typeof rawProfile.metadata === "object"
    ? { ...rawProfile.metadata }
    : {};
  if (!profile.id || !profile.provider) {
    return null;
  }
  return profile;
}

export function coerceStore(raw) {
  const store = defaultStore();
  if (!raw || typeof raw !== "object") {
    return store;
  }
  const record = raw;
  if (record.profiles && typeof record.profiles === "object") {
    for (const [profileId, profile] of Object.entries(record.profiles)) {
      const normalized = normalizeProfile(profileId, profile);
      if (normalized) {
        store.profiles[normalized.id] = normalized;
      }
    }
  }
  if (record.order && typeof record.order === "object") {
    for (const [provider, ids] of Object.entries(record.order)) {
      if (!Array.isArray(ids)) continue;
      store.order[String(provider)] = ids
        .map((id) => String(id ?? "").trim())
        .filter(Boolean);
    }
  }
  if (record.last_good && typeof record.last_good === "object") {
    for (const [provider, id] of Object.entries(record.last_good)) {
      if (typeof id === "string" && id.trim()) {
        store.last_good[String(provider)] = id.trim();
      }
    }
  }
  if (record.usage_stats && typeof record.usage_stats === "object") {
    for (const [profileId, stats] of Object.entries(record.usage_stats)) {
      if (!stats || typeof stats !== "object") continue;
      store.usage_stats[String(profileId)] = {
        last_used_ms: Number(stats.last_used_ms ?? 0) || 0,
        last_success_ms: Number(stats.last_success_ms ?? 0) || 0,
        last_failure_ms: Number(stats.last_failure_ms ?? 0) || 0,
        cooldown_until_ms: Number(stats.cooldown_until_ms ?? 0) || 0,
        consecutive_failures: Number(stats.consecutive_failures ?? 0) || 0,
        last_error_code:
          typeof stats.last_error_code === "string" ? stats.last_error_code : undefined,
        last_error_message:
          typeof stats.last_error_message === "string" ? stats.last_error_message : undefined,
        reauth_required: stats.reauth_required === true,
      };
    }
  }
  return store;
}

export async function loadStore() {
  const storePath = resolveStorePath();
  try {
    const raw = await fs.promises.readFile(storePath, "utf8");
    return coerceStore(JSON.parse(raw));
  } catch (error) {
    if (error?.code === "ENOENT") {
      return defaultStore();
    }
    throw error;
  }
}

export async function saveStore(store) {
  const storePath = resolveStorePath();
  await withFileLock(storePath, async () => {
    await fs.promises.mkdir(path.dirname(storePath), { recursive: true });
    const tmpPath = `${storePath}.${process.pid}.tmp`;
    await fs.promises.writeFile(`${tmpPath}`, `${JSON.stringify(store, null, 2)}\n`, {
      mode: 0o600,
    });
    await fs.promises.rename(tmpPath, storePath);
    await fs.promises.chmod(storePath, 0o600).catch(() => {});
  });
}

export async function updateStore(updater) {
  const storePath = resolveStorePath();
  return await withFileLock(storePath, async () => {
    let store = defaultStore();
    try {
      const raw = await fs.promises.readFile(storePath, "utf8");
      store = coerceStore(JSON.parse(raw));
    } catch (error) {
      if (error?.code !== "ENOENT") {
        throw error;
      }
    }
    const result = await updater(store);
    const tmpPath = `${storePath}.${process.pid}.tmp`;
    await fs.promises.mkdir(path.dirname(storePath), { recursive: true });
    await fs.promises.writeFile(tmpPath, `${JSON.stringify(store, null, 2)}\n`, { mode: 0o600 });
    await fs.promises.rename(tmpPath, storePath);
    await fs.promises.chmod(storePath, 0o600).catch(() => {});
    return result;
  });
}

export function getUsageStats(store, profileId) {
  if (!store.usage_stats[profileId]) {
    store.usage_stats[profileId] = {
      last_used_ms: 0,
      last_success_ms: 0,
      last_failure_ms: 0,
      cooldown_until_ms: 0,
      consecutive_failures: 0,
      last_error_code: undefined,
      last_error_message: undefined,
      reauth_required: false,
    };
  }
  return store.usage_stats[profileId];
}

export function upsertProfile(store, profile) {
  const normalized = normalizeProfile(profile.id, profile);
  if (!normalized) {
    throw new Error("Invalid profile payload");
  }
  store.profiles[normalized.id] = normalized;
  return normalized;
}

export function listProfiles(store, provider) {
  return Object.values(store.profiles)
    .filter((profile) => !provider || profile.provider === provider)
    .sort((a, b) => a.id.localeCompare(b.id));
}
