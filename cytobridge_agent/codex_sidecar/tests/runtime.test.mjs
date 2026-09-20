import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import {
  authCodexLogin,
  reportFailure,
  resolveToken,
  setRuntimeDepsForTests,
  setProfileOrder,
  status,
  usage,
} from "../lib/runtime.mjs";
import { chatComplete, setChatDepsForTests } from "../lib/chat.mjs";
import { loadStore, resolveStorePath, updateStore, upsertProfile } from "../lib/store.mjs";

function makeStorePath(name) {
  return path.join(
    fs.mkdtempSync(path.join(os.tmpdir(), "cytobridge-codex-test-")),
    `${name}.json`,
  );
}

test.afterEach(() => {
  setRuntimeDepsForTests({});
  setChatDepsForTests({});
  delete process.env.CYTOBRIDGE_AUTH_PROFILES_PATH;
});

test("auth.codex.login stores a profile and marks it last_good", async () => {
  process.env.CYTOBRIDGE_AUTH_PROFILES_PATH = makeStorePath("login");
  setRuntimeDepsForTests({
    loginOpenAICodex: async () => ({
      access: "access-token",
      refresh: "refresh-token",
      expires: Date.now() + 60_000,
      email: "alice@example.com",
      accountId: "acct-alice",
    }),
  });

  const summary = await authCodexLogin();
  const store = await loadStore();

  assert.equal(summary.profile_id, "openai-codex:alice@example.com");
  assert.equal(store.last_good["openai-codex"], "openai-codex:alice@example.com");
  assert.equal(store.profiles["openai-codex:alice@example.com"].refresh, "refresh-token");
});

test("resolve_token refreshes expired profile tokens", async () => {
  process.env.CYTOBRIDGE_AUTH_PROFILES_PATH = makeStorePath("refresh");
  await updateStore(async (store) => {
    upsertProfile(store, {
      id: "openai-codex:alice@example.com",
      provider: "openai-codex",
      type: "oauth",
      enabled: true,
      access: "expired-access",
      refresh: "refresh-token",
      expires: Date.now() - 1_000,
      email: "alice@example.com",
    });
    store.order["openai-codex"] = ["openai-codex:alice@example.com"];
  });
  setRuntimeDepsForTests({
    refreshOpenAICodexToken: async (refreshToken) => {
      assert.equal(refreshToken, "refresh-token");
      return {
        access: "fresh-access",
        refresh: "fresh-refresh",
        expires: Date.now() + 60_000,
        accountId: "acct-alice",
      };
    },
  });

  const resolved = await resolveToken();
  const store = await loadStore();

  assert.equal(resolved.access_token, "fresh-access");
  assert.equal(store.profiles["openai-codex:alice@example.com"].refresh, "fresh-refresh");
});

test("status excludes cooldown profiles and honors explicit order", async () => {
  process.env.CYTOBRIDGE_AUTH_PROFILES_PATH = makeStorePath("status");
  await updateStore(async (store) => {
    upsertProfile(store, {
      id: "openai-codex:alice@example.com",
      provider: "openai-codex",
      type: "oauth",
      enabled: true,
      access: "access-a",
      refresh: "refresh-a",
      expires: Date.now() + 10 * 60_000,
      email: "alice@example.com",
    });
    upsertProfile(store, {
      id: "openai-codex:bob@example.com",
      provider: "openai-codex",
      type: "oauth",
      enabled: true,
      access: "access-b",
      refresh: "refresh-b",
      expires: Date.now() + 10 * 60_000,
      email: "bob@example.com",
    });
  });

  await setProfileOrder({
    provider: "openai-codex",
    ids: ["openai-codex:bob@example.com", "openai-codex:alice@example.com"],
  });
  await reportFailure({
    profile_id: "openai-codex:bob@example.com",
    failure_class: "rate_limit",
    message: "too many requests",
  });

  const current = await status();

  assert.equal(current.available_count, 1);
  assert.deepEqual(
    current.profiles.map((profile) => profile.profile_id),
    ["openai-codex:bob@example.com", "openai-codex:alice@example.com"],
  );
  assert.equal(current.profiles[0].last_error_code, "rate_limit");
});

test("chat completion fails over after rate limit and succeeds on next profile", async () => {
  process.env.CYTOBRIDGE_AUTH_PROFILES_PATH = makeStorePath("chat-failover");
  await updateStore(async (store) => {
    upsertProfile(store, {
      id: "openai-codex:alice@example.com",
      provider: "openai-codex",
      type: "oauth",
      enabled: true,
      access: "access-a",
      refresh: "refresh-a",
      expires: Date.now() + 10 * 60_000,
      email: "alice@example.com",
    });
    upsertProfile(store, {
      id: "openai-codex:bob@example.com",
      provider: "openai-codex",
      type: "oauth",
      enabled: true,
      access: "access-b",
      refresh: "refresh-b",
      expires: Date.now() + 10 * 60_000,
      email: "bob@example.com",
    });
    store.order["openai-codex"] = [
      "openai-codex:alice@example.com",
      "openai-codex:bob@example.com",
    ];
  });
  setChatDepsForTests({
    getModel: (_provider, modelName) => ({ provider: "openai-codex", id: modelName }),
    complete: async (_model, _context, options) => {
      if (options.apiKey === "access-a") {
        throw new Error("429 rate limit");
      }
      return {
        model: "gpt-5.3-codex",
        content: [{ type: "text", text: "ok" }],
        usage: { input: 10, output: 5, totalTokens: 15 },
        stopReason: "stop",
      };
    },
  });

  const result = await chatComplete({
    model: "gpt-5.3-codex",
    messages: [{ role: "user", content: "hello" }],
  });
  const current = await status();

  assert.equal(result.profile_id, "openai-codex:bob@example.com");
  assert.equal(result.content, "ok");
  assert.equal(current.last_good_profile_id, "openai-codex:bob@example.com");
});

test("chat completion supports whitelisted forward-compat model ids", async () => {
  process.env.CYTOBRIDGE_AUTH_PROFILES_PATH = makeStorePath("chat-gpt54");
  await updateStore(async (store) => {
    upsertProfile(store, {
      id: "openai-codex:alice@example.com",
      provider: "openai-codex",
      type: "oauth",
      enabled: true,
      access: "access-a",
      refresh: "refresh-a",
      expires: Date.now() + 10 * 60_000,
      email: "alice@example.com",
    });
    store.order["openai-codex"] = ["openai-codex:alice@example.com"];
  });
  setChatDepsForTests({
    getModel: () => {
      throw new Error("unknown model");
    },
    complete: async (model) => ({
      model: model.id,
      content: [{ type: "text", text: "pong" }],
      usage: { input: 11, output: 5, totalTokens: 16 },
      stopReason: "stop",
    }),
  });

  const result = await chatComplete({
    model: "gpt-5.4",
    messages: [{ role: "user", content: "hello" }],
  });

  assert.equal(result.model, "gpt-5.4");
  assert.equal(result.content, "pong");
});

test("chat completion supports gpt-5.6-sol with xhigh reasoning", async () => {
  process.env.CYTOBRIDGE_AUTH_PROFILES_PATH = makeStorePath("chat-gpt56-sol");
  await updateStore(async (store) => {
    upsertProfile(store, {
      id: "openai-codex:alice@example.com",
      provider: "openai-codex",
      type: "oauth",
      enabled: true,
      access: "access-a",
      refresh: "refresh-a",
      expires: Date.now() + 10 * 60_000,
      email: "alice@example.com",
    });
    store.order["openai-codex"] = ["openai-codex:alice@example.com"];
  });

  let observedModel;
  let observedReasoning;
  setChatDepsForTests({
    getModel: () => {
      throw new Error("unknown model");
    },
    complete: async (model, _context, options) => {
      observedModel = model;
      observedReasoning = options.reasoning;
      return {
        model: model.id,
        content: [{ type: "text", text: "MODEL_OK" }],
        usage: { input: 11, output: 5, totalTokens: 16 },
        stopReason: "stop",
      };
    },
  });

  const result = await chatComplete({
    model: "gpt-5.6-sol",
    reasoning: "xhigh",
    messages: [{ role: "user", content: "hello" }],
  });

  assert.equal(observedModel.id, "gpt-5.6-sol");
  assert.equal(observedModel.api, "openai-codex-responses");
  assert.equal(observedReasoning, "xhigh");
  assert.equal(result.model, "gpt-5.6-sol");
  assert.equal(result.content, "MODEL_OK");
});

test("chat completion defaults Codex thinking to low and forwards explicit override", async () => {
  process.env.CYTOBRIDGE_AUTH_PROFILES_PATH = makeStorePath("chat-thinking");
  await updateStore(async (store) => {
    upsertProfile(store, {
      id: "openai-codex:alice@example.com",
      provider: "openai-codex",
      type: "oauth",
      enabled: true,
      access: "access-a",
      refresh: "refresh-a",
      expires: Date.now() + 10 * 60_000,
      email: "alice@example.com",
    });
    store.order["openai-codex"] = ["openai-codex:alice@example.com"];
  });

  const observed = [];
  setChatDepsForTests({
    getModel: (_provider, modelName) => ({ provider: "openai-codex", id: modelName, reasoning: true }),
    completeSimple: async (_model, _context, options) => {
      observed.push(options.reasoning ?? null);
      return {
        model: "gpt-5.4",
        content: [{ type: "text", text: "ok" }],
        usage: { input: 8, output: 4, totalTokens: 12 },
        stopReason: "stop",
      };
    },
  });

  await chatComplete({
    model: "gpt-5.4",
    messages: [{ role: "user", content: "hello" }],
  });
  await chatComplete({
    model: "gpt-5.4",
    reasoning: "medium",
    messages: [{ role: "user", content: "hello again" }],
  });
  await chatComplete({
    model: "gpt-5.4",
    reasoning: "off",
    messages: [{ role: "user", content: "hello once more" }],
  });

  assert.deepEqual(observed, ["low", "medium", null]);
});

test("usage fetch returns wham usage snapshot for profiles", async () => {
  process.env.CYTOBRIDGE_AUTH_PROFILES_PATH = makeStorePath("usage");
  await updateStore(async (store) => {
    upsertProfile(store, {
      id: "openai-codex:alice@example.com",
      provider: "openai-codex",
      type: "oauth",
      enabled: true,
      access: "header.payload.sig",
      refresh: "refresh-a",
      expires: Date.now() + 10 * 60_000,
      email: "alice@example.com",
      account_id: "acct-alice",
    });
    store.order["openai-codex"] = ["openai-codex:alice@example.com"];
  });
  setRuntimeDepsForTests({
    fetch: async (url, options) => {
      assert.equal(url, "https://chatgpt.com/backend-api/wham/usage");
      assert.equal(options.headers["ChatGPT-Account-Id"], "acct-alice");
      return new Response(
        JSON.stringify({
          plan_type: "free",
          credits: { balance: "1.25" },
          rate_limit: {
            primary_window: {
              limit_window_seconds: 10800,
              used_percent: 12.5,
              reset_at: 1772909999,
            },
          },
        }),
        { status: 200, headers: { "content-type": "application/json" } },
      );
    },
  });

  const result = await usage({ all_profiles: true });

  assert.equal(result.provider, "openai-codex");
  assert.equal(result.profiles.length, 1);
  assert.equal(result.profiles[0].plan, "free ($1.25)");
  assert.equal(result.profiles[0].windows[0].label, "3h");
  assert.equal(result.profiles[0].error, null);
});

test("stale legacy lock file is recovered automatically", async () => {
  process.env.CYTOBRIDGE_AUTH_PROFILES_PATH = makeStorePath("stale-lock");
  const storePath = resolveStorePath();
  const lockPath = `${storePath}.lock`;
  fs.writeFileSync(lockPath, "");
  const staleDate = new Date(Date.now() - 5 * 60_000);
  fs.utimesSync(lockPath, staleDate, staleDate);

  await updateStore(async (store) => {
    upsertProfile(store, {
      id: "openai-codex:alice@example.com",
      provider: "openai-codex",
      type: "oauth",
      enabled: true,
      access: "access-a",
      refresh: "refresh-a",
      expires: Date.now() + 10 * 60_000,
      email: "alice@example.com",
    });
  });

  const store = await loadStore();
  assert.ok(store.profiles["openai-codex:alice@example.com"]);
  assert.equal(fs.existsSync(lockPath), false);
});
