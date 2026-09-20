import {
  authCodexLogin,
  disableProfile,
  enableProfile,
  listProfileSummaries,
  reportFailure,
  reportSuccess,
  resolveToken,
  setProfileOrder,
  status,
  usage,
} from "./lib/runtime.mjs";
import { chatComplete, webSearch as chatWebSearch } from "./lib/chat.mjs";

const handlers = {
  "auth.codex.login": authCodexLogin,
  "auth.profiles.list": listProfileSummaries,
  "auth.profiles.enable": enableProfile,
  "auth.profiles.disable": disableProfile,
  "auth.profiles.order.set": setProfileOrder,
  "auth.runtime.resolve_token": resolveToken,
  "auth.runtime.report_success": reportSuccess,
  "auth.runtime.report_failure": reportFailure,
  "auth.runtime.status": status,
  "auth.runtime.usage": usage,
  "chat.codex.complete": chatComplete,
  "chat.codex.web_search": chatWebSearch,
};

async function dispatch(request) {
  const handler = handlers[request.method];
  if (!handler) {
    throw Object.assign(new Error(`Unknown method: ${request.method}`), { code: -32601 });
  }
  return await handler(request.params ?? {});
}

function respond(payload) {
  process.stdout.write(`${JSON.stringify(payload)}\n`);
}

async function main() {
  process.stdin.setEncoding("utf8");
  let buffer = "";
  process.stdin.on("data", async (chunk) => {
    buffer += chunk;
    let idx = buffer.indexOf("\n");
    while (idx !== -1) {
      const line = buffer.slice(0, idx).trim();
      buffer = buffer.slice(idx + 1);
      idx = buffer.indexOf("\n");
      if (!line) {
        continue;
      }
      let request;
      try {
        request = JSON.parse(line);
      } catch (error) {
        respond({
          id: null,
          error: { code: -32700, message: `Invalid JSON: ${error}` },
        });
        continue;
      }
      try {
        const result = await dispatch(request);
        respond({ id: request.id ?? null, result });
      } catch (error) {
        respond({
          id: request.id ?? null,
          error: {
            code: error?.code ?? -32000,
            message: error?.message ?? String(error),
          },
        });
      }
    }
  });
  process.stdin.resume();
}

await main();
