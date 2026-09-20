/**
 * demo-server.mjs — mock CellCompass backend for UI demos.
 *
 * Serves the REST endpoints + /ws WebSocket that the frontend expects, and
 * replays a realistic "LLM calls tools" turn (thoughts, tool calls, terminal,
 * Python sandbox streaming, training progress, plan updates, proposal review,
 * report, final answer) with human-paced delays.
 *
 * Usage:  node scripts/demo-server.mjs     (listens on :8000, vite proxies to it)
 */
import http from 'node:http';
import { WebSocketServer } from 'ws';
import crypto from 'node:crypto';

const PORT = 8000;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const now = () => new Date().toISOString();
const uid = () => crypto.randomUUID();

/* ------------------------------------------------------------------ */
/* WebSocket broadcast                                                  */
/* ------------------------------------------------------------------ */
const sockets = new Set();
function broadcast(type, data, turnId) {
  const payload = JSON.stringify({
    type,
    data,
    event_id: uid(),
    turn_id: turnId,
    timestamp: now(),
  });
  for (const ws of sockets) {
    if (ws.readyState === 1) ws.send(payload);
  }
}

/* ------------------------------------------------------------------ */
/* Scripted demo turn                                                   */
/* ------------------------------------------------------------------ */
let running = false;

const DEMO_PLOT_SVG = `<svg xmlns="http://www.w3.org/2000/svg" width="560" height="320">
  <rect width="560" height="320" fill="white"/>
  <g stroke="#e4e4e7"><line x1="60" y1="40" x2="60" y2="270"/><line x1="60" y1="270" x2="520" y2="270"/></g>
  <text x="290" y="28" font-family="Helvetica" font-size="14" text-anchor="middle" fill="#18181b">Validation loss vs. epoch</text>
  <polyline fill="none" stroke="#2563eb" stroke-width="2.5"
    points="60,80 110,120 160,150 210,175 260,196 310,212 360,226 410,237 460,245 520,250"/>
  <g fill="#71717a" font-family="Helvetica" font-size="10">
    <text x="60" y="285">0</text><text x="290" y="285">25</text><text x="510" y="285">50</text>
    <text x="40" y="274">0.1</text><text x="40" y="84">0.9</text>
  </g>
</svg>`;
const DEMO_PLOT_URI = `data:image/svg+xml;base64,${Buffer.from(DEMO_PLOT_SVG).toString('base64')}`;

async function runDemoTurn(userMessage) {
  if (running) return;
  running = true;
  const turnId = `turn-${Date.now()}`;

  try {
    await sleep(500);
    broadcast('agent_thought', {
      agent: 'Planner',
      content:
        `The user wants: **${userMessage.slice(0, 120)}**.\n\n` +
        'Plan: inspect the workspace, profile the peripheral blood mononuclear cells, then run a sandboxed cleaning pass and validate the result before reporting.',
    }, turnId);

    await sleep(800);
    broadcast('turn_plan_updated', {
      source: 'planner', scope: 'planner',
      explanation: 'Initial plan for the data-cleaning request.',
      updated_at: now(),
      plan: [
        { status: 'in_progress', step: 'Inspect workspace and profile peripheral blood mononuclear cells' },
        { status: 'pending', step: 'Draft cleaning pipeline in the Python sandbox' },
        { status: 'pending', step: 'Run validation checks against scientific quality gates' },
        { status: 'pending', step: 'Compile summary report' },
      ],
    }, turnId);

    await sleep(700);
    broadcast('stage', { stage: 'preprocessing', description: 'Profiling input workspace' }, turnId);

    await sleep(600);
    const toolCallId = uid();
    broadcast('tool_start', {
      tool_call_id: toolCallId,
      tool: 'list_workspace_files',
      args: { path: 'input/', recursive: true, reason: 'Discover candidate tables for cleaning' },
      reason: 'Discover candidate tables for cleaning',
    }, turnId);
    await sleep(1100);
    broadcast('tool_output', {
      tool_call_id: toolCallId,
      content: 'input/\n  orders_2024.csv      (1.2 GB, 8.4M rows)\n  customers.parquet    (310 MB)\n  returns_log.xlsx     (48 MB)\n  README.md',
    }, turnId);

    await sleep(700);
    broadcast('file_activity', {
      action: 'read', scope: 'planner', path: 'input/README.md',
      preview: 'Orders extracted nightly from the ERP. Known issues: duplicated order_ids on retries, mixed date formats before 2024-03.',
    }, turnId);

    await sleep(800);
    broadcast('terminal_command', {
      id: uid(), command: 'wc -l input/orders_2024.csv', argv: ['wc', '-l', 'input/orders_2024.csv'],
      cwd: '/workspace', success: true, exit_code: 0, duration_sec: 1.8,
      stdout: '8412339 input/orders_2024.csv',
    }, turnId);

    await sleep(900);
    broadcast('skill_loaded', {
      domain: 'planner', name: 'tabular-quality-gates',
      path: 'skills/tabular-quality-gates/SKILL.md',
      loaded: ['tabular-quality-gates'],
    }, turnId);

    /* ---- streaming python sandbox run ---- */
    await sleep(600);
    const pyId = uid();
    broadcast('item/pythonExecution/started', {
      id: pyId, toolName: 'execute_python',
      reason: 'Profile null rates and duplicate keys before cleaning',
      timeout: 120, timeoutMode: 'shared', status: 'inProgress',
      code: [
        'import pandas as pd',
        '',
        "df = pd.read_csv('input/orders_2024.csv', parse_dates=['created_at'])",
        "print(f'rows={len(df):,}')",
        "print(df.isna().mean().round(4).sort_values(ascending=False).head())",
        "dups = df.duplicated('order_id').sum()",
        "print(f'duplicate order_ids: {dups:,}')",
      ].join('\n'),
    }, turnId);
    await sleep(1400);
    broadcast('item/pythonExecution/outputDelta', { id: pyId, stream: 'stdout', delta: 'rows=8,412,338\n' }, turnId);
    await sleep(900);
    broadcast('item/pythonExecution/outputDelta', {
      id: pyId, stream: 'stdout',
      delta: 'discount_code    0.4112\nship_region      0.0671\ncreated_at       0.0008\n',
    }, turnId);
    await sleep(800);
    broadcast('item/pythonExecution/outputDelta', { id: pyId, stream: 'stdout', delta: 'duplicate order_ids: 14,202\n' }, turnId);
    await sleep(600);
    broadcast('item/pythonExecution/completed', {
      id: pyId, status: 'completed', durationMs: 4180, adataChanged: false,
    }, turnId);

    await sleep(800);
    broadcast('agent_thought', {
      agent: 'Planner',
      content: '14k duplicated `order_id`s match the retry pattern in the README — safe to keep last-write-wins. `discount_code` nulls are expected (no discount). Proceeding to the cleaning pass.',
    }, turnId);

    /* ---- proposal needing user review ---- */
    await sleep(900);
    broadcast('algorithm_proposal_review_requested', {
      algorithm_id: 'alg-clean-orders-v1',
      proposal_id: 'prop-001',
      title: 'Orders cleaning + dedup pipeline',
      status: 'pending_user_review',
      review_mode: 'always_user_review',
      requires_user_review: true,
      proposal_markdown: [
        '## Cleaning strategy',
        '',
        '1. **Deduplicate** on `order_id`, keep the latest `updated_at` (retry-safe).',
        '2. **Normalize dates** — coerce pre-2024-03 mixed formats to ISO-8601.',
        '3. **Null policy** — keep `discount_code` nulls; impute `ship_region` from customer record.',
        '',
        '| Gate | Threshold |',
        '|---|---|',
        '| Row loss | < 0.5% |',
        '| Key uniqueness | 100% |',
      ].join('\n'),
    }, turnId);

    /* ---- training-style progress ---- */
    await sleep(1200);
    broadcast('stage', { stage: 'training', description: 'Running cleaning + validation pass' }, turnId);
    for (const p of [0.12, 0.34, 0.58, 0.81, 1.0]) {
      await sleep(850);
      broadcast('training_progress', {
        id: 'clean-pass', progress: p,
        message: p >= 1 ? 'Validation gates passed (8/8)' : `Processing chunk ${Math.round(p * 86)}/86 · gate checks queued`,
      }, turnId);
    }

    await sleep(700);
    broadcast('image', { src: DEMO_PLOT_URI, filename: 'validation_loss.svg' }, turnId);

    await sleep(700);
    broadcast('llm_usage', {
      agent: 'planner', prompt_tokens: 18432, completion_tokens: 1210, total_tokens: 19642,
      cached_prompt_tokens: 14336, cache_hit_rate: 0.7778,
      response_phase: 'tool_loop', response_finish_reason: 'tool_calls',
    }, turnId);

    await sleep(600);
    broadcast('context_microcompacted', {
      agent: 'planner', before_tokens: 19642, after_tokens: 12110,
      tokens_saved: 7532, compacted_tool_messages: 6, reason: 'auto',
    }, turnId);

    await sleep(700);
    broadcast('turn_plan_updated', {
      source: 'planner', scope: 'planner',
      explanation: 'Cleaning pass complete; report stage remaining.',
      updated_at: now(),
      plan: [
        { status: 'done', step: 'Inspect workspace and profile peripheral blood mononuclear cells' },
        { status: 'done', step: 'Draft cleaning pipeline in the Python sandbox' },
        { status: 'done', step: 'Run validation checks against scientific quality gates' },
        { status: 'in_progress', step: 'Compile summary report' },
      ],
    }, turnId);

    await sleep(800);
    broadcast('stage', { stage: 'report', description: 'Compiling cleaning report' }, turnId);
    await sleep(900);
    broadcast('report_generated', {
      path: 'output/cleaning_report.html', filename: 'cleaning_report.html', url: '#demo-report',
    }, turnId);

    await sleep(700);
    broadcast('stage', { stage: 'complete', description: 'Turn complete' }, turnId);
    await sleep(400);
    broadcast('turn_complete', {
      response: [
        'Cleaning pipeline finished — **8,412,338 → 8,398,121 rows** (0.17% loss, within the 0.5% gate).',
        '',
        '**What was done**',
        '- Removed **14,202** retry-duplicated `order_id`s (kept latest `updated_at`)',
        '- Normalized 612k pre-March dates to ISO-8601',
        '- Imputed `ship_region` for 5.9% of rows from customer records',
        '',
        'All 8 validation gates passed. The full report is in `output/cleaning_report.html` — open it from the link above. Want me to schedule this as a nightly job?',
      ].join('\n'),
    }, turnId);
  } finally {
    running = false;
  }
}

/* ------------------------------------------------------------------ */
/* HTTP API                                                             */
/* ------------------------------------------------------------------ */
const json = (res, body, code = 200) => {
  res.writeHead(code, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify(body));
};
const readBody = (req) => new Promise((resolve) => {
  let raw = '';
  req.on('data', (c) => { raw += c; });
  req.on('end', () => { try { resolve(JSON.parse(raw || '{}')); } catch { resolve({}); } });
});

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, 'http://localhost');
  const path = url.pathname;

  if (path === '/api/config') {
    return json(res, {
      hasActiveSession: true,
      isAgentRunning: running,
      input_path: '/workspace/input',
      output_path: '/workspace/output',
      algorithm_proposal_review_mode: 'always_user_review',
      idea_review_mode: 'agent_decide',
      eventsLog: [],
      conversationHistory: [],
    });
  }
  if (path === '/api/chat' && req.method === 'POST') {
    const body = await readBody(req);
    const message = String(body.message || '');
    if (message.startsWith('/')) {
      return json(res, { status: 'success', command_handled: true, events: [], message: 'Demo server: slash commands are no-ops.' });
    }
    setTimeout(() => runDemoTurn(message), 50);
    return json(res, { status: 'success' });
  }
  if (path === '/api/stop' && req.method === 'POST') {
    return json(res, { status: 'success', thread_alive: false });
  }
  if (path === '/api/proposals/review' && req.method === 'POST') {
    const body = await readBody(req);
    return json(res, {
      status: 'success',
      message: `Proposal ${body.decision || 'reviewed'} recorded (demo).`,
      auto_continue: { status: 'skipped' },
      proposal: {
        algorithm_id: body.algorithm_id, proposal_id: body.proposal_id || 'prop-001',
        title: 'Orders cleaning + dedup pipeline',
      },
    });
  }
  if (path === '/api/ideas/review' && req.method === 'POST') {
    const body = await readBody(req);
    return json(res, { status: 'success', message: `Idea ${body.decision} recorded (demo).`, idea: { idea_id: body.idea_id } });
  }
  if (path === '/api/conversations') {
    return json(res, {
      status: 'success',
      conversations: [
        {
          session_id: 'demo-1',
          metadata: { title: 'Orders table cleaning', updated_at: now(), message_count: 12 },
          preview: 'Cleaning pipeline finished — 8.4M rows processed, all gates passed…',
        },
        {
          session_id: 'demo-2',
          metadata: { title: 'Quarterly returns analysis', updated_at: new Date(Date.now() - 3 * 864e5).toISOString(), message_count: 7 },
          preview: 'Returns clustered around two SKUs; drafted a validation plan…',
        },
      ],
    });
  }
  if (path === '/api/skills') {
    return json(res, { status: 'success', skills: [], algorithms: [] });
  }
  if (path === '/api/init' || path === '/api/new_session' || path === '/api/switch_model' || path === '/api/stop_hook_settings') {
    return json(res, { status: 'success' });
  }
  json(res, { status: 'error', message: `Demo server: no handler for ${path}` }, 404);
});

const wss = new WebSocketServer({ server, path: '/ws' });
wss.on('connection', (ws) => {
  sockets.add(ws);
  ws.on('close', () => sockets.delete(ws));
});

server.listen(PORT, () => {
  console.log(`Demo backend listening on http://localhost:${PORT} (REST + /ws)`);
});
