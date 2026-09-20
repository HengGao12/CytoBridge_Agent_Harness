import assert from 'node:assert/strict';
import {
  reduceTimelineEvent,
  reduceTimelineEvents,
  getStepKey,
  getTimelineItemKey,
  thinkingUpdateForEvent,
} from './timelineReducer.js';

const event = (type, data = {}, extra = {}) => ({
  type,
  timestamp: extra.timestamp || `2026-04-24T00:00:${String(extra.seq || 0).padStart(2, '0')}`,
  event_id: extra.event_id || `${type}-${extra.seq || 0}`,
  turn_id: extra.turn_id || '',
  data,
});

{
  let timeline = [];
  timeline = reduceTimelineEvent(timeline, event('user_message', { content: 'first' }, { turn_id: 'turn-a', seq: 1 }));
  timeline = reduceTimelineEvent(timeline, event('training_progress', { id: 'default', progress: 0.2, message: 'first progress' }, { turn_id: 'turn-a', seq: 2 }));
  timeline = reduceTimelineEvent(timeline, event('user_message', { content: 'second' }, { turn_id: 'turn-b', seq: 3 }));
  timeline = reduceTimelineEvent(timeline, event('training_progress', { id: 'default', progress: 0.4, message: 'second progress' }, { turn_id: 'turn-b', seq: 4 }));

  const processes = timeline.filter((item) => item.type === 'process');
  assert.equal(processes.length, 2);
  assert.equal(processes[0].isComplete, true);
  assert.equal(processes[0].steps[0].message, 'first progress');
  assert.equal(processes[1].steps[0].message, 'second progress');
}

{
  let timeline = [];
  timeline = reduceTimelineEvent(timeline, event('user_message', { content: 'same turn runs' }, { turn_id: 'turn-progress', seq: 21 }));
  timeline = reduceTimelineEvent(timeline, event('training_progress', { id: 'preview:algo:stage1', progress: 0.2, message: 'preview 1 start' }, { turn_id: 'turn-progress', seq: 22 }));
  timeline = reduceTimelineEvent(timeline, event('training_progress', { id: 'preview:algo:stage1', progress: 1.0, message: 'preview 1 done' }, { turn_id: 'turn-progress', seq: 23 }));
  timeline = reduceTimelineEvent(timeline, event('training_progress', { id: 'preview:algo:stage1', progress: 0.1, message: 'preview 2 start' }, { turn_id: 'turn-progress', seq: 24 }));
  timeline = reduceTimelineEvent(timeline, event('training_progress', { id: 'preview:algo:stage1', progress: 0.5, message: 'preview 2 running' }, { turn_id: 'turn-progress', seq: 25 }));

  const process = timeline.find((item) => item.type === 'process');
  const progressSteps = process.steps.filter((step) => step.type === 'training_progress');
  assert.equal(progressSteps.length, 2);
  assert.equal(progressSteps[0].message, 'preview 1 done');
  assert.equal(progressSteps[1].message, 'preview 2 running');
  assert.notEqual(getStepKey(0, progressSteps[0]), getStepKey(1, progressSteps[1]));
}

{
  let timeline = [];
  timeline = reduceTimelineEvent(timeline, event('user_message', { content: 'proposal' }, { turn_id: 'turn-c', seq: 5 }));
  timeline = reduceTimelineEvent(timeline, event('algorithm_proposal_submitted', {
    algorithm_id: 'algo_a',
    proposal_id: 'proposal_1',
    status: 'pending_user_review',
    proposal_markdown: '# Proposal',
  }, { turn_id: 'turn-c', seq: 6 }));
  timeline = reduceTimelineEvent(timeline, event('algorithm_proposal_review_requested', {
    algorithm_id: 'algo_a',
    proposal_id: 'proposal_1',
    status: 'pending_user_review',
    proposal_markdown: '# Proposal',
  }, { turn_id: 'turn-c', seq: 7 }));
  timeline = reduceTimelineEvent(timeline, event('algorithm_proposal_reviewed', {
    algorithm_id: 'algo_a',
    proposal_id: 'proposal_1',
    decision: 'approve',
    status: 'approved',
  }, { seq: 8 }));

  const process = timeline.find((item) => item.type === 'process');
  const proposalSteps = process.steps.filter((step) => step.type === 'proposal_event');
  assert.equal(proposalSteps.length, 1);
  assert.equal(proposalSteps[0].event_type, 'algorithm_proposal_reviewed');
  assert.equal(proposalSteps[0].data.status, 'approved');
}

{
  let timeline = [];
  timeline = reduceTimelineEvent(timeline, event('user_message', { content: 'tools' }, { turn_id: 'turn-d', seq: 9 }));
  timeline = reduceTimelineEvent(timeline, event('tool_start', { tool_call_id: 'tool-1', tool: 'read_file', args: {} }, { turn_id: 'turn-d', seq: 10 }));
  timeline = reduceTimelineEvent(timeline, event('tool_start', { tool_call_id: 'tool-2', tool: 'list_files', args: {} }, { turn_id: 'turn-d', seq: 11 }));
  timeline = reduceTimelineEvent(timeline, event('tool_output', { tool_call_id: 'tool-1', content: 'read output' }, { turn_id: 'turn-d', seq: 12 }));

  const process = timeline.find((item) => item.type === 'process');
  assert.equal(process.steps[0].output, 'read output');
  assert.equal(process.steps[1].status, 'running');
  assert.equal(getStepKey(0, process.steps[0]), 'tool-tool-1');
}

{
  const timeline = reduceTimelineEvents([
    event('user_message', { content: 'hello' }, { turn_id: 'turn-e', seq: 13 }),
    event('training_progress', { id: 'p', progress: 1, message: 'done' }, { turn_id: 'turn-e', seq: 14 }),
    event('turn_complete', { response: 'ok' }, { turn_id: 'turn-e', seq: 15 }),
  ]);
  assert.equal(timeline[0].type, 'system');
  assert.equal(timeline.some((item) => item.type === 'assistant' && item.content === 'ok'), true);
  assert.equal(timeline.find((item) => item.type === 'process').isComplete, true);
}

{
  const timeline = reduceTimelineEvents([
    event('user_message', { content: 'python' }, { turn_id: 'turn-stale', seq: 15 }),
    event('item/pythonExecution/started', {
      id: 'exec-stale',
      toolName: 'execute_python',
      status: 'inProgress',
      code: 'raise SystemExit("target block not found")',
    }, { turn_id: 'turn-stale', seq: 16 }),
    event('turn_finished', { message_count: 2 }, { turn_id: 'turn-stale', seq: 17 }),
  ]);
  const process = timeline.find((item) => item.type === 'process');
  const pyStep = process.steps.find((step) => step.type === 'python_execution');
  assert.equal(process.isComplete, true);
  assert.equal(pyStep.status, 'interrupted');
  assert.match(pyStep.error, /completion event/);
}

{
  const timeline = reduceTimelineEvent([], event('system', { message: 'Slash command handled.' }, { seq: 16 }));
  assert.equal(timeline.length, 1);
  assert.equal(timeline[0].type, 'system');
  assert.equal(timeline[0].message, 'Slash command handled.');
}

{
  let timeline = [];
  timeline = reduceTimelineEvent(timeline, event('user_message', { content: 'stop me' }, { turn_id: 'turn-f', seq: 17 }));
  timeline = reduceTimelineEvent(timeline, event('status', { message: 'Processing...' }, { turn_id: 'turn-f', seq: 18 }));
  timeline = reduceTimelineEvent(timeline, event('stopped', { message: 'Agent stopped by user' }, { turn_id: 'turn-f', seq: 19 }));
  timeline = reduceTimelineEvent(timeline, event('stopped', { message: 'Agent stopped by user' }, { turn_id: 'turn-f', seq: 20 }));

  const processes = timeline.filter((item) => item.type === 'process');
  const keys = timeline.map((item, idx) => getTimelineItemKey(idx, item));
  assert.equal(processes.length, 1);
  assert.equal(new Set(keys).size, keys.length);
  assert.equal(processes[0].steps.filter((step) => step.type === 'log').length, 3);
}

{
  assert.equal(thinkingUpdateForEvent(event('stopped', { thread_alive: true })), true);
  assert.equal(thinkingUpdateForEvent(event('stopped', { thread_alive: false })), false);
}
