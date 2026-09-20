const SYSTEM_RESUMED = { type: 'system', message: 'Session resumed from checkpoint.' };

const eventTimestamp = (payload) => payload?.timestamp || new Date().toISOString();
const eventTurnId = (payload) => String(payload?.turn_id || payload?.process_id || '').trim();
const eventId = (payload) => String(payload?.event_id || '').trim();

const findLastIndex = (items, predicate) => {
  for (let i = items.length - 1; i >= 0; i -= 1) {
    if (predicate(items[i], i)) return i;
  }
  return -1;
};

const finalizeProcessBlock = (process) => {
  const steps = Array.isArray(process?.steps)
    ? process.steps.map((step) => {
      if (step?.type === 'python_execution' && String(step.status || '').toLowerCase() === 'inprogress') {
        return {
          ...step,
          status: 'interrupted',
          error: step.error || 'Turn ended before a Python completion event was received.',
        };
      }
      if (step?.type === 'tool' && step.status === 'running') {
        return {
          ...step,
          status: 'done',
          output: step.output || 'Turn ended before a tool completion event was received.',
        };
      }
      return step;
    })
    : [];
  return { ...process, steps, isComplete: true };
};

const closeOpenProcessBlocks = (timeline) => {
  for (let i = 0; i < timeline.length; i += 1) {
    const item = timeline[i];
    if (item?.type === 'process' && !item?.isComplete) {
      timeline[i] = finalizeProcessBlock(item);
    }
  }
};

const cloneProcessAt = (timeline, processIndex) => {
  const currentProcess = timeline[processIndex];
  if (!currentProcess) return null;
  const cloned = {
    ...currentProcess,
    steps: Array.isArray(currentProcess.steps) ? [...currentProcess.steps] : [],
  };
  timeline[processIndex] = cloned;
  return cloned;
};

const findProcessIndexForEvent = (timeline, payload, { includeComplete = false } = {}) => {
  const turnId = eventTurnId(payload);
  if (turnId) {
    for (let i = timeline.length - 1; i >= 0; i -= 1) {
      const item = timeline[i];
      if (item?.type !== 'process') continue;
      if (!includeComplete && item?.isComplete) continue;
      if (String(item.turnId || '') === turnId) return i;
    }
  }

  const latestUserIndex = findLastIndex(timeline, (item) => item?.type === 'user');
  for (let i = timeline.length - 1; i >= 0; i -= 1) {
    const item = timeline[i];
    if (item?.type !== 'process') continue;
    if (!includeComplete && item?.isComplete) continue;
    if (latestUserIndex >= 0 && i < latestUserIndex) continue;
    return i;
  }
  return -1;
};

const ensureProcessBlock = (timeline, payload) => {
  const turnId = eventTurnId(payload);
  const openProcessIndex = findProcessIndexForEvent(
    timeline,
    payload,
    { includeComplete: Boolean(turnId) }
  );
  if (openProcessIndex < 0) {
    const timestamp = eventTimestamp(payload);
    const created = {
      type: 'process',
      id: turnId || eventId(payload) || `process-${timestamp}`,
      turnId,
      steps: [],
      isComplete: false,
      startTime: timestamp,
    };
    timeline.push(created);
    return { process: created, processIndex: timeline.length - 1 };
  }
  return { process: cloneProcessAt(timeline, openProcessIndex), processIndex: openProcessIndex };
};

const proposalEntityId = (eventItem) => String(
  eventItem?.data?.algorithm_id ||
  eventItem?.data?.idea_id ||
  ''
).trim();

const proposalRevisionId = (eventItem) => String(
  eventItem?.data?.proposal_id ||
  eventItem?.data?.current_revision_id ||
  eventItem?.data?.revision_id ||
  ''
).trim();

const proposalMarkdownPayload = (eventItem) => String(
  eventItem?.data?.proposal_markdown ||
  eventItem?.data?.idea_markdown ||
  ''
).trim();

const mergeProposalEvent = (existing, next) => {
  if (existing?.type !== 'proposal_event' || next?.type !== 'proposal_event') return null;
  const existingEntity = proposalEntityId(existing);
  const nextEntity = proposalEntityId(next);
  if (!existingEntity || existingEntity !== nextEntity) return null;

  const existingType = String(existing.event_type || '');
  const nextType = String(next.event_type || '');
  const existingData = existing.data || {};
  const nextData = next.data || {};
  const sameEventId = existing.event_id && next.event_id && existing.event_id === next.event_id;
  if (sameEventId) return 'skip';

  const existingRevision = proposalRevisionId(existing);
  const nextRevision = proposalRevisionId(next);
  const sameRevision = existingRevision && nextRevision && existingRevision === nextRevision;
  const hasRevisionBinding = Boolean(existingRevision || nextRevision);
  const existingMarkdown = proposalMarkdownPayload(existing);
  const nextMarkdown = proposalMarkdownPayload(next);
  const sameMarkdown = existingMarkdown && nextMarkdown && existingMarkdown === nextMarkdown;
  const submittedToReviewRequested =
    (
      existingType === 'algorithm_proposal_submitted' &&
      nextType === 'algorithm_proposal_review_requested'
    ) ||
    (
      existingType === 'research_idea_submitted' &&
      nextType === 'research_idea_review_requested'
    );
  const reviewUpdate =
    nextType === 'algorithm_proposal_reviewed' ||
    nextType === 'algorithm_proposal_gate_blocked' ||
    nextType === 'research_idea_reviewed' ||
    nextType === 'research_idea_progress_updated' ||
    nextType === 'research_idea_linked_algorithm';
  const sameEntityReviewUpdate = reviewUpdate && (!hasRevisionBinding || sameRevision);

  if (!sameRevision && !(submittedToReviewRequested && sameMarkdown) && !sameEntityReviewUpdate) {
    return null;
  }

  return {
    ...existing,
    ...next,
    data: {
      ...existingData,
      ...nextData,
    },
    timestamp: next.timestamp || existing.timestamp,
  };
};

const findProposalEventMergeInSteps = (steps, proposalEvent) => {
  for (let i = steps.length - 1; i >= 0; i -= 1) {
    const merged = mergeProposalEvent(steps[i], proposalEvent);
    if (!merged) continue;
    return { index: i, merged };
  }
  return null;
};

const upsertProposalEventInExistingTimeline = (timeline, proposalEvent) => {
  for (let i = timeline.length - 1; i >= 0; i -= 1) {
    const item = timeline[i];
    const merged = mergeProposalEvent(item, proposalEvent);
    if (merged) {
      if (merged !== 'skip') {
        timeline[i] = merged;
      }
      return true;
    }
    if (item?.type !== 'process' || !Array.isArray(item.steps)) continue;
    const stepMatch = findProposalEventMergeInSteps(item.steps, proposalEvent);
    if (stepMatch) {
      const cloned = cloneProcessAt(timeline, i);
      if (stepMatch.merged !== 'skip') {
        cloned.steps[stepMatch.index] = stepMatch.merged;
      }
      return true;
    }
  }
  return false;
};

const withStepEnvelope = (step, payload) => ({
  ...step,
  event_id: eventId(payload),
  turnId: eventTurnId(payload),
});

const ensurePythonExecutionStep = (timeline, payload, execId, seed = {}) => {
  const { process } = ensureProcessBlock(timeline, payload);
  const itemId = String(execId || '').trim();
  if (!itemId) return { item: null, process };
  const existingIndex = process.steps.findIndex(
    (step) => step?.type === 'python_execution' && String(step?.id || '') === itemId
  );
  if (existingIndex >= 0) {
    const existing = process.steps[existingIndex];
    const cloned = { ...existing };
    process.steps[existingIndex] = cloned;
    return { item: cloned, process };
  }
  const timestamp = eventTimestamp(payload);
  const created = withStepEnvelope({
    type: 'python_execution',
    id: itemId,
    toolName: seed.toolName || 'execute_python',
    reason: seed.reason || '',
    timeout: seed.timeout ?? null,
    timeoutMode: seed.timeoutMode || 'shared',
    status: seed.status || 'inProgress',
    code: seed.code || '',
    stdout: '',
    stderr: '',
    error: '',
    traceback: '',
    durationMs: null,
    adataChanged: false,
    scopeReused: false,
    dataPath: seed.dataPath || '',
    timeoutEnforced: true,
    switchNote: '',
    timestamp: seed.timestamp || timestamp,
  }, payload);
  process.steps.push(created);
  return { item: created, process };
};

const ensureSubagentLifecycleStep = (timeline, payload, subagentId, seed = {}) => {
  const { process } = ensureProcessBlock(timeline, payload);
  const itemId = String(subagentId || '').trim();
  if (!itemId) return { item: null, process };
  const existingIndex = process.steps.findIndex(
    (step) => step?.type === 'subagent_lifecycle' && String(step?.subagent_id || '') === itemId
  );
  if (existingIndex >= 0) {
    const existing = process.steps[existingIndex];
    const cloned = { ...existing };
    process.steps[existingIndex] = cloned;
    return { item: cloned, process };
  }
  const timestamp = eventTimestamp(payload);
  const created = withStepEnvelope({
    type: 'subagent_lifecycle',
    subagent_id: itemId,
    subagent_type: seed.subagent_type || 'general',
    parent_agent_id: seed.parent_agent_id || '',
    task: seed.task || '',
    status: seed.status || 'running',
    summary: seed.summary || '',
    findings: Array.isArray(seed.findings) ? [...seed.findings] : [],
    artifact_refs: Array.isArray(seed.artifact_refs) ? [...seed.artifact_refs] : [],
    proposed_state_updates: seed.proposed_state_updates ? { ...seed.proposed_state_updates } : {},
    needs_input_question: seed.needs_input_question || '',
    result_submitted: !!seed.result_submitted,
    started_at: seed.started_at || timestamp,
    finished_at: seed.finished_at || null,
    timestamp: seed.timestamp || timestamp,
  }, payload);
  process.steps.push(created);
  return { item: created, process };
};

const ensureTerminalCommandStep = (timeline, payload, signature, seed = {}) => {
  const { process } = ensureProcessBlock(timeline, payload);
  const itemId = String(signature || seed.command || eventTimestamp(payload)).trim();
  if (!itemId) return { item: null, process };
  const existingIndex = process.steps.findIndex(
    (step) => step?.type === 'terminal_command' && String(step?.id || '') === itemId
  );
  if (existingIndex >= 0) {
    const existing = process.steps[existingIndex];
    const cloned = { ...existing };
    process.steps[existingIndex] = cloned;
    return { item: cloned, process };
  }
  const created = withStepEnvelope({
    type: 'terminal_command',
    id: itemId,
    command: seed.command || '',
    argv: Array.isArray(seed.argv) ? [...seed.argv] : [],
    cwd: seed.cwd || '',
    success: typeof seed.success === 'boolean' ? seed.success : true,
    exit_code: seed.exit_code ?? null,
    timed_out: !!seed.timed_out,
    timeout_sec: seed.timeout_sec ?? null,
    duration_sec: seed.duration_sec ?? null,
    stdout: seed.stdout || '',
    stderr: '',
    error: '',
    stdout_truncated: !!seed.stdout_truncated,
    stderr_truncated: !!seed.stderr_truncated,
    clone_destination: seed.clone_destination || '',
    supported_commands: seed.supported_commands ? { ...seed.supported_commands } : {},
    supported_commands_summary: seed.supported_commands_summary || '',
    usage_hint: seed.usage_hint || '',
    timestamp: seed.timestamp || eventTimestamp(payload),
  }, payload);
  process.steps.push(created);
  return { item: created, process };
};

const appendProcessStep = (timeline, payload, step) => {
  const { process } = ensureProcessBlock(timeline, payload);
  process.steps.push(withStepEnvelope(step, payload));
  return process;
};

const progressInstanceId = (payload, process, progressId) => {
  const eventKey = eventId(payload) || eventTimestamp(payload);
  const index = Array.isArray(process?.steps) ? process.steps.length : 0;
  return `${progressId || 'progress'}:${eventKey}:${index}`;
};

const shouldStartNewProgressStep = (existing, nextProgress) => {
  if (!existing) return true;
  const previous = Number(existing.progress ?? 0);
  const next = Number(nextProgress ?? 0);
  if (!Number.isFinite(previous) || !Number.isFinite(next)) return false;
  if (previous >= 1 && next < 1) return true;
  return next + 0.05 < previous;
};

const findToolStep = (process, toolCallId) => {
  if (toolCallId) {
    const byId = process.steps.find(
      (step) => step?.type === 'tool' && String(step?.tool_call_id || '') === toolCallId
    );
    if (byId) return byId;
  }
  return process.steps.slice().reverse().find((step) => step.type === 'tool' && step.status === 'running') || null;
};

export function reduceTimelineEvent(prevTimeline, payload) {
  const type = payload?.type || '';
  const data = payload?.data || {};
  const timestamp = eventTimestamp(payload);
  const turnId = eventTurnId(payload);
  const id = eventId(payload);
  const newTimeline = [...(prevTimeline || [])];

  switch (type) {
    case 'user_message': {
      const clientRequestId = String(data.client_request_id || '').trim();
      const isOptimistic = Boolean(data._optimistic);
      const attachments = Array.isArray(data.attachments) ? data.attachments : [];

      // Reconcile a server echo against the optimistic bubble we already
      // rendered: match on client_request_id when present, otherwise fall back
      // to the most recent still-optimistic bubble with identical content.
      if (!isOptimistic) {
        const optimisticIndex = findLastIndex(newTimeline, (item) =>
          item?.type === 'user' &&
          item.optimistic &&
          (clientRequestId
            ? item.clientRequestId === clientRequestId
            : String(item.content || '') === String(data.content || '')),
        );
        if (optimisticIndex >= 0) {
          newTimeline[optimisticIndex] = {
            ...newTimeline[optimisticIndex],
            id: id || newTimeline[optimisticIndex].id,
            turnId: turnId || newTimeline[optimisticIndex].turnId,
            attachments: attachments.length
              ? attachments
              : newTimeline[optimisticIndex].attachments,
            optimistic: false,
          };
          break;
        }
      }

      closeOpenProcessBlocks(newTimeline);
      newTimeline.push({
        type: 'user',
        id: id || `user-${timestamp}`,
        turnId,
        content: data.content,
        attachments,
        timestamp,
        clientRequestId: clientRequestId || undefined,
        optimistic: isOptimistic,
      });
      break;
    }
    case 'assistant_message': {
      const content = data?.content || '';
      const currentLastItem = newTimeline[newTimeline.length - 1];
      if (
        currentLastItem &&
        currentLastItem.type === 'assistant' &&
        String(currentLastItem.content || '') === String(content)
      ) {
        break;
      }
      newTimeline.push({
        type: 'assistant',
        id: id || `assistant-${timestamp}`,
        turnId,
        content,
        timestamp,
      });
      break;
    }
    case 'item/pythonExecution/started': {
      ensurePythonExecutionStep(newTimeline, payload, data.id, { ...data, timestamp });
      break;
    }
    case 'item/pythonExecution/outputDelta': {
      const { item } = ensurePythonExecutionStep(newTimeline, payload, data.id, { timestamp });
      if (!item) break;
      const streamName = String(data.stream || 'stdout');
      const delta = String(data.delta || '');
      if (!delta) break;
      if (streamName === 'stderr') {
        item.stderr = `${item.stderr || ''}${delta}`;
      } else {
        item.stdout = `${item.stdout || ''}${delta}`;
      }
      break;
    }
    case 'item/pythonExecution/completed': {
      const { item } = ensurePythonExecutionStep(newTimeline, payload, data.id, { ...data, timestamp });
      if (!item) break;
      Object.assign(item, {
        toolName: data.toolName || item.toolName,
        reason: data.reason ?? item.reason,
        timeout: data.timeout ?? item.timeout,
        timeoutMode: data.timeoutMode || item.timeoutMode,
        status: data.status || item.status || 'completed',
        stdout: data.stdout ?? item.stdout,
        stderr: data.stderr ?? item.stderr,
        error: data.error ?? item.error,
        traceback: data.traceback ?? item.traceback,
        durationMs: data.durationMs ?? item.durationMs,
        adataChanged: !!data.adataChanged,
        scopeReused: !!data.scopeReused,
        dataPath: data.dataPath ?? item.dataPath,
        timeoutEnforced: typeof data.timeoutEnforced === 'boolean' ? data.timeoutEnforced : item.timeoutEnforced,
        switchNote: data.switchNote ?? item.switchNote,
      });
      break;
    }
    case 'tool_start': {
      appendProcessStep(newTimeline, payload, {
        type: 'tool',
        tool_call_id: data.tool_call_id || id,
        tool: data.tool,
        args: data.args,
        reason: data.reason || null,
        status: 'running',
        output: null,
        timestamp,
      });
      break;
    }
    case 'tool_output': {
      const { process } = ensureProcessBlock(newTimeline, payload);
      const toolStep = findToolStep(process, String(data.tool_call_id || ''));
      if (toolStep) {
        toolStep.status = 'done';
        toolStep.output = data.content;
        toolStep.output_event_id = id;
      }
      break;
    }
    case 'agent_thought': {
      appendProcessStep(newTimeline, payload, {
        type: 'thought',
        agent: data.agent,
        content: data.content,
        timestamp,
      });
      break;
    }
    case 'subagent_started': {
      const { item } = ensureSubagentLifecycleStep(newTimeline, payload, data.subagent_id, {
        ...data,
        status: 'running',
        started_at: timestamp,
        timestamp,
      });
      if (!item) break;
      Object.assign(item, {
        subagent_type: data.subagent_type || item.subagent_type,
        parent_agent_id: data.parent_agent_id || item.parent_agent_id,
        task: data.task ?? item.task,
        status: 'running',
        started_at: item.started_at || timestamp,
        timestamp,
      });
      break;
    }
    case 'subagent_thought': {
      appendProcessStep(newTimeline, payload, {
        type: 'subagent_thought',
        subagent_id: data.subagent_id || '',
        subagent_type: data.subagent_type || 'general',
        parent_agent_id: data.parent_agent_id || '',
        agent: data.agent || 'subagent',
        content: data.content || '',
        timestamp,
      });
      break;
    }
    case 'subagent_result_submitted': {
      const { item } = ensureSubagentLifecycleStep(newTimeline, payload, data.subagent_id, { timestamp });
      if (!item) break;
      item.result_submitted = true;
      item.summary = data.summary ?? item.summary;
      if (data.status) item.pending_status = data.status;
      break;
    }
    case 'proposal_review_submitted': {
      const { item } = ensureSubagentLifecycleStep(newTimeline, payload, data.subagent_id, {
        subagent_type: 'proposal_evaluator',
        timestamp,
      });
      if (!item) break;
      item.result_submitted = true;
      item.summary = data.summary ?? item.summary;
      item.proposed_state_updates = {
        ...(item.proposed_state_updates || {}),
        proposal_review: {
          ...((item.proposed_state_updates || {}).proposal_review || {}),
          decision: data.decision || '',
          implementation_risks: Array.isArray(data.implementation_risks) ? data.implementation_risks : [],
          confidence: data.confidence ?? null,
        },
      };
      break;
    }
    case 'subagent_completed':
    case 'subagent_failed':
    case 'subagent_needs_input': {
      const normalizedStatus =
        type === 'subagent_completed'
          ? 'completed'
          : (type === 'subagent_needs_input' ? 'needs_input' : 'failed');
      const { item } = ensureSubagentLifecycleStep(newTimeline, payload, data.subagent_id, {
        ...data,
        status: normalizedStatus,
        timestamp,
      });
      if (!item) break;
      Object.assign(item, {
        subagent_type: data.subagent_type || item.subagent_type,
        parent_agent_id: data.parent_agent_id || item.parent_agent_id,
        status: normalizedStatus,
        summary: data.summary ?? item.summary,
        findings: Array.isArray(data.findings) ? [...data.findings] : item.findings,
        artifact_refs: Array.isArray(data.artifact_refs) ? [...data.artifact_refs] : item.artifact_refs,
        proposed_state_updates: data.proposed_state_updates ? { ...data.proposed_state_updates } : item.proposed_state_updates,
        needs_input_question: data.needs_input_question ?? item.needs_input_question,
        finished_at: timestamp,
        result_submitted: true,
        timestamp,
      });
      break;
    }
    case 'code': {
      appendProcessStep(newTimeline, payload, {
        type: 'code',
        language: data.language,
        code: data.code,
        timestamp,
      });
      break;
    }
    case 'status': {
      if (data.message) {
        appendProcessStep(newTimeline, payload, { type: 'log', message: data.message, timestamp });
      }
      break;
    }
    case 'turn_complete':
    case 'turn_finished': {
      const processIndex = findProcessIndexForEvent(newTimeline, payload, { includeComplete: true });
      if (processIndex >= 0) {
        const currentProcess = newTimeline[processIndex];
        newTimeline[processIndex] = finalizeProcessBlock(currentProcess);
      }
      if (type === 'turn_finished') {
        break;
      }
      const response = data?.response || '';
      const latestAssistant = [...newTimeline].reverse().find((item) => item.type === 'assistant');
      const alreadyHasAssistant =
        !!latestAssistant &&
        String(latestAssistant.content || '') === String(response);
      if (response && !alreadyHasAssistant) {
        newTimeline.push({
          type: 'assistant',
          id: id || `assistant-${timestamp}`,
          turnId,
          content: response,
          timestamp,
        });
      }
      break;
    }
    case 'error': {
      newTimeline.push({
        type: 'error',
        id: id || `error-${timestamp}`,
        turnId,
        message: data.message,
        timestamp,
      });
      break;
    }
    case 'stopped': {
      appendProcessStep(newTimeline, payload, {
        type: 'log',
        message: 'Agent stopped by user',
        timestamp,
      });
      const processIndex = findProcessIndexForEvent(newTimeline, payload, { includeComplete: true });
      if (processIndex >= 0) {
        const currentProcess = newTimeline[processIndex];
        newTimeline[processIndex] = { ...currentProcess, isComplete: true };
      }
      break;
    }
    case 'image': {
      appendProcessStep(newTimeline, payload, {
        type: 'image',
        src: data.src,
        filename: data.filename,
        timestamp,
      });
      break;
    }
    case 'theory_selection': {
      appendProcessStep(newTimeline, payload, {
        type: 'theory_selection',
        model_family: data.model_family,
        reasoning: data.reasoning,
        theory_support: data.theory_support,
        config: data.config,
        timestamp,
      });
      break;
    }
    case 'report_generated': {
      appendProcessStep(newTimeline, payload, {
        type: 'report',
        path: data.path,
        filename: data.filename,
        url: data.url,
        timestamp,
      });
      break;
    }
    case 'stage': {
      const process = appendProcessStep(newTimeline, payload, {
        type: 'stage',
        stage: data.stage,
        description: data.description,
        timestamp,
      });
      process.currentStage = data.stage;
      process.stageDescription = data.description;
      break;
    }
    case 'progress':
    case 'training_progress': {
      const { process } = ensureProcessBlock(newTimeline, payload);
      const progressId = data.id || 'default';
      const existingIdx = findLastIndex(
        process.steps,
        (step) => step.type === 'training_progress' && step.id === progressId
      );
      const existingStep = existingIdx >= 0 ? process.steps[existingIdx] : null;
      if (existingIdx >= 0 && !shouldStartNewProgressStep(existingStep, data.progress)) {
        process.steps[existingIdx] = {
          ...existingStep,
          message: data.message,
          progress: data.progress,
          timestamp,
          event_id: id || existingStep.event_id,
        };
      } else {
        process.steps.push(withStepEnvelope({
          type: 'training_progress',
          id: progressId,
          progress_instance_id: progressInstanceId(payload, process, progressId),
          message: data.message,
          progress: data.progress,
          timestamp,
        }, payload));
      }
      break;
    }
    case 'context_compacted': {
      const beforeTokens = Number(data.before_tokens ?? 0);
      const afterTokens = Number(data.after_tokens ?? 0);
      const removedMessages = Number(data.removed_messages ?? 0);
      const summaryText = String(data.summary_text || data.summary_core || '').trim();
      const mode = data.mode || null;
      const isNoopCompact =
        (mode === 'none' || mode === null) &&
        beforeTokens <= afterTokens &&
        removedMessages <= 0 &&
        !summaryText;
      if (isNoopCompact) break;
      const compactStep = {
        type: 'context_compacted',
        agent: data.agent || 'agent',
        before_tokens: beforeTokens,
        after_tokens: afterTokens,
        reason: data.reason || 'auto',
        mode,
        summary_text: data.summary_text || '',
        summary_core: data.summary_core || '',
        timestamp,
      };
      if (typeof data.removed_messages === 'number') compactStep.removed_messages = data.removed_messages;
      if (typeof data.dropped_dangling_tool_results === 'number') {
        compactStep.dropped_dangling_tool_results = data.dropped_dangling_tool_results;
      }
      appendProcessStep(newTimeline, payload, compactStep);
      break;
    }
    case 'context_microcompacted': {
      appendProcessStep(newTimeline, payload, {
        type: 'context_microcompacted',
        agent: data.agent || 'agent',
        before_tokens: data.before_tokens ?? 0,
        after_tokens: data.after_tokens ?? 0,
        tokens_saved: data.tokens_saved ?? 0,
        compacted_tool_messages: data.compacted_tool_messages ?? 0,
        cleared_media_messages: data.cleared_media_messages ?? 0,
        reason: data.reason || 'auto',
        timestamp,
      });
      break;
    }
    case 'llm_usage': {
      appendProcessStep(newTimeline, payload, {
        type: 'llm_usage',
        agent: data.agent || 'agent',
        prompt_tokens: data.prompt_tokens ?? null,
        completion_tokens: data.completion_tokens ?? null,
        total_tokens: data.total_tokens ?? null,
        cached_prompt_tokens: data.cached_prompt_tokens ?? null,
        cache_hit_rate: data.cache_hit_rate ?? null,
        response_text: data.response_text || '',
        response_tool_calls: Array.isArray(data.response_tool_calls) ? data.response_tool_calls : [],
        response_phase: data.response_phase || null,
        response_finish_reason: data.response_finish_reason || null,
        timestamp,
      });
      break;
    }
    case 'resume_warning': {
      const warningText = Array.isArray(data?.warnings)
        ? data.warnings.join('\n')
        : (Array.isArray(data?.reasons) ? data.reasons.join('\n') : (data?.message || 'Resume completed with warnings.'));
      appendProcessStep(newTimeline, payload, {
        type: 'resume_warning',
        message: warningText,
        timestamp,
      });
      break;
    }
    case 'turn_plan_updated': {
      const { process } = ensureProcessBlock(newTimeline, payload);
      const plan = Array.isArray(data.plan) ? data.plan : [];
      const signature = JSON.stringify({
        source: data.source || 'unknown',
        scope: data.scope || 'planner',
        explanation: data.explanation || '',
        updated_at: data.updated_at || '',
        plan,
      });
      const duplicated = process.steps.some(
        (step) => step.type === 'plan_updated' && step.signature === signature
      );
      if (duplicated) break;
      process.steps.push(withStepEnvelope({
        type: 'plan_updated',
        source: data.source || 'unknown',
        scope: data.scope || 'planner',
        explanation: data.explanation || '',
        updated_at: data.updated_at || timestamp,
        plan,
        signature,
        timestamp,
      }, payload));
      break;
    }
    case 'tool_candidate_proposed':
    case 'tool_refined':
    case 'tool_gate_passed':
    case 'tool_gate_failed':
    case 'tool_activation_pending':
    case 'tool_activated':
    case 'tool_activation_mode_updated':
    case 'tool_review_updated': {
      appendProcessStep(newTimeline, payload, {
        type: 'tool_lifecycle',
        event_type: type,
        data: data || {},
        timestamp,
      });
      break;
    }
    case 'skill_loaded':
    case 'skill_unloaded':
    case 'skill_visibility_updated':
    case 'algorithm_visibility_updated':
    case 'planner_auto_skills_updated':
    case 'skills_context_updated': {
      appendProcessStep(newTimeline, payload, {
        type: 'skill_event',
        event_type: type,
        data: data || {},
        timestamp,
      });
      break;
    }
    case 'file_activity': {
      appendProcessStep(newTimeline, payload, {
        type: 'file_event',
        event_type: type,
        data: data || {},
        timestamp,
      });
      break;
    }
    case 'terminal_command': {
      const signature = data.id || data.command_id || JSON.stringify([
        data.command || '',
        data.cwd || '',
        timestamp,
      ]);
      const { item } = ensureTerminalCommandStep(newTimeline, payload, signature, { ...data, timestamp });
      if (!item) break;
      Object.assign(item, {
        command: data.command ?? item.command,
        argv: Array.isArray(data.argv) ? [...data.argv] : item.argv,
        cwd: data.cwd ?? item.cwd,
        success: typeof data.success === 'boolean' ? data.success : item.success,
        exit_code: data.exit_code ?? item.exit_code,
        timed_out: typeof data.timed_out === 'boolean' ? data.timed_out : item.timed_out,
        timeout_sec: data.timeout_sec ?? item.timeout_sec,
        duration_sec: data.duration_sec ?? item.duration_sec,
        stdout: data.stdout ?? item.stdout,
        stderr: data.stderr ?? item.stderr,
        error: data.error ?? item.error,
        stdout_truncated: typeof data.stdout_truncated === 'boolean' ? data.stdout_truncated : item.stdout_truncated,
        stderr_truncated: typeof data.stderr_truncated === 'boolean' ? data.stderr_truncated : item.stderr_truncated,
        clone_destination: data.clone_destination ?? item.clone_destination,
        supported_commands: data.supported_commands ? { ...data.supported_commands } : item.supported_commands,
        supported_commands_summary: data.supported_commands_summary ?? item.supported_commands_summary,
        usage_hint: data.usage_hint ?? item.usage_hint,
        timestamp,
      });
      break;
    }
    case 'planner_algorithm_workspace_initialized':
    case 'workspace_tree_listed':
    case 'workspace_file_read':
    case 'workspace_file_written':
    case 'workspace_diff_previewed':
    case 'workspace_patch_applied':
    case 'planner_workspace_updated':
    case 'algorithm_context_verified':
    case 'algorithm_context_gate_blocked':
    case 'experiment_training_run_registered':
    case 'algorithm_workspace_snapshot_created':
    case 'planner_algorithm_workspace_activated':
    case 'algorithm_campaign_started':
    case 'algorithm_campaign_reset':
    case 'algorithm_campaign_trial_started':
    case 'algorithm_campaign_trial_decided':
    case 'algorithm_campaign_stage_gate_checked': {
      appendProcessStep(newTimeline, payload, {
        type: 'workspace_event',
        event_type: type,
        data: data || {},
        timestamp,
      });
      break;
    }
    case 'algorithm_proposal_submitted':
    case 'algorithm_proposal_review_requested':
    case 'algorithm_proposal_reviewed':
    case 'algorithm_proposal_gate_blocked':
    case 'research_idea_submitted':
    case 'research_idea_review_requested':
    case 'research_idea_reviewed':
    case 'research_idea_progress_updated':
    case 'research_idea_linked_algorithm': {
      const proposalEvent = withStepEnvelope({
        type: 'proposal_event',
        event_type: type,
        data: data || {},
        timestamp,
      }, payload);
      const entityId = proposalEntityId(proposalEvent);
      if (!entityId) {
        appendProcessStep(newTimeline, payload, proposalEvent);
        break;
      }
      if (upsertProposalEventInExistingTimeline(newTimeline, proposalEvent)) break;
      appendProcessStep(newTimeline, payload, proposalEvent);
      break;
    }
    case 'model_switched': {
      newTimeline.push({
        type: 'system',
        id: id || `system-${timestamp}`,
        message: `Model switched: ${data?.old_model || 'unknown'} -> ${data?.new_model || 'unknown'}`,
        timestamp,
      });
      break;
    }
    case 'system': {
      newTimeline.push({
        type: 'system',
        id: id || `system-${timestamp}`,
        message: data?.message || payload.message || '',
        timestamp,
      });
      break;
    }
    default:
      break;
  }

  return newTimeline;
}

export function reduceTimelineEvents(eventsLog, initialTimeline = [SYSTEM_RESUMED]) {
  if (!Array.isArray(eventsLog) || eventsLog.length === 0) {
    return [{ type: 'system', message: 'Session resumed. Ready to continue.' }];
  }
  const seen = new Set();
  let timeline = [...initialTimeline];
  for (const event of eventsLog) {
    const stableId = eventId(event);
    const fallbackId = [
      event?.type || '',
      event?.timestamp || '',
      eventTurnId(event),
    ].join('|');
    const key = stableId || fallbackId;
    if (key && seen.has(key)) continue;
    if (key) seen.add(key);
    timeline = reduceTimelineEvent(timeline, event);
  }
  return timeline;
}

export function appendOptimisticUserMessage(prevTimeline, content, attachments = [], clientRequestId = '') {
  const timestamp = new Date().toISOString();
  return reduceTimelineEvent(prevTimeline, {
    type: 'user_message',
    event_id: `client-user-${timestamp}`,
    timestamp,
    data: {
      content,
      attachments,
      client_request_id: clientRequestId || undefined,
      _optimistic: true,
    },
  });
}

export function thinkingUpdateForEvent(payload) {
  if (payload?.type === 'stopped') {
    return payload?.data?.thread_alive ? true : false;
  }
  if (payload?.type === 'turn_complete' || payload?.type === 'error') {
    return false;
  }
  return null;
}

export function createClientTimelineEvent(type, data) {
  const timestamp = new Date().toISOString();
  const randomId = globalThis.crypto?.randomUUID?.() || `${timestamp}-${Math.random().toString(36).slice(2)}`;
  return {
    type,
    event_id: `client-${randomId}`,
    timestamp,
    data: data || {},
  };
}

export function getTimelineItemKey(index, item) {
  const value = item || {};
  if (value.id) return String(value.id);
  if (value.type === 'process') return `process-${value.turnId || value.startTime || index}`;
  if (value.type === 'user') return `user-${value.turnId || value.timestamp || index}`;
  if (value.type === 'assistant') return `assistant-${value.turnId || value.timestamp || index}`;
  if (value.type === 'proposal_event') {
    return `proposal-${value.data?.algorithm_id || value.data?.idea_id || 'entity'}-${value.data?.proposal_id || value.data?.current_revision_id || value.timestamp || index}`;
  }
  return `${index}-${value.type || 'item'}-${value.timestamp || ''}`;
}

export function getStepKey(index, step) {
  const value = step || {};
  if (value.type === 'tool' && value.tool_call_id) return `tool-${value.tool_call_id}`;
  if (value.type === 'python_execution' && value.id) return `python-${value.id}`;
  if (value.type === 'subagent_lifecycle' && value.subagent_id) return `subagent-${value.subagent_id}`;
  if (value.type === 'terminal_command' && value.id) return `terminal-${value.id}`;
  if (value.type === 'training_progress' && value.progress_instance_id) return `progress-${value.progress_instance_id}`;
  if (value.type === 'training_progress' && value.id) return `progress-${value.turnId || ''}-${value.id}`;
  if (value.type === 'proposal_event') {
    return `proposal-${value.data?.algorithm_id || value.data?.idea_id || 'entity'}-${value.data?.proposal_id || value.data?.current_revision_id || value.event_type || index}`;
  }
  if (value.event_id) return String(value.event_id);
  return `${index}-${value.type || 'step'}-${value.timestamp || ''}`;
}
