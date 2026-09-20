import React, { useState } from 'react';
import {
    ChevronDown, Terminal, Box, CheckCircle2, RotateCw, Circle,
    Image as ImageIcon, FileText, Database,
    Cpu, AlertTriangle, Layers, Wrench, Sparkles, Target,
    BarChart3, ListTree, BookOpen, ExternalLink, SquareTerminal,
    FolderOpen, Feather, HelpCircle, XCircle,
} from 'lucide-react';
import { CodeHighlight } from './CodeHighlight';
import { PythonExecutionDetails } from './PythonExecutionCard';
import { MarkdownContent } from './MarkdownContent';
import { isInlineCode } from '../lib/markdownCode';
import { Pill } from './ui';
import { cn } from './ui/cn';

/* ============================================================================
 * Activity rail primitives
 *
 * Every agent event renders as a quiet single-line row hung on a hairline
 * vertical rail (see <StepRail/> used by ProcessBlock). Color is reserved for
 * status semantics on the small leading icon — never for whole-card washes.
 * Details are collapsed by default and expand into a neutral inset panel.
 * ========================================================================== */

const TONES = {
    neutral: 'text-ink-soft',
    accent: 'text-accent',
    success: 'text-emerald-600 dark:text-emerald-400',
    warn: 'text-amber-600 dark:text-amber-400',
    danger: 'text-red-600 dark:text-red-400',
};

/** Continuous hairline rail behind a stack of StepRows. */
export function StepRail({ children, className }) {
    return (
        <div className={cn('relative', className)}>
            <div
                aria-hidden="true"
                className="absolute bottom-3 left-[11px] top-3 w-px bg-line"
            />
            {children}
        </div>
    );
}

/**
 * StepRow — one event on the rail.
 *
 *  icon / tone   leading badge (icon tinted by tone)
 *  title         short label, medium weight
 *  subtitle      truncated muted context (command, path, reason…)
 *  meta          right-aligned mono micro-text (duration, counters…)
 *  body          always-visible content under the header (thoughts, images…)
 *  children      expandable detail panel (collapsed unless defaultOpen)
 */
function StepRow({
    icon,
    tone = 'neutral',
    title,
    subtitle,
    meta,
    body,
    children,
    defaultOpen = false,
}) {
    const [isOpen, setIsOpen] = useState(defaultOpen);
    const expandable = Boolean(children);

    const header = (
        <>
            <span className="shrink-0 text-[13px] font-medium text-ink">{title}</span>
            {subtitle ? (
                <span className="min-w-0 flex-1 truncate text-xs text-ink-soft">{subtitle}</span>
            ) : (
                <span className="min-w-0 flex-1" />
            )}
            {meta && (
                <span className="shrink-0 font-mono text-[10px] tabular-nums text-ink-soft">{meta}</span>
            )}
            {expandable && (
                <ChevronDown
                    size={13}
                    className={cn(
                        'shrink-0 text-ink-soft transition-transform duration-200',
                        isOpen && 'rotate-180',
                    )}
                />
            )}
        </>
    );

    return (
        <div className="relative flex gap-2.5 py-px">
            <div className="relative z-10 mt-1.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-full border border-line bg-surface">
                <span className={cn('flex items-center justify-center', TONES[tone] || TONES.neutral)}>
                    {icon}
                </span>
            </div>
            <div className="min-w-0 flex-1">
                {expandable ? (
                    <button
                        type="button"
                        onClick={() => setIsOpen((v) => !v)}
                        aria-expanded={isOpen}
                        className="-ml-1.5 flex w-full items-center gap-2 rounded-lg px-2 py-1.5 text-left transition-colors hover:bg-surface-muted/80 focus:outline-none focus-visible:ring-2 focus-visible:ring-accent"
                    >
                        {header}
                    </button>
                ) : (
                    <div className="-ml-1.5 flex w-full items-center gap-2 px-2 py-1.5">
                        {header}
                    </div>
                )}
                {body && <div className="px-0.5 pb-1.5">{body}</div>}
                {expandable && isOpen && (
                    <div className="mb-2 mt-0.5 animate-fade-in px-0.5">{children}</div>
                )}
            </div>
        </div>
    );
}

/* ---------- Detail panel building blocks (all neutral) ---------- */

const DetailPanel = ({ children, className }) => (
    <div className={cn('space-y-2.5 rounded-xl border border-line bg-surface-muted/60 p-3 text-xs text-ink-muted', className)}>
        {children}
    </div>
);

const DetailLabel = ({ children }) => (
    <div className="text-[11px] font-medium text-ink-soft">{children}</div>
);

const MONO_TONES = {
    neutral: 'border-line bg-surface',
    stderr: 'border-amber-200 bg-amber-50/60 dark:border-amber-500/30 dark:bg-amber-500/5',
    error: 'border-red-200 bg-red-50/60 dark:border-red-500/30 dark:bg-red-500/5',
};

const MonoBlock = ({ title, text, tone = 'neutral', maxHeight = 'max-h-64' }) => {
    const value = String(text || '');
    if (!value) return null;
    return (
        <div className="space-y-1">
            {title && <DetailLabel>{title}</DetailLabel>}
            <pre className={cn(
                'overflow-auto custom-scrollbar whitespace-pre-wrap break-words rounded-lg border p-2.5 font-mono text-[11px] leading-relaxed text-ink',
                MONO_TONES[tone] || MONO_TONES.neutral,
                maxHeight,
            )}>
                {value}
            </pre>
        </div>
    );
};

const hasPlainObjectKeys = (value) =>
    !!value &&
    typeof value === 'object' &&
    !Array.isArray(value) &&
    Object.keys(value).length > 0;

const JsonBlock = ({ title, value }) => {
    if (!hasPlainObjectKeys(value) && !(Array.isArray(value) && value.length > 0)) return null;
    return <MonoBlock title={title} text={JSON.stringify(value, null, 2)} />;
};

/** Compact key/value listing; skips empty values. */
const KVList = ({ entries }) => {
    const rows = (entries || []).filter(([, v]) => v !== undefined && v !== null && v !== '');
    if (!rows.length) return null;
    return (
        <div className="space-y-0.5 font-mono text-[11px] leading-relaxed text-ink-soft">
            {rows.map(([k, v]) => (
                <div key={k} className="break-all">
                    <span className="text-ink-muted">{k}</span>: {String(v)}
                </div>
            ))}
        </div>
    );
};

const BulletList = ({ title, items }) => {
    if (!Array.isArray(items) || items.length === 0) return null;
    return (
        <div className="space-y-1">
            {title && <DetailLabel>{title}</DetailLabel>}
            <ul className="space-y-1">
                {items.map((item, idx) => (
                    <li key={idx} className="whitespace-pre-wrap leading-relaxed">• {String(item)}</li>
                ))}
            </ul>
        </div>
    );
};

const codeHighlightStyle = {
    margin: 0,
    fontSize: '12px',
    padding: '0.875rem',
    background: '#0d1117',
    borderRadius: 0,
};

/* ============================================================================
 * Step renderers
 * ========================================================================== */

const ToolStep = ({ step }) => {
    const isDone = step.status === 'done';
    const reason = step.reason || step.args?.reason || step.args?.thought || null;
    return (
        <StepRow
            icon={isDone ? <Wrench size={12} /> : <RotateCw size={12} className="animate-spin" />}
            tone={isDone ? 'neutral' : 'accent'}
            title={step.tool}
            subtitle={reason}
        >
            <DetailPanel>
                <MonoBlock title="Arguments" text={JSON.stringify(step.args, null, 2)} maxHeight="max-h-40" />
                <MonoBlock title="Output" text={step.output} />
            </DetailPanel>
        </StepRow>
    );
};

const CodeStep = ({ step }) => (
    <StepRow
        icon={<Terminal size={12} />}
        tone="accent"
        title="Code execution"
        subtitle={step.language || 'python'}
        defaultOpen
    >
        <div className="overflow-hidden rounded-xl border border-line">
            <CodeHighlight
                language={step.language || 'python'}
                customStyle={codeHighlightStyle}
                showLineNumbers
            >
                {step.code}
            </CodeHighlight>
        </div>
    </StepRow>
);

const PY_STATUS_META = {
    inProgress: { tone: 'accent', label: 'Running', icon: <RotateCw size={12} className="animate-spin" /> },
    completed: { tone: 'success', label: 'Completed', icon: <SquareTerminal size={12} /> },
    failed: { tone: 'danger', label: 'Failed', icon: <XCircle size={12} /> },
    timedOut: { tone: 'warn', label: 'Timed out', icon: <AlertTriangle size={12} /> },
    interrupted: { tone: 'neutral', label: 'Interrupted', icon: <SquareTerminal size={12} /> },
};

const PythonExecutionStep = ({ step }) => {
    const statusMeta = PY_STATUS_META[step.status] || PY_STATUS_META.completed;
    const metaBits = [];
    if (Number.isFinite(step.durationMs) && step.durationMs > 0) metaBits.push(`${step.durationMs} ms`);
    metaBits.push(statusMeta.label);
    return (
        <StepRow
            icon={statusMeta.icon}
            tone={statusMeta.tone}
            title="Python"
            subtitle={step.reason || 'execute_python run'}
            meta={metaBits.join(' · ')}
            defaultOpen={step.status === 'failed' || step.status === 'timedOut'}
        >
            <DetailPanel>
                <PythonExecutionDetails item={step} />
            </DetailPanel>
        </StepRow>
    );
};

const ThoughtStep = ({ step }) => (
    <StepRow
        icon={<Sparkles size={12} />}
        tone="accent"
        title={step.agent || 'Thinking'}
        body={
            <div className="text-[13px] leading-relaxed text-ink-muted">
                <MarkdownContent
                    content={step.content}
                    components={{
                        p: (props) => <p className="mb-2 last:mb-0" {...props} />,
                        strong: (props) => <strong className="font-semibold text-ink" {...props} />,
                        code: ({ className, children, ...props }) =>
                            isInlineCode(className, children)
                                ? <code className="rounded border border-line bg-surface-muted px-1 py-0.5 font-mono text-[11px] text-ink" {...props}>{children}</code>
                                : <code className={className} {...props}>{children}</code>,
                    }}
                />
            </div>
        }
    />
);

const SubagentThoughtStep = ({ step }) => {
    const typeLabel = String(step.subagent_type || 'general').trim() || 'general';
    return (
        <StepRow
            icon={<Box size={12} />}
            tone="accent"
            title={`Subagent · ${typeLabel}`}
            subtitle={String(step.subagent_id || '').trim() || null}
            body={
                <div className="text-[13px] leading-relaxed text-ink-muted">
                    <MarkdownContent
                        content={step.content || ''}
                        components={{
                            p: (props) => <p className="mb-2 last:mb-0" {...props} />,
                            strong: (props) => <strong className="font-semibold text-ink" {...props} />,
                            code: ({ className, children, ...props }) =>
                                isInlineCode(className, children)
                                    ? <code className="rounded border border-line bg-surface-muted px-1 py-0.5 font-mono text-[11px] text-ink" {...props}>{children}</code>
                                    : <code className={className} {...props}>{children}</code>,
                        }}
                    />
                </div>
            }
        />
    );
};

const SUBAGENT_STATUS = {
    running: { tone: 'accent', icon: <RotateCw size={12} className="animate-spin" />, label: 'running' },
    completed: { tone: 'success', icon: <CheckCircle2 size={12} />, label: 'completed' },
    needs_input: { tone: 'warn', icon: <HelpCircle size={12} />, label: 'needs input' },
    failed: { tone: 'danger', icon: <XCircle size={12} />, label: 'failed' },
};

const SubagentLifecycleStep = ({ step }) => {
    const status = String(step.status || 'running').trim().toLowerCase();
    const meta = SUBAGENT_STATUS[status] || { tone: 'neutral', icon: <Box size={12} />, label: status || 'unknown' };
    const findings = Array.isArray(step.findings) ? step.findings : [];
    const artifactRefs = Array.isArray(step.artifact_refs) ? step.artifact_refs : [];
    const proposedStateUpdates = hasPlainObjectKeys(step.proposed_state_updates) ? step.proposed_state_updates : {};
    const reviewPayload = hasPlainObjectKeys(proposedStateUpdates.proposal_review)
        ? proposedStateUpdates.proposal_review
        : null;
    const summaryLine = String(step.summary || step.task || '').trim();
    return (
        <StepRow
            icon={meta.icon}
            tone={meta.tone}
            title={`Subagent · ${String(step.subagent_type || 'general')}`}
            subtitle={summaryLine || String(step.subagent_id || '')}
            meta={meta.label}
            defaultOpen={status === 'needs_input' || status === 'failed'}
        >
            <DetailPanel>
                <KVList entries={[
                    ['subagent_id', step.subagent_id],
                    ['parent_agent_id', step.parent_agent_id],
                    ['started_at', step.started_at],
                    ['finished_at', step.finished_at],
                    ['result_schema_submitted', step.result_submitted ? 'true' : ''],
                ]} />
                {step.task && (
                    <div className="space-y-1">
                        <DetailLabel>Task</DetailLabel>
                        <div className="whitespace-pre-wrap text-sm leading-relaxed text-ink">{step.task}</div>
                    </div>
                )}
                {summaryLine && step.task !== summaryLine && (
                    <div className="space-y-1">
                        <DetailLabel>Summary</DetailLabel>
                        <div className="whitespace-pre-wrap text-sm leading-relaxed text-ink">{summaryLine}</div>
                    </div>
                )}
                {reviewPayload && (
                    <div className="space-y-1.5">
                        <DetailLabel>Proposal review</DetailLabel>
                        <KVList entries={[
                            ['decision', reviewPayload.decision],
                            ['confidence', typeof reviewPayload.confidence === 'number' ? reviewPayload.confidence.toFixed(2) : ''],
                        ]} />
                        {reviewPayload.reviewer_feedback && (
                            <MonoBlock text={reviewPayload.reviewer_feedback} />
                        )}
                        <BulletList title="Implementation risks" items={reviewPayload.implementation_risks} />
                    </div>
                )}
                <BulletList title="Findings" items={findings} />
                {step.needs_input_question && (
                    <div className="space-y-1">
                        <DetailLabel>Needs input</DetailLabel>
                        <div className="whitespace-pre-wrap rounded-lg border border-amber-200 bg-amber-50/60 p-2.5 text-sm leading-relaxed text-ink dark:border-amber-500/30 dark:bg-amber-500/5">
                            {step.needs_input_question}
                        </div>
                    </div>
                )}
                <JsonBlock title="Artifact refs" value={artifactRefs} />
                <JsonBlock title="Proposed state updates" value={proposedStateUpdates} />
            </DetailPanel>
        </StepRow>
    );
};

const TerminalCommandStep = ({ step }) => {
    const success = !!step.success;
    const timedOut = !!step.timed_out;
    const tone = timedOut ? 'warn' : success ? 'neutral' : 'danger';
    const metaBits = [];
    if (step.exit_code != null) metaBits.push(`exit ${step.exit_code}`);
    if (step.duration_sec != null) metaBits.push(`${step.duration_sec}s`);
    if (timedOut) metaBits.push('timed out');
    return (
        <StepRow
            icon={<Terminal size={12} />}
            tone={tone}
            title="Terminal"
            subtitle={String(step.command || '').trim() || 'command'}
            meta={metaBits.join(' · ')}
            defaultOpen={!success || timedOut}
        >
            <DetailPanel>
                <KVList entries={[
                    ['cwd', step.cwd],
                    ['timeout_sec', step.timeout_sec],
                    ['clone_destination', step.clone_destination],
                ]} />
                <JsonBlock title="argv" value={Array.isArray(step.argv) ? step.argv : []} />
                <MonoBlock title="Error" text={step.error} tone="error" />
                <MonoBlock title="stdout" text={step.stdout} />
                <MonoBlock title="stderr" text={step.stderr} tone="stderr" />
                <MonoBlock title="Supported commands" text={step.supported_commands_summary} />
                <MonoBlock title="Usage hint" text={step.usage_hint} />
                <JsonBlock title="Supported commands detail" value={step.supported_commands} />
            </DetailPanel>
        </StepRow>
    );
};

const LogStep = ({ step }) => (
    <div className="relative flex items-center gap-2.5 py-1">
        <span className="relative z-10 ml-[9px] mr-[5px] h-1.5 w-1.5 shrink-0 rounded-full border border-line-strong bg-surface" />
        <span className="truncate font-mono text-[11px] text-ink-soft">{step.message}</span>
    </div>
);

const ImageStep = ({ step }) => (
    <StepRow
        icon={<ImageIcon size={12} />}
        tone="accent"
        title="Generated image"
        subtitle={step.filename}
        body={
            <div className="mt-1 inline-block overflow-hidden rounded-xl border border-line bg-white shadow-xs">
                <img src={step.src} alt={step.filename} className="max-h-[400px] w-auto" />
            </div>
        }
    />
);

const TheorySelectionStep = ({ step }) => (
    <StepRow
        icon={<Target size={12} />}
        tone="accent"
        title="Theory selection"
        subtitle={step.model_family}
        defaultOpen
    >
        <DetailPanel>
            {step.reasoning && (
                <div className="space-y-1">
                    <DetailLabel>Reasoning</DetailLabel>
                    <p className="text-sm leading-relaxed text-ink">{step.reasoning}</p>
                </div>
            )}
            {step.theory_support && (
                <div className="space-y-1">
                    <DetailLabel>Theory support</DetailLabel>
                    <p className="text-sm italic leading-relaxed text-ink-muted">{step.theory_support}</p>
                </div>
            )}
            <JsonBlock title="Configuration" value={step.config} />
        </DetailPanel>
    </StepRow>
);

const ReportStep = ({ step }) => (
    <StepRow
        icon={<FileText size={12} />}
        tone="success"
        title="Report generated"
        subtitle={step.filename}
        body={
            <a
                href={step.url}
                target="_blank"
                rel="noopener noreferrer"
                className="mt-1 inline-flex items-center gap-1.5 rounded-lg border border-line bg-surface px-3 py-1.5 text-xs font-medium text-ink shadow-xs transition-all hover:border-line-strong hover:shadow-sm active:scale-[0.98] focus:outline-none focus-visible:ring-2 focus-visible:ring-accent"
            >
                Open report
                <ExternalLink size={12} className="text-ink-soft" />
            </a>
        }
    />
);

const STAGE_LABELS = {
    preprocessing: 'Preprocessing',
    theory_selection: 'Theory selection',
    training: 'Training',
    downstream: 'Downstream',
    report: 'Report',
    complete: 'Complete',
};

const StageStep = ({ step }) => (
    <StepRow
        icon={<Layers size={12} />}
        tone="accent"
        title={STAGE_LABELS[step.stage] || String(step.stage || 'Stage').replace(/_/g, ' ')}
        subtitle={step.description}
    />
);

const TrainingProgressStep = ({ step }) => {
    const progress = Number(step.progress ?? 0);
    const clamped = Math.min(1, Math.max(0, Number.isFinite(progress) ? progress : 0));
    const percentage = (clamped * 100).toFixed(1);
    const isComplete = clamped >= 1;
    return (
        <StepRow
            icon={isComplete ? <CheckCircle2 size={12} /> : <BarChart3 size={12} />}
            tone={isComplete ? 'success' : 'accent'}
            title={isComplete ? 'Training complete' : 'Training'}
            subtitle={step.message}
            meta={`${percentage}%`}
            body={
                <div className="mt-0.5 h-1 w-full max-w-md overflow-hidden rounded-full bg-surface-subtle">
                    <div
                        className={cn(
                            'h-full rounded-full transition-all duration-300 ease-out',
                            isComplete ? 'bg-emerald-500' : 'bg-accent',
                        )}
                        style={{ width: `${percentage}%` }}
                    />
                </div>
            }
        />
    );
};

const ContextCompactedStep = ({ step }) => {
    const before = Number(step.before_tokens || 0);
    const after = Number(step.after_tokens || 0);
    const saved = Math.max(0, before - after);
    const summaryText = String(step.summary_text || step.summary_core || '').trim();
    const detail = summaryText ? (
        <DetailPanel>
            <KVList entries={[
                ['reason', step.reason || 'auto'],
                ['mode', step.mode],
                ['removed_messages', typeof step.removed_messages === 'number' ? step.removed_messages : ''],
                ['dropped_dangling_tool_results', step.dropped_dangling_tool_results > 0 ? step.dropped_dangling_tool_results : ''],
            ]} />
            <MonoBlock title="Compaction summary" text={summaryText} />
        </DetailPanel>
    ) : null;
    return (
        <StepRow
            icon={<Database size={12} />}
            title="Context compacted"
            subtitle={`${before.toLocaleString()} → ${after.toLocaleString()} tokens`}
            meta={`saved ${saved.toLocaleString()}`}
        >
            {detail}
        </StepRow>
    );
};

const ContextMicrocompactedStep = ({ step }) => {
    const before = Number(step.before_tokens || 0);
    const after = Number(step.after_tokens || 0);
    const saved = Math.max(0, Number(step.tokens_saved || (before - after) || 0));
    return (
        <StepRow
            icon={<Feather size={12} />}
            title="Context trimmed"
            subtitle={`${before.toLocaleString()} → ${after.toLocaleString()} tokens · ${Number(step.compacted_tool_messages || 0)} tool messages`}
            meta={`saved ${saved.toLocaleString()}`}
        />
    );
};

const LlmUsageStep = ({ step }) => {
    const fmt = (v) => (Number.isFinite(v) ? v.toLocaleString() : 'n/a');
    const hit = (typeof step.cache_hit_rate === 'number' && Number.isFinite(step.cache_hit_rate))
        ? `${(step.cache_hit_rate * 100).toFixed(1)}% cache`
        : null;
    const hasResponseDetails = Boolean(step.response_text) ||
        (Array.isArray(step.response_tool_calls) && step.response_tool_calls.length > 0) ||
        step.response_phase || step.response_finish_reason;
    const detail = hasResponseDetails ? (
        <DetailPanel>
            <KVList entries={[
                ['agent', step.agent || 'agent'],
                ['prompt_tokens', fmt(step.prompt_tokens)],
                ['completion_tokens', fmt(step.completion_tokens)],
                ['cached_prompt_tokens', fmt(step.cached_prompt_tokens)],
                ['response_phase', step.response_phase],
                ['finish_reason', step.response_finish_reason],
            ]} />
            <MonoBlock title="Response text" text={step.response_text} />
            {Array.isArray(step.response_tool_calls) && step.response_tool_calls.length > 0 && (
                <MonoBlock title="Tool calls" text={JSON.stringify(step.response_tool_calls, null, 2)} />
            )}
        </DetailPanel>
    ) : null;
    return (
        <StepRow
            icon={<Cpu size={12} />}
            title="LLM usage"
            subtitle={`prompt ${fmt(step.prompt_tokens)} · completion ${fmt(step.completion_tokens)} · total ${fmt(step.total_tokens)}`}
            meta={hit}
        >
            {detail}
        </StepRow>
    );
};

const ResumeWarningStep = ({ step }) => (
    <StepRow
        icon={<AlertTriangle size={12} />}
        tone="warn"
        title="Resume warning"
        body={
            <div className="whitespace-pre-wrap text-xs leading-relaxed text-amber-700 dark:text-amber-300">
                {step.message}
            </div>
        }
    />
);

const PLAN_ITEM_ICONS = {
    done: <CheckCircle2 size={13} className="text-emerald-600 dark:text-emerald-400" />,
    completed: <CheckCircle2 size={13} className="text-emerald-600 dark:text-emerald-400" />,
    in_progress: <RotateCw size={13} className="text-accent" />,
    active: <RotateCw size={13} className="text-accent" />,
};

const PlanUpdatedStep = ({ step }) => (
    <StepRow
        icon={<ListTree size={12} />}
        tone="accent"
        title="Plan updated"
        subtitle={step.explanation || `${step.scope || 'planner'} · ${step.source || 'unknown'}`}
        defaultOpen
    >
        <DetailPanel>
            {step.explanation && (
                <p className="text-xs leading-relaxed text-ink">{step.explanation}</p>
            )}
            {step.plan?.length > 0 && (
                <ol className="space-y-1.5">
                    {step.plan.map((item, idx) => {
                        const status = String(item.status || '').toLowerCase();
                        const isDone = status === 'done' || status === 'completed';
                        return (
                            <li key={idx} className="flex items-start gap-2">
                                <span className="mt-0.5 shrink-0">
                                    {PLAN_ITEM_ICONS[status] || <Circle size={13} className="text-line-strong" />}
                                </span>
                                <span className={cn(
                                    'text-xs leading-relaxed',
                                    isDone ? 'text-ink-soft line-through decoration-line-strong' : 'text-ink',
                                )}>
                                    {item.step}
                                </span>
                            </li>
                        );
                    })}
                </ol>
            )}
        </DetailPanel>
    </StepRow>
);

const TOOL_LIFECYCLE_LABELS = {
    tool_candidate_proposed: 'Tool candidate proposed',
    tool_refined: 'Tool candidate refined',
    tool_gate_passed: 'Generalization gate passed',
    tool_gate_failed: 'Generalization gate failed',
    tool_activation_pending: 'Tool pending activation',
    tool_activated: 'Tool activated',
    tool_activation_mode_updated: 'Activation mode updated',
    tool_review_updated: 'Tool review updated',
};

const ToolLifecycleStep = ({ step }) => {
    const tone = step.event_type === 'tool_gate_failed'
        ? 'danger'
        : (step.event_type === 'tool_gate_passed' || step.event_type === 'tool_activated')
            ? 'success'
            : 'neutral';
    const data = step.data || {};
    return (
        <StepRow
            icon={<Wrench size={12} />}
            tone={tone}
            title={TOOL_LIFECYCLE_LABELS[step.event_type] || step.event_type}
            subtitle={data.name || data.tool_name || null}
        >
            <DetailPanel>
                <MonoBlock text={JSON.stringify(data, null, 2)} />
            </DetailPanel>
        </StepRow>
    );
};

const SKILL_EVENT_LABELS = {
    skill_loaded: 'Skill loaded',
    skill_unloaded: 'Skill unloaded',
    skill_visibility_updated: 'Skill visibility updated',
    algorithm_visibility_updated: 'Algorithm visibility updated',
    planner_auto_skills_updated: 'Auto skills updated',
    skills_context_updated: 'Skills context updated',
};

const SkillEventStep = ({ step }) => {
    const data = step.data || {};
    const domain = data.domain || data.scope || 'agent';
    const loaded = Array.isArray(data.loaded) ? data.loaded : [];
    const auto = Array.isArray(data.auto) ? data.auto : [];
    const mentioned = Array.isArray(data.mentioned) ? data.mentioned : [];
    const injected = Array.isArray(data.injected) ? data.injected : [];
    const autoRoutes = Array.isArray(data.auto_routes) ? data.auto_routes : [];
    const subtitle = data.name || data.algorithm || (loaded.length ? loaded.join(', ') : domain);
    return (
        <StepRow
            icon={<BookOpen size={12} />}
            title={SKILL_EVENT_LABELS[step.event_type] || step.event_type}
            subtitle={subtitle}
        >
            <DetailPanel>
                <KVList entries={[
                    ['domain', domain],
                    ['name', data.name],
                    ['algorithm', data.algorithm],
                    ['path', data.path],
                    ['exposed', typeof data.exposed === 'boolean' ? String(data.exposed) : ''],
                    ['loaded', loaded.join(', ')],
                    ['auto', auto.join(', ')],
                    ['mentioned', mentioned.join(', ')],
                    ['injected', injected.join(', ')],
                ]} />
                {autoRoutes.length > 0 && (
                    <BulletList
                        title="Auto routes"
                        items={autoRoutes.map((item) => `${item.name}: ${item.reason}`)}
                    />
                )}
            </DetailPanel>
        </StepRow>
    );
};

const WORKSPACE_EVENT_LABELS = {
    planner_algorithm_workspace_initialized: 'Algorithm workspace initialized',
    workspace_tree_listed: 'Workspace tree listed',
    workspace_file_read: 'Workspace file read',
    workspace_file_written: 'Workspace file written',
    workspace_diff_previewed: 'Workspace diff previewed',
    workspace_patch_applied: 'Workspace patch applied',
    planner_workspace_updated: 'Workspace updated',
    planner_algorithm_workspace_activated: 'Algorithm workspace activated',
    algorithm_workspace_snapshot_created: 'Workspace snapshot created',
    algorithm_context_verified: 'Algorithm context verified',
    algorithm_context_gate_blocked: 'Algorithm context blocked',
    experiment_training_run_registered: 'Training run registered',
    algorithm_campaign_started: 'Campaign started',
    algorithm_campaign_reset: 'Campaign reset',
    algorithm_campaign_trial_started: 'Campaign trial started',
    algorithm_campaign_trial_decided: 'Campaign trial decided',
    algorithm_campaign_stage_gate_checked: 'Stage gate checked',
};

const WorkspaceEventStep = ({ step }) => {
    const data = step.data || {};
    const diffs = Array.isArray(data.diffs) ? data.diffs : [];
    const diffText = typeof data.diff === 'string' ? data.diff : '';
    const activeAlgorithm = data.active_algorithm_id || data.active_algorithm?.algorithm_id || data.context?.algorithm_id || '';
    const targetAlgorithm = data.target_algorithm_id || data.algorithm_id || '';
    const proposalId = data.proposal_id || data.active_proposal_id || data.context?.proposal_id || '';
    const snapshotId = data.snapshot_id || data.active_workspace_snapshot_id || data.context?.active_snapshot_id || '';
    const dirtyValue = typeof data.dirty_since_snapshot === 'boolean'
        ? data.dirty_since_snapshot
        : (typeof data.context?.dirty_since_snapshot === 'boolean' ? data.context.dirty_since_snapshot : null);
    const tone = step.event_type === 'algorithm_context_gate_blocked' ? 'danger' : 'neutral';
    const subtitle = data.path
        || (Array.isArray(data.paths) && data.paths.length > 0 ? data.paths[0] : '')
        || targetAlgorithm
        || activeAlgorithm
        || data.decision
        || '';
    return (
        <StepRow
            icon={<FolderOpen size={12} />}
            tone={tone}
            title={WORKSPACE_EVENT_LABELS[step.event_type] || step.event_type}
            subtitle={subtitle}
            defaultOpen={tone === 'danger'}
        >
            <DetailPanel>
                <KVList entries={[
                    ['active_algorithm', activeAlgorithm],
                    ['target_algorithm', targetAlgorithm],
                    ['proposal_id', proposalId],
                    ['snapshot_id', snapshotId],
                    ['campaign_id', data.campaign_id],
                    ['stage', data.stage],
                    ['trial_id', data.trial_id],
                    ['decision', data.decision],
                    ['active_best_trial', data.active_best_trial_id],
                    ['restored_snapshot', data.restored_snapshot_id],
                    ['dirty_since_snapshot', dirtyValue !== null ? String(dirtyValue) : ''],
                    ['path', data.path],
                    ['reason', data.reason],
                ]} />
                {Array.isArray(data.paths) && data.paths.length > 0 && (
                    <BulletList title="Paths" items={data.paths} />
                )}
                <BulletList title="Decision reasons" items={data.decision_reasons} />
                <JsonBlock title="Metrics" value={data.metrics_summary} />
                <JsonBlock title="Checks" value={data.checks} />
                <BulletList title="Blockers" items={data.blockers} />
                <MonoBlock title="Preview" text={data.preview} />
                <MonoBlock title="Diff" text={diffText} maxHeight="max-h-80" />
                {diffs.map((item, idx) => (
                    <MonoBlock key={idx} title={`${item.kind}: ${item.path}`} text={item.diff} maxHeight="max-h-80" />
                ))}
            </DetailPanel>
        </StepRow>
    );
};

const FileEventStep = ({ step }) => {
    const data = step.data || {};
    const action = data.action || step.event_type || 'file_activity';
    return (
        <StepRow
            icon={<FileText size={12} />}
            title="File activity"
            subtitle={`${action}${data.path ? ` · ${data.path}` : ''}`}
        >
            <DetailPanel>
                <KVList entries={[
                    ['scope', data.scope || 'agent'],
                    ['action', action],
                    ['path', data.path],
                    ['pattern', data.pattern],
                    ['include', data.include],
                    ['start_line', typeof data.start_line === 'number' ? data.start_line : ''],
                    ['end_line', typeof data.end_line === 'number' ? data.end_line : ''],
                ]} />
                <MonoBlock title="Preview" text={data.preview} />
            </DetailPanel>
        </StepRow>
    );
};

/* ============================================================================
 * ProposalEventStep — the only elevated card on the rail. It can require an
 * explicit user decision (approve / revise / reject), so it earns a real
 * surface with clear actions. Still neutral: a single accent icon + status
 * pill, hairline borders, no color washes.
 * ========================================================================== */

const PROPOSAL_TITLES = {
    algorithm_proposal_submitted: 'Algorithm proposal submitted',
    algorithm_proposal_review_requested: 'Algorithm proposal review',
    algorithm_proposal_reviewed: 'Algorithm proposal reviewed',
    algorithm_proposal_gate_blocked: 'Algorithm proposal blocked',
    research_idea_submitted: 'Idea submitted',
    research_idea_review_requested: 'Idea review',
    research_idea_reviewed: 'Idea reviewed',
    research_idea_progress_updated: 'Idea progress updated',
    research_idea_linked_algorithm: 'Idea linked to algorithm',
};

const proposalStatusTone = (status, eventType) => {
    const s = String(status || '').toLowerCase();
    if (eventType === 'algorithm_proposal_gate_blocked') return 'danger';
    if (s.includes('pending')) return 'warn';
    if (s.includes('approved') || s.includes('accepted')) return 'success';
    if (s.includes('rejected')) return 'danger';
    return 'neutral';
};

const ProposalEventStep = ({ step, onProposalDecision, onIdeaDecision }) => {
    const data = step.data || {};
    const [feedback, setFeedback] = useState('');
    const [showFeedback, setShowFeedback] = useState(false);
    const [submitting, setSubmitting] = useState(false);
    const [localError, setLocalError] = useState('');
    const [localMessage, setLocalMessage] = useState('');

    const title = PROPOSAL_TITLES[step.event_type] || step.event_type;
    const isResearchIdea = String(data.idea_id || '').trim().length > 0;
    const proposalMarkdown = String(data.proposal_markdown || '').trim();
    const ideaMarkdown = String(data.idea_markdown || '').trim();
    const proposalObject = data.proposal && typeof data.proposal === 'object' ? data.proposal : null;
    const ideaObject = data.idea && typeof data.idea === 'object' ? data.idea : null;
    const markdown = proposalMarkdown || ideaMarkdown;
    const showActions = Boolean(
        (isResearchIdea ? typeof onIdeaDecision === 'function' : typeof onProposalDecision === 'function') &&
        String(isResearchIdea ? data.idea_id : data.algorithm_id || '').trim() &&
        (
            String(data.status || '').toLowerCase() === 'pending_user_review' ||
            step.event_type === 'algorithm_proposal_review_requested' ||
            step.event_type === 'research_idea_review_requested'
        )
    );

    const submitDecision = async (decision) => {
        if (!showActions || submitting) return;
        const entityId = String(isResearchIdea ? data.idea_id : data.algorithm_id || '').trim();
        if (!entityId) return;
        setSubmitting(true);
        setLocalError('');
        setLocalMessage('');
        try {
            const response = isResearchIdea
                ? await onIdeaDecision(entityId, decision, feedback)
                : await onProposalDecision(entityId, decision, feedback, String(data.proposal_id || '').trim());
            if (!response || response.status !== 'success') {
                const errMsg = response?.message || (isResearchIdea ? 'Research idea review failed.' : 'Proposal review failed.');
                setLocalError(errMsg);
            } else {
                const autoContinue = response?.auto_continue || {};
                const autoContinueStatus = String(autoContinue?.status || '').toLowerCase();
                if (autoContinueStatus && autoContinueStatus !== 'success' && autoContinueStatus !== 'skipped') {
                    const extra = autoContinue?.message ? ` Auto-continue: ${autoContinue.message}` : '';
                    setLocalError((response.message || `Proposal ${decision} submitted.`) + extra);
                } else {
                    const suffix = autoContinueStatus === 'success' ? ' Agent resumed automatically.' : '';
                    setLocalMessage((response.message || `Proposal ${decision} submitted.`) + suffix);
                }
                if (decision === 'approve') {
                    setShowFeedback(false);
                    setFeedback('');
                }
            }
        } catch (error) {
            setLocalError(String(error?.message || error || 'Proposal review failed.'));
        } finally {
            setSubmitting(false);
        }
    };

    const actionButton = 'inline-flex h-8 items-center justify-center gap-1.5 rounded-lg px-3 text-xs font-medium transition-all active:scale-[0.98] disabled:opacity-50 disabled:pointer-events-none focus:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 focus-visible:ring-offset-surface';

    return (
        <div className="my-2 overflow-hidden rounded-xl border border-line bg-surface shadow-sm">
            <div className="flex items-center justify-between gap-3 border-b border-line px-4 py-3">
                <div className="flex min-w-0 items-center gap-2.5">
                    <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-accent-soft/60 text-accent dark:text-accent-strong">
                        <Sparkles size={14} />
                    </span>
                    <div className="min-w-0">
                        <div className="truncate text-sm font-medium text-ink">{title}</div>
                        {data.title && (
                            <div className="truncate text-xs text-ink-soft">{data.title}</div>
                        )}
                    </div>
                </div>
                {data.status && (
                    <Pill tone={proposalStatusTone(data.status, step.event_type)} className="shrink-0">
                        {String(data.status).replace(/_/g, ' ')}
                    </Pill>
                )}
            </div>

            <div className="space-y-3 px-4 py-3 text-xs text-ink-muted">
                <KVList entries={[
                    ['algorithm_id', data.algorithm_id],
                    ['idea_id', data.idea_id],
                    ['review_mode', data.review_mode],
                    ['primary_track', data.primary_track],
                    ['alternate_track', data.alternate_track],
                    ['requires_user_review', typeof data.requires_user_review === 'boolean' ? String(data.requires_user_review) : ''],
                    ['decision', data.decision],
                    ['resolution_status', data.resolution_status],
                    ['execution_status', data.execution_status],
                    ['portfolio_status', data.portfolio_status],
                    ['linked_algorithm_id', data.linked_algorithm_id],
                    ['path', data.path],
                ]} />
                {data.progress_summary && <p className="leading-relaxed text-ink">{data.progress_summary}</p>}
                {data.reason && <p className="leading-relaxed text-ink">{data.reason}</p>}
                {data.question && <p className="leading-relaxed text-ink">{data.question}</p>}
                {data.review_feedback && (
                    <p className="leading-relaxed"><span className="font-medium text-ink">Feedback:</span> {data.review_feedback}</p>
                )}
                <BulletList title="Implementation risks" items={data.implementation_risks} />

                {markdown && (
                    <div className="space-y-1">
                        <DetailLabel>{isResearchIdea && !proposalMarkdown ? 'Idea' : 'Proposal'}</DetailLabel>
                        <div className="prose prose-sm dark:prose-invert max-h-96 max-w-none overflow-auto custom-scrollbar rounded-xl border border-line bg-surface-muted/60 p-3.5">
                            <MarkdownContent content={markdown} />
                        </div>
                    </div>
                )}
                {!proposalMarkdown && proposalObject && <JsonBlock title="Proposal" value={proposalObject} />}
                {!ideaMarkdown && ideaObject && <JsonBlock title="Idea" value={ideaObject} />}

                {showActions && (
                    <div className="space-y-2.5 border-t border-line pt-3">
                        <DetailLabel>Your review</DetailLabel>
                        <div className="flex flex-wrap items-center gap-2">
                            <button
                                type="button"
                                disabled={submitting}
                                onClick={() => submitDecision('approve')}
                                className={cn(actionButton, 'bg-emerald-600 text-white shadow-sm hover:bg-emerald-500')}
                            >
                                <CheckCircle2 size={13} /> Approve
                            </button>
                            <button
                                type="button"
                                disabled={submitting}
                                onClick={() => submitDecision('revise')}
                                className={cn(actionButton, 'border border-line bg-surface text-ink hover:border-line-strong hover:bg-surface-muted')}
                            >
                                Request revise
                            </button>
                            <button
                                type="button"
                                disabled={submitting}
                                onClick={() => submitDecision('reject')}
                                className={cn(actionButton, 'border border-red-200 bg-surface text-red-600 hover:bg-red-50 dark:border-red-500/30 dark:text-red-400 dark:hover:bg-red-500/10')}
                            >
                                <XCircle size={13} /> Reject
                            </button>
                            <button
                                type="button"
                                disabled={submitting}
                                onClick={() => setShowFeedback(!showFeedback)}
                                className={cn(actionButton, 'text-ink-muted hover:bg-surface-muted hover:text-ink')}
                            >
                                {showFeedback ? 'Hide feedback' : 'Add feedback'}
                            </button>
                        </div>
                        {showFeedback && (
                            <textarea
                                value={feedback}
                                onChange={(e) => setFeedback(e.target.value)}
                                placeholder="Optional feedback for revise / reject / approve…"
                                className="block min-h-[90px] w-full rounded-lg border border-line bg-surface p-2.5 text-sm text-ink placeholder:text-ink-soft transition-colors hover:border-line-strong focus:border-accent focus:outline-none focus:ring-2 focus:ring-accent/20"
                            />
                        )}
                        {localError && (
                            <p className="text-xs leading-relaxed text-red-600 dark:text-red-400">{localError}</p>
                        )}
                        {localMessage && (
                            <p className="text-xs leading-relaxed text-emerald-700 dark:text-emerald-400">{localMessage}</p>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
};

/* ============================================================================
 * StepItem — dispatch table
 * ========================================================================== */

function StepItemImpl({ step, onProposalDecision, onIdeaDecision }) {
    switch (step.type) {
        case 'tool': return <ToolStep step={step} />;
        case 'terminal_command': return <TerminalCommandStep step={step} />;
        case 'python_execution': return <PythonExecutionStep step={step} />;
        case 'code': return <CodeStep step={step} />;
        case 'thought': return <ThoughtStep step={step} />;
        case 'subagent_thought': return <SubagentThoughtStep step={step} />;
        case 'subagent_lifecycle': return <SubagentLifecycleStep step={step} />;
        case 'log': return <LogStep step={step} />;
        case 'image': return <ImageStep step={step} />;
        case 'theory_selection': return <TheorySelectionStep step={step} />;
        case 'report': return <ReportStep step={step} />;
        case 'stage': return <StageStep step={step} />;
        case 'training_progress': return <TrainingProgressStep step={step} />;
        case 'context_compacted': return <ContextCompactedStep step={step} />;
        case 'context_microcompacted': return <ContextMicrocompactedStep step={step} />;
        case 'llm_usage': return <LlmUsageStep step={step} />;
        case 'resume_warning': return <ResumeWarningStep step={step} />;
        case 'plan_updated': return <PlanUpdatedStep step={step} />;
        case 'tool_lifecycle': return <ToolLifecycleStep step={step} />;
        case 'skill_event': return <SkillEventStep step={step} />;
        case 'workspace_event': return <WorkspaceEventStep step={step} />;
        case 'proposal_event': return <ProposalEventStep step={step} onProposalDecision={onProposalDecision} onIdeaDecision={onIdeaDecision} />;
        case 'file_event': return <FileEventStep step={step} />;
        default: return null;
    }
}

export const StepItem = React.memo(
    StepItemImpl,
    (prev, next) =>
        prev.step === next.step &&
        prev.onProposalDecision === next.onProposalDecision &&
        prev.onIdeaDecision === next.onIdeaDecision
);
