import React, { useState, useEffect } from 'react';
import { Virtuoso } from 'react-virtuoso';
import { CheckCircle2, ChevronDown, ExternalLink, RotateCw } from 'lucide-react';
import { StepItem, StepRail } from './ActivityItem';
import { getStepKey } from '../lib/timelineReducer';
import { cn } from './ui/cn';

const VIRTUALIZE_STEPS_THRESHOLD = 80;
const VIRTUALIZED_STEPS_HEIGHT = 'min(760px, 62vh)';

const STAGE_LABELS = {
    preprocessing: 'Preprocessing',
    theory_selection: 'Theory selection',
    training: 'Training',
    downstream: 'Downstream',
    report: 'Report',
    complete: 'Complete',
};

/** "25s" / "4m 12s" from first → last step timestamps. */
const formatWorkDuration = (steps) => {
    if (!Array.isArray(steps) || steps.length < 2) return null;
    const first = Date.parse(steps[0]?.timestamp || '');
    const last = Date.parse(steps[steps.length - 1]?.timestamp || '');
    if (!Number.isFinite(first) || !Number.isFinite(last) || last <= first) return null;
    const sec = Math.max(1, Math.round((last - first) / 1000));
    if (sec < 60) return `${sec}s`;
    const m = Math.floor(sec / 60);
    const s = sec % 60;
    return s ? `${m}m ${s}s` : `${m}m`;
};

/**
 * ProcessBlock — one agent turn's activity trace.
 *
 * Collapsed, it is a single quiet disclosure line ("Worked for 25s · 29
 * steps") rather than a card, so the assistant's final answer below reads as
 * the hero content. Expanding reveals the hairline activity rail.
 */
function ProcessBlockImpl({ steps, isComplete, isActive, currentStage, stageDescription, onProposalDecision, onIdeaDecision }) {
    // Start expanded if not complete, collapsed if complete.
    const [isExpanded, setIsExpanded] = useState(!isComplete);

    const hasPendingReview = steps.some((step) => {
        if (step?.type !== 'proposal_event') return false;
        const eventType = String(step.event_type || '');
        const status = String(step.data?.status || '').toLowerCase();
        return (
            eventType === 'algorithm_proposal_review_requested' ||
            eventType === 'research_idea_review_requested' ||
            status === 'pending_user_review'
        );
    });
    // Live turns and unanswered reviews stay open; once the turn completes the
    // trace always collapses to a single line (a "Review needed" badge keeps
    // pending decisions discoverable).
    const effectiveExpanded = isExpanded || (!isComplete && (isActive || hasPendingReview));

    // Auto-collapse on completion after a brief settle.
    useEffect(() => {
        if (isComplete) {
            const timer = setTimeout(() => setIsExpanded(false), 600);
            return () => clearTimeout(timer);
        }
    }, [isComplete]);

    const reportStep = steps.find((step) => step.type === 'report');
    const stepsToShow = steps.filter((step) => step.type !== 'report');

    const displayStage = currentStage || (isComplete ? 'complete' : 'analyzing');
    const stageLabel = STAGE_LABELS[displayStage] || (currentStage ? currentStage.replace(/_/g, ' ') : null);
    const workDuration = formatWorkDuration(steps);
    const displayTitle = isComplete
        ? (workDuration ? `Worked for ${workDuration}` : 'Completed')
        : hasPendingReview
            ? 'Waiting for your review'
            : (stageDescription || stageLabel || 'Working…');

    const shouldVirtualizeSteps = stepsToShow.length > VIRTUALIZE_STEPS_THRESHOLD;
    const showStepLogHeader = stepsToShow.length > 15;

    const activeThinkingIndicator = (
        <div className="relative flex items-center gap-2.5 py-2 pl-9 text-xs text-ink-soft">
            <div className="flex gap-1">
                <span className="h-1.5 w-1.5 rounded-full bg-accent thinking-dot" style={{ animationDelay: '0ms' }} />
                <span className="h-1.5 w-1.5 rounded-full bg-accent thinking-dot" style={{ animationDelay: '150ms' }} />
                <span className="h-1.5 w-1.5 rounded-full bg-accent thinking-dot" style={{ animationDelay: '300ms' }} />
            </div>
            <span>Thinking…</span>
        </div>
    );

    return (
        <div className="px-2">
            {/* Disclosure line */}
            <div className="flex items-center gap-2">
                <button
                    type="button"
                    onClick={() => setIsExpanded(!effectiveExpanded)}
                    aria-expanded={effectiveExpanded}
                    className={cn(
                        'group -mx-2 flex min-w-0 items-center gap-2 rounded-lg px-2 py-1.5 text-left transition-colors',
                        'hover:bg-surface-muted/80 focus:outline-none focus-visible:ring-2 focus-visible:ring-accent',
                    )}
                >
                    {isComplete ? (
                        <CheckCircle2 size={14} className="shrink-0 text-ink-soft" />
                    ) : (
                        <RotateCw size={14} className="shrink-0 animate-spin text-accent" />
                    )}
                    <span
                        className={cn(
                            'truncate text-[13px]',
                            isComplete
                                ? 'text-ink-muted'
                                : (isActive && !hasPendingReview
                                    ? 'font-medium text-shimmer'
                                    : 'font-medium text-ink'),
                        )}
                    >
                        {displayTitle}
                    </span>
                    {!isComplete && stageLabel && !hasPendingReview && (
                        <span className="shrink-0 rounded border border-line px-1.5 py-px text-[11px] font-medium text-ink-soft">
                            {stageLabel}
                        </span>
                    )}
                    <span className="shrink-0 text-xs text-ink-soft">
                        · {steps.length} step{steps.length !== 1 ? 's' : ''}
                    </span>
                    {hasPendingReview && (
                        <span className="shrink-0 rounded-full border border-amber-200 bg-amber-50 px-2 py-px text-[10px] font-medium text-amber-700 dark:border-amber-500/30 dark:bg-amber-500/10 dark:text-amber-300">
                            Review needed
                        </span>
                    )}
                    <ChevronDown
                        size={13}
                        className={cn(
                            'shrink-0 text-ink-soft transition-transform duration-200',
                            effectiveExpanded && 'rotate-180',
                        )}
                    />
                </button>

                {isComplete && reportStep && (
                    <a
                        href={reportStep.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="inline-flex shrink-0 items-center gap-1 rounded-md px-1.5 py-1 text-xs font-medium text-accent transition-colors hover:bg-accent-soft/50 hover:text-accent-strong focus:outline-none focus-visible:ring-2 focus-visible:ring-accent"
                    >
                        Open report
                        <ExternalLink size={11} />
                    </a>
                )}
            </div>

            {/* Activity rail */}
            {effectiveExpanded && (
                <div className="mt-1 pl-1 animate-fade-in">
                    {shouldVirtualizeSteps ? (
                        <div className="overflow-hidden rounded-xl border border-line bg-surface">
                            <div className="border-b border-line px-3 py-2 text-[11px] font-medium text-ink-soft">
                                Virtualized step log · {stepsToShow.length} items
                            </div>
                            <Virtuoso
                                style={{ height: VIRTUALIZED_STEPS_HEIGHT }}
                                data={stepsToShow}
                                followOutput={isActive && !isComplete ? 'auto' : false}
                                atBottomThreshold={96}
                                initialTopMostItemIndex={{ index: Math.max(0, stepsToShow.length - 1), align: 'end' }}
                                increaseViewportBy={{ top: 320, bottom: 560 }}
                                computeItemKey={(index, step) => getStepKey(index, step)}
                                itemContent={(_, step) => (
                                    <div className="px-3 py-px">
                                        <StepItem step={step} onProposalDecision={onProposalDecision} onIdeaDecision={onIdeaDecision} />
                                    </div>
                                )}
                                components={{
                                    Footer: () => (
                                        isActive && !isComplete ? (
                                            <div className="px-3">{activeThinkingIndicator}</div>
                                        ) : <div className="h-2" />
                                    ),
                                }}
                            />
                        </div>
                    ) : showStepLogHeader ? (
                        <div className="overflow-hidden rounded-xl border border-line bg-surface">
                            <div className="border-b border-line px-3 py-2 text-[11px] font-medium text-ink-soft">
                                Step log · {stepsToShow.length} items
                            </div>
                            <StepRail className="px-3 py-2">
                                {stepsToShow.map((step, idx) => (
                                    <StepItem
                                        key={getStepKey(idx, step)}
                                        step={step}
                                        onProposalDecision={onProposalDecision}
                                        onIdeaDecision={onIdeaDecision}
                                    />
                                ))}
                                {isActive && !isComplete && activeThinkingIndicator}
                            </StepRail>
                        </div>
                    ) : (
                        <StepRail>
                            {stepsToShow.map((step, idx) => (
                                <StepItem
                                    key={getStepKey(idx, step)}
                                    step={step}
                                    onProposalDecision={onProposalDecision}
                                    onIdeaDecision={onIdeaDecision}
                                />
                            ))}
                            {isActive && !isComplete && activeThinkingIndicator}
                        </StepRail>
                    )}
                </div>
            )}
        </div>
    );
}

export const ProcessBlock = React.memo(
    ProcessBlockImpl,
    (prev, next) =>
        prev.steps === next.steps &&
        prev.isComplete === next.isComplete &&
        prev.isActive === next.isActive &&
        prev.currentStage === next.currentStage &&
        prev.stageDescription === next.stageDescription &&
        prev.onProposalDecision === next.onProposalDecision &&
        prev.onIdeaDecision === next.onIdeaDecision
);
