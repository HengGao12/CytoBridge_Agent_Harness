import React, { useState } from 'react';
import { Bot, AlertTriangle, Info, Check, Copy, RotateCw } from 'lucide-react';
import { ChatMarkdown } from './ChatMarkdown';
import { StepItem } from './ActivityItem';
import { Pill } from './ui';

/**
 * Small, presentational timeline items. Each item receives a single timeline
 * `item` shape from `lib/timelineReducer.js` and renders one row in the chat
 * column. Process / Python / Proposal / Assistant items are heavier and live
 * in their own files.
 */


const formatMessageTime = (timestamp) => {
    if (!timestamp) return '';
    const date = new Date(timestamp);
    if (Number.isNaN(date.getTime())) return '';
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
};

export const UserTimelineItem = React.memo(function UserTimelineItem({ content, attachments, timestamp }) {
    const timeLabel = formatMessageTime(timestamp);
    return (
        <div className="group flex flex-col items-end px-2 animate-fade-in">
            <div className="user-chat-bubble max-w-[42rem] rounded-2xl rounded-br-md border border-line bg-surface-muted px-4 py-3 text-ink">
                <ChatMarkdown content={content} variant="user" />
                {Array.isArray(attachments) && attachments.length > 0 && (
                    <div className="mt-3 grid grid-cols-1 sm:grid-cols-2 gap-2">
                        {attachments.map((attachment, aidx) => (
                            <div
                                key={aidx}
                                className="overflow-hidden rounded-lg border border-line bg-surface"
                            >
                                <img
                                    src={attachment.data_url}
                                    alt={attachment.file_name || `attachment-${aidx + 1}`}
                                    className="max-h-64 w-full object-contain bg-surface"
                                />
                                <div className="px-2 py-1 text-[10px] font-mono text-ink-muted truncate bg-surface-muted">
                                    {attachment.file_name || `image_${aidx + 1}`}
                                </div>
                            </div>
                        ))}
                    </div>
                )}
            </div>
            {timeLabel && (
                <div className="mt-1 pr-1 text-[10px] font-mono text-ink-soft opacity-0 transition-opacity duration-150 group-hover:opacity-100">
                    {timeLabel}
                </div>
            )}
        </div>
    );
});

export const AssistantTimelineItem = React.memo(function AssistantTimelineItem({ content, timestamp }) {
    const [copied, setCopied] = useState(false);

    const handleCopy = async () => {
        try {
            await navigator.clipboard.writeText(String(content || ''));
            setCopied(true);
            window.setTimeout(() => setCopied(false), 1600);
        } catch {
            /* clipboard unavailable — ignore */
        }
    };

    return (
        <div className="group px-2 animate-fade-in">
            <div className="mb-2 flex items-center gap-2">
                <span className="flex h-5 w-5 items-center justify-center rounded-md bg-accent text-white">
                    <Bot size={12} strokeWidth={2.25} />
                </span>
                <span className="text-xs font-semibold text-ink-muted">CellCompass</span>
                <button
                    type="button"
                    onClick={handleCopy}
                    aria-label={copied ? 'Copied' : 'Copy response'}
                    title="Copy response"
                    className="rounded-md p-1 text-ink-soft opacity-0 transition-all duration-150 group-hover:opacity-100 focus-visible:opacity-100 hover:bg-surface-muted hover:text-ink active:scale-95 focus:outline-none focus-visible:ring-2 focus-visible:ring-accent"
                >
                    {copied
                        ? <Check size={12} className="text-emerald-600 dark:text-emerald-400" />
                        : <Copy size={12} />}
                </button>
                {formatMessageTime(timestamp) && (
                    <span className="text-[10px] font-mono text-ink-soft opacity-0 transition-opacity duration-150 group-hover:opacity-100">
                        {formatMessageTime(timestamp)}
                    </span>
                )}
            </div>
            <div className="min-w-0">
                <ChatMarkdown content={content} variant="assistant" />
            </div>
        </div>
    );
});

export const ProposalTimelineItem = React.memo(function ProposalTimelineItem({
    item,
    onProposalDecision,
    onIdeaDecision,
}) {
    return (
        <div className="px-2">
            <StepItem
                step={item}
                onProposalDecision={onProposalDecision}
                onIdeaDecision={onIdeaDecision}
            />
        </div>
    );
});

export const ErrorTimelineItem = React.memo(function ErrorTimelineItem({ message, onRetry }) {
    return (
        <div className="flex justify-center my-3 px-2">
            <div className="inline-flex items-center gap-2 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-700 dark:border-red-500/30 dark:bg-red-500/10 dark:text-red-300">
                <AlertTriangle size={14} className="shrink-0" />
                <span>{message}</span>
                {onRetry && (
                    <button
                        type="button"
                        onClick={onRetry}
                        className="ml-1 inline-flex shrink-0 items-center gap-1 rounded-md border border-red-200 bg-white/60 px-2 py-1 font-medium text-red-700 transition-all hover:bg-white active:scale-[0.97] dark:border-red-500/30 dark:bg-red-500/10 dark:text-red-300 dark:hover:bg-red-500/20 focus:outline-none focus-visible:ring-2 focus-visible:ring-red-500"
                    >
                        <RotateCw size={11} />
                        Retry
                    </button>
                )}
            </div>
        </div>
    );
});

export const SystemTimelineItem = React.memo(function SystemTimelineItem({ message }) {
    return (
        <div className="flex justify-center my-3 px-2">
            <span className="text-[11px] font-medium text-ink-soft px-2 py-1 rounded-md bg-surface-muted border border-line">
                {message}
            </span>
        </div>
    );
});

export const ResumeDiagnosticsTimelineItem = React.memo(function ResumeDiagnosticsTimelineItem({ diagnostics }) {
    const d = diagnostics || {};
    return (
        <div className="px-2 my-3">
            <div className="mx-auto max-w-3xl rounded-xl border border-amber-200 bg-amber-50 p-4 text-amber-900 dark:border-amber-500/30 dark:bg-amber-500/10 dark:text-amber-100">
                <div className="flex items-center gap-2 text-sm font-semibold">
                    <Info size={14} />
                    Resume diagnostics
                </div>
                <div className="mt-1 flex flex-wrap gap-1.5 text-[11px] font-mono">
                    <Pill tone="warn">source: {d.source || 'snapshot'}</Pill>
                    <Pill tone="warn">migrated_v1: {String(!!d.migrated_from_v1)}</Pill>
                    <Pill tone={d.degraded ? 'danger' : 'success'}>
                        {d.degraded ? 'DEGRADED' : 'OK'}
                    </Pill>
                </div>
                {Array.isArray(d.warnings) && d.warnings.length > 0 && (
                    <ul className="mt-2 list-disc pl-5 space-y-0.5 text-xs">
                        {d.warnings.map((w, i) => (
                            <li key={i}>{w}</li>
                        ))}
                    </ul>
                )}
            </div>
        </div>
    );
});
