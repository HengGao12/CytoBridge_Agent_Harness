import React from 'react';
import {
    Bot,
    Cpu,
    BookText,
    ShieldCheck,
    BarChart3,
    Download,
    LineChart,
    ShieldAlert,
    ArrowUpRight,
    Sparkles,
} from 'lucide-react';
import { Card } from './ui';

/**
 * WelcomeScreen — empty-state landing inside the chat column.
 *
 * Calm two-column information layout on neutral surfaces with a single accent.
 * Sections stagger in with a quiet rise animation; cards lift slightly on
 * hover to signal interactivity without saturated treatments.
 */
const PILLARS = [
    {
        icon: Cpu,
        title: 'Pipeline Design & Sandbox',
        body: 'Propose bioinformatics algorithms, customize tools, and run Python sandboxes to automate single-cell fitting & enrichment.',
    },
    {
        icon: BookText,
        title: 'Literature & Skill Mining',
        body: 'Query arXiv / BioRxiv, parse references, and inject required bio-computational skills into the runtime on demand.',
    },
    {
        icon: ShieldCheck,
        title: 'Scientific Claim Validation',
        body: 'Enforce scientific quality gates and stop hooks. Validate outputs via feasibility checks, validation trials, and tuning cycles.',
    },
    {
        icon: BarChart3,
        title: 'Reports & Artifact Delivery',
        body: 'Compile run logs, plots, and fit parameters into polished summaries inside your output workspace.',
    },
];

const QUICK_PROMPTS = [
    {
        icon: Download,
        label: 'PBMC Pipeline',
        body: 'Design a marker-gene enrichment workflow for peripheral blood mononuclear cells.',
        prompt: 'Design a marker-gene enrichment pipeline for PBMC data.',
    },
    {
        icon: LineChart,
        label: 'Cell Annotation',
        body: 'Parse raw RNA-seq clusters and query literature to recommend cell types.',
        prompt: 'Inspect my current single-cell clustering script and suggest cell-type annotations based on literature.',
    },
    {
        icon: ShieldAlert,
        label: 'Claims Validation',
        body: 'Verify fitting parameters and algorithms against rigorous multi-phase scientific quality gates.',
        prompt: 'Run a multi-stage Claim Validation workflow on my fit results.',
    },
];

export function WelcomeScreen({ onQuickPrompt }) {
    return (
        <div className="mx-auto flex w-full max-w-3xl flex-col items-center px-6 py-16 sm:py-24">
            {/* Hero */}
            <div className="animate-rise-in relative mb-6">
                <div
                    aria-hidden="true"
                    className="absolute -inset-6 rounded-full bg-accent/15 blur-2xl"
                />
                <div className="relative flex h-16 w-16 items-center justify-center rounded-2xl bg-accent text-white shadow-md ring-1 ring-accent/20 ring-offset-4 ring-offset-surface">
                    <Bot size={30} strokeWidth={1.75} />
                </div>
            </div>
            <div
                className="animate-rise-in mb-4 inline-flex items-center gap-1.5 rounded-full border border-line bg-surface-muted px-3 py-1 text-[11px] font-medium text-ink-muted"
                style={{ animationDelay: '40ms' }}
            >
                <Sparkles size={11} className="text-accent" />
                Autonomous workspace agent
            </div>
            <h1
                className="animate-rise-in text-hero-gradient text-balance text-center text-3xl font-semibold tracking-tight sm:text-[2.75rem] sm:leading-[1.15]"
                style={{ animationDelay: '80ms' }}
            >
                What should we automate today?
            </h1>
            <p
                className="animate-rise-in mt-3 max-w-xl text-balance text-center text-sm leading-relaxed text-ink-soft sm:text-[15px]"
                style={{ animationDelay: '120ms' }}
            >
                CellCompass designs pipelines, mines literature, executes sandboxed
                analyses, and validates claims — end to end.
            </p>

            {/* Quick prompts */}
            <div className="animate-rise-in mt-10 w-full" style={{ animationDelay: '180ms' }}>
                <div className="mb-3 flex items-center gap-3">
                    <span className="text-xs font-medium text-ink-muted">
                        Quick start
                    </span>
                    <span className="h-px flex-1 bg-line" aria-hidden="true" />
                </div>
                <div className="grid grid-cols-1 gap-2.5 sm:grid-cols-3">
                    {QUICK_PROMPTS.map((qp) => (
                        <button
                            key={qp.label}
                            type="button"
                            onClick={() => onQuickPrompt(qp.prompt)}
                            className="group relative flex flex-col items-start rounded-xl border border-line bg-surface p-3.5 text-left transition-all duration-200 hover:-translate-y-0.5 hover:border-line-strong hover:shadow-md focus:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 focus-visible:ring-offset-surface"
                        >
                            <ArrowUpRight
                                size={14}
                                className="absolute right-3 top-3 text-ink-soft opacity-0 transition-all duration-200 group-hover:translate-x-0.5 group-hover:opacity-100"
                                aria-hidden="true"
                            />
                            <div className="mb-1.5 flex items-center gap-2 text-ink">
                                <span className="flex h-6 w-6 items-center justify-center rounded-md bg-accent-soft/60 text-accent dark:text-accent-strong">
                                    <qp.icon size={13} />
                                </span>
                                <span className="text-xs font-semibold">{qp.label}</span>
                            </div>
                            <span className="text-xs leading-relaxed text-ink-soft">{qp.body}</span>
                        </button>
                    ))}
                </div>
            </div>

            {/* Pillars */}
            <div className="animate-rise-in mt-9 w-full" style={{ animationDelay: '240ms' }}>
                <div className="mb-3 flex items-center gap-3">
                    <span className="text-xs font-medium text-ink-muted">
                        Capabilities
                    </span>
                    <span className="h-px flex-1 bg-line" aria-hidden="true" />
                </div>
                <div className="grid w-full grid-cols-1 gap-2.5 sm:grid-cols-2">
                    {PILLARS.map((pillar) => (
                        <Card key={pillar.title} padded={false} interactive className="p-4">
                            <div className="flex items-start gap-3">
                                <div className="mt-0.5 flex h-9 w-9 shrink-0 items-center justify-center rounded-lg border border-line bg-surface-muted text-ink-muted">
                                    <pillar.icon size={16} strokeWidth={1.75} />
                                </div>
                                <div className="min-w-0">
                                    <div className="text-sm font-medium text-ink">{pillar.title}</div>
                                    <p className="mt-1 text-xs leading-relaxed text-ink-soft">{pillar.body}</p>
                                </div>
                            </div>
                        </Card>
                    ))}
                </div>
            </div>

            {/* Footer hint */}
            <p
                className="animate-rise-in mt-10 text-center text-[11px] text-ink-soft"
                style={{ animationDelay: '300ms' }}
            >
                Type below to begin —{' '}
                <kbd className="rounded border border-line bg-surface-muted px-1.5 py-0.5 font-mono text-[10px] text-ink-muted">
                    Enter
                </kbd>{' '}
                to send,{' '}
                <kbd className="rounded border border-line bg-surface-muted px-1.5 py-0.5 font-mono text-[10px] text-ink-muted">
                    Shift + Enter
                </kbd>{' '}
                for a new line.
            </p>
        </div>
    );
}
