import React, { useMemo, useState } from 'react';
import { ChevronDown, ChevronRight, CircleAlert, CircleCheckBig, CircleSlash, LoaderCircle, SquareTerminal } from 'lucide-react';
import { CodeHighlight } from './CodeHighlight';

const STATUS_META = {
  inProgress: {
    label: 'Running',
    chip: 'bg-accent-soft/60 text-accent dark:text-accent-strong border-accent/20',
    icon: <LoaderCircle size={14} className="animate-spin" />,
  },
  completed: {
    label: 'Completed',
    chip: 'bg-emerald-50 text-emerald-700 border-emerald-200 dark:bg-emerald-500/10 dark:text-emerald-300 dark:border-emerald-500/30',
    icon: <CircleCheckBig size={14} />,
  },
  failed: {
    label: 'Failed',
    chip: 'bg-red-50 text-red-700 border-red-200 dark:bg-red-500/10 dark:text-red-300 dark:border-red-500/30',
    icon: <CircleAlert size={14} />,
  },
  timedOut: {
    label: 'Timed Out',
    chip: 'bg-amber-50 text-amber-700 border-amber-200 dark:bg-amber-500/10 dark:text-amber-300 dark:border-amber-500/30',
    icon: <CircleAlert size={14} />,
  },
  interrupted: {
    label: 'Interrupted',
    chip: 'bg-surface-muted text-ink-muted border-line',
    icon: <CircleSlash size={14} />,
  },
};

function OutputBlock({ title, content, tone = 'neutral' }) {
  if (!content) return null;
  const toneClass =
    tone === 'stderr'
      ? 'border-amber-200 bg-amber-50/60 dark:border-amber-500/30 dark:bg-amber-500/5'
      : tone === 'error'
        ? 'border-red-200 bg-red-50/60 dark:border-red-500/30 dark:bg-red-500/5'
        : 'border-line bg-surface';
  return (
    <div className={`overflow-hidden rounded-lg border ${toneClass}`}>
      <div className="border-b border-inherit px-3 py-1.5 text-[11px] font-medium text-ink-soft">
        {title}
      </div>
      <pre className="max-h-96 overflow-auto custom-scrollbar px-3 py-2.5 text-xs whitespace-pre-wrap break-words font-mono text-ink">
        {content}
      </pre>
    </div>
  );
}

function CodeBlock({ content }) {
  if (!content) return null;
  return (
    <div className="overflow-hidden rounded-lg border border-line bg-surface">
      <div className="border-b border-line px-3 py-1.5 text-[11px] font-medium text-ink-soft">
        Code
      </div>
      <CodeHighlight
        language="python"
        customStyle={{
          margin: 0,
          padding: '0.875rem',
          fontSize: '12px',
          maxHeight: '24rem',
          overflow: 'auto',
          borderRadius: 0,
          background: '#0d1117',
        }}
        showLineNumbers
        wrapLongLines
      >
        {content}
      </CodeHighlight>
    </div>
  );
}

/**
 * PythonExecutionDetails — notes + code + output blocks without a card shell.
 * Reused by the standalone card below and the activity-rail step row.
 */
export function PythonExecutionDetails({ item, showCodeToggle = false }) {
  const [showCode, setShowCode] = useState(true);
  const [showTraceback, setShowTraceback] = useState(!showCodeToggle);
  const combinedOutput = useMemo(() => {
    const parts = [];
    if (item.stdout) parts.push(item.stdout);
    if (item.stderr) parts.push(`[STDERR]\n${item.stderr}`);
    return parts.join(parts.length > 1 ? '\n' : '');
  }, [item.stdout, item.stderr]);

  return (
    <div className="space-y-2.5">
      {item.timeoutEnforced === false && (
        <div className="rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-800 dark:border-amber-500/30 dark:bg-amber-500/10 dark:text-amber-200">
          Hard timeout enforcement was unavailable for this execution mode/thread. Execution still ran in shared in-memory mode.
        </div>
      )}

      {item.switchNote && (
        <div className="rounded-lg border border-accent/30 bg-accent-soft px-3 py-2 text-xs text-accent-strong">
          {item.switchNote}
        </div>
      )}

      {showCodeToggle && (
        <div className="flex flex-wrap gap-1.5">
          <button
            type="button"
            onClick={() => setShowCode((v) => !v)}
            className="inline-flex items-center gap-1 rounded-md border border-line bg-surface px-2.5 py-1 text-xs font-medium text-ink-muted transition-colors hover:border-line-strong hover:text-ink"
          >
            {showCode ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
            Code
          </button>
          {!!item.traceback && (
            <button
              type="button"
              onClick={() => setShowTraceback((v) => !v)}
              className="inline-flex items-center gap-1 rounded-md border border-line bg-surface px-2.5 py-1 text-xs font-medium text-ink-muted transition-colors hover:border-line-strong hover:text-ink"
            >
              {showTraceback ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
              Traceback
            </button>
          )}
        </div>
      )}

      {showCode && <CodeBlock content={item.code || ''} />}

      <OutputBlock title="Output" content={combinedOutput} />
      <OutputBlock title="Error" content={item.error || ''} tone="error" />
      {showTraceback && <OutputBlock title="Traceback" content={item.traceback || ''} tone="stderr" />}
    </div>
  );
}

function PythonExecutionCardImpl({ item, embedded = false }) {
  const statusMeta = STATUS_META[item.status] || STATUS_META.completed;
  const subtitle = item.reason
    ? item.reason
    : 'Interactive execute_python run';
  const metaBits = [];
  if (Number.isFinite(item.durationMs) && item.durationMs > 0) {
    metaBits.push(`${item.durationMs} ms`);
  }
  if (item.status === 'timedOut' && item.timeout) {
    metaBits.push(`timeout ${item.timeout}s`);
  }
  if (item.status === 'interrupted') {
    metaBits.push('stopped');
  }
  if (item.status === 'failed') {
    metaBits.push('execution failed');
  }

  return (
    <div className={embedded ? '' : 'px-2'}>
      <div className={`w-full ${embedded ? '' : 'max-w-3xl'} overflow-hidden rounded-xl border border-line bg-surface`}>
        <div className="flex items-start justify-between gap-4 border-b border-line px-4 py-3">
          <div className="min-w-0">
            <div className="flex items-center gap-2 text-sm font-medium text-ink">
              <SquareTerminal size={14} className="text-accent" />
              <span>Python execution</span>
            </div>
            <div className="mt-0.5 text-xs text-ink-soft">{subtitle}</div>
            {metaBits.length > 0 && (
              <div className="mt-1 font-mono text-[11px] text-ink-soft">
                {metaBits.join(' · ')}
              </div>
            )}
          </div>
          <div className={`inline-flex items-center gap-1.5 rounded-md border px-2 py-0.5 text-[11px] font-medium ${statusMeta.chip}`}>
            {statusMeta.icon}
            <span>{statusMeta.label}</span>
          </div>
        </div>

        <div className="px-4 py-3">
          <PythonExecutionDetails item={item} showCodeToggle />
        </div>
      </div>
    </div>
  );
}

export const PythonExecutionCard = React.memo(
  PythonExecutionCardImpl,
  (prev, next) => prev.item === next.item && prev.embedded === next.embedded,
);
