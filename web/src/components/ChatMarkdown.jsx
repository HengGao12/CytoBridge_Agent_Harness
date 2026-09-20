import React, { useMemo, useRef, useState } from 'react';
import { Check, Copy } from 'lucide-react';
import { MarkdownContent } from './MarkdownContent';
import { isInlineCode } from '../lib/markdownCode';
import { cn } from './ui/cn';

/**
 * Code block with a hover copy button. Reads the rendered text content from
 * the DOM so it works for any nested markdown/code structure.
 */
function CopyablePre({ className, children, ...props }) {
  const preRef = useRef(null);
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    const text = preRef.current?.innerText || '';
    if (!text) return;
    try {
      await navigator.clipboard.writeText(text.replace(/\n$/, ''));
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1600);
    } catch {
      /* clipboard unavailable (e.g. insecure context) — silently ignore */
    }
  };

  return (
    <div className="group relative my-3">
      <pre
        ref={preRef}
        className={cn('overflow-x-auto rounded-lg px-3 py-2.5 text-[12.5px] leading-relaxed font-mono', className)}
        {...props}
      >
        {children}
      </pre>
      <button
        type="button"
        onClick={handleCopy}
        aria-label={copied ? 'Copied' : 'Copy code'}
        title="Copy code"
        className={cn(
          'absolute right-1.5 top-1.5 rounded-md border p-1.5 opacity-0 transition-all duration-150',
          'group-hover:opacity-100 focus-visible:opacity-100 active:scale-95',
          'focus:outline-none focus-visible:ring-2 focus-visible:ring-accent',
          'border-line bg-surface text-ink-soft shadow-xs hover:text-ink',
        )}
      >
        {copied
          ? <Check size={12} className="text-emerald-600 dark:text-emerald-400" />
          : <Copy size={12} />}
      </button>
    </div>
  );
}

/**
 * Markdown component overrides. User and assistant bubbles now share the same
 * neutral surface, so a single style set covers both. Memoized so
 * `ReactMarkdown` doesn't re-render every node when only `content` changes.
 */
function buildComponents() {
  const codeBlockClass = 'bg-surface-subtle border border-line text-ink';
  const inlineCodeClass = 'bg-surface-muted text-ink border border-line';
  const blockquoteClass = 'border-l-line-strong text-ink-muted';
  const linkClass = 'text-accent underline underline-offset-2 decoration-accent/40 hover:decoration-accent';
  const hrClass = 'border-t border-line';

  return {
    p: (props) => <p className="my-2.5 first:mt-0 last:mb-0 leading-relaxed" {...props} />,
    h1: (props) => <h1 className="mt-5 mb-2.5 text-xl font-semibold tracking-tight" {...props} />,
    h2: (props) => <h2 className="mt-5 mb-2.5 text-lg font-semibold tracking-tight" {...props} />,
    h3: (props) => <h3 className="mt-4 mb-2 text-base font-semibold tracking-tight" {...props} />,
    h4: (props) => <h4 className="mt-4 mb-2 text-sm font-semibold" {...props} />,
    ul: (props) => <ul className="my-2.5 list-disc pl-5 space-y-1" {...props} />,
    ol: (props) => <ol className="my-2.5 list-decimal pl-5 space-y-1" {...props} />,
    li: (props) => <li className="pl-1 leading-relaxed" {...props} />,
    blockquote: (props) => (
      <blockquote className={`my-3 border-l-2 pl-3 italic ${blockquoteClass}`} {...props} />
    ),
    a: (props) => <a className={linkClass} target="_blank" rel="noreferrer noopener" {...props} />,
    hr: (props) => <hr className={`my-4 border-0 ${hrClass}`} {...props} />,
    table: (props) => (
      <div className="my-3 overflow-x-auto rounded-lg border border-line">
        <table className="min-w-full border-collapse text-xs" {...props} />
      </div>
    ),
    th: (props) => (
      <th className="border-b border-line bg-surface-muted px-3 py-2 text-left font-semibold text-ink" {...props} />
    ),
    td: (props) => (
      <td className="border-b border-line px-3 py-2 align-top text-ink-muted last:border-b-0" {...props} />
    ),
    pre: (props) => (
      <CopyablePre className={codeBlockClass} {...props} />
    ),
    code: ({ className, children, ...props }) => {
      if (isInlineCode(className, children)) {
        return (
          <code className={`rounded-md px-1 py-0.5 text-[0.85em] font-mono ${inlineCodeClass}`} {...props}>
            {children}
          </code>
        );
      }
      return (
        <code className={className} {...props}>
          {children}
        </code>
      );
    },
    strong: (props) => <strong className="font-semibold" {...props} />,
  };
}

function ChatMarkdownImpl({ content, variant = 'assistant' }) {
  const components = useMemo(() => buildComponents(), []);

  return (
    <div className="chat-markdown text-[15px] leading-7 break-words text-ink" data-variant={variant}>
      <MarkdownContent content={content} components={components} />
    </div>
  );
}

export const ChatMarkdown = React.memo(
  ChatMarkdownImpl,
  (prev, next) => prev.content === next.content && prev.variant === next.variant,
);
