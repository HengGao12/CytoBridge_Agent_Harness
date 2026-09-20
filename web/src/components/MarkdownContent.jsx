import React, { Suspense } from 'react';

/**
 * MarkdownContent — thin lazy wrapper around the markdown pipeline.
 *
 * Two renderers are code-split: a light GFM-only one for the common case and
 * the heavier math pipeline (remark-math + rehype-katex + KaTeX CSS, ~120 kB
 * gzip) that is fetched only when the content actually contains math. Until a
 * chunk resolves the raw text shows in the same typographic position, so the
 * swap is invisible in practice.
 */
const MarkdownRenderer = React.lazy(() => import('./MarkdownRenderer'));
const MarkdownRendererPlain = React.lazy(() => import('./MarkdownRendererPlain'));

// Cheap heuristic: anything that could be a TeX math delimiter. False positives
// (e.g. a stray `$`) only mean we load the heavier renderer — never wrong output.
const MATH_HINT_RE = /\$|\\\(|\\\[|\\begin\{/;

function hasMath(content) {
    return typeof content === 'string' && MATH_HINT_RE.test(content);
}

export function MarkdownContent({ content, components }) {
    const Renderer = hasMath(content) ? MarkdownRenderer : MarkdownRendererPlain;
    return (
        <Suspense fallback={<div className="whitespace-pre-wrap">{content}</div>}>
            <Renderer content={content} components={components} />
        </Suspense>
    );
}
