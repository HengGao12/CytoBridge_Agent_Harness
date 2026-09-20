import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { normalizeMarkdownForRendering } from '../lib/markdownMath';

/**
 * MarkdownRendererPlain — the common case: GFM markdown without the KaTeX
 * stack. The math renderer (`MarkdownRenderer`) pulls in remark-math,
 * rehype-katex and KaTeX's CSS (~120 kB gzip), so we only load it when the
 * content actually contains math (see `MarkdownContent`). Table/inline
 * normalization is shared so output matches the math path for non-math text.
 */
const plainMarkdownOptions = {
    remarkPlugins: [remarkGfm],
};

export default function MarkdownRendererPlain({ content, components }) {
    return (
        <ReactMarkdown
            {...plainMarkdownOptions}
            components={components}
        >
            {normalizeMarkdownForRendering(content)}
        </ReactMarkdown>
    );
}
