import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import 'katex/dist/katex.min.css';
import { normalizeMarkdownForRendering } from '../lib/markdownMath';

/**
 * MarkdownRenderer — the heavy markdown pipeline (react-markdown + GFM +
 * KaTeX). Loaded lazily via MarkdownContent so the math/markdown stack stays
 * out of the initial bundle.
 */
const mathMarkdownOptions = {
    remarkPlugins: [remarkGfm, remarkMath],
    rehypePlugins: [[rehypeKatex, { throwOnError: false, strict: false }]],
};

export default function MarkdownRenderer({ content, components }) {
    return (
        <ReactMarkdown
            {...mathMarkdownOptions}
            components={components}
        >
            {normalizeMarkdownForRendering(content)}
        </ReactMarkdown>
    );
}
