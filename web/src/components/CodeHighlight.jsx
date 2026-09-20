import React, { Suspense } from 'react';

/**
 * CodeHighlight — lazily loaded Prism syntax highlighter.
 *
 * Uses `PrismLight` with an explicit allow-list of grammars instead of
 * `prism-async-light`. The async variant code-splits a separate chunk for
 * every Prism language (~9.8k tiny files in `dist`); registering only the
 * languages the agent actually emits keeps the build to a handful of chunks.
 * Unknown languages degrade gracefully to unhighlighted (but still styled)
 * code, and a plain <pre> fallback preserves layout while the chunk loads.
 */
const LazyPrism = React.lazy(async () => {
    const [
        { default: PrismLight },
        { default: oneDark },
        python,
        bash,
        json,
        yaml,
        sql,
        markup,
        clike,
        javascript,
        jsx,
        typescript,
        tsx,
        css,
        markdown,
        rLang,
        diff,
    ] = await Promise.all([
        import('react-syntax-highlighter/dist/esm/prism-light'),
        import('react-syntax-highlighter/dist/esm/styles/prism/one-dark'),
        import('react-syntax-highlighter/dist/esm/languages/prism/python'),
        import('react-syntax-highlighter/dist/esm/languages/prism/bash'),
        import('react-syntax-highlighter/dist/esm/languages/prism/json'),
        import('react-syntax-highlighter/dist/esm/languages/prism/yaml'),
        import('react-syntax-highlighter/dist/esm/languages/prism/sql'),
        import('react-syntax-highlighter/dist/esm/languages/prism/markup'),
        import('react-syntax-highlighter/dist/esm/languages/prism/clike'),
        import('react-syntax-highlighter/dist/esm/languages/prism/javascript'),
        import('react-syntax-highlighter/dist/esm/languages/prism/jsx'),
        import('react-syntax-highlighter/dist/esm/languages/prism/typescript'),
        import('react-syntax-highlighter/dist/esm/languages/prism/tsx'),
        import('react-syntax-highlighter/dist/esm/languages/prism/css'),
        import('react-syntax-highlighter/dist/esm/languages/prism/markdown'),
        import('react-syntax-highlighter/dist/esm/languages/prism/r'),
        import('react-syntax-highlighter/dist/esm/languages/prism/diff'),
    ]);

    PrismLight.registerLanguage('python', python.default);
    PrismLight.registerLanguage('py', python.default);
    PrismLight.registerLanguage('bash', bash.default);
    PrismLight.registerLanguage('sh', bash.default);
    PrismLight.registerLanguage('shell', bash.default);
    PrismLight.registerLanguage('json', json.default);
    PrismLight.registerLanguage('yaml', yaml.default);
    PrismLight.registerLanguage('yml', yaml.default);
    PrismLight.registerLanguage('sql', sql.default);
    PrismLight.registerLanguage('markup', markup.default);
    PrismLight.registerLanguage('html', markup.default);
    PrismLight.registerLanguage('xml', markup.default);
    PrismLight.registerLanguage('clike', clike.default);
    PrismLight.registerLanguage('javascript', javascript.default);
    PrismLight.registerLanguage('js', javascript.default);
    PrismLight.registerLanguage('jsx', jsx.default);
    PrismLight.registerLanguage('typescript', typescript.default);
    PrismLight.registerLanguage('ts', typescript.default);
    PrismLight.registerLanguage('tsx', tsx.default);
    PrismLight.registerLanguage('css', css.default);
    PrismLight.registerLanguage('markdown', markdown.default);
    PrismLight.registerLanguage('md', markdown.default);
    PrismLight.registerLanguage('r', rLang.default);
    PrismLight.registerLanguage('diff', diff.default);

    const Highlighter = ({ children, customStyle, ...props }) => (
        <PrismLight style={oneDark} customStyle={customStyle} {...props}>
            {children}
        </PrismLight>
    );
    return { default: Highlighter };
});

export function CodeHighlight({ children, customStyle, ...props }) {
    return (
        <Suspense
            fallback={
                <pre
                    className="overflow-auto whitespace-pre-wrap break-words font-mono text-xs text-zinc-200 custom-scrollbar"
                    style={{ margin: 0, padding: '0.875rem', background: '#0d1117', ...customStyle }}
                >
                    {children}
                </pre>
            }
        >
            <LazyPrism customStyle={customStyle} {...props}>{children}</LazyPrism>
        </Suspense>
    );
}
