import typography from '@tailwindcss/typography';

/**
 * Design tokens (Claude / Linear minimalist direction):
 *  - Surface palette: neutral-first; single blue accent.
 *  - Radius scale: md / lg / xl / 2xl only (no arbitrary radii in components).
 *  - Shadow: subtle, no colored glows.
 *  - Animations: tight, low-amplitude.
 *
 * @type {import('tailwindcss').Config}
 */
export default {
    darkMode: 'class',
    content: [
        './index.html',
        './src/**/*.{js,ts,jsx,tsx}',
    ],
    theme: {
        extend: {
            fontFamily: {
                sans: [
                    'Inter',
                    '"Inter Variable"',
                    'ui-sans-serif',
                    'system-ui',
                    '-apple-system',
                    'BlinkMacSystemFont',
                    '"Segoe UI"',
                    'Roboto',
                    '"Helvetica Neue"',
                    'Arial',
                    'sans-serif',
                ],
                mono: [
                    '"JetBrains Mono"',
                    '"Fira Code"',
                    'ui-monospace',
                    'SFMono-Regular',
                    'Menlo',
                    'Monaco',
                    'Consolas',
                    '"Liberation Mono"',
                    '"Courier New"',
                    'monospace',
                ],
            },
            colors: {
                // Semantic surface tokens reused via CSS variables in index.css.
                surface: {
                    DEFAULT: 'rgb(var(--surface) / <alpha-value>)',
                    muted: 'rgb(var(--surface-muted) / <alpha-value>)',
                    subtle: 'rgb(var(--surface-subtle) / <alpha-value>)',
                    inverted: 'rgb(var(--surface-inverted) / <alpha-value>)',
                },
                line: {
                    DEFAULT: 'rgb(var(--line) / <alpha-value>)',
                    strong: 'rgb(var(--line-strong) / <alpha-value>)',
                },
                ink: {
                    DEFAULT: 'rgb(var(--ink) / <alpha-value>)',
                    muted: 'rgb(var(--ink-muted) / <alpha-value>)',
                    soft: 'rgb(var(--ink-soft) / <alpha-value>)',
                },
                accent: {
                    DEFAULT: 'rgb(var(--accent) / <alpha-value>)',
                    soft: 'rgb(var(--accent-soft) / <alpha-value>)',
                    strong: 'rgb(var(--accent-strong) / <alpha-value>)',
                },
            },
            boxShadow: {
                // Designer-curated shadow ramp; everything else routes through these.
                xs: '0 1px 1px 0 rgb(15 23 42 / 0.04)',
                sm: '0 1px 2px 0 rgb(15 23 42 / 0.05), 0 1px 1px 0 rgb(15 23 42 / 0.03)',
                md: '0 4px 12px -2px rgb(15 23 42 / 0.06), 0 2px 4px -2px rgb(15 23 42 / 0.04)',
                lg: '0 12px 28px -8px rgb(15 23 42 / 0.12), 0 4px 8px -4px rgb(15 23 42 / 0.06)',
                'lg-dark': '0 12px 28px -8px rgb(0 0 0 / 0.45), 0 4px 8px -4px rgb(0 0 0 / 0.3)',
                ring: '0 0 0 1px rgb(var(--line) / 1)',
            },
            keyframes: {
                fadeIn: {
                    from: { opacity: '0', transform: 'translateY(4px)' },
                    to: { opacity: '1', transform: 'translateY(0)' },
                },
                scaleIn: {
                    from: { opacity: '0', transform: 'scale(0.97) translateY(6px)' },
                    to: { opacity: '1', transform: 'scale(1) translateY(0)' },
                },
                slideInLeft: {
                    from: { opacity: '0', transform: 'translateX(-12px)' },
                    to: { opacity: '1', transform: 'translateX(0)' },
                },
                riseIn: {
                    from: { opacity: '0', transform: 'translateY(12px)' },
                    to: { opacity: '1', transform: 'translateY(0)' },
                },
                shimmer: {
                    '0%': { backgroundPosition: '-200% 0' },
                    '100%': { backgroundPosition: '200% 0' },
                },
            },
            animation: {
                'fade-in': 'fadeIn 200ms ease-out',
                'scale-in': 'scaleIn 220ms cubic-bezier(0.16, 1, 0.3, 1)',
                'slide-in-left': 'slideInLeft 240ms cubic-bezier(0.16, 1, 0.3, 1)',
                'rise-in': 'riseIn 400ms cubic-bezier(0.16, 1, 0.3, 1) both',
                'spin-slow': 'spin 10s linear infinite',
                shimmer: 'shimmer 2s linear infinite',
            },
        },
    },
    plugins: [
        typography,
    ],
};
