import React, { useEffect, useRef } from 'react';
import { X } from 'lucide-react';
import { cn } from './cn';

/* ===========================================================================
 * Button
 *
 * Variants:
 *   - primary:   solid accent, white text (filled call-to-action)
 *   - secondary: surface-muted card with subtle border
 *   - ghost:     transparent, hover reveals subtle background
 *   - danger:    red surface for destructive actions
 *
 * Sizes: sm / md / lg
 * ========================================================================= */
const BUTTON_VARIANTS = {
    primary:
        'bg-accent text-white hover:bg-accent-strong active:bg-accent-strong shadow-sm hover:shadow-md',
    secondary:
        'bg-surface text-ink border border-line hover:bg-surface-muted hover:border-line-strong',
    ghost:
        'bg-transparent text-ink-muted hover:bg-surface-muted hover:text-ink',
    danger:
        'bg-red-600 text-white hover:bg-red-500 shadow-sm',
};

const BUTTON_SIZES = {
    sm: 'h-8 px-3 text-xs gap-1.5',
    md: 'h-9 px-4 text-sm gap-2',
    lg: 'h-11 px-5 text-base gap-2',
};

export const Button = React.forwardRef(function Button(
    { variant = 'secondary', size = 'md', className, type = 'button', children, ...props },
    ref,
) {
    return (
        <button
            ref={ref}
            type={type}
            className={cn(
                'inline-flex items-center justify-center rounded-lg font-medium transition-all duration-150',
                'active:scale-[0.98]',
                'disabled:opacity-50 disabled:cursor-not-allowed disabled:pointer-events-none',
                'focus:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 focus-visible:ring-offset-surface',
                BUTTON_VARIANTS[variant],
                BUTTON_SIZES[size],
                className,
            )}
            {...props}
        >
            {children}
        </button>
    );
});

/* ===========================================================================
 * IconButton — square-sized button for icon-only affordances.
 * ========================================================================= */
const ICON_SIZES = {
    sm: 'h-7 w-7',
    md: 'h-9 w-9',
    lg: 'h-10 w-10',
};

export const IconButton = React.forwardRef(function IconButton(
    { variant = 'ghost', size = 'md', className, type = 'button', children, active = false, ...props },
    ref,
) {
    return (
        <button
            ref={ref}
            type={type}
            className={cn(
                'inline-flex items-center justify-center rounded-lg transition-all duration-150',
                'active:scale-95',
                'disabled:opacity-50 disabled:cursor-not-allowed disabled:pointer-events-none',
                'focus:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 focus-visible:ring-offset-surface',
                active
                    ? 'bg-accent-soft text-accent-strong'
                    : BUTTON_VARIANTS[variant],
                ICON_SIZES[size],
                className,
            )}
            {...props}
        >
            {children}
        </button>
    );
});

/* ===========================================================================
 * Card — neutral surface container with consistent radius/border/shadow.
 * ========================================================================= */
export function Card({ className, padded = true, interactive = false, children, ...props }) {
    return (
        <div
            className={cn(
                'rounded-xl border border-line bg-surface',
                padded && 'p-5',
                interactive && 'transition-all duration-200 hover:-translate-y-0.5 hover:border-line-strong hover:bg-surface-muted hover:shadow-md',
                className,
            )}
            {...props}
        >
            {children}
        </div>
    );
}

/* ===========================================================================
 * Pill — compact label / status chip.
 *
 * Tones map to status semantics. Use neutral by default.
 * ========================================================================= */
const PILL_TONES = {
    neutral: 'bg-surface-muted text-ink-muted border border-line',
    accent: 'bg-accent-soft text-accent-strong border border-accent/20',
    success: 'bg-emerald-50 text-emerald-700 border border-emerald-200 dark:bg-emerald-500/10 dark:text-emerald-300 dark:border-emerald-500/30',
    warn: 'bg-amber-50 text-amber-700 border border-amber-200 dark:bg-amber-500/10 dark:text-amber-300 dark:border-amber-500/30',
    danger: 'bg-red-50 text-red-700 border border-red-200 dark:bg-red-500/10 dark:text-red-300 dark:border-red-500/30',
};

export function Pill({ tone = 'neutral', className, children, ...props }) {
    return (
        <span
            className={cn(
                'inline-flex items-center gap-1.5 rounded-md px-2 py-0.5 text-xs font-medium',
                PILL_TONES[tone],
                className,
            )}
            {...props}
        >
            {children}
        </span>
    );
}

/* ===========================================================================
 * Field / Label / Help — form layout primitives.
 * ========================================================================= */
export function Field({ children, className, ...props }) {
    return (
        <div className={cn('flex flex-col gap-1.5', className)} {...props}>
            {children}
        </div>
    );
}

export function Label({ icon, children, htmlFor, className, ...props }) {
    return (
        <label
            htmlFor={htmlFor}
            className={cn(
                'flex items-center gap-1.5 text-xs font-semibold text-ink-muted',
                className,
            )}
            {...props}
        >
            {icon}
            {children}
        </label>
    );
}

export function Help({ tone = 'muted', children, className, ...props }) {
    const toneCls =
        tone === 'warn'
            ? 'text-amber-700 dark:text-amber-300'
            : tone === 'danger'
                ? 'text-red-700 dark:text-red-300'
                : 'text-ink-soft';
    return (
        <p className={cn('text-xs leading-relaxed', toneCls, className)} {...props}>
            {children}
        </p>
    );
}

/* ===========================================================================
 * Input / Select / Textarea — shared chrome.
 * ========================================================================= */
const FIELD_CHROME =
    'block w-full rounded-lg border border-line bg-surface text-sm text-ink placeholder:text-ink-soft transition-colors ' +
    'hover:border-line-strong focus:border-accent focus:outline-none focus:ring-2 focus:ring-accent/20 ' +
    'disabled:cursor-not-allowed disabled:opacity-60';

export const Input = React.forwardRef(function Input({ className, ...props }, ref) {
    return (
        <input
            ref={ref}
            className={cn(FIELD_CHROME, 'h-10 px-3', className)}
            {...props}
        />
    );
});

export const Select = React.forwardRef(function Select({ className, children, ...props }, ref) {
    return (
        <select
            ref={ref}
            className={cn(FIELD_CHROME, 'h-10 px-3 pr-8 appearance-none bg-no-repeat bg-right', className)}
            style={{
                backgroundImage:
                    "url(\"data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2.5' stroke-linecap='round' stroke-linejoin='round'><polyline points='6 9 12 15 18 9'/></svg>\")",
                backgroundPosition: 'right 0.75rem center',
                backgroundSize: '0.75rem',
            }}
            {...props}
        >
            {children}
        </select>
    );
});

export const Textarea = React.forwardRef(function Textarea({ className, rows = 3, ...props }, ref) {
    return (
        <textarea
            ref={ref}
            rows={rows}
            className={cn(FIELD_CHROME, 'px-3 py-2 resize-y min-h-[5rem]', className)}
            {...props}
        />
    );
});

/* ===========================================================================
 * Toggle — pill switch with label/description rows.
 * ========================================================================= */
export function Toggle({ id, checked, onChange, disabled = false, ariaLabel }) {
    return (
        <button
            id={id}
            type="button"
            role="switch"
            aria-checked={checked}
            aria-label={ariaLabel}
            disabled={disabled}
            onClick={() => onChange?.(!checked)}
            className={cn(
                'relative inline-flex h-6 w-11 shrink-0 cursor-pointer rounded-full border transition-colors duration-200',
                'focus:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 focus-visible:ring-offset-surface',
                'disabled:cursor-not-allowed disabled:opacity-50',
                checked
                    ? 'bg-accent border-accent'
                    : 'bg-surface-muted border-line',
            )}
        >
            <span
                className={cn(
                    'pointer-events-none inline-block h-4 w-4 transform rounded-full bg-white shadow-sm transition-transform duration-200',
                    'translate-y-0.5',
                    checked ? 'translate-x-[1.4rem]' : 'translate-x-0.5',
                )}
            />
        </button>
    );
}

export function ToggleRow({ icon, title, description, checked, onChange, disabled = false }) {
    return (
        <div className="flex items-start justify-between gap-4 rounded-lg border border-line bg-surface px-4 py-3 transition-colors hover:border-line-strong">
            <div className="flex items-start gap-3 min-w-0">
                {icon && (
                    <div className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-md bg-surface-muted text-ink-muted">
                        {icon}
                    </div>
                )}
                <div className="min-w-0">
                    <div className="text-sm font-medium text-ink">{title}</div>
                    {description && (
                        <div className="mt-0.5 text-xs text-ink-soft leading-relaxed">{description}</div>
                    )}
                </div>
            </div>
            <Toggle checked={checked} onChange={onChange} disabled={disabled} ariaLabel={title} />
        </div>
    );
}

/* ===========================================================================
 * Section — labeled container used inside Setup / Skills modals.
 * ========================================================================= */
export function Section({ icon, eyebrow, title, description, children, className }) {
    return (
        <section className={cn('flex flex-col gap-4 rounded-xl border border-line bg-surface p-5', className)}>
            <header className="flex items-start gap-3">
                {icon && (
                    <div className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-md bg-surface-muted text-accent">
                        {icon}
                    </div>
                )}
                <div className="min-w-0">
                    {eyebrow && (
                        <div className="text-[11px] font-medium text-ink-soft mb-1">
                            {eyebrow}
                        </div>
                    )}
                    <h3 className="text-sm font-semibold text-ink leading-tight">{title}</h3>
                    {description && (
                        <p className="mt-1 text-xs text-ink-soft leading-relaxed">{description}</p>
                    )}
                </div>
            </header>
            <div className="flex flex-col gap-4">{children}</div>
        </section>
    );
}

/* ===========================================================================
 * Divider — semantic separator using the line token.
 * ========================================================================= */
export function Divider({ orientation = 'horizontal', className }) {
    return (
        <div
            role="separator"
            aria-orientation={orientation}
            className={cn(
                orientation === 'horizontal' ? 'h-px w-full bg-line' : 'w-px h-full bg-line',
                className,
            )}
        />
    );
}

/* ===========================================================================
 * Modal — focus-trap + escape-to-close shell.
 *
 * Layout-agnostic: the caller provides the inner card via children. Use the
 * `size` prop to pick a width preset; pass `size="custom"` for full control.
 * ========================================================================= */
const MODAL_SIZES = {
    sm: 'max-w-md',
    md: 'max-w-xl',
    lg: 'max-w-3xl',
    xl: 'max-w-5xl',
    '2xl': 'max-w-6xl',
    custom: '',
};

export function Modal({
    isOpen,
    onClose,
    title,
    description,
    size = 'md',
    initialFocusRef,
    children,
    footer,
    showCloseButton = true,
    contentClassName,
}) {
    const dialogRef = useRef(null);
    const lastActiveRef = useRef(null);

    useEffect(() => {
        if (!isOpen) return undefined;

        lastActiveRef.current = document.activeElement;
        // Defer focus to the next tick so child content has rendered.
        const focusTimer = window.setTimeout(() => {
            const target =
                initialFocusRef?.current ||
                dialogRef.current?.querySelector('[data-autofocus]') ||
                dialogRef.current;
            target?.focus?.();
        }, 0);

        const FOCUSABLE_SELECTOR = [
            'a[href]',
            'button:not([disabled])',
            'textarea:not([disabled])',
            'input:not([disabled])',
            'select:not([disabled])',
            '[tabindex]:not([tabindex="-1"])',
        ].join(',');

        const onKeyDown = (event) => {
            if (event.key === 'Escape') {
                event.stopPropagation();
                onClose?.();
                return;
            }
            if (event.key !== 'Tab') return;
            const dialog = dialogRef.current;
            if (!dialog) return;
            const focusable = Array.from(dialog.querySelectorAll(FOCUSABLE_SELECTOR))
                .filter((el) => el.offsetParent !== null || el === document.activeElement);
            if (focusable.length === 0) {
                // Nothing to tab to — keep focus pinned on the dialog shell.
                event.preventDefault();
                dialog.focus();
                return;
            }
            const first = focusable[0];
            const last = focusable[focusable.length - 1];
            const active = document.activeElement;
            if (event.shiftKey) {
                if (active === first || active === dialog || !dialog.contains(active)) {
                    event.preventDefault();
                    last.focus();
                }
            } else if (active === last) {
                event.preventDefault();
                first.focus();
            }
        };
        document.addEventListener('keydown', onKeyDown);

        const previousOverflow = document.body.style.overflow;
        document.body.style.overflow = 'hidden';

        return () => {
            window.clearTimeout(focusTimer);
            document.removeEventListener('keydown', onKeyDown);
            document.body.style.overflow = previousOverflow;
            if (lastActiveRef.current && typeof lastActiveRef.current.focus === 'function') {
                lastActiveRef.current.focus();
            }
        };
    }, [isOpen, onClose, initialFocusRef]);

    if (!isOpen) return null;

    return (
        <div
            role="dialog"
            aria-modal="true"
            aria-label={typeof title === 'string' ? title : undefined}
            className="fixed inset-0 z-50 flex items-center justify-center p-4 sm:p-6 animate-fade-in"
        >
            <div
                className="absolute inset-0 bg-ink/50 backdrop-blur-[6px]"
                onClick={onClose}
                aria-hidden="true"
            />
            <div
                ref={dialogRef}
                tabIndex={-1}
                className={cn(
                    'relative w-full rounded-2xl border border-line bg-surface shadow-lg dark:shadow-lg-dark',
                    'max-h-[90vh] flex flex-col overflow-hidden animate-scale-in',
                    MODAL_SIZES[size],
                    contentClassName,
                )}
            >
                {(title || showCloseButton) && (
                    <header className="flex items-start justify-between gap-4 px-6 pt-5 pb-4 border-b border-line">
                        <div className="min-w-0">
                            {title && (
                                <h2 className="text-base font-semibold text-ink leading-tight">{title}</h2>
                            )}
                            {description && (
                                <p className="mt-1 text-sm text-ink-soft leading-relaxed">{description}</p>
                            )}
                        </div>
                        {showCloseButton && (
                            <IconButton onClick={onClose} aria-label="Close" size="sm">
                                <X size={16} />
                            </IconButton>
                        )}
                    </header>
                )}
                <div className="flex-1 overflow-y-auto custom-scrollbar">{children}</div>
                {footer && (
                    <footer className="px-6 py-4 border-t border-line bg-surface-muted">
                        {footer}
                    </footer>
                )}
            </div>
        </div>
    );
}

/* ===========================================================================
 * StatusDot — small colored circle indicating live connection state.
 * ========================================================================= */
const STATUS_TONES = {
    connected: 'bg-emerald-500 animate-pulse-status',
    disconnected: 'bg-red-500',
    error: 'bg-red-500',
    pending: 'bg-amber-400',
};

export function StatusDot({ status = 'pending', className }) {
    return (
        <span
            role="status"
            aria-label={status}
            className={cn('inline-block h-2 w-2 rounded-full', STATUS_TONES[status] || STATUS_TONES.pending, className)}
        />
    );
}
