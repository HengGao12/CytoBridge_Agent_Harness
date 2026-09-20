import React, { useEffect, useRef } from 'react';
import { Send, Square, BookOpen, ImagePlus, X } from 'lucide-react';
import { IconButton } from './ui';
import { cn } from './ui/cn';

/**
 * MessageComposer — sticky bottom input area.
 *
 * Owns its own textarea ref + auto-grow behavior. The parent passes the input
 * value/setters and submit/stop callbacks; this component is presentational.
 */
export const MessageComposer = React.forwardRef(function MessageComposer(
    {
        value,
        onChange,
        onSubmit,
        onStop,
        onOpenSkills,
        onPickAttachment,
        onPaste,
        onRemoveAttachment,
        attachments = [],
        canSend,
        isThinking,
        connected,
    },
    ref,
) {
    const textareaRef = useRef(null);
    const fileInputRef = useRef(null);

    // Expose textarea focus to the parent (used after quick-prompt clicks).
    React.useImperativeHandle(ref, () => ({
        focus: () => textareaRef.current?.focus(),
    }));

    // Auto-grow textarea height as the user types.
    useEffect(() => {
        const ta = textareaRef.current;
        if (!ta) return;
        ta.style.height = 'auto';
        ta.style.height = `${Math.min(ta.scrollHeight, 240)}px`;
    }, [value]);

    const handleKeyDown = (event) => {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            onSubmit?.(event);
        }
    };

    const handleAttachmentChange = async (event) => {
        await onPickAttachment?.(event.target.files);
        event.target.value = '';
    };

    return (
        <div className="sticky bottom-0 z-20 px-4 pb-4 pt-2 sm:px-6 bg-gradient-to-t from-surface via-surface to-transparent">
            <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                multiple
                className="hidden"
                onChange={handleAttachmentChange}
            />

            {attachments.length > 0 && (
                <div className="mx-auto mb-3 max-w-3xl grid grid-cols-2 sm:grid-cols-4 gap-2">
                    {attachments.map((attachment, idx) => (
                        <div
                            key={`${attachment.file_name}-${idx}`}
                            className="group relative overflow-hidden rounded-lg border border-line bg-surface shadow-xs transition-shadow hover:shadow-md animate-fade-in"
                        >
                            <img
                                src={attachment.data_url}
                                alt={attachment.file_name}
                                className="h-20 w-full object-cover"
                            />
                            <button
                                type="button"
                                onClick={() => onRemoveAttachment?.(idx)}
                                className="absolute right-1.5 top-1.5 rounded-md bg-ink/70 p-1 text-white hover:bg-ink/90 focus:outline-none focus-visible:ring-2 focus-visible:ring-accent"
                                aria-label={`Remove ${attachment.file_name}`}
                            >
                                <X size={12} />
                            </button>
                            <div className="px-2 py-1 text-[10px] font-mono truncate text-ink-muted bg-surface-muted">
                                {attachment.file_name}
                            </div>
                        </div>
                    ))}
                </div>
            )}

            <form
                onSubmit={onSubmit}
                className={cn(
                    'mx-auto flex max-w-3xl items-end gap-1.5 rounded-2xl border bg-surface/85 p-2 backdrop-blur-xl',
                    'shadow-md border-line',
                    'transition-all duration-200',
                    'focus-within:border-accent/50 focus-within:bg-surface focus-within:shadow-lg focus-within:ring-4 focus-within:ring-accent/10',
                )}
            >
                <IconButton
                    onClick={onOpenSkills}
                    aria-label="Skills"
                    title="Skills"
                    size="md"
                >
                    <BookOpen size={16} />
                </IconButton>

                <textarea
                    ref={textareaRef}
                    value={value}
                    onChange={(event) => onChange(event.target.value)}
                    onKeyDown={handleKeyDown}
                    onPaste={onPaste}
                    placeholder={isThinking ? 'Agent is working — type / for a command…' : 'Ask CellCompass to analyze your data...'}
                    rows={1}
                    className={cn(
                        'flex-1 resize-none border-0 bg-transparent px-2 py-2 text-sm text-ink placeholder:text-ink-soft',
                        'focus:outline-none focus:ring-0',
                        'min-h-[2.5rem] max-h-60 leading-6',
                    )}
                />

                <IconButton
                    onClick={() => fileInputRef.current?.click()}
                    aria-label="Attach image"
                    title="Attach image"
                    size="md"
                >
                    <ImagePlus size={16} />
                </IconButton>

                {isThinking ? (
                    <button
                        type="button"
                        onClick={onStop}
                        className="inline-flex h-9 w-9 items-center justify-center rounded-xl bg-red-600 text-white shadow-sm transition-all hover:bg-red-500 active:scale-95 focus:outline-none focus-visible:ring-2 focus-visible:ring-red-500 focus-visible:ring-offset-2 focus-visible:ring-offset-surface"
                        aria-label="Stop agent"
                        title="Stop agent"
                    >
                        <Square size={14} fill="currentColor" />
                    </button>
                ) : (
                    <button
                        type="submit"
                        disabled={!canSend || !connected}
                        className={cn(
                            'inline-flex h-9 w-9 items-center justify-center rounded-xl transition-all duration-150',
                            'focus:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 focus-visible:ring-offset-surface',
                            canSend && connected
                                ? 'bg-accent text-white shadow-sm hover:bg-accent-strong hover:shadow-md active:scale-95'
                                : 'bg-surface-muted text-ink-soft cursor-not-allowed',
                        )}
                        aria-label="Send message"
                        title="Send"
                    >
                        <Send size={15} />
                    </button>
                )}
            </form>

            <p className="mx-auto mt-2 flex max-w-3xl items-center justify-center gap-2 text-center text-[11px] text-ink-soft">
                <span>CellCompass can make mistakes. Review generated analysis.</span>
                <span aria-hidden="true" className="hidden sm:inline text-line-strong">·</span>
                <span className="hidden sm:inline">
                    <kbd className="rounded border border-line bg-surface-muted px-1 py-px font-mono text-[10px] text-ink-muted">Enter</kbd>
                    {' '}to send
                </span>
            </p>
        </div>
    );
});
