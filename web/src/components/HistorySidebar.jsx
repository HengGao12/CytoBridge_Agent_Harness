import React, { useState, useEffect } from 'react';
import { History, ChevronLeft, Trash2, MessageSquare, Clock, Loader2, Check, X } from 'lucide-react';
import { IconButton } from './ui';
import { cn } from './ui/cn';

export function HistorySidebar({ onConversationResume, isOpen, onToggle }) {
    const [conversations, setConversations] = useState([]);
    const [loading, setLoading] = useState(true);
    const [selectedId, setSelectedId] = useState(null);
    const [resumingId, setResumingId] = useState(null);
    const [pendingDeleteId, setPendingDeleteId] = useState(null);
    const [errorMessage, setErrorMessage] = useState('');

    // Fetch conversations on mount and when sidebar opens
    useEffect(() => {
        if (isOpen) {
            fetchConversations();
        }
    }, [isOpen]);

    const fetchConversations = async () => {
        setLoading(true);
        try {
            const response = await fetch('/api/conversations');
            const data = await response.json();
            if (data.status === 'success') {
                setConversations(data.conversations || []);
            }
        } catch (error) {
            console.error('Failed to fetch conversations:', error);
        } finally {
            setLoading(false);
        }
    };

    const requestDelete = (e, sessionId) => {
        e.stopPropagation();
        setErrorMessage('');
        setPendingDeleteId(sessionId);
    };

    const cancelDelete = (e) => {
        e.stopPropagation();
        setPendingDeleteId(null);
    };

    const confirmDelete = async (e, sessionId) => {
        e.stopPropagation();
        setPendingDeleteId(null);
        try {
            const response = await fetch(`/api/conversations/${sessionId}`, {
                method: 'DELETE'
            });
            const data = await response.json();
            if (data.status === 'success') {
                setConversations(prev => prev.filter(c => c.session_id !== sessionId));
            } else {
                setErrorMessage(data.message || 'Failed to delete conversation');
            }
        } catch (error) {
            console.error('Failed to delete conversation:', error);
            setErrorMessage(error?.message || 'Failed to delete conversation');
        }
    };

    const resumeConversation = async (sessionId) => {
        if (resumingId) return;
        setErrorMessage('');
        setSelectedId(sessionId);
        setResumingId(sessionId);
        try {
            const response = await fetch(`/api/resume/${sessionId}`, {
                method: 'POST'
            });
            const data = await response.json();
            if (data.status === 'success') {
                if (onConversationResume) {
                    await onConversationResume(data);
                }
                return;
            }
            console.error('Failed to resume conversation:', data.message || data);
            setErrorMessage(data.message || 'Failed to resume conversation');
        } catch (error) {
            console.error('Failed to resume conversation:', error);
            setErrorMessage(error?.message || 'Failed to resume conversation');
        } finally {
            setResumingId(null);
        }
    };

    const handleSelect = async (sessionId) => {
        await resumeConversation(sessionId);
    };

    const handleResume = async (e, sessionId) => {
        e.stopPropagation();
        await resumeConversation(sessionId);
    };

    const formatDate = (dateString) => {
        if (!dateString) return '';
        const date = new Date(dateString);
        const now = new Date();
        const diff = now - date;

        // Within 24 hours
        if (diff < 86400000) {
            return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        }
        // Within 7 days
        if (diff < 604800000) {
            const days = Math.floor(diff / 86400000);
            return `${days}d ago`;
        }
        // Older
        return date.toLocaleDateString([], { month: 'short', day: 'numeric' });
    };

    if (!isOpen) {
        return null;
    }

    return (
        <>
            {/* Mobile backdrop */}
            <div
                className="fixed inset-0 z-30 bg-ink/30 backdrop-blur-sm md:hidden"
                onClick={onToggle}
                aria-hidden="true"
            />
            <aside className="fixed inset-y-0 left-0 z-40 flex h-full w-72 shrink-0 flex-col border-r border-line bg-surface shadow-lg animate-slide-in-left md:static md:z-30 md:shadow-none">
            {/* Header */}
            <header className="flex h-12 shrink-0 items-center justify-between border-b border-line bg-surface-muted/60 px-4 backdrop-blur">
                <div className="flex items-center gap-2 text-sm font-medium text-ink">
                    <History size={15} className="text-ink-muted" />
                    <span>History</span>
                    {!loading && conversations.length > 0 && (
                        <span className="rounded-full border border-line bg-surface px-1.5 py-px text-[10px] font-mono text-ink-soft">
                            {conversations.length}
                        </span>
                    )}
                </div>
                <IconButton onClick={onToggle} size="sm" aria-label="Close history" title="Close">
                    <ChevronLeft size={16} />
                </IconButton>
            </header>

            {errorMessage && (
                <div className="mx-2 mt-2 flex items-start gap-2 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-700 dark:border-red-500/30 dark:bg-red-500/10 dark:text-red-300">
                    <span className="min-w-0 flex-1 break-words">{errorMessage}</span>
                    <button
                        type="button"
                        onClick={() => setErrorMessage('')}
                        className="shrink-0 rounded p-0.5 hover:bg-red-100 dark:hover:bg-red-500/20"
                        aria-label="Dismiss error"
                    >
                        <X size={12} />
                    </button>
                </div>
            )}

            {/* Conversation List */}
            <div className="flex-1 overflow-y-auto custom-scrollbar">
                {loading ? (
                    <div className="flex items-center justify-center h-32" aria-busy="true">
                        <Loader2 size={18} className="animate-spin text-ink-soft" />
                    </div>
                ) : conversations.length === 0 ? (
                    <div className="flex flex-col items-center justify-center h-48 px-6 text-center text-ink-soft">
                        <div className="mb-3 flex h-12 w-12 items-center justify-center rounded-full border border-line bg-surface-muted">
                            <MessageSquare size={20} className="opacity-70" />
                        </div>
                        <p className="text-sm font-medium text-ink-muted">No saved conversations</p>
                        <p className="text-xs mt-1">Sessions are saved automatically.</p>
                    </div>
                ) : (
                    <ul className="p-2 space-y-0.5" role="list">
                        {conversations.map((conv) => {
                            const isResuming = resumingId === conv.session_id;
                            const isDisabled = Boolean(resumingId);
                            const isSelected = selectedId === conv.session_id;
                            return (
                                <li
                                    key={conv.session_id}
                                    className={cn(
                                        'group relative overflow-hidden rounded-lg transition-colors',
                                        isSelected ? 'bg-accent-soft' : 'hover:bg-surface-muted',
                                    )}
                                >
                                    {isResuming && (
                                        <div className="absolute left-0 top-0 z-10 h-0.5 w-full overflow-hidden bg-accent/20">
                                            <div className="h-full w-1/3 animate-[pulse_1s_ease-in-out_infinite] bg-accent" />
                                        </div>
                                    )}
                                    <button
                                        type="button"
                                        onClick={() => handleSelect(conv.session_id)}
                                        disabled={isDisabled}
                                        aria-busy={isResuming}
                                        className={cn(
                                            'block w-full px-3 py-2.5 pr-16 text-left',
                                            isDisabled ? 'cursor-wait' : 'cursor-pointer',
                                            isResuming && 'opacity-70',
                                            'focus:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-inset rounded-lg',
                                        )}
                                    >
                                        <span className="flex items-center gap-1.5 min-w-0">
                                            {isResuming && (
                                                <Loader2 size={12} className="shrink-0 animate-spin text-accent" />
                                            )}
                                            <span className={cn('truncate text-sm font-medium', isSelected ? 'text-accent-strong' : 'text-ink')}>
                                                {isResuming ? 'Restoring conversation…' : (conv.metadata?.title || 'Untitled')}
                                            </span>
                                        </span>
                                        <span className="mt-1 flex items-center gap-1.5 text-[11px] text-ink-muted">
                                            <Clock size={10} />
                                            <span>{formatDate(conv.metadata?.updated_at)}</span>
                                            <span aria-hidden="true">·</span>
                                            <span>{conv.metadata?.message_count || 0} msg</span>
                                        </span>
                                        {conv.preview && (
                                            <span className="mt-1.5 block text-xs text-ink-soft line-clamp-2 leading-relaxed">
                                                {conv.preview}
                                            </span>
                                        )}
                                    </button>
                                    {pendingDeleteId === conv.session_id ? (
                                        <div className="absolute right-2 top-2 flex items-center gap-0.5">
                                            <button
                                                type="button"
                                                onClick={(e) => confirmDelete(e, conv.session_id)}
                                                className="rounded p-1.5 text-red-600 transition-colors hover:bg-red-50 dark:text-red-300 dark:hover:bg-red-500/10"
                                                title="Confirm delete"
                                                aria-label="Confirm delete conversation"
                                            >
                                                <Check size={13} />
                                            </button>
                                            <button
                                                type="button"
                                                onClick={cancelDelete}
                                                className="rounded p-1.5 text-ink-soft transition-colors hover:bg-surface-muted hover:text-ink"
                                                title="Cancel"
                                                aria-label="Cancel delete"
                                            >
                                                <X size={13} />
                                            </button>
                                        </div>
                                    ) : (
                                        <div className="absolute right-2 top-2 flex items-center gap-0.5 opacity-0 transition-opacity group-hover:opacity-100 group-focus-within:opacity-100">
                                            <button
                                                type="button"
                                                onClick={(e) => handleResume(e, conv.session_id)}
                                                disabled={isDisabled}
                                                className="rounded p-1.5 text-ink-soft transition-colors hover:bg-emerald-50 hover:text-emerald-600 disabled:opacity-40 disabled:hover:bg-transparent dark:hover:bg-emerald-500/10 dark:hover:text-emerald-300"
                                                title="Resume"
                                                aria-label="Resume conversation"
                                            >
                                                {isResuming ? <Loader2 size={13} className="animate-spin" /> : <MessageSquare size={13} />}
                                            </button>
                                            <button
                                                type="button"
                                                onClick={(e) => requestDelete(e, conv.session_id)}
                                                disabled={isDisabled}
                                                className="rounded p-1.5 text-ink-soft transition-colors hover:bg-red-50 hover:text-red-600 disabled:opacity-40 disabled:hover:bg-transparent dark:hover:bg-red-500/10 dark:hover:text-red-300"
                                                title="Delete"
                                                aria-label="Delete conversation"
                                            >
                                                <Trash2 size={13} />
                                            </button>
                                        </div>
                                    )}
                                </li>
                            );
                        })}
                    </ul>
                )}
            </div>
            </aside>
        </>
    );
}
