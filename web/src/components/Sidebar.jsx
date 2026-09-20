import React from 'react';
import { Bot, History, Settings as SettingsIcon, BookOpen, Plus, Moon, Sun } from 'lucide-react';
import { IconButton, StatusDot } from './ui';
import { cn } from './ui/cn';
import { useTheme } from '../context/useTheme';

/**
 * Sidebar — left rail used on >=md viewports.
 *
 * Hosts: brand mark, primary navigation icons (history / settings / skills /
 * new chat), connection status, theme toggle. Width is intentionally narrow
 * (`w-14`) — content lives in the main column.
 */
export function Sidebar({
    status,
    showHistory,
    onToggleHistory,
    onOpenSettings,
    onOpenSkills,
    onNewChat,
}) {
    const { theme, toggleTheme } = useTheme();
    const isDark = theme === 'dark';

    return (
        <aside className="hidden md:flex flex-col w-14 items-center py-4 border-r border-line bg-surface-muted/60 backdrop-blur-md">
            <button
                type="button"
                onClick={onNewChat}
                className={cn(
                    'flex h-9 w-9 items-center justify-center rounded-xl mb-4',
                    'bg-accent text-white shadow-sm transition-all duration-200',
                    'hover:scale-105 hover:shadow-md active:scale-95',
                    'focus:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 focus-visible:ring-offset-surface',
                )}
                aria-label="CellCompass — start new chat"
                title="New chat"
            >
                <Bot size={18} strokeWidth={2.25} />
            </button>

            <div className="mb-4 h-px w-8 bg-line" aria-hidden="true" />

            <nav className="flex flex-col gap-1.5" aria-label="Primary">
                <IconButton onClick={onToggleHistory} active={showHistory} aria-label="Conversation history" title="History">
                    <History size={18} />
                </IconButton>
                <IconButton onClick={onOpenSettings} aria-label="Settings" title="Settings">
                    <SettingsIcon size={18} />
                </IconButton>
                <IconButton onClick={onOpenSkills} aria-label="Skills" title="Skills">
                    <BookOpen size={18} />
                </IconButton>
                <IconButton onClick={onNewChat} aria-label="New chat" title="New chat">
                    <Plus size={18} />
                </IconButton>
            </nav>

            <div className="mt-auto flex flex-col items-center gap-3">
                <div
                    className="flex h-7 w-7 items-center justify-center rounded-full border border-line bg-surface"
                    title={`Status: ${status}`}
                >
                    <StatusDot status={status === 'connected' ? 'connected' : 'disconnected'} />
                </div>
                <IconButton
                    onClick={toggleTheme}
                    aria-label={isDark ? 'Switch to light theme' : 'Switch to dark theme'}
                    title="Toggle theme"
                >
                    {isDark ? <Sun size={18} /> : <Moon size={18} />}
                </IconButton>
            </div>
        </aside>
    );
}
