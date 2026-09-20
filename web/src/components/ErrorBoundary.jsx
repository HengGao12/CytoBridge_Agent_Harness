import React from 'react';
import { AlertTriangle } from 'lucide-react';

/**
 * ErrorBoundary — contains render-time crashes (e.g. a malformed markdown /
 * KaTeX node) to a single timeline item instead of white-screening the app.
 * Pass a stable `resetKey`; when it changes the boundary clears its error so a
 * recovered item can render again.
 */
export class ErrorBoundary extends React.Component {
    constructor(props) {
        super(props);
        this.state = { error: null };
    }

    static getDerivedStateFromError(error) {
        return { error };
    }

    componentDidUpdate(prevProps) {
        if (this.state.error && prevProps.resetKey !== this.props.resetKey) {
            this.setState({ error: null });
        }
    }

    componentDidCatch(error, info) {
        console.error('Timeline item failed to render:', error, info);
    }

    render() {
        if (this.state.error) {
            if (this.props.fallback) return this.props.fallback;
            return (
                <div className="flex justify-center px-2">
                    <div className="inline-flex items-center gap-2 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-700 dark:border-red-500/30 dark:bg-red-500/10 dark:text-red-300">
                        <AlertTriangle size={14} className="shrink-0" />
                        <span>This item could not be displayed.</span>
                    </div>
                </div>
            );
        }
        return this.props.children;
    }
}
