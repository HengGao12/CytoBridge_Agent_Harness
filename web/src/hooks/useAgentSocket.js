import { useState, useEffect, useCallback, useRef } from 'react';
import {
    appendOptimisticUserMessage,
    createClientTimelineEvent,
    reduceTimelineEvent,
    reduceTimelineEvents,
    thinkingUpdateForEvent,
} from '../lib/timelineReducer';
import {
    mapResumedAttachments,
    shouldShowResumeDiagnostics,
    normalizeResumeDiagnostics,
} from '../lib/restoreHelpers';

// Build WebSocket URL dynamically based on current location
const getDefaultWsUrl = () => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    return `${protocol}//${window.location.host}/ws`;
};

// Read once and cache — this is hit on every inbound WS event.
let cachedDebugSocket;
const shouldDebugSocket = () => {
    if (cachedDebugSocket === undefined) {
        try {
            cachedDebugSocket = window.localStorage?.getItem('debugAgentSocket') === '1';
        } catch {
            cachedDebugSocket = false;
        }
    }
    return cachedDebugSocket;
};

const proposalStatusFromDecision = (decision) => {
    const normalized = String(decision || '').toLowerCase();
    if (normalized === 'approve') return 'approved';
    if (normalized === 'reject') return 'rejected';
    if (normalized === 'revise') return 'revise';
    return normalized || 'reviewed';
};

export function useAgentSocket(url = getDefaultWsUrl()) {
    const [timeline, setTimeline] = useState([]);
    const [status, setStatus] = useState('disconnected');
    const [isThinking, setIsThinking] = useState(false);
    const wsRef = useRef(null);
    const pendingSendRef = useRef(false);

    const setRunningState = useCallback((running) => {
        setIsThinking(!!running);
    }, []);

    const applyTimelineEvent = useCallback((payload) => {
        if (shouldDebugSocket()) {
            console.log('[WS Event]', payload?.type, payload?.data || {});
        }
        setTimeline(prev => reduceTimelineEvent(prev, payload));
        const thinkingUpdate = thinkingUpdateForEvent(payload);
        if (typeof thinkingUpdate === 'boolean') {
            setIsThinking(thinkingUpdate);
        }
    }, []);

    useEffect(() => {
        let disposed = false;
        let retryCount = 0;
        let retryTimer = null;
        let ws = null;

        const connect = () => {
            if (disposed) return;
            ws = new WebSocket(url);
            wsRef.current = ws;

            ws.onopen = () => {
                retryCount = 0;
                setStatus('connected');
                if (shouldDebugSocket()) {
                    console.log('Connected to Agent Socket');
                }
            };

            ws.onclose = () => {
                if (disposed) return;
                setStatus('disconnected');
                // Auto-reconnect with capped exponential backoff (1s → 15s).
                const delay = Math.min(15000, 1000 * 2 ** retryCount);
                retryCount += 1;
                if (shouldDebugSocket()) {
                    console.log(`Agent Socket closed — reconnecting in ${delay}ms`);
                }
                retryTimer = setTimeout(connect, delay);
            };

            ws.onerror = (error) => {
                if (disposed) return;
                console.error('WebSocket Error:', error);
                setStatus('error');
            };

            ws.onmessage = (event) => {
                try {
                    const payload = JSON.parse(event.data);
                    applyTimelineEvent(payload);
                } catch (e) {
                    console.error('Failed to parse message', e);
                }
            };
        };

        connect();

        return () => {
            disposed = true;
            if (retryTimer) clearTimeout(retryTimer);
            ws?.close();
        };
    }, [url, applyTimelineEvent]);

    const applyResumedConfig = useCallback((data, resumeResult = {}) => {
        if (!data || typeof data !== 'object') return;
        setRunningState(!!data.isAgentRunning);
        if (Array.isArray(data.eventsLog) && data.eventsLog.length > 0) {
            setTimeline(reduceTimelineEvents(data.eventsLog));
        } else if (Array.isArray(data.conversationHistory) && data.conversationHistory.length > 0) {
            const restoredTimeline = [];
            restoredTimeline.push({ type: 'system', message: 'Session resumed from checkpoint.' });
            for (const msg of data.conversationHistory) {
                if (msg.role === 'user') {
                    restoredTimeline.push({
                        type: 'user',
                        content: msg.content,
                        attachments: mapResumedAttachments(msg.attachments),
                        timestamp: msg.timestamp || new Date().toISOString()
                    });
                } else if (msg.role === 'assistant') {
                    restoredTimeline.push({
                        type: 'assistant',
                        content: msg.content,
                        timestamp: msg.timestamp || new Date().toISOString()
                    });
                }
            }
            setTimeline(restoredTimeline);
        } else {
            setTimeline([{ type: 'system', message: 'Session resumed. Ready to continue.' }]);
        }

        const diagnostics = data.resumeDiagnostics || resumeResult?.resume_diagnostics;
        if (shouldShowResumeDiagnostics(diagnostics)) {
            setTimeline(prev => [
                ...prev,
                {
                    type: 'resume_diagnostics',
                    diagnostics: normalizeResumeDiagnostics(diagnostics),
                    timestamp: new Date().toISOString()
                }
            ]);
        }
    }, [setRunningState]);

    const sendMessage = useCallback((text, attachments = []) => {
        if (pendingSendRef.current) {
            return false;
        }
        pendingSendRef.current = true;
        const clientRequestId = globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random().toString(36).slice(2)}`;
        setTimeline(prev => appendOptimisticUserMessage(prev, text, attachments, clientRequestId));
        setIsThinking(true);

        fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: text, attachments, client_request_id: clientRequestId })
        }).then(res => res.json())
            .then(data => {
                if (data.command_handled) {
                    setIsThinking(false);
                    if (data.clear_timeline) {
                        setTimeline([]);
                    }
                    const events = Array.isArray(data.events) ? data.events : [];
                    for (const event of events) {
                        applyTimelineEvent(event);
                    }
                    if (data.resume?.config) {
                        applyResumedConfig(data.resume.config, data.resume.resume_result);
                    }
                    if (data.status !== 'success' && events.length === 0) {
                        setTimeline(prev => [...prev, { type: 'error', message: data.message || 'Command failed' }]);
                    }
                    return;
                }
                if (data.status !== 'success') {
                    setTimeline(prev => [...prev, {
                        type: 'error',
                        message: data.message || 'Failed to send',
                        retry_text: text,
                        retry_attachments: attachments,
                    }]);
                    setIsThinking(false);
                }
            })
            .catch(err => {
                setTimeline(prev => [...prev, {
                    type: 'error',
                    message: err.message,
                    retry_text: text,
                    retry_attachments: attachments,
                }]);
                setIsThinking(false);
            })
            .finally(() => {
                pendingSendRef.current = false;
            });
        return true;
    }, [applyResumedConfig, applyTimelineEvent]);

    const stopAgent = useCallback(async () => {
        try {
            const res = await fetch('/api/stop', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });
            const data = await res.json();
            if (data.status !== 'success') {
                console.error('Failed to stop agent:', data.message);
                return;
            }
            setIsThinking(!!data.thread_alive);
        } catch (err) {
            console.error('Stop request failed:', err);
        }
    }, []);

    const initSession = useCallback(async (config) => {
        try {
            const res = await fetch('/api/init', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(config)
            });
            const data = await res.json();
            if (data.status === 'success') {
                setTimeline([{ type: 'system', message: 'Session initialized. Ready to chat.' }]);
                return true;
            }
            setTimeline(prev => [...prev, { type: 'error', message: `Init Failed: ${data.message}` }]);
            return false;
        } catch (e) {
            setTimeline(prev => [...prev, { type: 'error', message: `Init Error: ${e.message}` }]);
            return false;
        }
    }, []);

    const switchModel = useCallback(async (config) => {
        try {
            const res = await fetch('/api/switch_model', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    llm_model: config.llm_model,
                    llm_provider: config.llm_provider,
                    llm_base_url: config.llm_base_url,
                    openai_api_key: config.openai_api_key,
                    llm_auth_mode: config.llm_auth_mode,
                    llm_profile_id: config.llm_profile_id,
                    llm_thinking_level: config.llm_thinking_level,
                    allow_large_context_window: !!config.allow_large_context_window,
                    algorithm_proposal_review_mode: config.algorithm_proposal_review_mode,
                    idea_review_mode: config.idea_review_mode,
                    stop_hook_enabled: !!config.stop_hook_enabled,
                    stop_hook_mode: config.stop_hook_mode || 'prompt',
                    stop_hook_prompt: config.stop_hook_prompt || '',
                    stop_hook_max_triggers: Number(config.stop_hook_max_triggers ?? 20),
                    persist: true,
                })
            });
            const data = await res.json();
            if (data.status === 'success') {
                return true;
            }
            setTimeline(prev => [...prev, { type: 'error', message: `Switch Model Failed: ${data.message}` }]);
            return false;
        } catch (e) {
            setTimeline(prev => [...prev, { type: 'error', message: `Switch Model Error: ${e.message}` }]);
            return false;
        }
    }, []);

    const updateStopHookSettings = useCallback(async (config) => {
        try {
            const res = await fetch('/api/stop_hook_settings', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    stop_hook_enabled: !!config.stop_hook_enabled,
                    stop_hook_mode: config.stop_hook_mode || 'prompt',
                    stop_hook_prompt: config.stop_hook_prompt || '',
                    stop_hook_max_triggers: Number(config.stop_hook_max_triggers ?? 20),
                })
            });
            const data = await res.json();
            if (data.status === 'success') {
                return true;
            }
            setTimeline(prev => [...prev, { type: 'error', message: `Stop Hook Settings Failed: ${data.message}` }]);
            return false;
        } catch (e) {
            setTimeline(prev => [...prev, { type: 'error', message: `Stop Hook Settings Error: ${e.message}` }]);
            return false;
        }
    }, []);

    const newChatSession = useCallback(async () => {
        try {
            const res = await fetch('/api/new_session', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });
            const data = await res.json();
            if (data.status === 'success') {
                setTimeline([]);
                setIsThinking(false);
                return true;
            }
            setTimeline(prev => [...prev, { type: 'error', message: `New Chat Failed: ${data.message}` }]);
            return false;
        } catch (e) {
            setTimeline(prev => [...prev, { type: 'error', message: `New Chat Error: ${e.message}` }]);
            return false;
        }
    }, []);

    const listSkills = useCallback(async (scope = 'planner') => {
        const params = new URLSearchParams({ scope });
        const res = await fetch(`/api/skills?${params.toString()}`);
        return await res.json();
    }, []);

    const setSkillExposure = useCallback(async (scope, name, exposed) => {
        const res = await fetch('/api/skills/exposure', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ scope, name, exposed: !!exposed })
        });
        return await res.json();
    }, []);

    const setAlgorithmExposure = useCallback(async (name, exposed) => {
        const res = await fetch('/api/algorithms/exposure', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name, exposed: !!exposed })
        });
        return await res.json();
    }, []);

    const reviewAlgorithmProposal = useCallback(async (algorithm_id, decision, reviewer_feedback = '', proposal_id = '') => {
        const res = await fetch('/api/proposals/review', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ algorithm_id, proposal_id, decision, reviewer_feedback })
        });
        const data = await res.json();
        if (data.status === 'success') {
            const proposal = data.proposal && typeof data.proposal === 'object' ? data.proposal : {};
            applyTimelineEvent(createClientTimelineEvent('algorithm_proposal_reviewed', {
                ...proposal,
                algorithm_id: proposal.algorithm_id || algorithm_id,
                proposal_id: proposal.proposal_id || proposal.active_proposal_id || proposal_id,
                decision,
                status: proposal.status || proposalStatusFromDecision(decision),
                review_feedback: proposal.review_feedback || reviewer_feedback,
            }));
        }
        return data;
    }, [applyTimelineEvent]);

    const reviewResearchIdea = useCallback(async (idea_id, decision, reviewer_feedback = '') => {
        const res = await fetch('/api/ideas/review', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ idea_id, decision, reviewer_feedback })
        });
        const data = await res.json();
        if (data.status === 'success') {
            const idea = data.idea && typeof data.idea === 'object' ? data.idea : {};
            applyTimelineEvent(createClientTimelineEvent('research_idea_reviewed', {
                ...idea,
                idea_id: idea.idea_id || idea_id,
                decision,
                status: idea.review_status || idea.status || proposalStatusFromDecision(decision),
                review_feedback: idea.review_feedback || reviewer_feedback,
            }));
        }
        return data;
    }, [applyTimelineEvent]);

    const restoreHistory = useCallback((conversationHistory) => {
        if (!conversationHistory || conversationHistory.length === 0) {
            setTimeline([{ type: 'system', message: 'Session resumed. Ready to continue.' }]);
            return;
        }

        const restoredTimeline = [];
        restoredTimeline.push({ type: 'system', message: 'Session resumed from checkpoint.' });

        for (const msg of conversationHistory) {
            if (msg.role === 'user') {
                restoredTimeline.push({
                    type: 'user',
                    content: msg.content,
                    attachments: mapResumedAttachments(msg.attachments),
                    timestamp: msg.timestamp || new Date().toISOString()
                });
            } else if (msg.role === 'assistant') {
                restoredTimeline.push({
                    type: 'assistant',
                    content: msg.content,
                    timestamp: msg.timestamp || new Date().toISOString()
                });
            }
        }

        setTimeline(restoredTimeline);
    }, []);

    const restoreFromEvents = useCallback((eventsLog) => {
        setTimeline(reduceTimelineEvents(eventsLog));
    }, []);

    const appendResumeDiagnostics = useCallback((diagnostics) => {
        if (!diagnostics) return;
        setTimeline(prev => [
            ...prev,
            {
                type: 'resume_diagnostics',
                diagnostics: normalizeResumeDiagnostics(diagnostics),
                timestamp: new Date().toISOString()
            }
        ]);
    }, []);

    return {
        timeline,
        status,
        isThinking,
        setRunningState,
        sendMessage,
        stopAgent,
        initSession,
        newChatSession,
        switchModel,
        updateStopHookSettings,
        listSkills,
        setSkillExposure,
        setAlgorithmExposure,
        reviewAlgorithmProposal,
        reviewResearchIdea,
        restoreHistory,
        restoreFromEvents,
        appendResumeDiagnostics
    };
}
