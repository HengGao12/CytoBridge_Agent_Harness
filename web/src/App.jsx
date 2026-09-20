import React, { useRef, useEffect, useState, useCallback } from 'react';
import { Bot, Settings as SettingsIcon, History, Plus, ArrowDown, Loader2 } from 'lucide-react';
import { useAgentSocket } from './hooks/useAgentSocket';
import { SetupModal } from './components/SetupModal';
import { ProcessBlock } from './components/ProcessBlock';
import { HistorySidebar } from './components/HistorySidebar';
import { SkillsModal } from './components/SkillsModal';
import { PythonExecutionCard } from './components/PythonExecutionCard';
import { Sidebar } from './components/Sidebar';
import { WelcomeScreen } from './components/WelcomeScreen';
import { MessageComposer } from './components/MessageComposer';
import {
  UserTimelineItem,
  AssistantTimelineItem,
  ProposalTimelineItem,
  ErrorTimelineItem,
  SystemTimelineItem,
  ResumeDiagnosticsTimelineItem,
} from './components/timelineItems';
import { ThemeProvider } from './context/ThemeProvider';
import { ErrorBoundary } from './components/ErrorBoundary';
import { IconButton } from './components/ui';
import { getTimelineItemKey } from './lib/timelineReducer';
import { buildSessionConfigFromData, shouldShowResumeDiagnostics } from './lib/restoreHelpers';

const TIMELINE_INITIAL_RENDER_COUNT = 80;
const TIMELINE_LOAD_CHUNK = 80;

const STOP_HOOK_CONFIG_KEYS = new Set([
  'stop_hook_enabled',
  'stop_hook_mode',
  'stop_hook_prompt',
  'stop_hook_max_triggers',
]);

const normalizeConfigValue = (value) => {
  if (value === undefined || value === null) return '';
  if (typeof value === 'boolean') return value ? 'true' : 'false';
  return String(value);
};

const isStopHookOnlyConfigChange = (prevConfig, nextConfig) => {
  const explicitChangedKeys = Array.isArray(nextConfig?._changed_keys) ? nextConfig._changed_keys : null;
  if (explicitChangedKeys) {
    return explicitChangedKeys.length > 0 && explicitChangedKeys.every((key) => STOP_HOOK_CONFIG_KEYS.has(key));
  }
  const keys = new Set([...Object.keys(prevConfig || {}), ...Object.keys(nextConfig || {})]);
  let sawStopHookChange = false;
  for (const key of keys) {
    if (key === 'openai_api_key') continue;
    const changed = normalizeConfigValue(prevConfig?.[key]) !== normalizeConfigValue(nextConfig?.[key]);
    if (!changed) continue;
    if (!STOP_HOOK_CONFIG_KEYS.has(key)) {
      return false;
    }
    sawStopHookChange = true;
  }
  return sawStopHookChange;
};

function MainLayout() {
  const {
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
    appendResumeDiagnostics,
  } = useAgentSocket();

  const [input, setInput] = useState('');
  const [showConfig, setShowConfig] = useState(true);
  const [showSkills, setShowSkills] = useState(false);
  const [config, setConfig] = useState(null);
  const [hasActiveSession, setHasActiveSession] = useState(false);
  const [showHistory, setShowHistory] = useState(false);
  const [isCheckingSession, setIsCheckingSession] = useState(true);
  const [attachments, setAttachments] = useState([]);
  const [visibleTimelineCount, setVisibleTimelineCount] = useState(TIMELINE_INITIAL_RENDER_COUNT);
  const [showJumpToBottom, setShowJumpToBottom] = useState(false);


  const composerRef = useRef(null);
  const messagesEndRef = useRef(null);
  const chatScrollRef = useRef(null);
  const preserveBottomOnNextTimelineRef = useRef(true);
  const pendingAutoFollowRef = useRef(false);
  const isAtBottomRef = useRef(true);
  const pendingTimelinePrependAnchorRef = useRef(null);
  const scrollRafRef = useRef(0);

  const handleQuickPrompt = useCallback((promptText) => {
    setInput(promptText);
    // Defer focus to next tick so the composer textarea is mounted/updated.
    window.setTimeout(() => composerRef.current?.focus(), 0);
  }, []);

  const resetTimelineWindow = useCallback(() => {
    pendingTimelinePrependAnchorRef.current = null;
    setVisibleTimelineCount(TIMELINE_INITIAL_RENDER_COUNT);
  }, []);

  const effectiveVisibleTimelineCount = Math.min(
    Math.max(visibleTimelineCount, TIMELINE_INITIAL_RENDER_COUNT),
    Math.max(timeline.length, TIMELINE_INITIAL_RENDER_COUNT),
  );
  const hiddenTimelineCount = Math.max(0, timeline.length - effectiveVisibleTimelineCount);
  const visibleTimeline = timeline.slice(hiddenTimelineCount);

  // Check for active session on mount
  useEffect(() => {
    const checkActiveSession = async () => {
      try {
        const res = await fetch('/api/config');
        const data = await res.json();

        if (data.hasActiveSession) {
          setHasActiveSession(true);
          setRunningState(data.isAgentRunning);
          preserveBottomOnNextTimelineRef.current = true;
          resetTimelineWindow();
          // Session already exists (resumed), skip setup modal
          setShowConfig(false);
          setConfig(buildSessionConfigFromData(data));
          // Prefer events log for full timeline restoration (includes tool calls, thoughts, etc.)
          if (data.eventsLog && data.eventsLog.length > 0) {
            restoreFromEvents(data.eventsLog);
          } else if (data.conversationHistory && data.conversationHistory.length > 0) {
            restoreHistory(data.conversationHistory);
          }
          if (shouldShowResumeDiagnostics(data.resumeDiagnostics)) {
            appendResumeDiagnostics(data.resumeDiagnostics);
          }
        } else {
          setHasActiveSession(false);
          setRunningState(false);
        }
      } catch (error) {
        console.error('Failed to check session:', error);
      } finally {
        setIsCheckingSession(false);
      }
    };

    checkActiveSession();
  }, [restoreHistory, restoreFromEvents, appendResumeDiagnostics, resetTimelineWindow, setRunningState]);

  const scrollToBottom = useCallback(() => {
    const scroller = chatScrollRef.current;
    if (scroller) {
      scroller.scrollTop = scroller.scrollHeight;
    }
    messagesEndRef.current?.scrollIntoView({ behavior: 'auto', block: 'end', inline: 'nearest' });
  }, []);

  const hydrateActiveSessionFromConfig = useCallback(async (resumeResult = {}) => {
    const res = await fetch('/api/config');
    const data = await res.json();
    setHasActiveSession(true);
    setRunningState(!!data.isAgentRunning);
    setShowConfig(false);
    setConfig(buildSessionConfigFromData(data));
    preserveBottomOnNextTimelineRef.current = true;
    resetTimelineWindow();
    if (Array.isArray(data.eventsLog) && data.eventsLog.length > 0) {
      restoreFromEvents(data.eventsLog);
    } else if (Array.isArray(data.conversationHistory) && data.conversationHistory.length > 0) {
      restoreHistory(data.conversationHistory);
    }
    const diagnostics = data.resumeDiagnostics || resumeResult?.resume_diagnostics;
    if (shouldShowResumeDiagnostics(diagnostics)) {
      appendResumeDiagnostics(diagnostics);
    }
    return data;
  }, [appendResumeDiagnostics, resetTimelineWindow, restoreFromEvents, restoreHistory, setRunningState]);

  const scheduleBottomFollow = useCallback(() => {
    pendingAutoFollowRef.current = true;
    const snap = () => {
      scrollToBottom();
    };
    requestAnimationFrame(() => {
      snap();
      requestAnimationFrame(snap);
    });
    setTimeout(snap, 40);
    setTimeout(() => {
      pendingAutoFollowRef.current = false;
    }, 220);
  }, [scrollToBottom]);

  useEffect(() => {
    const shouldScroll = timeline.length > 0 && (preserveBottomOnNextTimelineRef.current || isAtBottomRef.current);
    if (shouldScroll) {
      scheduleBottomFollow();
    }
    preserveBottomOnNextTimelineRef.current = false;
  }, [timeline, scheduleBottomFollow]);

  useEffect(() => {
    const anchor = pendingTimelinePrependAnchorRef.current;
    if (!anchor) return;
    pendingTimelinePrependAnchorRef.current = null;
    const scroller = chatScrollRef.current;
    if (!scroller) return;
    scroller.scrollTop = scroller.scrollHeight - anchor.scrollHeight + anchor.scrollTop;
  }, [hiddenTimelineCount, visibleTimelineCount]);

  // Scroll fires rapidly; coalesce the measurement into one rAF per frame.
  const handleTimelineScroll = useCallback(() => {
    if (scrollRafRef.current) return;
    scrollRafRef.current = requestAnimationFrame(() => {
      scrollRafRef.current = 0;
      const scroller = chatScrollRef.current;
      if (!scroller) {
        isAtBottomRef.current = true;
        setShowJumpToBottom(false);
        return;
      }
      const distanceFromBottom = scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight;
      const atBottom = distanceFromBottom < 96;
      isAtBottomRef.current = atBottom;
      setShowJumpToBottom(!atBottom && distanceFromBottom > 240);
    });
  }, []);

  useEffect(() => () => {
    if (scrollRafRef.current) cancelAnimationFrame(scrollRafRef.current);
  }, []);

  const handleJumpToBottom = useCallback(() => {
    isAtBottomRef.current = true;
    setShowJumpToBottom(false);
    scrollToBottom();
  }, [scrollToBottom]);

  const handleLoadEarlierTimeline = useCallback(() => {
    const scroller = chatScrollRef.current;
    pendingTimelinePrependAnchorRef.current = scroller
      ? { scrollHeight: scroller.scrollHeight, scrollTop: scroller.scrollTop }
      : null;
    preserveBottomOnNextTimelineRef.current = false;
    isAtBottomRef.current = false;
    setVisibleTimelineCount((prev) => Math.min(timeline.length, prev + TIMELINE_LOAD_CHUNK));
  }, [timeline.length]);

  const fileToAttachment = useCallback((file) => new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const dataUrl = String(reader.result || '');
      const [, base64 = ''] = dataUrl.split(',', 2);
      resolve({
        type: 'image',
        mime_type: file.type,
        file_name: file.name || 'image',
        content: base64,
        data_url: dataUrl
      });
    };
    reader.onerror = () => reject(reader.error || new Error('Failed to read file'));
    reader.readAsDataURL(file);
  }), []);

  const appendImageFiles = useCallback(async (files) => {
    const imageFiles = Array.from(files || []).filter((file) => String(file.type || '').startsWith('image/'));
    if (!imageFiles.length) return;
    try {
      const next = await Promise.all(imageFiles.map(fileToAttachment));
      setAttachments((prev) => [...prev, ...next]);
    } catch (error) {
      console.error('Failed to load image attachment:', error);
    }
  }, [fileToAttachment]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    const trimmedInput = input.trim();
    const isSlashCommand = trimmedInput.startsWith('/');
    if ((!trimmedInput && attachments.length === 0) || status !== 'connected') return;
    if (isThinking && !isSlashCommand) return;
    preserveBottomOnNextTimelineRef.current = true;
    const accepted = sendMessage(input, attachments);
    if (accepted) {
      setInput('');
      setAttachments([]);
    }
  };

  const handlePaste = useCallback(async (e) => {
    const items = Array.from(e.clipboardData?.items || []);
    const imageItems = items.filter((item) => String(item.type || '').startsWith('image/'));
    if (!imageItems.length) return;
    e.preventDefault();
    const files = imageItems.map((item) => item.getAsFile()).filter(Boolean);
    await appendImageFiles(files);
  }, [appendImageFiles]);

  const removeAttachment = useCallback((idx) => {
    setAttachments((prev) => prev.filter((_, index) => index !== idx));
  }, []);

  const handleConfigSubmit = async (newConfig) => {
    const effectiveConfig = { ...newConfig };
    delete effectiveConfig._changed_keys;
    let success = false;
    if (!hasActiveSession) {
      success = await initSession(effectiveConfig);
    } else if (isStopHookOnlyConfigChange(config, newConfig)) {
      success = await updateStopHookSettings(effectiveConfig);
    } else {
      success = await switchModel(effectiveConfig);
    }
    if (success) {
      setConfig(effectiveConfig);
      setShowConfig(false);
      setHasActiveSession(true);
    }
  };

  const handleProposalDecision = useCallback(async (algorithmId, decision, feedback = '', proposalId = '') => {
    try {
      const resp = await reviewAlgorithmProposal(algorithmId, decision, feedback, proposalId);
      if (!resp || resp.status !== 'success') {
        console.error('Failed to review proposal:', resp?.message || 'unknown error');
      }
      return resp;
    } catch (error) {
      console.error('Proposal review request failed:', error);
      return { status: 'error', message: String(error?.message || error) };
    }
  }, [reviewAlgorithmProposal]);

  const handleIdeaDecision = useCallback(async (ideaId, decision, feedback = '') => {
    try {
      const resp = await reviewResearchIdea(ideaId, decision, feedback);
      if (!resp || resp.status !== 'success') {
        console.error('Failed to review research idea:', resp?.message || 'unknown error');
      }
      return resp;
    } catch (error) {
      console.error('Research idea review request failed:', error);
      return { status: 'error', message: String(error?.message || error) };
    }
  }, [reviewResearchIdea]);

  const handleNewChat = async () => {
    const ok = await newChatSession();
    if (!ok) return;
    preserveBottomOnNextTimelineRef.current = true;
    resetTimelineWindow();
    setShowConfig(true);
    setHasActiveSession(false);
  };

  const handleConversationResume = useCallback(async (resumeResult) => {
    try {
      await hydrateActiveSessionFromConfig(resumeResult);
    } catch (error) {
      console.error('Failed to hydrate resumed conversation:', error);
    }
  }, [hydrateActiveSessionFromConfig]);

  const renderTimelineItem = useCallback((index, item) => {
    switch (item.type) {
      case 'process':
        return (
          <ProcessBlock
            steps={item.steps}
            isComplete={item.isComplete}
            isActive={!item.isComplete && index === timeline.length - 1}
            currentStage={item.currentStage}
            stageDescription={item.stageDescription}
            onProposalDecision={handleProposalDecision}
            onIdeaDecision={handleIdeaDecision}
          />
        );
      case 'user':
        return <UserTimelineItem content={item.content} attachments={item.attachments} timestamp={item.timestamp} />;
      case 'assistant':
        return <AssistantTimelineItem content={item.content} timestamp={item.timestamp} />;
      case 'python_execution':
        return <PythonExecutionCard item={item} />;
      case 'proposal_event':
        return (
          <ProposalTimelineItem
            item={item}
            onProposalDecision={handleProposalDecision}
            onIdeaDecision={handleIdeaDecision}
          />
        );
      case 'error':
        return (
          <ErrorTimelineItem
            message={item.message}
            onRetry={item.retry_text ? () => {
              preserveBottomOnNextTimelineRef.current = true;
              sendMessage(item.retry_text, item.retry_attachments || []);
            } : undefined}
          />
        );
      case 'system':
        return <SystemTimelineItem message={item.message} />;
      case 'resume_diagnostics':
        return <ResumeDiagnosticsTimelineItem diagnostics={item.diagnostics} />;
      default:
        return null;
    }
  }, [timeline.length, handleProposalDecision, handleIdeaDecision, sendMessage]);

  const canSend = !!input.trim() || attachments.length > 0;
  const isEmpty = timeline.length === 0;

  return (
    <div className="relative flex h-screen w-full overflow-hidden bg-surface text-ink">
      {/* Subtle ambient backdrop — masked dot grid + faint top accent wash */}
      <div className="pointer-events-none fixed inset-0 -z-10 bg-ambient-glow" />
      <div className="pointer-events-none fixed inset-0 -z-10 bg-dot-grid opacity-50 dark:opacity-30" />

      {/* History drawer */}
      <HistorySidebar
        isOpen={showHistory}
        onToggle={() => setShowHistory(!showHistory)}
        onConversationResume={handleConversationResume}
      />

      {/* Left rail */}
      <Sidebar
        status={status}
        showHistory={showHistory}
        onToggleHistory={() => setShowHistory(!showHistory)}
        onOpenSettings={() => setShowConfig(true)}
        onOpenSkills={() => setShowSkills(true)}
        onNewChat={handleNewChat}
      />

      {/* Main column */}
      <div className="relative flex flex-1 flex-col w-full h-full">
        {/* Modals */}
        {showConfig && !isCheckingSession && (
          <SetupModal
            onConfigSubmit={handleConfigSubmit}
            onClose={() => setShowConfig(false)}
            initialConfig={config}
          />
        )}
        <SkillsModal
          isOpen={showSkills}
          onClose={() => setShowSkills(false)}
          listSkills={listSkills}
          setSkillExposure={setSkillExposure}
          setAlgorithmExposure={setAlgorithmExposure}
          hasActiveSession={hasActiveSession}
        />

        {/* Mobile header */}
        <div className="md:hidden sticky top-0 z-10 flex h-12 items-center justify-between border-b border-line bg-surface/80 px-4 backdrop-blur-md">
          <span className="flex items-center gap-2 text-sm font-semibold tracking-tight text-ink">
            <span className="flex h-6 w-6 items-center justify-center rounded-md bg-accent text-white shadow-sm">
              <Bot size={14} strokeWidth={2.25} />
            </span>
            CellCompass
          </span>
          <div className="flex items-center gap-0.5">
            <IconButton onClick={() => setShowHistory(!showHistory)} size="sm" aria-label="History" active={showHistory}>
              <History size={16} />
            </IconButton>
            <IconButton onClick={handleNewChat} size="sm" aria-label="New chat">
              <Plus size={16} />
            </IconButton>
            <IconButton onClick={() => setShowConfig(true)} size="sm" aria-label="Settings">
              <SettingsIcon size={16} />
            </IconButton>
          </div>
        </div>

        {/* Chat / welcome scroller */}
        <div
          ref={chatScrollRef}
          onScroll={handleTimelineScroll}
          className="flex-1 overflow-y-auto custom-scrollbar"
        >
          {isCheckingSession ? (
            <div className="flex h-full flex-col items-center justify-center gap-3 text-ink-soft animate-fade-in">
              <Loader2 size={22} className="animate-spin text-accent" />
              <span className="text-xs">Restoring your workspace…</span>
            </div>
          ) : isEmpty ? (
            <WelcomeScreen onQuickPrompt={handleQuickPrompt} />
          ) : (
            <div className="pt-6 px-2 sm:px-0">
              {hiddenTimelineCount > 0 && (
                <div className="mx-auto max-w-3xl py-3 flex justify-center">
                  <button
                    type="button"
                    onClick={handleLoadEarlierTimeline}
                    className="rounded-full border border-line bg-surface px-3.5 py-1.5 text-xs font-medium text-ink-muted shadow-xs transition-all hover:border-line-strong hover:text-ink hover:shadow-sm active:scale-[0.98] focus:outline-none focus-visible:ring-2 focus-visible:ring-accent"
                  >
                    Load earlier messages ({hiddenTimelineCount} hidden)
                  </button>
                </div>
              )}
              {visibleTimeline.map((item, localIndex) => {
                const index = hiddenTimelineCount + localIndex;
                const itemKey = getTimelineItemKey(index, item);
                return (
                  <div key={itemKey} className="mx-auto max-w-3xl py-3">
                    <ErrorBoundary resetKey={itemKey}>
                      {renderTimelineItem(index, item)}
                    </ErrorBoundary>
                  </div>
                );
              })}
              {hiddenTimelineCount > 0 && (
                <div className="mx-auto max-w-3xl pb-2 text-center text-[11px] font-mono text-ink-soft">
                  Showing latest {visibleTimeline.length} of {timeline.length} timeline items
                </div>
              )}
              <div ref={messagesEndRef} className="h-4" />
            </div>
          )}
        </div>

        {/* Jump to bottom */}
        {!isEmpty && showJumpToBottom && (
          <button
            type="button"
            onClick={handleJumpToBottom}
            aria-label="Jump to latest"
            className="absolute bottom-36 left-1/2 z-10 flex h-9 w-9 -translate-x-1/2 items-center justify-center rounded-full border border-line bg-surface text-ink-muted shadow-md transition-all animate-fade-in hover:text-ink hover:shadow-lg active:scale-95 focus:outline-none focus-visible:ring-2 focus-visible:ring-accent"
          >
            <ArrowDown size={16} />
          </button>
        )}

        {/* Composer */}
        <MessageComposer
          ref={composerRef}
          value={input}
          onChange={setInput}
          onSubmit={handleSubmit}
          onStop={stopAgent}
          onOpenSkills={() => setShowSkills(true)}
          onPickAttachment={appendImageFiles}
          onPaste={handlePaste}
          onRemoveAttachment={removeAttachment}
          attachments={attachments}
          canSend={canSend}
          isThinking={isThinking}
          connected={status === 'connected'}
        />
      </div>
    </div>
  );
}

function App() {
  return (
    <ThemeProvider>
      <MainLayout />
    </ThemeProvider>
  );
}

export default App;
