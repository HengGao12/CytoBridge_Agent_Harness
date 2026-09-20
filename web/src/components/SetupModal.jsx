import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
    AlertTriangle,
    ArrowRight,
    CheckCircle2,
    Cpu,
    Loader2,
    Eye,
    FileText,
    FolderOpen,
    Key,
    Server,
    Settings,
    SlidersHorizontal,
    X,
    ChevronDown,
    ChevronRight,
} from 'lucide-react';
import {
    Button,
    Help,
    IconButton,
    Input,
    Label,
    Pill,
    Section,
    Select,
    Textarea,
    ToggleRow,
} from './ui';
import { cn } from './ui/cn';

const DEFAULT_CONFIG = {
    input_path: '',
    output_path: '',
    openai_api_key: '',
    llm_base_url: '',
    llm_provider: 'auto',
    llm_auth_mode: 'auto',
    llm_profile_id: '',
    llm_thinking_level: 'low',
    llm_model: 'gpt-4o',
    allow_large_context_window: false,
    device: 'cuda',
    enable_multimodal: true,
    algorithm_proposal_review_mode: 'agent_decide',
    idea_review_mode: 'agent_decide',
    stop_hook_enabled: false,
    stop_hook_mode: 'prompt',
    stop_hook_prompt:
        'Before allowing the planner to finish, check whether an automated custom algorithm-design task has completed a meaningful algorithm lifecycle: the proposal was reviewed or intentionally skipped by policy, the workspace implementation matches the approved proposal, required implementation/inference reviews passed, campaign evidence was collected through the configured gates, Stage 1 feasibility passed, Stage 2 claim validation passed, Stage 3 tuning/generalization was completed or intentionally advanced according to policy, and final_regression produced a final result / locked release. Passing Stage 1 alone, or having only a promoted trial without a formal stage gate, is not completion. This stop hook is an autonomy guard: do not ask the user for clarification or confirmation, and do not allow the planner to stop just because it wants user input. If the current user request is not an algorithm-lifecycle task, pass only when that concrete request is complete. Return block whenever the planner is trying to stop, ask the user, or wait for confirmation before the relevant lifecycle or requested task is actually complete. A terminal response that only says the planner will inspect, edit, run, test, train, evaluate, or otherwise do work next is not an acceptable final answer; return block so the planner continues executing.',
    stop_hook_max_triggers: 20,
};

const MODEL_SUGGESTIONS = [
    'gpt-5.5',
    'gpt-5.4',
    'gpt-5.4-mini',
    'gpt-4o',
    'gemini-2.5-pro',
    'gemini-2.5-flash',
];

const authLabels = {
    auto: 'Auto',
    api_key: 'API Key',
    gemini_oauth: 'Google OAuth',
    codex_oauth: 'Codex OAuth',
};

const FALLBACK_PROVIDER_OPTIONS = [
    { id: 'auto', display_name: 'Auto' },
    { id: 'openai-compatible', display_name: 'OpenAI-compatible' },
    { id: 'openai', display_name: 'OpenAI API' },
    { id: 'openrouter', display_name: 'OpenRouter' },
    { id: 'xiaomi', display_name: 'Xiaomi MiMo' },
    { id: 'deepseek', display_name: 'DeepSeek' },
    { id: 'zai', display_name: 'Z.AI / GLM' },
    { id: 'kimi-for-coding', display_name: 'Kimi / Moonshot' },
    { id: 'openai-codex', display_name: 'OpenAI Codex OAuth' },
    { id: 'google-gemini-cli', display_name: 'Google Gemini OAuth' },
];

const reviewLabels = {
    always_user_review: 'User review required',
    agent_decide: 'Agent decides',
    auto_approve: 'Auto approve',
};

/* ---------- Local helpers (design-token only) ---------- */

function SidePanel({ label, children }) {
    return (
        <div className="rounded-xl border border-line bg-surface/80 p-4 shadow-xs backdrop-blur-sm">
            <div className="mb-2 text-[11px] font-medium text-ink-soft">
                {label}
            </div>
            {children}
        </div>
    );
}

function UsageCard({ usage }) {
    const windows = Array.isArray(usage.windows) ? usage.windows : [];
    return (
        <div className="rounded-lg border border-line bg-surface p-3">
            <div className="flex items-start justify-between gap-3">
                <div className="min-w-0">
                    <div className="truncate font-mono text-xs font-semibold text-ink">
                        {usage.profile_id}
                    </div>
                    <div
                        className={cn(
                            'mt-1 text-xs',
                            usage.error ? 'text-amber-700 dark:text-amber-300' : 'text-ink-soft',
                        )}
                    >
                        {usage.error
                            ? `Usage unavailable: ${usage.error}`
                            : `Plan: ${usage.plan || 'unknown'}`}
                    </div>
                </div>
                {usage.error ? (
                    <AlertTriangle size={14} className="flex-shrink-0 text-amber-500" />
                ) : (
                    <CheckCircle2 size={14} className="flex-shrink-0 text-emerald-500" />
                )}
            </div>
            {!usage.error && windows.length > 0 && (
                <div className="mt-3 space-y-2">
                    {windows.map((window) => {
                        const percent = Math.max(
                            0,
                            Math.min(100, Math.round(Number(window.used_percent || 0))),
                        );
                        return (
                            <div key={`${usage.profile_id}-${window.label}`}>
                                <div className="mb-1 flex justify-between text-[11px] text-ink-soft">
                                    <span>{window.label}</span>
                                    <span>{percent}%</span>
                                </div>
                                <div className="h-1.5 overflow-hidden rounded-full bg-line">
                                    <div
                                        className={cn(
                                            'h-full rounded-full',
                                            percent > 85 ? 'bg-amber-400' : 'bg-accent',
                                        )}
                                        style={{ width: `${percent}%` }}
                                    />
                                </div>
                            </div>
                        );
                    })}
                </div>
            )}
        </div>
    );
}

/* ---------- Component ---------- */

export function SetupModal({ onConfigSubmit, onClose, initialConfig }) {
    const [profileRenderNow] = useState(() => Date.now());
    const [saving, setSaving] = useState(false);
    const [saveError, setSaveError] = useState('');
    const [config, setConfig] = useState(DEFAULT_CONFIG);
    const [baselineConfig, setBaselineConfig] = useState(DEFAULT_CONFIG);
    const [isConfigLoading, setIsConfigLoading] = useState(true);
    const [showAdvanced, setShowAdvanced] = useState(false);
    const [logoFailed, setLogoFailed] = useState(false);
    const codexUsageFetchStarted = useRef(false);

    useEffect(() => {
        let isMounted = true;
        fetch('/api/config')
            .then((res) => res.json())
            .then((data) => {
                if (!isMounted) return;
                if (data && Object.keys(data).length > 0) {
                    const merged = { ...DEFAULT_CONFIG, ...data };
                    setConfig(merged);
                    setBaselineConfig(merged);
                } else if (initialConfig) {
                    const merged = { ...DEFAULT_CONFIG, ...initialConfig };
                    setConfig(merged);
                    setBaselineConfig(merged);
                }
            })
            .catch((err) => {
                console.error('Failed to fetch initial config', err);
                if (isMounted && initialConfig) {
                    const merged = { ...DEFAULT_CONFIG, ...initialConfig };
                    setConfig(merged);
                    setBaselineConfig(merged);
                }
            })
            .finally(() => {
                if (isMounted) {
                    setIsConfigLoading(false);
                }
            });
        return () => {
            isMounted = false;
        };
    }, [initialConfig]);

    useEffect(() => {
        if (isConfigLoading || !config.has_codex_oauth_profiles || codexUsageFetchStarted.current) {
            return undefined;
        }

        codexUsageFetchStarted.current = true;
        let isMounted = true;
        setConfig((prev) => ({
            ...prev,
            codex_oauth_usage: {
                provider: 'openai-codex',
                profiles: [],
                ...(prev.codex_oauth_usage || {}),
                status: 'loading',
            },
        }));

        fetch('/api/auth/codex/usage')
            .then((res) => res.json())
            .then((data) => {
                if (!isMounted) return;
                const usage = data?.usage && typeof data.usage === 'object'
                    ? data.usage
                    : { provider: 'openai-codex', profiles: [], status: 'unavailable', error: 'Usage response was empty.' };
                setConfig((prev) => ({
                    ...prev,
                    codex_oauth_usage: {
                        provider: 'openai-codex',
                        profiles: [],
                        ...usage,
                        status: usage.status || (usage.error ? 'unavailable' : 'loaded'),
                    },
                }));
            })
            .catch((err) => {
                if (!isMounted) return;
                setConfig((prev) => ({
                    ...prev,
                    codex_oauth_usage: {
                        provider: 'openai-codex',
                        profiles: [],
                        status: 'unavailable',
                        error: err?.message || String(err),
                    },
                }));
            });

        return () => {
            isMounted = false;
        };
    }, [config.has_codex_oauth_profiles, isConfigLoading]);

    const codexProfiles = Array.isArray(config.codex_oauth_status?.profiles)
        ? config.codex_oauth_status.profiles
        : [];
    const codexUsage = Array.isArray(config.codex_oauth_usage?.profiles)
        ? config.codex_oauth_usage.profiles
        : [];
    const codexUsageStatus = config.codex_oauth_usage?.status || '';
    const codexUsageMessage = codexUsageStatus === 'loading' || codexUsageStatus === 'pending'
        ? 'Checking Codex usage in background...'
        : config.codex_oauth_usage?.error
            ? `Codex usage unavailable: ${config.codex_oauth_usage.error}`
            : 'Codex usage data is not available yet.';
    const providerOptions = Array.isArray(config.llm_provider_options) && config.llm_provider_options.length
        ? config.llm_provider_options
        : FALLBACK_PROVIDER_OPTIONS;
    const selectedProvider = config.llm_provider || 'auto';
    const selectedProviderOption = providerOptions.find((provider) => provider.id === selectedProvider);
    const hasCurrentProviderApiKey = !!(
        config.has_current_provider_api_key ||
        config.has_provider_api_keys?.[selectedProvider] ||
        config.has_openai_api_key
    );
    const modelSuggestions = Array.isArray(selectedProviderOption?.model_ids) && selectedProviderOption.model_ids.length
        ? selectedProviderOption.model_ids
        : MODEL_SUGGESTIONS;
    const effectiveOAuthProvider = selectedProvider === 'openai-codex'
        ? 'codex_oauth'
        : (selectedProvider === 'google-gemini-cli' ? 'gemini_oauth' : config.llm_auth_mode);
    const showCodexProfiles = effectiveOAuthProvider === 'codex_oauth' || config.has_codex_oauth_profiles;
    const usesApiKeyInput = !['codex_oauth', 'gemini_oauth'].includes(effectiveOAuthProvider);

    const authSummary = useMemo(() => {
        if (effectiveOAuthProvider === 'codex_oauth') {
            return config.has_codex_oauth_profiles
                ? `${codexProfiles.length} Codex profile${codexProfiles.length === 1 ? '' : 's'} detected`
                : 'CLI login required';
        }
        if (effectiveOAuthProvider === 'gemini_oauth') {
            return config.has_gemini_oauth_credentials ? 'Google OAuth ready' : 'OAuth credentials not found';
        }
        if (config.llm_auth_mode === 'api_key' || selectedProvider !== 'auto') {
            return hasCurrentProviderApiKey || config.openai_api_key ? 'API key configured' : 'API key required';
        }
        return config.has_any_api_key || config.has_codex_oauth_profiles || config.has_gemini_oauth_credentials
            ? 'Auto can resolve credentials'
            : 'No saved credentials detected';
    }, [
        codexProfiles.length,
        config.has_codex_oauth_profiles,
        config.has_gemini_oauth_credentials,
        config.has_any_api_key,
        effectiveOAuthProvider,
        hasCurrentProviderApiKey,
        config.llm_auth_mode,
        config.openai_api_key,
        selectedProvider,
    ]);

    const authTone = useMemo(() => {
        if (effectiveOAuthProvider === 'codex_oauth') return config.has_codex_oauth_profiles ? 'success' : 'warn';
        if (effectiveOAuthProvider === 'gemini_oauth') return config.has_gemini_oauth_credentials ? 'success' : 'warn';
        if (config.llm_auth_mode === 'api_key' || selectedProvider !== 'auto') return hasCurrentProviderApiKey || config.openai_api_key ? 'success' : 'warn';
        return config.has_any_api_key || config.has_codex_oauth_profiles || config.has_gemini_oauth_credentials ? 'success' : 'warn';
    }, [
        config.has_codex_oauth_profiles,
        config.has_gemini_oauth_credentials,
        config.has_any_api_key,
        hasCurrentProviderApiKey,
        effectiveOAuthProvider,
        config.llm_auth_mode,
        config.openai_api_key,
        selectedProvider,
    ]);

    const handleProviderChange = (e) => {
        const provider = e.target.value;
        const option = providerOptions.find((p) => p.id === provider);

        const defaultModel = option?.default_model || '';
        let defaultBaseUrl = '';

        if (provider === 'openai') {
            defaultBaseUrl = 'https://api.openai.com/v1';
        } else if (provider === 'openrouter') {
            defaultBaseUrl = 'https://openrouter.ai/api/v1';
        } else if (provider === 'deepseek') {
            defaultBaseUrl = 'https://api.deepseek.com';
        } else if (provider === 'zai') {
            defaultBaseUrl = 'https://open.bigmodel.cn/api/paas/v4';
        } else if (provider === 'kimi-for-coding') {
            defaultBaseUrl = 'https://api.moonshot.cn/v1';
        } else if (provider === 'openai-compatible') {
            defaultBaseUrl = 'http://localhost:8000/v1';
        }

        setConfig((prev) => ({
            ...prev,
            llm_provider: provider,
            llm_model: defaultModel || prev.llm_model,
            llm_base_url: defaultBaseUrl,
            llm_auth_mode: (provider === 'google-gemini-cli') ? 'gemini_oauth'
                           : (provider === 'openai-codex') ? 'codex_oauth'
                           : prev.llm_auth_mode,
        }));
    };

    const handleChange = (e) => {
        const { name, value, type, checked } = e.target;
        setConfig((prev) => ({ ...prev, [name]: type === 'checkbox' ? checked : value }));
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setSaving(true);
        setSaveError('');

        const payload = { ...config };
        if (
            String(payload.llm_model || '').startsWith('gemini') &&
            ['https://api.openai.com/v1', 'https://api.openai.com/v1/'].includes(String(payload.llm_base_url || '').trim())
        ) {
            payload.llm_base_url = '';
        }
        if (!String(payload.llm_profile_id || '').trim()) {
            payload.llm_profile_id = '';
        }
        payload.stop_hook_enabled = !!payload.stop_hook_enabled;
        payload.allow_large_context_window = !!payload.allow_large_context_window;
        payload.stop_hook_mode = ['builtin', 'prompt', 'agent'].includes(String(payload.stop_hook_mode || '').trim())
            ? String(payload.stop_hook_mode || '').trim()
            : 'prompt';
        payload.stop_hook_prompt = String(payload.stop_hook_prompt || '').trim();
        payload.stop_hook_max_triggers = Math.max(
            1,
            Math.min(20, Number.parseInt(String(payload.stop_hook_max_triggers || '20'), 10) || 20),
        );
        // Legacy generated-tool harvest settings are no longer part of the runtime-v2 skill flow.
        // Keep old backend defaults stable, but do not let stale saved UI state re-enable them.
        payload.tool_activation_mode = 'auto_next_turn';
        payload.tool_harvest_enabled = false;
        const changedKeys = Object.keys(payload).filter((key) => {
            if (key === 'openai_api_key') return false;
            return String(payload[key] ?? '') !== String(baselineConfig?.[key] ?? '');
        });

        try {
            const res = await fetch('/api/config/save', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });
            if (!res.ok) {
                throw new Error(`HTTP ${res.status}`);
            }
        } catch (err) {
            console.error('Failed to save config persistently:', err);
            setSaveError(`Config save failed: ${err?.message || err}`);
        } finally {
            setSaving(false);
        }

        onConfigSubmit({ ...payload, _changed_keys: changedKeys });
    };

    if (isConfigLoading) {
        return (
            <div
                role="dialog"
                aria-modal="true"
                aria-label="Loading configuration"
                className="fixed inset-0 z-50 flex flex-col items-center justify-center bg-surface/95 backdrop-blur-sm animate-fade-in"
            >
                <div className="flex flex-col items-center">
                    <div className="mb-5 flex h-14 w-14 items-center justify-center">
                        {logoFailed ? (
                            <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-accent text-white shadow-sm">
                                <Cpu size={24} />
                            </div>
                        ) : (
                            <img
                                src="/logo.png"
                                alt="CellCompass Logo"
                                className="h-full w-full object-contain"
                                onError={() => setLogoFailed(true)}
                            />
                        )}
                    </div>
                    <div
                        className="mb-3 h-1 w-28 rounded-full animate-shimmer"
                        style={{
                            backgroundImage:
                                'linear-gradient(90deg, rgb(var(--line)) 0%, rgb(var(--accent)) 50%, rgb(var(--line)) 100%)',
                            backgroundSize: '200% 100%',
                        }}
                    />
                    <p className="text-[11px] font-medium text-ink-soft">
                        Loading control deck…
                    </p>
                </div>
            </div>
        );
    }

    return (
        <div
            role="dialog"
            aria-modal="true"
            aria-label="CellCompass configuration"
            className="fixed inset-0 z-50 flex items-center justify-center p-4 sm:p-6 animate-fade-in"
        >
            <div
                className="absolute inset-0 bg-ink/50 backdrop-blur-[6px]"
                onClick={onClose}
                aria-hidden="true"
            />
            {/*
             * The shell is a flex column with a definite height. That definite height
             * is what lets the inner aside / form establish their own internal scroll
             * via overflow-y-auto + min-h-0 + flex-1.
             */}
            <div className="relative flex w-full max-w-6xl flex-col overflow-hidden rounded-2xl border border-line bg-surface shadow-lg dark:shadow-lg-dark h-[min(88vh,880px)] min-h-[560px] animate-scale-in">
                <IconButton
                    onClick={onClose}
                    size="sm"
                    aria-label="Close configuration"
                    className="absolute right-4 top-4 z-20"
                >
                    <X size={16} />
                </IconButton>

                <div className="flex min-h-0 flex-1 flex-col lg:flex-row">
                    {/* Aside (sticky summary on lg+) */}
                    <aside className="relative hidden min-h-0 w-full shrink-0 flex-col overflow-y-auto custom-scrollbar border-r border-line bg-surface-muted p-8 lg:flex lg:w-[42%]">
                        {/* Ambient atmosphere */}
                        <div className="pointer-events-none absolute inset-0 bg-ambient-glow" aria-hidden="true" />
                        <div className="pointer-events-none absolute inset-0 bg-dot-grid opacity-40 dark:opacity-25" aria-hidden="true" />
                        <div className="relative">
                            {logoFailed ? (
                                <div className="mb-5 flex h-10 w-10 items-center justify-center rounded-xl bg-accent text-white shadow-md ring-1 ring-accent/30 ring-offset-2 ring-offset-surface-muted">
                                    <Cpu size={20} />
                                </div>
                            ) : (
                                <img
                                    src="/logo.png"
                                    alt="CellCompass Logo"
                                    className="mb-5 h-10 w-10 object-contain"
                                    onError={() => setLogoFailed(true)}
                                />
                            )}
                            <p className="mb-2 text-[11px] font-medium text-ink-soft">
                                CellCompass · Control Deck
                            </p>
                            <h2 className="text-hero-gradient max-w-sm text-2xl font-semibold leading-tight tracking-tight">
                                Navigate single-cell research with AI autonomy.
                            </h2>
                            <p className="mt-3 max-w-sm text-sm leading-relaxed text-ink-muted">
                                CellCompass is an autonomous agent that automates literature mining, pipeline design, cluster annotation and claim validation for single-cell genomics research.
                            </p>
                        </div>

                        <div className="relative mt-6 space-y-3">
                            <SidePanel label="Runtime engine">
                                <div className="flex flex-wrap gap-1.5">
                                    <Pill tone="accent">{config.llm_model || 'model unset'}</Pill>
                                    <Pill tone="neutral">{selectedProvider}</Pill>
                                    <Pill tone={authTone}>
                                        {authLabels[config.llm_auth_mode] || config.llm_auth_mode}
                                    </Pill>
                                    <Pill tone="neutral">
                                        {config.llm_thinking_level || 'low'} reasoning
                                    </Pill>
                                </div>
                                <div className="mt-2 text-xs text-ink-soft">{authSummary}</div>
                            </SidePanel>

                            <SidePanel label="Data workspace">
                                <div className="truncate font-mono text-xs text-ink">
                                    {config.input_path || 'No raw dataset bound'}
                                </div>
                                <div className="mt-1 truncate font-mono text-[11px] text-ink-soft">
                                    {config.output_path || 'Default output directory'}
                                </div>
                            </SidePanel>

                            <SidePanel label="Scientific quality gates">
                                <div className="flex flex-wrap gap-1.5">
                                    <Pill tone="neutral">
                                        Proposal: {reviewLabels[config.algorithm_proposal_review_mode] || config.algorithm_proposal_review_mode}
                                    </Pill>
                                    <Pill tone={config.stop_hook_enabled ? 'accent' : 'neutral'}>
                                        Stop hook {config.stop_hook_enabled ? 'on' : 'off'}
                                    </Pill>
                                </div>
                            </SidePanel>
                        </div>

                        <div className="relative mt-6 border-t border-line pt-4 text-xs leading-5 text-ink-soft">
                            Deploy default configurations for stable bio-computational loops. You can still modify individual variables live in chat.
                        </div>
                    </aside>

                    {/* Main column */}
                    <main className="relative flex min-h-0 flex-1 flex-col overflow-hidden">
                        {/* Mobile header */}
                        <header className="border-b border-line bg-surface px-6 py-5 lg:hidden">
                            {logoFailed ? (
                                <div className="mb-3 flex h-9 w-9 items-center justify-center rounded-md bg-accent text-white">
                                    <Cpu size={18} />
                                </div>
                            ) : (
                                <img
                                    src="/logo.png"
                                    alt="CellCompass Logo"
                                    className="mb-3 h-9 w-9 object-contain"
                                    onError={() => setLogoFailed(true)}
                                />
                            )}
                            <h2 className="text-lg font-semibold tracking-tight text-ink">
                                CellCompass Configuration
                            </h2>
                            <p className="mt-1 text-xs text-ink-soft">
                                Configure LLM reasoning parameters, dataset bindings, and scientific quality gates.
                            </p>
                        </header>

                        <form
                            onSubmit={handleSubmit}
                            className="flex min-h-0 flex-1 flex-col overflow-hidden"
                        >
                            <div className="min-h-0 flex-1 space-y-5 overflow-y-auto custom-scrollbar px-6 py-6 sm:px-8">
                            {/* 01 — Model runtime */}
                            <Section
                                icon={<Cpu size={16} />}
                                eyebrow="01"
                                title="Model runtime"
                                description="Choose provider, model, base URL and API key that powers the planner."
                            >
                                <div className="grid gap-4 md:grid-cols-2">
                                    <div className="flex flex-col gap-1.5">
                                        <Label icon={<Server size={13} />}>Provider</Label>
                                        <Select
                                            name="llm_provider"
                                            value={config.llm_provider || 'auto'}
                                            onChange={handleProviderChange}
                                        >
                                            {providerOptions.map((provider) => (
                                                <option
                                                    key={provider.id}
                                                    value={provider.id}
                                                    disabled={provider.supported === 'false'}
                                                >
                                                    {provider.display_name || provider.id}
                                                    {provider.supported === 'false' ? ' (unsupported)' : ''}
                                                </option>
                                            ))}
                                        </Select>
                                        <Help>
                                            Provider controls default base URL, credential env vars, transport, and reasoning parameter style.
                                        </Help>
                                    </div>

                                    <div className="flex flex-col gap-1.5">
                                        <Label icon={<Cpu size={13} />}>Model</Label>
                                        <Input
                                            type="text"
                                            name="llm_model"
                                            list="llm-model-suggestions"
                                            value={config.llm_model || ''}
                                            onChange={handleChange}
                                            placeholder="gpt-5.4"
                                        />
                                        <datalist id="llm-model-suggestions">
                                            {modelSuggestions.map((model) => (
                                                <option key={model} value={model} />
                                            ))}
                                        </datalist>
                                        {selectedProviderOption?.default_model ? (
                                            <Help>
                                                Default for this provider: <span className="font-mono text-[11px]">{selectedProviderOption.default_model}</span>
                                            </Help>
                                        ) : null}
                                    </div>

                                    <div className="flex flex-col gap-1.5">
                                        <Label icon={<Server size={13} />}>Base URL</Label>
                                        <Input
                                            type="text"
                                            name="llm_base_url"
                                            value={config.llm_base_url || ''}
                                            onChange={handleChange}
                                            placeholder={String(config.llm_model || '').startsWith('gemini')
                                                ? 'Optional for Gemini OAuth'
                                                : 'https://api.openai.com/v1'}
                                        />
                                    </div>

                                    <div className="flex flex-col gap-1.5">
                                        <Label icon={<Key size={13} />}>API key</Label>
                                        <Input
                                            type="password"
                                            name="openai_api_key"
                                            value={config.openai_api_key || ''}
                                            onChange={handleChange}
                                            disabled={!usesApiKeyInput}
                                            placeholder={
                                                effectiveOAuthProvider === 'gemini_oauth'
                                                    ? 'Saved separately when using Google OAuth'
                                                    : (effectiveOAuthProvider === 'codex_oauth'
                                                        ? 'Not used in Codex OAuth mode'
                                                        : 'sk-...')
                                            }
                                        />
                                        <Help>
                                            {effectiveOAuthProvider === 'gemini_oauth'
                                                ? 'Google OAuth credentials are stored separately.'
                                                : (effectiveOAuthProvider === 'codex_oauth'
                                                    ? 'Codex OAuth profiles are stored separately.'
                                                    : (hasCurrentProviderApiKey
                                                        ? 'A saved API key exists. Leave blank to keep using it.'
                                                        : 'No saved API key detected for this provider.'))}
                                        </Help>
                                    </div>
                                </div>
                            </Section>

                            {/* Advanced toggle */}
                            <button
                                type="button"
                                onClick={() => setShowAdvanced(!showAdvanced)}
                                aria-expanded={showAdvanced}
                                className="flex w-full items-center justify-between gap-2 rounded-lg border border-line bg-surface px-4 py-3 text-sm font-medium text-ink-muted transition-colors hover:border-line-strong hover:bg-surface-muted hover:text-ink"
                            >
                                <span className="flex items-center gap-2">
                                    <SlidersHorizontal size={14} className="text-accent" />
                                    Advanced settings
                                </span>
                                {showAdvanced ? (
                                    <ChevronDown size={14} />
                                ) : (
                                    <ChevronRight size={14} />
                                )}
                            </button>

                            {showAdvanced && (
                                <div className="space-y-5 animate-fade-in">
                                    {/* 01-A — Model auth & details */}
                                    <Section
                                        icon={<Cpu size={16} />}
                                        eyebrow="01-A"
                                        title="Model auth & details"
                                        description="Refine authorization mode and reasoning level parameters."
                                    >
                                        <div className="grid gap-4 md:grid-cols-2">
                                            <div className="flex flex-col gap-1.5">
                                                <Label icon={<Key size={13} />}>Auth mode</Label>
                                                <Select
                                                    name="llm_auth_mode"
                                                    value={config.llm_auth_mode || 'auto'}
                                                    onChange={handleChange}
                                                >
                                                    <option value="auto">Auto</option>
                                                    <option value="api_key">API Key</option>
                                                    <option value="gemini_oauth">Google OAuth</option>
                                                    <option value="codex_oauth">Codex OAuth</option>
                                                </Select>
                                                <Help tone={authTone === 'warn' ? 'warn' : 'muted'}>
                                                    {authSummary}
                                                </Help>
                                            </div>

                                            <div className="flex flex-col gap-1.5">
                                                <Label icon={<SlidersHorizontal size={13} />}>Reasoning effort</Label>
                                                <Select
                                                    name="llm_thinking_level"
                                                    value={config.llm_thinking_level || 'low'}
                                                    onChange={handleChange}
                                                >
                                                    <option value="off">Off</option>
                                                    <option value="minimal">Minimal</option>
                                                    <option value="low">Low</option>
                                                    <option value="medium">Medium</option>
                                                    <option value="high">High</option>
                                                    <option value="xhigh">Extra high</option>
                                                </Select>
                                                <Help>
                                                    Applied to Codex OAuth and known reasoning models; ordinary chat models ignore it.
                                                </Help>
                                            </div>

                                            <div className="md:col-span-2">
                                                <ToggleRow
                                                    icon={<SlidersHorizontal size={14} />}
                                                    title="Allow 1M context before compact"
                                                    description="Off by default: compact still triggers around 256k × 80%, even for 1M-context models. Turn on only for explicit long-context runs."
                                                    checked={!!config.allow_large_context_window}
                                                    onChange={(value) =>
                                                        setConfig((prev) => ({ ...prev, allow_large_context_window: value }))
                                                    }
                                                />
                                            </div>
                                        </div>

                                        {showCodexProfiles && (
                                            <div className="rounded-lg border border-line bg-surface-muted p-4">
                                                <div className="grid gap-4 md:grid-cols-[1fr_1.15fr]">
                                                    <div className="flex flex-col gap-1.5">
                                                        <Label icon={<Key size={13} />}>Codex profile</Label>
                                                        <Select
                                                            name="llm_profile_id"
                                                            value={config.llm_profile_id || ''}
                                                            onChange={handleChange}
                                                        >
                                                            <option value="">Auto-select available profile</option>
                                                            {codexProfiles.map((profile) => {
                                                                const suffix = [];
                                                                if (profile.email) suffix.push(profile.email);
                                                                if (profile.reauth_required) suffix.push('reauth required');
                                                                else if ((profile.cooldown_until_ms || 0) > profileRenderNow) suffix.push('cooldown');
                                                                else if (profile.enabled === false) suffix.push('disabled');
                                                                return (
                                                                    <option key={profile.profile_id} value={profile.profile_id}>
                                                                        {profile.profile_id}{suffix.length ? ` (${suffix.join(', ')})` : ''}
                                                                    </option>
                                                                );
                                                            })}
                                                        </Select>
                                                        <Help>
                                                            {config.has_codex_oauth_profiles
                                                                ? `Detected ${codexProfiles.length} saved profile${codexProfiles.length === 1 ? '' : 's'}. Auto-select keeps provider failover available.`
                                                                : 'No saved Codex OAuth profile detected. Login once from CLI with: cytobridge-agent auth codex-login'}
                                                        </Help>
                                                    </div>
                                                    <div className="grid gap-3">
                                                        {codexUsage.length > 0 ? (
                                                            codexUsage.map((usage) => (
                                                                <UsageCard key={usage.profile_id} usage={usage} />
                                                            ))
                                                        ) : (
                                                            <div className="rounded-lg border border-line bg-surface px-4 py-3 text-sm text-ink-soft">
                                                                {codexUsageMessage}
                                                            </div>
                                                        )}
                                                    </div>
                                                </div>
                                            </div>
                                        )}
                                    </Section>

                                    {/* 02 — Workspace */}
                                    <Section
                                        icon={<FolderOpen size={16} />}
                                        eyebrow="02"
                                        title="Workspace"
                                        description="Bind the starting data path and output directory. Both can still be updated later in chat."
                                    >
                                        <div className="grid gap-4 md:grid-cols-2">
                                            <div className="flex flex-col gap-1.5">
                                                <Label icon={<FileText size={13} />}>Data file path</Label>
                                                <Input
                                                    type="text"
                                                    name="input_path"
                                                    value={config.input_path || ''}
                                                    onChange={handleChange}
                                                    className="font-mono"
                                                    placeholder="Optional. Provide a path later in chat if needed."
                                                />
                                            </div>
                                            <div className="flex flex-col gap-1.5">
                                                <Label icon={<FolderOpen size={13} />}>Output directory</Label>
                                                <Input
                                                    type="text"
                                                    name="output_path"
                                                    value={config.output_path || ''}
                                                    onChange={handleChange}
                                                    className="font-mono"
                                                    placeholder="Leave empty for cytobridge_output"
                                                />
                                            </div>
                                            <div className="flex flex-col gap-1.5">
                                                <Label icon={<Cpu size={13} />}>Default compute device</Label>
                                                <Select
                                                    name="device"
                                                    value={config.device || 'cuda'}
                                                    onChange={handleChange}
                                                >
                                                    <option value="cuda">CUDA GPU</option>
                                                    <option value="cpu">CPU</option>
                                                    <option value="mps">MPS</option>
                                                </Select>
                                                <Help>
                                                    Used as the session-level default for training tools unless a run overrides it.
                                                </Help>
                                            </div>
                                            <div className="md:col-span-1">
                                                <ToggleRow
                                                    icon={<Eye size={14} />}
                                                    title="Vision input"
                                                    description="Allow image attachments and image/PDF rendering paths to reach the model."
                                                    checked={!!config.enable_multimodal}
                                                    onChange={(value) =>
                                                        setConfig((prev) => ({ ...prev, enable_multimodal: value }))
                                                    }
                                                />
                                            </div>
                                        </div>
                                    </Section>

                                    {/* 03 — Review & stop hook */}
                                    <Section
                                        icon={<Settings size={16} />}
                                        eyebrow="03"
                                        title="Review and stop hook"
                                        description="Keep proposal gates explicit while leaving the stop hook off by default."
                                    >
                                        <div className="grid gap-4 md:grid-cols-2">
                                            <div className="flex flex-col gap-1.5">
                                                <Label icon={<Settings size={13} />}>Algorithm proposal review</Label>
                                                <Select
                                                    name="algorithm_proposal_review_mode"
                                                    value={config.algorithm_proposal_review_mode || 'agent_decide'}
                                                    onChange={handleChange}
                                                >
                                                    <option value="always_user_review">Always require user review</option>
                                                    <option value="agent_decide">Agent decides when uncertain</option>
                                                    <option value="auto_approve">Auto-approve proposal</option>
                                                </Select>
                                            </div>
                                            <div className="flex flex-col gap-1.5">
                                                <Label icon={<Settings size={13} />}>Research idea review</Label>
                                                <Select
                                                    name="idea_review_mode"
                                                    value={config.idea_review_mode || 'agent_decide'}
                                                    onChange={handleChange}
                                                >
                                                    <option value="always_user_review">Always require user review</option>
                                                    <option value="agent_decide">Agent decides when uncertain</option>
                                                    <option value="auto_approve">Auto-approve idea</option>
                                                </Select>
                                            </div>
                                        </div>

                                        <ToggleRow
                                            icon={<Settings size={14} />}
                                            title="Planner stop hook"
                                            description="Run a bounded final guard before accepting final_answer or needs_input."
                                            checked={!!config.stop_hook_enabled}
                                            onChange={(value) =>
                                                setConfig((prev) => ({ ...prev, stop_hook_enabled: value }))
                                            }
                                        />

                                        <div className="grid gap-4 md:grid-cols-[1fr_0.55fr]">
                                            <div className="flex flex-col gap-1.5">
                                                <Label icon={<Settings size={13} />}>Stop hook mode</Label>
                                                <Select
                                                    name="stop_hook_mode"
                                                    value={config.stop_hook_mode || 'prompt'}
                                                    onChange={handleChange}
                                                    disabled={!config.stop_hook_enabled}
                                                >
                                                    <option value="builtin">Builtin guards only</option>
                                                    <option value="prompt">LLM prompt evaluator</option>
                                                    <option value="agent">Subagent evaluator</option>
                                                </Select>
                                            </div>
                                            <div className="flex flex-col gap-1.5">
                                                <Label icon={<SlidersHorizontal size={13} />}>Max triggers</Label>
                                                <Input
                                                    type="number"
                                                    name="stop_hook_max_triggers"
                                                    min="1"
                                                    max="20"
                                                    step="1"
                                                    disabled={!config.stop_hook_enabled}
                                                    value={config.stop_hook_max_triggers ?? 20}
                                                    onChange={handleChange}
                                                />
                                            </div>
                                        </div>

                                        {config.stop_hook_enabled && (config.stop_hook_mode === 'prompt' || config.stop_hook_mode === 'agent') && (
                                            <div className="flex flex-col gap-1.5">
                                                <Label icon={<Settings size={13} />}>Stop hook instructions</Label>
                                                <Textarea
                                                    name="stop_hook_prompt"
                                                    value={config.stop_hook_prompt || ''}
                                                    onChange={handleChange}
                                                    rows={5}
                                                    className="font-mono"
                                                    placeholder="Describe what the stop hook should verify before accepting the planner final answer."
                                                />
                                            </div>
                                        )}
                                    </Section>
                                </div>
                            )}

                            {saveError && (
                                <div className="rounded-lg border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-800 dark:border-amber-500/30 dark:bg-amber-500/10 dark:text-amber-200">
                                    {saveError}
                                </div>
                            )}
                        </div>

                            {/* Footer */}
                            <footer className="flex shrink-0 flex-col gap-3 border-t border-line bg-surface-muted px-6 py-4 sm:flex-row sm:items-center sm:justify-between sm:px-8">
                                <div className="max-w-sm text-xs leading-5 text-ink-soft">
                                    Settings are persisted locally then used to spin up your single-cell analysis workspace.
                                </div>
                                <Button
                                    type="submit"
                                    variant="primary"
                                    size="md"
                                    disabled={saving}
                                    className="min-w-[10.5rem]"
                                >
                                    {saving ? (
                                        <>
                                            <Loader2 size={15} className="animate-spin" />
                                            Deploying…
                                        </>
                                    ) : (
                                        <>
                                            Initialize Agent
                                            <ArrowRight size={15} />
                                        </>
                                    )}
                                </Button>
                            </footer>
                        </form>
                    </main>
                </div>
            </div>
        </div>
    );
}
