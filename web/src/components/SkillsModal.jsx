import React, { useCallback, useEffect, useState } from 'react';
import { ChevronDown, ChevronUp, RefreshCw, PackageOpen } from 'lucide-react';
import { Button, Modal, Pill, Toggle } from './ui';
import { cn } from './ui/cn';

const IDEA_DETAIL_FIELDS = [
  ['Problem Definition', 'problem_definition'],
  ['Scientific Object', 'scientific_object'],
  ['Current-Method Failure Mode', 'current_method_failure_mode'],
  ['Prior Work', 'prior_work'],
  ['Why This Matters', 'why_this_matters'],
  ['Why Now', 'why_now'],
  ['Falsifiable Success Criteria', 'falsifiable_success_criteria'],
  ['Non-goals', 'non_goals'],
  ['Evidence Basis', 'evidence_basis'],
  ['Feasible Direction Families', 'feasible_direction_families'],
  ['Feasibility Constraints', 'feasibility_constraints'],
  ['Review Feedback', 'review_feedback'],
];

const SCOPE_TABS = [
  { id: 'workflow', label: 'Workflow' },
  { id: 'planner', label: 'Planner' },
  { id: 'downstream', label: 'Downstream' },
];

function IdeaDetailField({ label, value }) {
  const text = String(value || '').trim();
  if (!text) return null;
  return (
    <div className="rounded-lg border border-line bg-surface-muted p-3">
      <div className="text-[11px] font-medium text-ink-soft">
        {label}
      </div>
      <div className="mt-2 whitespace-pre-wrap text-sm leading-6 text-ink">
        {text}
      </div>
    </div>
  );
}

function EmptyState({ children }) {
  return (
    <div className="flex flex-col items-center gap-2 rounded-xl border border-dashed border-line bg-surface-muted/60 px-4 py-7 text-center text-sm text-ink-soft">
      <PackageOpen size={20} className="opacity-60" />
      {children}
    </div>
  );
}

function SectionHeader({ title, count, suffix }) {
  return (
    <div className="mb-3 flex items-center justify-between">
      <div className="text-sm font-semibold text-ink">{title}</div>
      {(count !== undefined || suffix) && (
        <div className="text-xs text-ink-soft">
          {count !== undefined ? `${count} ${suffix || 'detected'}` : suffix}
        </div>
      )}
    </div>
  );
}

export function SkillsModal({
  isOpen,
  onClose,
  listSkills,
  setSkillExposure,
  setAlgorithmExposure,
  hasActiveSession = false,
}) {
  const [scope, setScope] = useState('planner');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [message, setMessage] = useState('');
  const [expandedIdeas, setExpandedIdeas] = useState({});
  const [ideaDetails, setIdeaDetails] = useState({});
  const [ideaDetailLoading, setIdeaDetailLoading] = useState({});
  const [ideaDetailErrors, setIdeaDetailErrors] = useState({});
  const [payload, setPayload] = useState({
    available_skills: [],
    loaded_skills: [],
    available_algorithms: [],
    research_ideas: [],
    hidden_skills: { workflow: [], planner: [], downstream: [] },
    hidden_training_algorithms: [],
    workspace_policy: {},
  });

  const refresh = useCallback(async (selectedScope = scope) => {
    setLoading(true);
    setError('');
    try {
      const data = await listSkills(selectedScope);
      if (data.status !== 'success') {
        setError(data.message || 'Failed to load skills.');
        return;
      }
      setPayload({
        available_skills: data.available_skills || [],
        loaded_skills: data.loaded_skills || [],
        available_algorithms: data.available_algorithms || [],
        research_ideas: data.research_ideas || [],
        hidden_skills: data.hidden_skills || { workflow: [], planner: [], downstream: [] },
        hidden_training_algorithms: data.hidden_training_algorithms || [],
        workspace_policy: data.workspace_policy || {},
      });
    } catch (err) {
      setError(err.message || 'Failed to load skills.');
    } finally {
      setLoading(false);
    }
  }, [listSkills, scope]);

  useEffect(() => {
    if (!isOpen) return;
    refresh();
  }, [isOpen, refresh]);

  const fetchIdeaDetail = async (ideaId) => {
    const normalized = String(ideaId || '').trim();
    if (!normalized) return;
    setIdeaDetailLoading(prev => ({ ...prev, [normalized]: true }));
    setIdeaDetailErrors(prev => ({ ...prev, [normalized]: '' }));
    try {
      const res = await fetch(`/api/ideas/${encodeURIComponent(normalized)}`);
      const data = await res.json();
      if (!res.ok || data.status !== 'success') {
        throw new Error(data.message || 'Failed to load idea details.');
      }
      setIdeaDetails(prev => ({ ...prev, [normalized]: data.idea || {} }));
    } catch (err) {
      setIdeaDetailErrors(prev => ({
        ...prev,
        [normalized]: err.message || 'Failed to load idea details.',
      }));
    } finally {
      setIdeaDetailLoading(prev => ({ ...prev, [normalized]: false }));
    }
  };

  const onToggleIdeaDetails = async (ideaId) => {
    const normalized = String(ideaId || '').trim();
    if (!normalized) return;
    const willExpand = !expandedIdeas[normalized];
    setExpandedIdeas(prev => ({ ...prev, [normalized]: willExpand }));
    if (willExpand && !ideaDetails[normalized] && !ideaDetailLoading[normalized]) {
      await fetchIdeaDetail(normalized);
    }
  };

  const onToggleSkill = async (skillName, currentEnabled) => {
    if (!setSkillExposure) return;
    setLoading(true);
    setError('');
    setMessage('');
    try {
      const data = await setSkillExposure(scope, skillName, !currentEnabled);
      if (data.status !== 'success') {
        setError(data.message || 'Failed to update skill visibility.');
        return;
      }
      setMessage(data.message || 'Skill visibility updated.');
        setPayload(prev => ({
          ...prev,
          available_skills: data.available_skills || prev.available_skills,
          loaded_skills: data.loaded_skills || prev.loaded_skills,
          hidden_skills: data.hidden_skills || prev.hidden_skills,
          available_algorithms: data.available_algorithms || prev.available_algorithms,
          research_ideas: data.research_ideas || prev.research_ideas,
        }));
    } catch (err) {
      setError(err.message || 'Failed to update skill visibility.');
    } finally {
      setLoading(false);
    }
  };

  const onToggleAlgorithm = async (algorithmName, currentEnabled) => {
    if (!setAlgorithmExposure) return;
    setLoading(true);
    setError('');
    setMessage('');
    try {
      const data = await setAlgorithmExposure(algorithmName, !currentEnabled);
      if (data.status !== 'success') {
        setError(data.message || 'Failed to update algorithm visibility.');
        return;
      }
      setMessage(data.message || 'Algorithm visibility updated.');
      setPayload(prev => ({
        ...prev,
        available_algorithms: data.available_algorithms || prev.available_algorithms,
        hidden_training_algorithms: data.hidden_training_algorithms || prev.hidden_training_algorithms,
      }));
    } catch (err) {
      setError(err.message || 'Failed to update algorithm visibility.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Modal
      isOpen={isOpen}
      onClose={onClose}
      title="Skills"
      description="Detect available skills and custom algorithms, then control whether they are exposed to the agent."
      size="xl"
      contentClassName="max-h-[92vh]"
    >
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-line bg-surface-muted/60 px-6 py-3">
        <div role="tablist" className="inline-flex items-center gap-1 rounded-lg border border-line bg-surface p-1">
          {SCOPE_TABS.map(tab => {
            const active = scope === tab.id;
            return (
              <button
                key={tab.id}
                role="tab"
                aria-selected={active}
                onClick={() => setScope(tab.id)}
                className={cn(
                  'rounded-md px-3 py-1.5 text-xs font-medium transition-colors',
                  active
                    ? 'bg-accent text-white shadow-xs'
                    : 'text-ink-muted hover:bg-surface-muted hover:text-ink',
                )}
              >
                {tab.label}
              </button>
            );
          })}
        </div>
        <div className="flex items-center gap-2">
          {!hasActiveSession && (
            <span className="text-xs text-amber-700 dark:text-amber-300">
              No active session. Listing still works.
            </span>
          )}
          <Button
            variant="secondary"
            size="sm"
            onClick={() => refresh(scope)}
            disabled={loading}
          >
            <RefreshCw size={13} className={loading ? 'animate-spin' : ''} />
            Refresh
          </Button>
        </div>
      </div>

      <div className="grid gap-5 p-6 md:grid-cols-[1.15fr_0.85fr]">
        <div className="min-w-0">
          <SectionHeader
            title="Available Skills"
            count={payload.available_skills.length}
            suffix="detected"
          />
          <div className="space-y-2">
            {payload.available_skills.map((skill) => {
              const skillName = String(skill.name || '');
              const enabled = skill.enabled !== false;
              return (
                <div
                  key={`${scope}-${skillName}`}
                  className="rounded-lg border border-line bg-surface p-4 transition-colors hover:border-line-strong"
                >
                  <div className="flex items-start justify-between gap-3">
                    <div className="min-w-0 flex-1">
                      <div className="flex flex-wrap items-center gap-2">
                        <span className="text-sm font-semibold text-ink">{skillName}</span>
                        <Pill tone={enabled ? 'success' : 'neutral'}>
                          {enabled ? 'exposed' : 'hidden'}
                        </Pill>
                        <Pill tone={skill.source === 'user' ? 'success' : 'accent'}>
                          {skill.source || 'builtin'}
                        </Pill>
                      </div>
                      <div className="mt-1.5 text-sm text-ink-muted leading-relaxed">
                        {skill.description || 'No description'}
                      </div>
                      <div className="mt-2 break-all font-mono text-[11px] text-ink-soft">
                        {skill.skill_md_path}
                      </div>
                    </div>
                    <Toggle
                      checked={enabled}
                      onChange={() => onToggleSkill(skillName, enabled)}
                      disabled={loading}
                      ariaLabel={`Toggle ${skillName}`}
                    />
                  </div>
                </div>
              );
            })}
            {!payload.available_skills.length && !loading && (
              <EmptyState>No skills detected for this scope.</EmptyState>
            )}
          </div>

          {scope === 'planner' && (
            <>
              <div className="mt-6">
                <SectionHeader
                  title="Custom Training Algorithms"
                  count={payload.available_algorithms.length}
                  suffix="detected"
                />
                <div className="space-y-2">
                  {payload.available_algorithms.map((algo) => {
                    const algoName = String(algo.name || '');
                    const enabled = algo.enabled !== false;
                    return (
                      <div
                        key={`algo-${algoName}`}
                        className="rounded-lg border border-line bg-surface p-4 transition-colors hover:border-line-strong"
                      >
                        <div className="flex items-start justify-between gap-3">
                          <div className="min-w-0 flex-1">
                            <div className="flex flex-wrap items-center gap-2">
                              <span className="text-sm font-semibold text-ink">{algoName}</span>
                              <Pill tone={enabled ? 'success' : 'neutral'}>
                                {enabled ? 'exposed' : 'hidden'}
                              </Pill>
                            </div>
                            <div className="mt-1.5 text-sm text-ink-muted leading-relaxed">
                              {algo.description || 'No description'}
                            </div>
                            <div className="mt-1.5 text-xs text-amber-700 dark:text-amber-300">
                              requirements: {algo.requirements || 'No requirements declared'}
                            </div>
                            <div className="mt-2 break-all font-mono text-[11px] text-ink-soft">
                              {algo.path}
                            </div>
                          </div>
                          <Toggle
                            checked={enabled}
                            onChange={() => onToggleAlgorithm(algoName, enabled)}
                            disabled={loading}
                            ariaLabel={`Toggle ${algoName}`}
                          />
                        </div>
                      </div>
                    );
                  })}
                  {!payload.available_algorithms.length && !loading && (
                    <EmptyState>
                      No custom algorithms detected under <span className="font-mono">~/.cellcompass/training_algorithms</span>.
                    </EmptyState>
                  )}
                </div>
              </div>

              <div className="mt-6">
                <SectionHeader
                  title="Ideas"
                  count={payload.research_ideas.length}
                  suffix="recorded"
                />
                <div className="space-y-2">
                  {payload.research_ideas.map((idea) => {
                    const ideaId = String(idea.idea_id || '');
                    const preview = String(idea.problem_definition || idea.scientific_object || '').trim();
                    const details = ideaDetails[ideaId];
                    const isExpanded = !!expandedIdeas[ideaId];
                    const isLoadingDetail = !!ideaDetailLoading[ideaId];
                    const detailError = String(ideaDetailErrors[ideaId] || '').trim();
                    const linkedAlgorithms = Array.isArray(details?.linked_algorithms)
                      ? details.linked_algorithms.filter(Boolean)
                      : [];
                    const reviewHistory = Array.isArray(details?.review_history)
                      ? details.review_history
                      : [];
                    return (
                      <div
                        key={`idea-${ideaId}`}
                        className="rounded-lg border border-line bg-surface p-4 transition-colors hover:border-line-strong"
                      >
                        <div className="flex items-start justify-between gap-3">
                          <div className="min-w-0 flex-1">
                            <div className="flex flex-wrap items-center gap-2">
                              <span className="text-sm font-semibold text-ink">{idea.title || ideaId}</span>
                              <Pill tone={idea.active ? 'success' : 'neutral'}>
                                {idea.active ? 'active' : 'inactive'}
                              </Pill>
                              <Pill tone="accent">{idea.primary_track || 'track?'}</Pill>
                            </div>
                            <div className="mt-1 break-all font-mono text-[11px] text-ink-soft">
                              {ideaId}
                            </div>
                            <div className="mt-2 text-xs text-ink-muted">
                              review={idea.review_status || '(unset)'} · portfolio={idea.portfolio_status || '(unset)'} · resolution={idea.resolution_status || '(unset)'} · execution={idea.execution_status || '(unset)'}
                            </div>
                            <div className="mt-1 text-xs text-amber-700 dark:text-amber-300">
                              linked algorithms: {Number(idea.linked_algorithm_count || 0)} · attempts: {Number(idea.attempt_count || 0)}
                            </div>
                            {preview && (
                              <div className="mt-2 whitespace-pre-wrap text-sm leading-6 text-ink-muted">
                                {preview}
                              </div>
                            )}
                          </div>
                          <Button
                            type="button"
                            variant="secondary"
                            size="sm"
                            onClick={() => onToggleIdeaDetails(ideaId)}
                          >
                            {isExpanded ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
                            {isExpanded ? 'Hide details' : 'View details'}
                          </Button>
                        </div>
                        {isExpanded && (
                          <div className="mt-4 space-y-3 border-t border-line pt-4">
                            {isLoadingDetail && (
                              <div className="text-sm text-ink-soft">Loading idea details…</div>
                            )}
                            {detailError && !isLoadingDetail && (
                              <div className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700 dark:border-red-500/30 dark:bg-red-500/10 dark:text-red-300">
                                {detailError}
                              </div>
                            )}
                            {details && !isLoadingDetail && !detailError && (
                              <>
                                <div className="grid gap-3 md:grid-cols-2">
                                  <div className="rounded-lg border border-line bg-surface-muted p-3">
                                    <div className="text-[11px] font-medium text-ink-soft">Metadata</div>
                                    <div className="mt-2 space-y-1 text-sm text-ink-muted">
                                      <div>review mode: {details.review_mode || '(unset)'}</div>
                                      <div>current revision: {details.current_revision_id || '(unset)'}</div>
                                      <div>created at: {details.created_at || '(unset)'}</div>
                                      <div>updated at: {details.updated_at || '(unset)'}</div>
                                      {details.reviewed_at && <div>reviewed at: {details.reviewed_at}</div>}
                                    </div>
                                  </div>
                                  <div className="rounded-lg border border-line bg-surface-muted p-3">
                                    <div className="text-[11px] font-medium text-ink-soft">Links</div>
                                    <div className="mt-2 space-y-1.5 text-sm text-ink-muted">
                                      <div>linked algorithms: {linkedAlgorithms.length ? linkedAlgorithms.join(', ') : '(none)'}</div>
                                      <div>review history: {reviewHistory.length}</div>
                                      {details.idea_path && (
                                        <div className="break-all font-mono text-[11px] text-ink-soft">{details.idea_path}</div>
                                      )}
                                      {details.idea_json_path && (
                                        <div className="break-all font-mono text-[11px] text-ink-soft">{details.idea_json_path}</div>
                                      )}
                                    </div>
                                  </div>
                                </div>
                                <div className="space-y-3">
                                  {IDEA_DETAIL_FIELDS.map(([label, key]) => (
                                    <IdeaDetailField
                                      key={`${ideaId}-${key}`}
                                      label={label}
                                      value={details[key]}
                                    />
                                  ))}
                                </div>
                                {reviewHistory.length > 0 && (
                                  <div className="rounded-lg border border-line bg-surface-muted p-3">
                                    <div className="text-[11px] font-medium text-ink-soft">Review History</div>
                                    <div className="mt-2 space-y-2">
                                      {reviewHistory.map((item, idx) => (
                                        <div
                                          key={`${ideaId}-review-${idx}`}
                                          className="rounded-md border border-line bg-surface px-3 py-2 text-sm text-ink-muted"
                                        >
                                          <div>
                                            {item.at || '(unknown time)'} · status={item.review_status || '(unset)'}
                                          </div>
                                          {String(item.feedback || '').trim() && (
                                            <div className="mt-1 whitespace-pre-wrap text-ink">
                                              {item.feedback}
                                            </div>
                                          )}
                                        </div>
                                      ))}
                                    </div>
                                  </div>
                                )}
                              </>
                            )}
                          </div>
                        )}
                      </div>
                    );
                  })}
                  {!payload.research_ideas.length && !loading && (
                    <EmptyState>No ideas recorded yet.</EmptyState>
                  )}
                </div>
              </div>
            </>
          )}
        </div>

        <aside className="space-y-3">
          <div className="rounded-lg border border-line bg-surface-muted p-4">
            <div className="mb-2 text-sm font-semibold text-ink">Usage</div>
            <div className="text-sm leading-relaxed text-ink-muted">
              Toggle <span className="font-medium text-ink">Expose</span> to control whether a skill or custom algorithm is shown to the agent in prompt catalogs.
              Settings are persisted to <span className="font-mono text-[12px]">~/.cellcompass/config.json</span> and kept across conversations.
            </div>
          </div>

          {scope === 'planner' && (
            <div className="rounded-lg border border-line bg-surface-muted p-4">
              <div className="mb-2 text-sm font-semibold text-ink">Planner Skill Policy</div>
              <div className="text-sm leading-relaxed text-ink-muted">
                Automatic routing is disabled. Skills are browsed here and read on demand; they are not injected into the system prompt.
              </div>
            </div>
          )}

          {scope === 'planner' && payload.workspace_policy && Object.keys(payload.workspace_policy).length > 0 && (
            <div className="rounded-lg border border-line bg-surface-muted p-4">
              <div className="mb-2 text-sm font-semibold text-ink">Planner Workspace Policy</div>
              <pre className="overflow-x-auto whitespace-pre-wrap font-mono text-[11px] text-ink-muted">
                {JSON.stringify(payload.workspace_policy, null, 2)}
              </pre>
            </div>
          )}

          {(error || message) && (
            <div
              className={cn(
                'rounded-lg border px-4 py-3 text-sm',
                error
                  ? 'border-red-200 bg-red-50 text-red-700 dark:border-red-500/30 dark:bg-red-500/10 dark:text-red-300'
                  : 'border-emerald-200 bg-emerald-50 text-emerald-700 dark:border-emerald-500/30 dark:bg-emerald-500/10 dark:text-emerald-300',
              )}
            >
              {error || message}
            </div>
          )}
        </aside>
      </div>
    </Modal>
  );
}
