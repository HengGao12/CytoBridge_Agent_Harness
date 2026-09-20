/**
 * Shared helpers for restoring a session from the backend `/api/config`
 * payload (or a resume result). These were previously copy-pasted across
 * `App.jsx` and `useAgentSocket.js`.
 */

/** Map raw stored attachments into the client `{ type, mime_type, ... }` shape. */
export function mapResumedAttachments(attachments) {
    if (!Array.isArray(attachments)) return [];
    return attachments
        .map((item, idx) => {
            const mimeType = item.mime_type || item.mimeType || '';
            const content = item.content || '';
            const fileName = item.file_name || item.fileName || `image_${idx + 1}`;
            return {
                type: 'image',
                mime_type: mimeType,
                file_name: fileName,
                data_url: mimeType && content ? `data:${mimeType};base64,${content}` : '',
            };
        })
        .filter((item) => item.data_url);
}

/** Whether resume diagnostics are noteworthy enough to surface to the user. */
export function shouldShowResumeDiagnostics(diagnostics) {
    return Boolean(
        diagnostics &&
        (
            diagnostics.degraded ||
            diagnostics.migrated_from_v1 ||
            (Array.isArray(diagnostics.warnings) && diagnostics.warnings.length > 0) ||
            diagnostics.source === 'replay'
        ),
    );
}

/** Normalize a diagnostics object into the timeline-item shape. */
export function normalizeResumeDiagnostics(diagnostics) {
    return {
        source: diagnostics?.source || 'snapshot',
        migrated_from_v1: !!diagnostics?.migrated_from_v1,
        degraded: !!diagnostics?.degraded,
        warnings: Array.isArray(diagnostics?.warnings) ? diagnostics.warnings : [],
    };
}

/** Build the UI session-config object from a `/api/config` response. */
export function buildSessionConfigFromData(data) {
    return {
        input_path: data.input_path,
        output_path: data.output_path,
        algorithm_proposal_review_mode: data.algorithm_proposal_review_mode || 'agent_decide',
        idea_review_mode: data.idea_review_mode || 'agent_decide',
        allow_large_context_window: !!data.allow_large_context_window,
        stop_hook_enabled: !!data.stop_hook_enabled,
        stop_hook_mode: data.stop_hook_mode || 'prompt',
        stop_hook_prompt: data.stop_hook_prompt || '',
        stop_hook_max_triggers: data.stop_hook_max_triggers ?? 20,
    };
}
