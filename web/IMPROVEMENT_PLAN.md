# Frontend Improvement Plan

Tracking doc for the review-driven cleanup of the CellCompass frontend.
Scope is UI/robustness only — API routes, event payloads, and runtime field
names stay stable (per `README.md`).

Status legend: `[ ]` todo · `[~]` in progress · `[x]` done

## P0 — Correctness bugs

- [x] **1. Inline code rendering** — `react-markdown` v10 no longer passes the
  `inline` prop to the `code` component, so inline code spans lose their
  pill styling. Detect inline vs block without relying on `inline`.
  Files: `src/components/ChatMarkdown.jsx` (+ any other `code` overrides).
  Done: added `src/lib/markdownCode.js#isInlineCode`; wired into ChatMarkdown
  and the two ActivityItem thought renderers.
- [x] **2. Input disabled while thinking** — the composer textarea is
  `disabled` during `isThinking`, but `handleSubmit` explicitly allows slash
  commands mid-run, so that branch is dead. Keep the textarea usable; only
  gate the normal send button.
  Files: `src/components/MessageComposer.jsx`, `src/App.jsx`.
  Done: textarea no longer disabled while thinking (send still gated by the
  Stop-button swap + `handleSubmit` slash-command guard); placeholder hints
  that only `/` commands go through mid-run.
- [x] **3. Optimistic user-message dedup** — `sendMessage` injects an
  optimistic user bubble with a `client_request_id`, but the reducer's
  `user_message` branch never dedups, risking duplicate bubbles if the
  backend echoes the event. Reconcile by `client_request_id`.
  Files: `src/hooks/useAgentSocket.js`, `src/lib/timelineReducer.js`.
  Done: optimistic bubbles now carry `clientRequestId`/`optimistic` flags; the
  reducer reconciles a non-optimistic `user_message` echo onto the matching
  optimistic bubble (by client_request_id, else identical content) instead of
  pushing a duplicate. Backward-compatible — no dedup happens if the backend
  never echoes.

## P1 — Accessibility & build

- [x] **4. Modal focus trap** — `Modal` only sets initial focus + Esc; Tab can
  leave the dialog. Add a proper focus trap.
  Files: `src/components/ui/index.jsx`.
  Done: the keydown handler now cycles Tab/Shift+Tab within the dialog's
  focusable elements and pins focus to the shell when none exist.
- [x] **5. HistorySidebar nested interactives** — a `div role="button"` wraps
  real `<button>`s (Resume/Delete). Restructure so interactive elements are
  not nested.
  Files: `src/components/HistorySidebar.jsx`.
  Done: row is now a real `<button>`; Resume/Delete are absolutely-positioned
  siblings on the `<li>`; nested `<p>` swapped to block `<span>` for valid
  button content.
- [x] **6. Prism language registration** — `prism-async-light` code-splits a
  grammar chunk per language (~9.8k files in `dist`). Switch to `PrismLight`
  with manual `registerLanguage` for the languages actually used.
  Files: `src/components/CodeHighlight.jsx`.
  Done: now `PrismLight` + an explicit allow-list (python/bash/json/yaml/sql/
  markup/js/jsx/ts/tsx/css/markdown/r/diff and aliases). Unknown languages
  degrade to unhighlighted code. Verified module paths exist in node_modules.

## P2 — Robustness & quality

- [x] **7. ErrorBoundary** — markdown/KaTeX render errors currently white-screen
  the app. Wrap timeline items in an ErrorBoundary.
  Files: new `src/components/ErrorBoundary.jsx`, `src/App.jsx`.
  Done: per-item `ErrorBoundary` (keyed by the timeline item key so it resets
  when the item changes) renders an inline "could not be displayed" notice.
- [x] **8. Dedup attachment/diagnostics mapping** — the attachment `.map(...)`
  and resume-diagnostics gate are copy-pasted 3× across `App.jsx` and
  `useAgentSocket.js`. Extract shared helpers.
  Files: new `src/lib/restoreHelpers.js`, `src/App.jsx`, `src/hooks/useAgentSocket.js`.
  Done: added `mapResumedAttachments`, `shouldShowResumeDiagnostics`,
  `normalizeResumeDiagnostics`, `buildSessionConfigFromData`; replaced all
  duplicated blocks in both files.
- [x] **9. Native confirm/alert + i18n** — replace blocking `confirm()`/`alert()`
  in HistorySidebar with inline UI; unify mixed zh-CN/en strings to English.
  Files: `src/components/HistorySidebar.jsx`.
  Done: delete now uses an inline confirm/cancel cluster; resume/delete errors
  render in a dismissible banner; `formatDate` uses the default locale + "Nd
  ago" instead of zh-CN / "N天前".
- [x] **10. Misc polish** — cache the `debugAgentSocket` flag instead of reading
  localStorage per WS event; make the sidebar brand mark not full-reload the
  page; bump a few `text-ink-soft` micro-labels to `ink-muted` for contrast.
  Files: `src/hooks/useAgentSocket.js`, `src/components/Sidebar.jsx`, `src/index.css`.
  Done: `shouldDebugSocket` caches its first read; sidebar brand mark is now a
  button that starts a new chat (no full reload); light-mode `--ink-soft`
  darkened (zinc-500 → ~zinc-600) so small muted labels clear AA; History
  meta line bumped to `ink-muted`.

## Verification

- [x] `npm run lint` clean
- [x] `npm run build` succeeds; `dist/assets` JS chunks dropped from ~9.85k
  (one per Prism grammar) to ~two dozen after #6. Main + markdown bundles
  unchanged (~125 kB + ~123 kB gzip).

## Aesthetic calm pass (blue kept as primary)

- [x] **A1. Refined blue accent** — kept the blue family but moved to a cooler
  cobalt in light mode (`30 88 224`) and a calmer, less-neon blue in dark mode
  (`122 162 247`); softened accent-soft fills. File: `src/index.css`.
- [x] **A2. Neutral user bubble** — user messages are now a neutral
  `surface-muted` card with a hairline border instead of a full accent-blue
  fill, so blue reads as an accent (buttons/links/icons/status) rather than
  saturating the conversation. Files: `src/components/timelineItems.jsx`,
  `src/components/ChatMarkdown.jsx` (user/assistant now share one neutral
  style set; dropped white-on-accent theming).
- [x] **A3. Calmer welcome screen** — removed the spinning conic-gradient halo
  (replaced with a static soft glow) and unified the hero glyph to `Bot` to
  match the sidebar brand mark. File: `src/components/WelcomeScreen.jsx`.

## Next-step pass (perf + aesthetic refinement)

- [x] **B1. Self-host fonts** — installed `@fontsource-variable/inter` and
  `@fontsource/jetbrains-mono`, imported them in `src/main.jsx`, and removed
  the remote font links from `index.html`.
- [x] **B2. Conditional KaTeX** — split a light GFM-only renderer
  (`MarkdownRendererPlain`) from the math pipeline; `MarkdownContent` only loads
  the heavy KaTeX chunk (~katex CSS + rehype-katex) when the text actually has
  math delimiters. Files: `src/components/MarkdownContent.jsx`, new
  `src/components/MarkdownRendererPlain.jsx`.
- [x] **B3. Scroll handler throttle** — timeline scroll measurement is now
  coalesced into one `requestAnimationFrame` per frame. File: `src/App.jsx`.
- [x] **B4. Quieter labels** — softened the loud `uppercase tracking-[…] font-bold`
  10px eyebrows to sentence-case `font-medium` across WelcomeScreen, the shared
  `Label`/`Section` primitives, ProcessBlock, PythonExecutionCard, ActivityItem,
  SetupModal, SkillsModal, and the system chip.
- [x] **B5. De-doubled cards** — removed resting `shadow-xs` from the `Card`
  primitive, the Python execution card, welcome quick-prompt cards, and the user
  bubble; cards now lean on the hairline border (floating chrome — composer,
  modals, drawer — keeps its shadow).
- [x] **B6. Chat chrome** — refined the assistant attribution badge (larger,
  rounded-md, sentence-case label). File: `src/components/timelineItems.jsx`.
- [x] **B7. Responsive + loading** — initial session check now shows a centered
  "Restoring your workspace…" spinner and no longer flashes the Setup modal;
  added mobile horizontal gutters to the timeline. File: `src/App.jsx`.

## Status: all in-scope items complete
