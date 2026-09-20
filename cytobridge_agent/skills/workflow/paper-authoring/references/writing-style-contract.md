# Writing Style Contract

Use this contract after `paper_plan.md` exists and before declaring the paper
complete.

## Research Paper Voice

- Write like a paper, not like an agent report.
- Lead with the contribution and the mechanism.
- Write in a confident, direct, natural voice. The manuscript should sound like
  authors explaining a discovery or method they understand, not a system
  defending why a workflow passed.
- Make the paper understandable to a reader who does not know the project,
  CytoBridge, WFR-FM, or the previous experiment history.
- Start from the biological or mathematical motivation, then introduce the
  method as the natural representation of that motivation.
- Explain the key idea in plain language before technical notation, and define
  notation before relying on it.
- Use positive but bounded claims: state what the work achieves, then place
  limitations in Discussion or supplement.
- Do not repeatedly undercut claims with immediate caveats.
- Avoid repeated antithetical sentence patterns such as "not X, but Y" or
  "rather than X, we Y". Use them only when the contrast is the central
  scientific point. If many consecutive sentences use this pattern, rewrite the
  passage as a positive explanation.
- Do not describe internal workflow decisions unless they matter scientifically.

## Narrative Coherence

- Every main section should answer the same central question introduced in the
  Introduction. If Introduction, Method, Experiments, and Results feel like
  separate reports, rewrite the plan.
- The Introduction should establish: problem, consequence, gap, idea, and
  evidence. Avoid opening with abstract machinery before the reader knows why it
  is needed.
- Method writing should be derivational: failure mode -> representation ->
  objective -> training -> inference -> evaluation. Do not present equations as
  disconnected definitions.
- Related Work should explain what existing method families can and cannot
  represent, then position the paper. Do not list papers without explaining the
  gap.
- Results should test the paper's promised object. Do not present metrics whose
  relationship to the contribution is unclear.
- Metric definitions should not crowd out insight. Introduce metrics as tools
  for testing paper claims, not as the main story.
- Discussion should synthesize what the evidence establishes and where it
  stops. It should not be the first place where the paper's motivation becomes
  clear.

## Packaging the Contribution

- Use confident, bounded framing. Say what the method enables, then state the
  evidence level.
- Avoid vague novelty claims such as "new framework" unless the concrete new
  object, assumption, or evaluation target is named.
- Name the method as if it were intended for publication. Prefer a short,
  pronounceable name with a meaningful expansion over a registry id or keyword
  concatenation. Use internal ids only in supplement/provenance.
- If evidence is limited, package the paper as a method/theory or
  proof-of-concept contribution rather than a benchmark victory.
- A reader should be able to summarize the paper in one sentence after reading
  the abstract and first two Introduction paragraphs.

## Main-Body Banned Language

The main paper body must not use agent, workflow, or internal engineering
vocabulary. These terms are banned in `main.tex` and `sections/*.tex` unless
the user explicitly asks for a technical report or the term is quoted from a
source:

- rollout
- artifact
- runtime
- contract
- preflight
- hard gate
- gate
- campaign
- trial
- trial id
- trial_
- preserved run
- fallback
- agent
- workspace
- implementation map
- local path
- sealed trajectory
- tool call
- approved
- final locked
- locked release
- final regression
- Stage 1
- Stage 2
- Stage 3

Prefer manuscript language:

- generated trajectory or trajectory simulation
- evidence, figure, metric, or saved result
- implementation or training procedure
- evaluation protocol or evaluation criterion
- ablation or experiment
- selected configuration
- validation threshold
- validation experiment
- final evaluation
- trained model

When translating source documents, rewrite internal terms instead of copying
them. Examples:

- `rollout` -> `generated trajectory` or `trajectory simulation`
- `artifact` -> `trained model`, `saved result`, `figure`, or `output`
- `runtime` -> `implementation` or `inference procedure`
- `contract` -> `protocol`, `definition`, or `criterion`
- `hard gate` -> `validation threshold` or `required criterion`
- `campaign/trial` -> `experiment`, `run`, or `evaluation`
- `agent` -> `method`, `model`, or `writing process`, depending on context
- `approved/final locked/locked release/final regression` -> `selected model`,
  `final evaluation`, or `trained method`, depending on context

The style audit must run a literal search for the banned terms. Any main-body
hit is a failing audit unless it is explicitly justified as a quoted source or
scientific term. Do not send the paper to reviewer with unresolved main-body
hits.

## Report-Like Writing To Remove

Rewrite the manuscript before review when it has these symptoms:

- paragraphs are ordered by workflow chronology rather than by claims;
- the main text repeatedly mentions approval, gates, stages, internal ids, or
  final locked state;
- the Abstract is mostly a list of components and metrics rather than a problem,
  idea, result, and implication;
- the Results section defines many metrics before giving a reader takeaway;
- equations use symbols that are never defined or change meaning across
  sections;
- the paper mainly says what the method avoids, does not assume, or does not
  claim, instead of explaining what it contributes.

## Sentence-Level Rules

- Avoid dash-heavy sentences when a period or comma is clearer.
- Avoid quotation marks around ordinary technical terms.
- Avoid colon-heavy lists in prose unless the list is genuinely easier to read.
- Avoid generic AI phrases such as "it is worth noting", "delve",
  "landscape", "pivotal", "underscore", "notably", and "comprehensive".
- Prefer concrete verbs over nominalizations.
- Keep technical terms consistent. Do not rename a defined object to avoid
  repetition.

## Results Writing

- Start each Results paragraph with the claim or reader takeaway.
- Then give the evidence: metric, table, figure, or comparison.
- Put run ids, local paths, configuration details, and failure bookkeeping in
  supplement or `evidence_ledger.md`.
- If a comparison is weak, state the measured result and reserve the boundary
  for Discussion.

## Final Passes

- Reverse outline: read the first sentence of every paragraph; the sequence
  should tell the paper story.
- Claim coverage: every contribution in the Introduction must be supported in
  Method, Theory, or Results.
- De-report pass: remove main-body local paths, long ids, workflow terms, and
  repeated defensive caveats.
- Banned-term pass: run a literal search over `main.tex` and `sections/*.tex`
  for the banned vocabulary above and rewrite every hit into manuscript
  language before reviewer submission.
- Notation pass: every symbol in each displayed equation must have been defined
  before use.
- Main-PDF pass: `main.tex` must not input appendix or supplement files unless
  the user explicitly requested a combined PDF.
- Audit-file pass: write `checks/style_audit.md` with the concrete terms found,
  fixes made, and any terms intentionally retained.
