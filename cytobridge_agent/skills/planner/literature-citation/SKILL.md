---
name: literature-citation
description: Literature citation standards and deep utilization guidelines. Use before workflow stages that require literature support (e.g., theory selection, downstream analysis) to ensure all conclusions are evidence-backed and properly cited.
---

# Literature Citation Skill

## When to Use
- Before entering theory selection (`theory-selection`) or downstream analysis (`downstream-analysis`) stages, when those stages require literature evidence.
- The workflow orchestrator automatically determines whether to invoke this skill based on stage properties and current state.

## Literature Retrieval and Citation Standards

### 1. Basic Citation Format Requirements
- **In-text citations**: Use `[1]`, `[2]` format when referencing literature.
- **Multiple citations**: Use `[1,2,3]` or `[1-3]` format for multiple references.
- **References section**: Always include a `## References` section at the end with complete APA citations (authors, year, title, journal/preprint, DOI/arXiv ID). Citations must be taken exactly from retrieval results—do not modify.

### 2. Deep Utilization Requirements (Critical!)
Simply listing references is NOT enough. Literature must be integrated into reasoning:

- **Quote specific excerpts**: When making an argument, directly quote the original sentence from the literature using double quotes, followed by the citation number.
  
  ✅ CORRECT: Literature [2] states: "TIGON reconstructs dynamic trajectories and population growth simultaneously", which perfectly matches our dataset's 142% growth rate.
  
  ❌ INCORRECT: Literature [2] supports using unbalanced OT. (Too vague)

- **Every reasoning step must be literature-supported**:
  - **Dataset characteristic analysis**: Compare dataset features (number of time points, growth rate, label structure, etc.) with method applicability conditions described in the literature.
  - **Model exclusion reasoning**: Explain why certain models are NOT suitable, citing literature.
  - **Final selection justification**: Explain why a specific model is the best match, citing literature.

- **Analyze each candidate model**: For every candidate model (e.g., dynamical_ot, unbalanced_ot, vgfm, wfrfm, ruot, crufm, cyto_simulation), extract from literature:
  - Applicability conditions, strengths, and limitations.
  - Compare these conditions with our actual dataset characteristics.
  - Clearly state why the model is suitable or unsuitable, citing corresponding literature excerpts.
  - Even for excluded models, provide literature-based reasoning for exclusion.

### 3. Literature Retrieval Execution Steps
When literature retrieval is required, follow this process:

1. **Construct search queries**: Based on current task (e.g., theory selection) and dataset characteristics (time points, proliferation rate, cell types, etc.), generate 2-3 precise query combinations. Examples:
   - "unbalanced optimal transport single-cell multi-timepoint"
   - "TIGON cell fate prediction proliferation"
   - "regularized unbalanced OT flow matching comparison"

2. **Execute search**: Call `search_literature(query="your query here")` to retrieve relevant papers. The tool returns pre-formatted strings containing full APA citations, abstracts, and key excerpts.

3. **Check retrieval sufficiency**: Ensure returned literature covers all candidate models and relevant methodological comparisons. If coverage is insufficient, refine queries and search again.

4. **Append citation guide**: After retrieval, load this skill's guidelines (already in context) to ensure final output complies with all citation requirements.

### 4. In-Depth Reading (Optional)
When abstracts/excerpts are insufficient, request full-text reading:

1. **Trigger**: Need detailed methodology, experimental setup, or technical specifics not covered in summaries.

2. **Action**:
- Use the `pdf_path` or `pdf_filename` returned by `search_literature(...)`.
- Read the paper directly with `read_file(...)`.

3. **Recommended modes**:
- `read_file(file_path="{pdf_path}", pdf_mode="text", pages="1-3")` when you want extracted text.
- `read_file(file_path="{pdf_path}", pdf_mode="render", pages="1-3")` when layout, tables, or figures matter.

4. **Strategy**: 
- Read incrementally instead of dumping the whole paper into context.
- Use `pages="..."` to stay targeted, especially on long PDFs.
- When using `pdf_mode="text"`, use `offset` / `limit` to inspect only the relevant extracted ranges.
- Locate relevant sections (Methods, Results, etc.) using prior knowledge from abstracts.
- Extract only task-relevant details.
- Integrate new insights with existing retrieval results.

5. **Example**:
- User: `Why is unbalanced_OT better for proliferating cells?`
- Assistant: 
  - [After search_literature] The abstract mentions it handles growth, but I need details.
  - [Call: read_file(file_path="2024_tigon.pdf", pdf_mode="text", pages="5-6", offset=1, limit=80)]
  - From paper [2] p.5: "WFR distance explicitly models mass creation/destruction through unbalanced transport, unlike classical OT which enforces mass conservation."
  - Thus, unbalanced_OT fits our 142% growth data while dynamical_OT doesn't.

### 5. Example: Integrating Literature into Reasoning

Assume you have the following literature excerpts:
- [1] "MultistageOT extends bimarginal optimal transport so that transport occurs across multiple marginals..."
- [2] "TIGON reconstructs dynamic trajectories and population growth simultaneously from multiple snapshots using unbalanced optimal transport."
- [3] "WFR-FM is entirely simulation-free, eliminating costly ODE integration..."
- [4] "CellRank 2...Waddington optimal transport (WOT) for experimental time points"

Your reasoning should be structured like:

"**Dataset Characteristic Analysis**: Our dataset contains 5 time points with 142% total growth, indicating significant cell proliferation. Literature [2] explicitly states that TIGON 'reconstructs dynamic trajectories and population growth simultaneously from multiple snapshots', which perfectly matches our multi-timepoint, unbalanced data characteristics.

**Candidate Model Analysis**:
- **dynamical_ot**: Assumes mass conservation, but our data shows 142% growth, violating this assumption. Therefore, it is excluded.
- **unbalanced_ot**: Both [2] and [4] support using unbalanced OT for proliferating systems. [2] states that TIGON uses the WFR distance to handle mass creation/destruction, while [4] notes that Waddington OT is designed for experimental time points and handles unbalanced growth effectively.
- **ruot**: Requires more complex regularization; literature does not explicitly show advantages for deterministic growth patterns like ours.
- **wfrfm**: Literature [3] highlights WFR-FM's simulation-free WFR dynamic OT derivation, but its biological growth-velocity prior should still be checked against the target data.
- **crufm**: Stochastic unbalanced FM surrogate; compare it when stochasticity matters, but do not cite WFR-FM as the direct theory for CRUFM.
- **cyto_simulation**: Requires spatial data for cell-cell interaction modeling; our dataset lacks spatial information, making this model unsuitable.

**Final Selection**: Based on the above analysis, **unbalanced_ot** is the most appropriate choice, as it directly addresses our key dataset features—multiple time points and significant proliferation—with clear support from the literature [2,4]."

### 6. Final Checklist Before Submitting

- [ ] Does every key claim have a **direct quote or specific reference** from the literature?
- [ ] Have you explicitly linked dataset characteristics to method applicability conditions described in the literature?
- [ ] Have you discussed **all** candidate models, including reasons for exclusion?
- [ ] Are in-text citations correctly formatted as `[1]`, `[2]`, etc.?
- [ ] Is the `## References` section complete and correctly formatted in APA style?
- [ ] Have you included at least **2-3 distinct literature citations**?

### 7. Summary of DOs and DON'Ts

| DOs | DON'Ts |
|-----|--------|
| ✅ Quote specific excerpts from literature | ❌ Just list references at the end without using them |
| ✅ Link dataset features to method conditions | ❌ Make claims without literature support |
| ✅ Analyze each candidate model with evidence | ❌ Skip analysis of excluded models |
| ✅ Use correct citation format `[1]`, `[1-3]` | ❌ Use vague references like "as mentioned in the literature" |
| ✅ Include complete APA references section | ❌ Modify provided APA citations |

Remember: **Literature is your evidence base—use it explicitly, quote it directly, and cite it correctly.**
