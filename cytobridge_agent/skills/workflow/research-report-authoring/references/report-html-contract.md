# Report HTML Contract

Use this contract for `report.html` deliverables.

## HTML Skeleton

- Write a complete HTML document with `<!DOCTYPE html>`, `<html>`, `<head>`,
  and `<body>`.
- Include responsive CSS in the head.
- Use semantic sections: `summary`, `evidence`, `results`, `figures`,
  `limitations`, and `artifact-index` when applicable.
- Use valid relative paths for local figures when possible.

## Math

If the report contains formulas, include MathJax in the head:

```html
<script>
window.MathJax = {
  tex: {
    inlineMath: [['\\(', '\\)']],
    displayMath: [['\\[', '\\]'], ['$$', '$$']],
    processEscapes: true
  },
  svg: { fontCache: 'global' }
};
</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
```

Rules:

- Inline math: `\( ... \)`.
- Display math: `\[ ... \]`.
- Do not put equations in `<pre>`, `<code>`, or Markdown fences.
- Use code blocks only for literal code, commands, or logs.

## Tables

- Use real HTML tables.
- Do not write one-line Markdown pipe tables inside HTML.
- Add column headers.
- Keep long text readable with line breaks or short cells.

Example:

```html
<table>
  <thead>
    <tr><th>Claim</th><th>Evidence</th><th>Status</th></tr>
  </thead>
  <tbody>
    <tr><td>Example claim</td><td>metrics.json</td><td>supported</td></tr>
  </tbody>
</table>
```

## Evidence And Citations

- Cite concrete artifacts inline: paths, figure names, RAG chunk ids, run ids,
  config names, metric files, or paper titles.
- For literature-heavy reports, include a sources section with paper title,
  local PDF path or URL, and the claim supported by that source.
- Do not cite a source for a claim unless the source was inspected or retrieved.

## Required Checks Before Saving

- Search the draft for `<pre>` and confirm every occurrence is literal code/logs.
- Search the draft for `|---|` and replace Markdown tables with HTML tables.
- Confirm every image path exists or mark it as missing.
- Confirm limitations are explicit.
