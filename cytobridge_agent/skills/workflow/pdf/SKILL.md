---
name: "pdf"
description: "Use when a task involves reading, rendering, creating, or reviewing PDF files where page layout matters. Adapts the general PDF workflow to the current CytoBridge tool surface."
---

# PDF

Use this skill when a task involves a `.pdf` artifact and the agent needs to inspect visual layout, extract text, or generate a new PDF.

## Goal
Handle PDF work in a way that is faithful to page rendering, not just raw extracted text.

## What CytoBridge Can Do Today
The current CytoBridge agent can inspect text files, images, and PDFs through the unified `read_file(...)` tool, and can still fall back to `execute_python` when extraction or rendering needs tighter control.

Current practical mapping:
- locate PDF files with `list_path`, `find_files`, and `grep_files`
- inspect text packs or small documents with `read_file(file_path=...)`
- inspect PDF text with `read_file(file_path="paper.pdf", pdf_mode="text", pages="...")`
- inspect PDF pages visually with `read_file(file_path="paper.pdf", pdf_mode="render", pages="...")`
- inspect rendered PNG outputs with `read_file(file_path=...)`
- use `execute_python` with `pdfplumber` or `pypdf` only when structured extraction is needed
- create new PDFs with `execute_python` plus `reportlab` only if `reportlab` is installed

## Current Environment Reality
At the time this skill was installed:
- `pdftoppm` is available
- `pdfplumber` is available
- `pypdf` is available
- `reportlab` is not installed by default

This means:
- PDF reading and rendering are currently feasible
- PDF generation is possible only after installing `reportlab`

## Recommended Workflow
1. Locate the target PDF.
   - Use `list_path` / `find_files`.

2. If content understanding is enough, use `read_file(...)` first.
   - Use `pdf_mode="text"` when you want extracted text.
   - Use `pdf_mode="render"` when layout, figures, or table geometry matter.
   - Small PDFs can be read directly; large PDFs require `pages="..."`.
   - Do not treat extracted text and rendered page images as interchangeable; text is for content, page images are for layout.

3. If layout, tables, alignment, clipping, or figure placement matter, render pages to PNG.
   - Prefer `read_file(file_path="paper.pdf", pdf_mode="render", pages="1-3")` first.
   - Use `execute_python` to call `pdftoppm` only when you need persistent render artifacts under a custom directory.

4. After each meaningful PDF update, re-render and inspect again.
   - Do not rely on the previous render after changing the PDF.

5. If generating a PDF, prefer a reproducible programmatic path.
   - Use `reportlab` when available.
   - If `reportlab` is missing, say so explicitly.

## Execution Patterns
### Extract text
Use `read_file(file_path="paper.pdf", pdf_mode="text")` for small PDFs or `read_file(file_path="paper.pdf", pdf_mode="text", pages="1-3")` for targeted extraction.

Use `execute_python` with one of:
- `pdfplumber`
- `pypdf`

Preferred rule:
- `pdfplumber` for page-wise extraction and quick checks
- `pypdf` as a simpler fallback

### Render pages
Use `read_file(file_path="paper.pdf", pdf_mode="render", pages="1-3")` when transient visual inspection is enough.

Use `execute_python` and call:

```python
import subprocess
from pathlib import Path

pdf = Path("input.pdf")
out_dir = Path("tmp/pdfs/rendered")
out_dir.mkdir(parents=True, exist_ok=True)
prefix = out_dir / pdf.stem
subprocess.run(
    ["pdftoppm", "-png", str(pdf), str(prefix)],
    check=True,
)
```

### Create a PDF
Use `execute_python` with `reportlab` only if import succeeds.

If `reportlab` is unavailable, do not pretend PDF creation is supported natively in the current environment.

## Quality Rules
- Prefer rendered-page inspection over raw text when layout matters.
- Do not claim a PDF looks correct unless pages have been rendered and checked.
- Watch for:
  - clipped text
  - overlapping elements
  - table overflow
  - broken figure placement
  - unreadable glyphs
  - page-number / header / footer mistakes

## Current Tool Gaps
CytoBridge now has a unified `read_file(...)` surface for PDF inspection, but dedicated PDF utilities would still make repeated rendering/extraction cleaner. The most useful additions would be:

1. `render_pdf_pages(pdf_path, pages=None, dpi=150, output_dir=None)`
   - highest-value missing tool
   - removes the need to shell out through `execute_python`

2. `extract_pdf_text(pdf_path, page_range=None, backend="pdfplumber")`
   - structured text extraction without writing ad hoc scripts each time

3. `inspect_pdf_artifact(pdf_path)`
   - return page count, dimensions, and optionally recent render artifact paths

4. `create_pdf_artifact(...)`
   - only worth adding if PDF generation becomes common in reports or manuscript workflows

## Decision Rule
- If the task is PDF reading or PDF review: current CytoBridge tools are sufficient.
- If the task is reliable PDF generation: current CytoBridge tools are only conditionally sufficient because `reportlab` is not guaranteed.
- If PDF work becomes frequent, promote rendering and extraction into first-class tools rather than relying on `execute_python`.

## Agent Reminder
- The `workflow/pdf` skill is exposed through the workflow skill catalog.
- Start with `read_file(...)` for normal PDF reads.
- If PDF inspection needs more control than `read_file(...)` provides, follow this skill and use `execute_python` for explicit extraction/rendering.
