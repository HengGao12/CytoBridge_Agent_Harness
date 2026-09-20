# Approved Panel Workflow

This document standardizes the per-panel archiving workflow for manuscript figures.

## Why Per-Panel Scripts

A single monolithic script that generates all panels in one figure has several problems:

- Changing one panel requires re-running the entire figure.
- Layout is coupled to content: adjusting panel D's UMAP limits can break panel A's spacing.
- Iterative approval is difficult because the user cannot approve panels independently.
- Final assembly in Illustrator/Inkscape requires separate vector PDFs anyway.

The per-panel workflow solves these by treating each panel as an independent, self-contained unit.

## Directory Structure

```
<dataset>/approved/
├── README.md                            # Describes all panels
├── panel_A_<short_name>/
│   ├── make_<dataset>_panel_A_<short_name>.py
│   ├── <dataset>_panel_A_<short_name>.png
│   └── <dataset>_panel_A_<short_name>.pdf
├── panel_B_<short_name>/
│   ├── make_<dataset>_panel_B_<short_name>.py
│   ├── <dataset>_panel_B_<short_name>.png
│   └── <dataset>_panel_B_<short_name>.pdf
├── ...
```

### Naming Conventions

- Directory: `panel_<LETTER>_<descriptive_name>` (e.g., `panel_D_trajectory_divergence`)
- Script: `make_<dataset>_panel_<LETTER>_<descriptive_name>.py`
- Output: `<dataset>_panel_<LETTER>_<descriptive_name>.{png,pdf}`
- Panel letters follow the final figure layout order. If panels are swapped, rename the
  directories and files accordingly.

## Per-Panel Script Requirements

Each script must be **self-contained and runnable independently**:

1. **Path resolution**: Use `pathlib.Path(__file__).resolve().parent` to find its own location,
   then navigate to data files using relative paths. Do not hardcode absolute paths.
2. **Data loading**: Load only the data files this panel needs (not the entire AnnData if only
   a CSV is required).
3. **Color constants**: Define domain-specific colors at the top of the script as named
   constants. These should be consistent across all panel scripts.
4. **No external template imports** unless the template is stable and in the template library.
   Prefer inlining plotting code for robustness.
5. **Dual output**: Save both PNG (for quick preview, dpi=400) and PDF (for editing, dpi=300).
6. **Vector PDF by default**: Do not use `rasterized=True` on scatter plots unless point count
   exceeds ~50,000. See PDF Export Rules in SKILL.md.
7. **No panel letter in the figure** (optional): Some users prefer to add panel letters in
   Illustrator during final assembly. Check with the user.

### Example Script Header

```python
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]   # <dataset>/ directory

# Domain colors (consistent across all panels)
BM_C  = "#d62728"
AL_C  = "#1f77b4"
FP_C  = "#2ca02c"
BG_C  = "#d0d0d0"
INK   = "#222222"

# Data paths
CACHE = BASE / "main_figure_materials_YYYYMMDD" / "03_support_results" / "cache.npz"
```

## README Requirements

The `approved/README.md` must document each panel with:

1. **Content description**: What the panel shows visually.
2. **Scientific message**: What conclusion the panel supports.
3. **Data sources**: Exact file paths (relative to the materials directory) with column names.
4. **Analysis principle**: How the underlying numbers were computed (not just "loaded from CSV",
   but "Jacobian perturbation of latent dynamics" or "kNN expression estimation along ODE
   trajectory").
5. **Script locations**: Both the panel plotting script and the upstream analysis script.

Include a data path summary table at the end for quick reference.

## Approval Process

1. **Draft phase**: Generate panels into a `04_candidate_figures/` directory (not `approved/`).
   Show to the user for feedback.
2. **Iteration**: Modify and regenerate until the user says "approve" or "可以".
3. **Save to approved**: Move the script and outputs to the `approved/` directory. Generate
   both PNG and PDF. Verify PDF quality at 400% zoom.
4. **README update**: Add or update the panel entry in `README.md`.
5. **Post-approval changes**: If the user requests changes to an approved panel, modify the
   script in-place, regenerate, and verify. Do not create a new directory.

## Swapping Panels

If the user wants to swap panel letters (e.g., swap B and C):

1. Rename directories: `panel_B_...` ↔ `panel_C_...`
2. Rename files inside each directory to match the new letter.
3. Update `README.md` to reflect the new order.
4. The script contents (internal logic) do not need to change—only file/directory names.

## Final Assembly

The approved per-panel PDFs are designed to be imported into Illustrator/Inkscape for final
page-level composition. This means:

- Each PDF should have tight bounding boxes (`bbox_inches="tight"`).
- Panel letters can be omitted from the script if the user prefers to add them during assembly.
- All elements should be vector (editable) unless explicitly rasterized for performance.
- Font sizes should follow the global typography hierarchy so they remain consistent after
  import at the same scale.
