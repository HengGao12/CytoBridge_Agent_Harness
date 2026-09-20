#!/usr/bin/env python
"""
Generate a self-contained task package for any agent to solve.

Usage:
    python -m benchmark.make_task_package \
        --config benchmark/configs/weinreb_2020.yaml \
        --fold middle_holdout \
        --output-dir benchmark/task_packages/weinreb_middle

The task package contains:
  - train.h5ad (training data)
  - TASK.md (instructions for the agent)
  - eval_prediction.py (self-check script the agent can optionally run)

Give this folder to any agent (Codex App, Antigravity, Biomni, etc.)
Then evaluate the output with:
    python -m benchmark.eval_external \
        --prediction <path>/predicted_heldout.h5ad \
        --config benchmark/configs/weinreb_2020.yaml \
        --fold middle_holdout \
        --agent codex --seed 42
"""
from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

from .benchmark_utils import load_task_card, get_fold_data_dir

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TASK_MD_TEMPLATE = """# Benchmark Task: Cell Distribution Prediction

## Goal
{task_description}

## Input
- **`train.h5ad`**: scRNA-seq data with cells at time points [{training_timepoints}]
- Time column: `obs["{time_key}"]`

## Environment
Use the **CytoCompass** conda environment which has all required packages pre-installed:
```bash
conda activate CytoCompass
# Or prefix commands with: conda run -n CytoCompass python <script.py>
```

Installed packages: scanpy, anndata, numpy, scipy, sklearn, torch, POT (Python Optimal Transport)

## Output Requirements
Create an `output/` subdirectory and save ALL of the following files there:
```bash
mkdir -p output
```

1. **`output/predicted_heldout.h5ad`** (REQUIRED) — your prediction AnnData with:
   - `.X`: predicted gene expression in **normalized + log1p space**
     (i.e., after `sc.pp.normalize_total(target_sum=1e4)` then `sc.pp.log1p()`)
   - `.obs["weight"]`: per-cell weights (unnormalized). If your method produces per-cell weights, save them in .obs['weight']. Do NOT normalize them to sum to 1 — the evaluation pipeline handles normalization internally.
   - The `.var_names` must match the genes in `train.h5ad`

2. **`output/predict.py`** (REQUIRED) — your final prediction script (for reproducibility)

3. **`output/run_log.json`** (REQUIRED) — timing and metadata, create at the end of your script:
```python
import json, time
# At the start of your script:
_start_time = time.time()
# ... your code ...
# At the end:
with open("output/run_log.json", "w") as f:
    json.dump({{
        "runtime_sec": time.time() - _start_time,
        "method": "<brief description of your method>",
        "seed": {seed},
    }}, f, indent=2)
```

## Rules
1. Use ONLY the data in `train.h5ad` — do NOT access any held-out or test data
2. Do NOT use CytoBridge or any specialized agent framework
3. You may use any method: optimal transport, neural ODEs, interpolation, flow matching, etc.
4. Random seed: {seed}
5. **Prediction must start from the EARLIEST available time point.** Your model should predict the held-out distribution by simulating/interpolating from the first observed time point all the way to the target. Do NOT simply project from the nearest time point.

## Self-check (REQUIRED)
After generating your prediction, run this to verify the output format:
```bash
conda run -n CytoCompass python eval_prediction.py
```
If it prints "OK", your prediction is correctly formatted.

## Report (REQUIRED)
After generating your prediction, create a file `output/report.md` summarizing:
1. **Method**: what approach you used (e.g. optimal transport, neural ODE, interpolation)
2. **Preprocessing**: how you preprocessed the data (normalization, HVG, PCA, etc.)
3. **Runtime**: total time from `run_log.json`
4. **Caveats**: any limitations or warnings encountered

## Final Evaluation (REQUIRED — run this LAST)
After self-check passes and report is written, run this to compute evaluation metrics:
```bash
cd {repo_root} && conda run -n CytoCompass python -m benchmark.eval_external \\
    --prediction {task_package_dir}/output/predicted_heldout.h5ad \\
    --config {config_path} \\
    --fold {fold_id} \\
    --agent AGENT_NAME \\
    --seed {seed}
```
Replace AGENT_NAME with your agent name (e.g. codex, antigravity).
This will print W1/W2 metrics and copy all outputs (including report.md) to the benchmark results directory.
"""



EVAL_SCRIPT = '''#!/usr/bin/env python
"""Quick self-check: verify predicted_heldout.h5ad format."""
import sys
from pathlib import Path

try:
    import anndata as ad
except ImportError:
    print("ERROR: anndata not installed")
    sys.exit(1)

pred_path = Path(__file__).parent / "output" / "predicted_heldout.h5ad"
if not pred_path.exists():
    print(f"ERROR: {pred_path} not found")
    sys.exit(1)

adata = ad.read_h5ad(pred_path)
print(f"Shape: {adata.shape}")
print(f"Has .X: {adata.X is not None}")
print(f"Has X_latent: {'X_latent' in adata.obsm}")
print(f"Has weight: {'weight' in adata.obs.columns}")
print("OK — prediction file looks valid!")
'''


def main():
    parser = argparse.ArgumentParser(description="Generate task package for external agents")
    parser.add_argument("--config", required=True, help="Task-card YAML")
    parser.add_argument("--fold", required=True, help="Fold ID")
    parser.add_argument("--output-dir", required=True, help="Output directory for task package")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    card = load_task_card(args.config)
    fold = next((f for f in card["folds"] if f["fold_id"] == args.fold), None)
    if not fold:
        raise ValueError(f"Fold '{args.fold}' not found")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Copy train data
    fold_dir = get_fold_data_dir(card["dataset_id"], args.fold)
    train_src = fold_dir / "train.h5ad"
    if not train_src.exists():
        raise FileNotFoundError(f"Run benchmark_prepare first: {train_src}")

    logger.info(f"Copying training data to {out / 'train.h5ad'}")
    shutil.copy2(train_src, out / "train.h5ad")

    # Generate TASK.md
    training_tps = ", ".join(str(t) for t in fold["training_timepoints"])
    task_desc = card["prompt_template"].format(
        training_timepoints=training_tps,
        held_out_timepoint=str(fold["held_out_timepoint"]),
        time_key=card["time_key"],
    )
    task_md = TASK_MD_TEMPLATE.format(
        task_description=task_desc,
        training_timepoints=training_tps,
        time_key=card["time_key"],
        seed=args.seed,
        repo_root=str(Path(args.config).resolve().parent.parent.parent),
        task_package_dir=str(out.resolve()),
        config_path=str(Path(args.config).resolve()),
        fold_id=args.fold,
    )
    (out / "TASK.md").write_text(task_md)
    logger.info(f"Wrote TASK.md")

    # Generate eval script
    (out / "eval_prediction.py").write_text(EVAL_SCRIPT)
    logger.info(f"Wrote eval_prediction.py")

    print(f"\n✓ Task package ready at: {out}")
    print(f"  1. Give this folder to any agent")
    print(f"  2. Agent reads TASK.md + train.h5ad → saves predicted_heldout.h5ad")
    print(f"  3. Evaluate: python -m benchmark.eval_external --prediction {out}/predicted_heldout.h5ad --config {args.config} --fold {args.fold} --agent <name> --seed {args.seed}")


if __name__ == "__main__":
    main()
