#!/bin/bash
# Phase 2 rerun script - 2026-05-13
# Organized by batch, runs all needed agents in one pass per batch
set -euo pipefail

PROJ="/home/zty/CytoBridge-agent-benchmark_v2"
RESULTS="$PROJ/benchmark/results"
LOGDIR="$RESULTS/run_logs_phase2"
mkdir -p "$LOGDIR"

eval "$(/home/zty/miniconda3/bin/conda shell.bash hook 2>/dev/null)"
conda activate cytobridge
cd "$PROJ"

rm -f ~/.cellcompass/auth_profiles.json.lock

RUNNER="python benchmark/batch_benchmark_runner.py"
TIMEOUT=1800

run_batch() {
    local BATCH="$1" TOPO="$2" SCENARIO="$3" AGENTS="$4" SEEDS="$5"
    local LOGFILE="$LOGDIR/${BATCH}_$(date +%Y%m%d_%H%M%S).log"
    echo "[$(date '+%H:%M:%S')] START: $BATCH | $AGENTS | seeds=$SEEDS"
    $RUNNER \
        --topology-spec-file "$TOPO" \
        --difficulties medium \
        --seeds 42 \
        --run-seeds "$SEEDS" \
        --agents "$AGENTS" \
        --scenarios "$SCENARIO" \
        --skip-generate \
        --timeout-sec $TIMEOUT \
        --output-dir "$RESULTS/$BATCH" \
        2>&1 | tee "$LOGFILE"
    echo "[$(date '+%H:%M:%S')] END: $BATCH | rc=${PIPESTATUS[0]}"
}

echo "=========================================================="
echo "Phase 2 rerun START $(date)"
echo "=========================================================="

# --- A1: cytobridge re-run seed42 + biomni/codex all seeds ---
run_batch "batch_A1_gene_gradient_low_20260513_003947" \
    "benchmark/dynbench/simulators/test_configs/phase2_A1_gene_gradient_low.json" \
    "branching_tree_medium_seed42" \
    "cytobridge,biomni,codex" "42,137,256"

# --- A2: cytobridge seed256 re-run + biomni/codex all seeds ---
run_batch "batch_A2_gene_gradient_mid" \
    "benchmark/dynbench/simulators/test_configs/phase2_A2_gene_gradient_mid.json" \
    "branching_tree_medium_seed42" \
    "cytobridge,biomni,codex" "42,137,256"

# --- A3: cytobridge all seeds (quota failed) ---
run_batch "batch_A3_gene_gradient_match" \
    "benchmark/dynbench/simulators/test_configs/phase2_A3_gene_gradient_match.json" \
    "asymmetric_tree_medium_seed42" \
    "cytobridge" "42,137,256"

# --- A4: cytobridge all seeds ---
run_batch "batch_A4_gene_gradient_high" \
    "benchmark/dynbench/simulators/test_configs/phase2_A4_gene_gradient_high.json" \
    "branching_tree_medium_seed42" \
    "cytobridge" "42,137,256"

# --- B1: cytobridge all seeds ---
run_batch "batch_B1_fate_gradient_simple" \
    "benchmark/dynbench/simulators/test_configs/phase2_B1_fate_gradient_simple.json" \
    "compact_toggle_medium_seed42" \
    "cytobridge" "42,137,256"

# --- B2: cytobridge all seeds ---
run_batch "batch_B2_fate_gradient_medium" \
    "benchmark/dynbench/simulators/test_configs/phase2_B2_fate_gradient_medium.json" \
    "branching_tree_medium_seed42" \
    "cytobridge" "42,137,256"

# --- B3: cytobridge all seeds ---
run_batch "batch_B3_fate_gradient_match" \
    "benchmark/dynbench/simulators/test_configs/phase2_B3_fate_gradient_match.json" \
    "asymmetric_tree_medium_seed42" \
    "cytobridge" "42,137,256"

# --- B4: cytobridge all seeds ---
run_batch "batch_B4_fate_gradient_complex" \
    "benchmark/dynbench/simulators/test_configs/phase2_B4_fate_gradient_complex.json" \
    "branching_tree_medium_seed42" \
    "cytobridge" "42,137,256"

# --- C1: cytobridge all seeds ---
run_batch "batch_C1_topology_symmetric" \
    "benchmark/dynbench/simulators/test_configs/phase2_C1_topology_symmetric.json" \
    "branching_tree_medium_seed42" \
    "cytobridge" "42,137,256"

# --- C2: cytobridge all seeds ---
run_batch "batch_C2_topology_asymmetric" \
    "benchmark/dynbench/simulators/test_configs/phase2_C2_topology_asymmetric.json" \
    "asymmetric_tree_medium_seed42" \
    "cytobridge" "42,137,256"

# --- C3: cytobridge seed137,256 re-run + biomni/codex all seeds ---
run_batch "batch_C3_topology_feedback" \
    "benchmark/dynbench/simulators/test_configs/phase2_C3_topology_feedback.json" \
    "branching_with_feedback_medium_seed42" \
    "cytobridge,biomni,codex" "42,137,256"

# --- C4: cytobridge all seeds ---
run_batch "batch_C4_topology_competitive" \
    "benchmark/dynbench/simulators/test_configs/phase2_C4_topology_competitive.json" \
    "competitive_multistable_medium_seed42" \
    "cytobridge" "42,137,256"

echo ""
echo "=========================================================="
echo "Phase 2 ALL DONE $(date)"
echo "=========================================================="
