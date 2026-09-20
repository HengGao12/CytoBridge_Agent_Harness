#!/bin/bash
# Batch A1-A3: all 3 agents, 3 seeds, fresh run with seed fix
set -euo pipefail

PROJ="/home/zty/CytoBridge-agent-benchmark_v2"
RESULTS="$PROJ/benchmark/results"
LOGDIR="$RESULTS/run_logs_phase3"
mkdir -p "$LOGDIR"

eval "$(/home/zty/miniconda3/bin/conda shell.bash hook 2>/dev/null)"
conda activate cytobridge
cd "$PROJ"
rm -f ~/.cellcompass/auth_profiles.json.lock

RUNNER="python benchmark/batch_benchmark_runner.py"
TIMEOUT=1800

run_batch() {
    local NAME="$1" TOPO="$2" SCENARIO="$3"
    local LOGFILE="$LOGDIR/${NAME}_$(date +%Y%m%d_%H%M%S).log"
    echo "[$(date '+%H:%M:%S')] START: $NAME"
    $RUNNER \
        --topology-spec-file "$TOPO" \
        --difficulties medium \
        --seeds 42 \
        --run-seeds 42,137,256 \
        --agents cytobridge,biomni,codex \
        --scenarios "$SCENARIO" \
        --skip-generate \
        --timeout-sec $TIMEOUT \
        --output-dir "$RESULTS/$NAME" \
        2>&1 | tee "$LOGFILE"
    echo "[$(date '+%H:%M:%S')] END: $NAME | rc=${PIPESTATUS[0]}"
}

echo "=========================================================="
echo "Batch A1-A3 rerun (seed fix + 3 agents) $(date)"
echo "=========================================================="

# A1: gene gradient low (branching_tree, 8 fates, low)
run_batch "batch_A1_gene_gradient_low" \
    "benchmark/dynbench/simulators/test_configs/phase2_A1_gene_gradient_low.json" \
    "branching_tree_medium_seed42"

# A2: gene gradient mid (branching_tree, 8 fates, mid)
run_batch "batch_A2_gene_gradient_mid" \
    "benchmark/dynbench/simulators/test_configs/phase2_A2_gene_gradient_mid.json" \
    "branching_tree_medium_seed42"

# A3: gene gradient match (asymmetric_tree, 8 fates, match)
run_batch "batch_A3_gene_gradient_match" \
    "benchmark/dynbench/simulators/test_configs/phase2_A3_gene_gradient_match.json" \
    "asymmetric_tree_medium_seed42"

echo ""
echo "=========================================================="
echo "ALL DONE $(date)"
echo "=========================================================="
