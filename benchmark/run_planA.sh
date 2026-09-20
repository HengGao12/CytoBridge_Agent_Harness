#!/bin/bash
# Plan A (Phase 2): 12 controlled-variable scenarios
# 3 groups × 4 scenarios each
# Group A: gene gradient (A1-A4)
# Group B: fate gradient (B1-B4)
# Group C: topology comparison (C1-C4)
set -e

source /home/zty/miniconda3/etc/profile.d/conda.sh
conda activate cytobridge
cd /home/zty/CytoBridge-agent-benchmark_v2

RESULTS_DIR="benchmark/results"
CONFIG_DIR="benchmark/dynbench/simulators/test_configs"
LOG_DIR="$RESULTS_DIR/run_logs_planA"
mkdir -p "$LOG_DIR"

AGENTS="cytobridge"
SEEDS="42"
RUN_SEEDS="42,137,256"
TIMEOUT=1800

# 12 scenarios: Group A (gene gradient) + Group B (fate gradient) + Group C (topology)
SCENARIOS=(
  "A1_gene_gradient_low|phase2_A1_gene_gradient_low.json|batch_A1_gene_gradient_low"
  "A2_gene_gradient_mid|phase2_A2_gene_gradient_mid.json|batch_A2_gene_gradient_mid"
  "A3_gene_gradient_match|phase2_A3_gene_gradient_match.json|batch_A3_gene_gradient_match"
  "A4_gene_gradient_high|phase2_A4_gene_gradient_high.json|batch_A4_gene_gradient_high"
  "B1_fate_gradient_simple|phase2_B1_fate_gradient_simple.json|batch_B1_fate_gradient_simple"
  "B2_fate_gradient_medium|phase2_B2_fate_gradient_medium.json|batch_B2_fate_gradient_medium"
  "B3_fate_gradient_match|phase2_B3_fate_gradient_match.json|batch_B3_fate_gradient_match"
  "B4_fate_gradient_complex|phase2_B4_fate_gradient_complex.json|batch_B4_fate_gradient_complex"
  "C1_topology_symmetric|phase2_C1_topology_symmetric.json|batch_C1_topology_symmetric"
  "C2_topology_asymmetric|phase2_C2_topology_asymmetric.json|batch_C2_topology_asymmetric"
  "C3_topology_feedback|phase2_C3_topology_feedback.json|batch_C3_topology_feedback"
  "C4_topology_competitive|phase2_C4_topology_competitive.json|batch_C4_topology_competitive"
)

TOTAL=${#SCENARIOS[@]}
COUNT=0
FAILED=0
START_ALL=$(date +%s)

for entry in "${SCENARIOS[@]}"; do
  IFS='|' read -r NAME CONFIG OUTPUT <<< "$entry"
  COUNT=$((COUNT + 1))
  LOGFILE="$LOG_DIR/${OUTPUT}.log"
  OUTDIR="$RESULTS_DIR/$OUTPUT"

  echo "[$(date '+%H:%M:%S')] [$COUNT/$TOTAL] Starting: $NAME"
  START_TIME=$(date +%s)

  python benchmark/batch_benchmark_runner.py \
    --topology-spec-file "$CONFIG_DIR/$CONFIG" \
    --difficulties medium \
    --seeds $SEEDS \
    --run-seeds $RUN_SEEDS \
    --agents "$AGENTS" \
    --timeout-sec $TIMEOUT \
    --force-regenerate \
    --output-dir "$OUTDIR" \
    2>&1 | tee "$LOGFILE"

  RC=$?
  ELAPSED=$(( $(date +%s) - START_TIME ))
  [ $RC -ne 0 ] && FAILED=$((FAILED + 1))
  echo "[$(date '+%H:%M:%S')] [$COUNT/$TOTAL] $([ $RC -eq 0 ] && echo 'OK' || echo 'FAIL') — ${ELAPSED}s"
done

TOTAL_ELAPSED=$(( $(date +%s) - START_ALL ))
echo ""
echo "=== Plan A Complete ==="
echo "Finished: $((TOTAL - FAILED))/$TOTAL succeeded, $FAILED failed"
echo "Total time: $((TOTAL_ELAPSED / 3600))h $((TOTAL_ELAPSED % 3600 / 60))m $((TOTAL_ELAPSED % 60))s"
echo "Results: $RESULTS_DIR/"
