#!/bin/bash
# Force re-run seed 42 for cytobridge + run codex and biomini on all scenarios
set -e

source /home/zty/miniconda3/etc/profile.d/conda.sh
conda activate cytobridge
cd /home/zty/CytoBridge-agent-benchmark_v2

RESULTS_DIR="benchmark/results"
CONFIG_DIR="benchmark/dynbench/simulators/test_configs"
LOG_DIR="$RESULTS_DIR/run_logs_supplement"
mkdir -p "$LOG_DIR"

# All 12 Phase 2 batch directories
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

echo "=== Supplement Run: cytobridge seed42 re-run + codex + biomini ==="
echo ""

for entry in "${SCENARIOS[@]}"; do
  IFS='|' read -r NAME CONFIG BATCH <<< "$entry"
  BATCH_DIR="$RESULTS_DIR/$BATCH"
  SPEC_FILE="$CONFIG_DIR/$CONFIG"
  
  # Find the actual scenario name from the batch directory
  if [ ! -d "$BATCH_DIR/native_runs/synthetic" ]; then
    echo "[$NAME] No native_runs found, skipping (will run after cytobridge batch completes)"
    continue
  fi
  
  SCENARIO=$(ls "$BATCH_DIR/native_runs/synthetic/" 2>/dev/null | head -1)
  if [ -z "$SCENARIO" ]; then
    echo "[$NAME] No scenario found, skipping"
    continue
  fi
  
  echo "[$NAME] Scenario: $SCENARIO"
  
  # 1. Force re-run cytobridge seed 42
  echo "  [1/3] cytobridge seed 42 (re-run)..."
  # Remove old seed 42 eval to force re-evaluation
  rm -f "$BATCH_DIR/native_runs/synthetic/$SCENARIO/agent_cytobridge_skills-on_seed42/eval_results.json"
  LOGFILE="$LOG_DIR/${NAME}_cytobridge_seed42.log"
  python benchmark/batch_benchmark_runner.py \
    --topology-spec-file "$SPEC_FILE" \
    --difficulties medium \
    --seeds 42 \
    --run-seeds 42 \
    --agents cytobridge \
    --scenarios "$SCENARIO" \
    --skip-generate \
    --timeout-sec 1800 \
    --output-dir "$BATCH_DIR" \
    2>&1 | tee -a "$LOGFILE"
  
  # 2. Codex agent (3 seeds)
  echo "  [2/3] codex (seeds 42,137,256)..."
  LOGFILE="$LOG_DIR/${NAME}_codex.log"
  python benchmark/batch_benchmark_runner.py \
    --topology-spec-file "$SPEC_FILE" \
    --difficulties medium \
    --seeds 42 \
    --run-seeds 42,137,256 \
    --agents codex \
    --scenarios "$SCENARIO" \
    --skip-generate \
    --timeout-sec 600 \
    --output-dir "$BATCH_DIR" \
    2>&1 | tee "$LOGFILE"
  
  # 3. Biomini agent (3 seeds)
  echo "  [3/3] biomini (seeds 42,137,256)..."
  LOGFILE="$LOG_DIR/${NAME}_biomini.log"
  python benchmark/batch_benchmark_runner.py \
    --topology-spec-file "$SPEC_FILE" \
    --difficulties medium \
    --seeds 42 \
    --run-seeds 42,137,256 \
    --agents biomini \
    --scenarios "$SCENARIO" \
    --skip-generate \
    --timeout-sec 1800 \
    --output-dir "$BATCH_DIR" \
    2>&1 | tee "$LOGFILE"
  
  echo "  [$NAME] Done"
  echo ""
done

echo "=== All supplement runs complete ==="
