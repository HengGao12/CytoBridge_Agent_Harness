#!/bin/bash
# Run all Phase 2 scenarios sequentially
# Auto-switch accounts when quota is exhausted

set -e
source /home/zty/miniconda3/etc/profile.d/conda.sh
conda activate cytobridge
cd /home/zty/CytoBridge-agent-benchmark_v2

# All 12 Phase 2 scenarios
SCENARIOS=(
    "phase2_A1_gene_gradient_low"
    "phase2_A2_gene_gradient_mid"
    "phase2_A3_gene_gradient_match"
    "phase2_A4_gene_gradient_high"
    "phase2_B1_fate_gradient_simple"
    "phase2_B2_fate_gradient_medium"
    "phase2_B3_fate_gradient_match"
    "phase2_B4_fate_gradient_complex"
    "phase2_C1_topology_symmetric"
    "phase2_C2_topology_asymmetric"
    "phase2_C3_topology_feedback"
    "phase2_C4_topology_competitive"
)

# Known team accounts (plus plan)
ACCOUNTS=(
    "openai-codex:bac1e754-4ee1-443e-a33e-bff09804afe3"
    "openai-codex:eca89c6a-e7dc-40e8-84ea-2ff2dfedb7e0"
)
ACCOUNT_NAMES=(
    "2251005164@qq.com (team)"
    "lynnhill5656@outlook.com (plus)"
)

current_account=0

switch_account() {
    local idx=$1
    local profile=${ACCOUNTS[$idx]}
    local name=${ACCOUNT_NAMES[$idx]}
    
    echo "[$(date)] Switching to account: $name"
    
    # Update config.json
    python3 -c "
import json
cfg = json.load(open('/home/zty/.cellcompass/config.json'))
cfg['llm_profile_id'] = '$profile'
json.dump(cfg, open('/home/zty/.cellcompass/config.json', 'w'), indent=2)
"
    
    # Sync codex CLI auth
    python3 -c "
import json
cc = json.load(open('/home/zty/.cellcompass/auth_profiles.json'))
prof = cc['profiles']['$profile']
codex_auth = json.load(open('/home/zty/.codex/auth.json'))
codex_auth['tokens']['access_token'] = prof['access']
codex_auth['tokens']['refresh_token'] = prof['refresh']
codex_auth['tokens']['account_id'] = prof.get('account_id', '')
json.dump(codex_auth, open('/home/zty/.codex/auth.json', 'w'), indent=2)
"
    
    rm -f ~/.cellcompass/auth_profiles.json.lock
    echo "[$(date)] Account switched to: $name"
}

check_quota() {
    # Quick test with codex CLI
    echo "say hi" | codex exec -m gpt-5.4 --sandbox read-only --skip-git-repo-check - 2>&1 | grep -q "usage limit"
    return $?
}

TOTAL=${#SCENARIOS[@]}
COUNT=0
FAILED=0

for scenario in "${SCENARIOS[@]}"; do
    COUNT=$((COUNT + 1))
    CONFIG="benchmark/dynbench/simulators/test_configs/${scenario}.json"
    OUTPUT="benchmark/results/batch_${scenario}_no_seed"
    LOG="benchmark/results/batch_${scenario}_no_seed.log"
    
    # Skip if already completed (check for summary)
    if [ -f "$OUTPUT/summary.md" ]; then
        completed=$(grep -c "passed:" "$LOG" 2>/dev/null || echo 0)
        if [ "$completed" -ge 20 ]; then  # 3 agents × 3 runs = 9, but some may fail
            echo "[$(date)] [$COUNT/$TOTAL] $scenario — already completed, skipping"
            continue
        fi
    fi
    
    echo "[$(date)] [$COUNT/$TOTAL] Starting: $scenario"
    START_TIME=$(date +%s)
    
    # Check quota before each run
    if check_quota; then
        echo "[$(date)] Quota OK"
    else
        echo "[$(date)] Quota exhausted, switching account..."
        current_account=$(( (current_account + 1) % ${#ACCOUNTS[@]} ))
        switch_account $current_account
    fi
    
    rm -f ~/.cellcompass/auth_profiles.json.lock
    
    python benchmark/batch_benchmark_runner.py \
        --topology-spec-file "$CONFIG" \
        --difficulties medium \
        --seeds 42 \
        --repeat 3 \
        --agents codex,cytobridge,biomini \
        --force-regenerate \
        --timeout-sec 1800 \
        --output-dir "$OUTPUT" \
        2>&1 | tee "$LOG"
    
    RC=$?
    ELAPSED=$(( $(date +%s) - START_TIME ))
    ELAPSED_MIN=$(( ELAPSED / 60 ))
    
    if [ $RC -ne 0 ]; then
        FAILED=$((FAILED + 1))
        echo "[$(date)] [$COUNT/$TOTAL] FAILED: $scenario — ${ELAPSED_MIN}min"
        
        # Check if it was a quota issue
        if grep -q "usage limit" "$LOG" 2>/dev/null; then
            echo "[$(date)] Quota exhausted during run, switching account..."
            current_account=$(( (current_account + 1) % ${#ACCOUNTS[@]} ))
            switch_account $current_account
            # Re-run this scenario
            echo "[$(date)] Retrying $scenario with new account..."
            rm -f ~/.cellcompass/auth_profiles.json.lock
            python benchmark/batch_benchmark_runner.py \
                --topology-spec-file "$CONFIG" \
                --difficulties medium \
                --seeds 42 \
                --repeat 3 \
                --agents codex,cytobridge,biomini \
                --force-regenerate \
                --timeout-sec 1800 \
                --output-dir "$OUTPUT" \
                2>&1 | tee -a "$LOG"
        fi
    else
        echo "[$(date)] [$COUNT/$TOTAL] OK: $scenario — ${ELAPSED_MIN}min"
    fi
done

echo ""
echo "[$(date)] ============================================"
echo "[$(date)] ALL DONE: $((TOTAL - FAILED))/$TOTAL succeeded, $FAILED failed"
echo "[$(date)] ============================================"
