#!/bin/bash
# Loop runner: keeps calling run_all_8_scenarios.py until all scenarios done
PYTHON="/home/os/miniconda3/envs/cytobridge/bin/python"
SCRIPT="/mnt/e/work/CytoBridge-agent-benchmark_p1/benchmark/run_all_8_scenarios.py"
RESULTS_ROOT="/mnt/e/work/CytoBridge-agent-benchmark_p1/benchmark/results/three_agent_full_rerun_20260502_155815"
LOG="$RESULTS_ROOT/loop_runner.log"

echo "[$(date -Iseconds)] Loop runner started" | tee -a "$LOG"

while true; do
    echo "[$(date -Iseconds)] Running scenario..." | tee -a "$LOG"
    $PYTHON "$SCRIPT" 2>&1 | tee -a "$LOG"
    EXIT=${PIPESTATUS[0]}
    
    # Check if all done
    REMAINING=$(python3 -c "
import json
s = json.load(open('$RESULTS_ROOT/state.json'))
pending = sum(1 for x in s['scenarios'] if x['status'] in ('pending','retry'))
print(pending)
")
    
    echo "[$(date -Iseconds)] Remaining: $REMAINING scenarios" | tee -a "$LOG"
    
    if [ "$REMAINING" -eq 0 ]; then
        echo "[$(date -Iseconds)] All scenarios completed!" | tee -a "$LOG"
        break
    fi
    
    sleep 5
done
