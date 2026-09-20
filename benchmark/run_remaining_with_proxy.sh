#!/bin/bash
set -e

# Proxy configuration
export HTTP_PROXY=http://172.26.176.1:10090
export HTTPS_PROXY=http://172.26.176.1:10090
export http_proxy=http://172.26.176.1:10090
export https_proxy=http://172.26.176.1:10090

cd /mnt/e/work/CytoBridge-agent-benchmark_p1

PYTHON="/home/os/miniconda3/envs/cytobridge/bin/python"
SCRIPT="benchmark/run_all_8_scenarios.py"
STATE="benchmark/results/three_agent_full_rerun_20260502_155815/state.json"

MAX_ROUNDS=10
for i in $(seq 1 $MAX_ROUNDS); do
    # Check if all scenarios are completed
    DONE=$($PYTHON -c "
import json
state = json.load(open('$STATE'))
remaining = sum(1 for s in state['scenarios'] if s['status'] not in ('completed','failed'))
print(remaining)
")
    echo "[$(date)] Remaining scenarios: $DONE"
    if [ "$DONE" = "0" ]; then
        echo "All scenarios completed!"
        break
    fi
    
    echo "[$(date)] Running round $i..."
    $PYTHON $SCRIPT 2>&1
    RC=$?
    echo "[$(date)] run_all_8_scenarios.py exited with code $RC"
    
    # Brief pause between rounds
    sleep 5
done

echo "[$(date)] Wrapper finished."
