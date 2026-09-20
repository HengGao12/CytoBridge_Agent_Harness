#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
  if [[ ! -t 0 ]]; then
    echo "DEEPSEEK_API_KEY is not set and no interactive terminal is available." >&2
    exit 1
  fi
  read -r -s -p "DeepSeek API Key: " DEEPSEEK_API_KEY
  echo
fi

if [[ -z "$DEEPSEEK_API_KEY" ]]; then
  echo "DEEPSEEK_API_KEY cannot be empty." >&2
  exit 1
fi

export DEEPSEEK_API_KEY
export OPENAI_API_KEY="$DEEPSEEK_API_KEY"

python benchmark/dynbench/run_dynbench.py \
  --scenario S_balanced_easy_01_seed42 \
  --agent-type cytobridge \
  --mode skills-on \
  --device cpu \
  --seed 42 \
  --llm-provider deepseek \
  --llm-model deepseek-chat \
  --llm-auth-mode api_key \
  --harness-revisions 0 \
  --run-label deepseek_smoke \
  --run-root benchmark/results/deepseek_smoke
