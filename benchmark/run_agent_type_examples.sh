#!/usr/bin/env bash
set -euo pipefail

# Example agent-type switches for the same DynBench sanity task.
# Run from the repository root:
#   bash benchmark/run_agent_type_examples.sh codex

AGENT_TYPE="${1:-codex}"
SCENARIO="${SCENARIO:-S3_no_holdout_sanity}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-cpu}"

case "$AGENT_TYPE" in
  cytobridge|cellcompass)
    # Uses the modern CellCompass/CytoBridge runtime through the benchmark adapter.
    # The adapter shares the same session-open path as `cellcompass exec` while
    # preserving benchmark sandboxing and result collection.
    python -m benchmark.dynbench.run_dynbench \
      --scenario "$SCENARIO" \
      --agent-type cytobridge \
      --mode skills-on \
      --device "$DEVICE" \
      --seed "$SEED"
    ;;
  codex)
    # Uses Codex OAuth profiles from ~/.cellcompass/auth_profiles.json.
    python -m benchmark.dynbench.run_dynbench \
      --scenario "$SCENARIO" \
      --agent-type codex \
      --mode skills-on \
      --device "$DEVICE" \
      --seed "$SEED"
    ;;
  gemini)
    # Uses the same Codex OAuth login as codex; run `cellcompass auth codex-login` first.
    python -m benchmark.dynbench.run_dynbench \
      --scenario "$SCENARIO" \
      --agent-type gemini \
      --mode skills-on \
      --device "$DEVICE" \
      --seed "$SEED"
    ;;
  biomini|biomni)
    # Uses the same Codex OAuth login as codex; run `cellcompass auth codex-login` first.
    python -m benchmark.dynbench.run_dynbench \
      --scenario "$SCENARIO" \
      --agent-type biomini \
      --biomni-path "${BIOMNI_REPO:-$(dirname "$0")/../Biomni}" \
      --mode skills-on \
      --device "$DEVICE" \
      --seed "$SEED"
    ;;
  *)
    echo "Usage: $0 {cytobridge|cellcompass|codex|gemini|biomini}" >&2
    exit 2
    ;;
esac
