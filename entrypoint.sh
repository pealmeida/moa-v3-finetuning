#!/usr/bin/env bash
# MOA v3 Finetuning entrypoint for RunPod Serverless
set -e

echo "=== GateSwarm MoA Router Entrypoint ==="
echo "Starting at: $(date -u)"

# Install dependencies if not present
pip install --quiet scipy numpy scikit-learn datasets runpod requests 2>/dev/null || true

echo "Dependencies installed"

# Determine which handler to run
# Accept handler from job input (via RUNPOD_HANDLER env var set by runpod SDK)
HANDLER="${RUNPOD_HANDLER:-${HANDLER:-handler_v33_label_correction.py}}"

echo "Running: $HANDLER"
python3 /workspace/$HANDLER

echo "Completed at: $(date -u)"
