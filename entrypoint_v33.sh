#!/usr/bin/env bash
# MOA v3.3 — Label Correction Handler
# Uses cascade trained on formula labels + LLM-validated correction on sample
# to produce clean labels for the full 100K dataset.
set -e
cd /workspace

echo "=== GateSwarm MoA Router v0.3.5 Label Correction ==="
echo "Starting: $(date -u)"

pip install --quiet scipy numpy scikit-learn datasets runpod requests 2>/dev/null || true

python3 -u handler_v33_label_correction.py

echo "Done: $(date -u)"
