#!/bin/bash
# Entrypoint for RunPod Serverless — MoA v3.1 Massive Per-Tier Test
exec /opt/venv/bin/python -c "
import runpod
import handler
runpod.serverless.start({'handler': handler.handler})
"
