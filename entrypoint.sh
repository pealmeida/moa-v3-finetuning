#!/bin/bash
# Entrypoint for RunPod Serverless — MoA v3.1
# Uses handler_v31.py (v3.1 with LLMFit support, GPD, multi-dataset)
exec /opt/venv/bin/python -c "
import runpod
import handler_v31
runpod.serverless.start({'handler': handler_v31.handler})
"
