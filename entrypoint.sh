#!/bin/bash
exec /opt/venv/bin/python -c "
import runpod
import handler
runpod.serverless.start({'handler': handler.handler})
"
