"""
MoA v3.3 — Minimal RunPod Serverless Handler (test)
Verifies the endpoint works before running the full pipeline.
"""
import runpod
import json
import time

def handler(event):
    """Simple test handler that returns environment info."""
    inp = event.get("input", {})
    
    return {
        "status": "ok",
        "timestamp": time.time(),
        "input": inp,
        "message": "RunPod serverless handler is working!",
        "python_version": "3.12",
        "working_dir": "/workspace",
    }

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
