"""
RunPod Serverless wrapper for MoA v3.3 Label Correction Handler.
This wraps the handler to work with RunPod's serverless job queue.
"""
import os, sys, subprocess, json

def ensure_deps():
    for pkg in ["scipy", "numpy", "scikit-learn", "datasets", "runpod", "requests"]:
        try: __import__(pkg.replace("-", "_"))
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", pkg])
ensure_deps()

import runpod

def handler(event):
    """RunPod serverless handler that executes handler.py with job input."""
    inp = event.get("input", {})
    
    # Pass input as JSON arg to handler.py
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    
    cmd = [
        sys.executable, "/workspace/handler.py",
        json.dumps(inp)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=1800)
    
    if result.returncode != 0:
        return {
            "status": "failed",
            "error": result.stderr[-2000:] if result.stderr else "Unknown error",
            "stdout": result.stdout[-2000:] if result.stdout else "",
        }
    
    # Parse result from JSON output
    try:
        # Find JSON in stdout
        import re
        json_match = re.search(r'\{[\s\S]*"version"[\s\S]*\}', result.stdout)
        if json_match:
            return json.loads(json_match.group())
    except:
        pass
    
    return {
        "status": "completed",
        "stdout": result.stdout[-5000:] if result.stdout else "",
    }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
