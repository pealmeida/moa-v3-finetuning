"""
GateSwarm LLMFit — Dataset Factory CLI

Usage:
    python -m llmfit <command> [options]

Commands:
    generate   Extract prompts from workspace/chat logs
    label      Label prompts with complexity estimation
    validate   Validate dataset quality
"""

import sys


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    command = sys.argv[1]

    if command == "generate":
        print("Usage: python -m llmfit generate --source workspace --output datasets/raw.jsonl")
        print("See llmfit/llmfit.py for the full generate API.")
    elif command == "label":
        print("Usage: python -m llmfit label --input datasets/raw.jsonl --mode rule")
        print("See llmfit/llmfit.py for the full label API.")
    elif command == "validate":
        print("Usage: python -m llmfit validate --input datasets/labeled.jsonl")
        print("See llmfit/llmfit.py for the full validate API.")
    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
