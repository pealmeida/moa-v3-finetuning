# GateSwarm MoA Router

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)

**A self-optimizing complexity classifier for Mixture-of-Agents routing.**

Classifies prompt complexity into 6 tiers (trivial → extreme) using a pre-trained binary cascade, then recommends the cheapest model that can handle it. **74.7% accuracy** across all 6 tiers on 75K prompts. Zero GPU required.

---

## Quick Start

### Install

```bash
pip install numpy
```

That's it. No GPU, no heavy ML frameworks. The pre-trained weights ship with the repo.

### Use as a Library

```python
from router import score_prompt

# Score a single prompt
result = score_prompt("Write a REST API in Python")
# → {"tier": "heavy", "score": 0.42, "confidence": 0.82,
#    "model": "glm-5.1", "provider": "zai", "max_tokens": 2048}

# Score multiple prompts at once
from router import score_prompts
results = score_prompts([
    "hello",
    "Explain quantum computing",
    "Design a distributed event-driven microservice architecture",
])
```

### CLI

```bash
# Score a prompt
python router.py "Write a REST API in Python"

# Start HTTP API server
python router.py --serve --port 8080

# Batch score from file
python router.py --file prompts.jsonl --output scored.jsonl
```

### HTTP API

```bash
# Start the server
python router.py --serve

# Score a prompt
curl -X POST http://localhost:8080/score \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Write a REST API in Python"}'

# Response:
# {
#   "tier": "heavy",
#   "score": 0.42,
#   "confidence": 0.82,
#   "model": "glm-5.1",
#   "provider": "zai",
#   "max_tokens": 2048,
#   "method": "cascade",
#   "latency_ms": 12
# }
```

---

## How It Works

### 6-Tier Complexity Classification

| Tier | Score Range | Typical Prompt | Routed Model |
|------|-------------|----------------|--------------|
| **trivial** | 0.00 – 0.08 | Greetings, simple facts | glm-4.5-air (FREE) |
| **light** | 0.08 – 0.18 | Definitions, formatting | glm-4.7-flash ($0.02/M) |
| **moderate** | 0.18 – 0.32 | Code help, explanations | glm-4.7 ($0.10/M) |
| **heavy** | 0.32 – 0.52 | API design, refactoring | glm-5.1 ($0.13/M) |
| **intensive** | 0.52 – 0.72 | Architecture, deep analysis | qwen3.6-plus ($0.26/M) |
| **extreme** | 0.72 – 1.00 | Strategic thinking, research | qwen3.6-plus ($0.26/M) |

### 15-Feature Complexity Vector

Each prompt is scored on 15 features extracted locally (no LLM calls):

| Feature | Type | What it measures |
|---------|------|-----------------|
| `word_count` | int | Prompt length |
| `sentence_count` | int | Structural complexity |
| `avg_word_length` | float | Vocabulary sophistication |
| `has_code` | bool | Contains code blocks |
| `has_question` | bool | Is a question |
| `has_imperative` | bool | Starts with a command verb |
| `technical_terms` | int | Tech keyword density |
| `question_technical` | bool | Technical question |
| `architecture` | bool | System design keywords |
| `technical_design` | bool | Implementation plan keywords |
| `multi_step` | bool | Multiple steps implied |
| `requires_context` | bool | Needs external context |
| `domain_specificity` | float | Domain jargon density |
| `ambiguity_score` | float | Vague language density |
| `four_plus` | bool | ≥4 complexity signals active |

### Binary Cascade Classifier

Instead of one global model for 6 tiers, the cascade trains 5 independent binary classifiers:

```
trivial? ──Yes──→ TRIVIAL (100% acc)
    │ No
    ▼
light? ──Yes──→ LIGHT (93% acc)
    │ No
    ▼
moderate? ──Yes──→ MODERATE (42% acc)
    │ No
    ▼
heavy? ──Yes──→ HEAVY (39% acc)
    │ No
    ▼
intensive? ──Yes──→ INTENSIVE (19% acc)
    │ No
    ▼
EXTREME (69% acc)
```

Each classifier is trained on balanced 1:1 data and finds its own optimal feature weights. This avoids class imbalance and lets each tier specialize independently.

---

## Customization

### Override Model Assignments

The default model routing uses cost-optimized Chinese LLM providers. Override with your own:

```python
from router import set_tier_models, score_prompt

set_tier_models({
    "trivial":   {"model": "gpt-4o-mini",      "provider": "openai",    "max_tokens": 256},
    "light":     {"model": "gpt-4o-mini",      "provider": "openai",    "max_tokens": 512},
    "moderate":  {"model": "gpt-4o",           "provider": "openai",    "max_tokens": 1024},
    "heavy":     {"model": "claude-sonnet-4-6", "provider": "anthropic", "max_tokens": 2048},
    "intensive": {"model": "claude-sonnet-4-6", "provider": "anthropic", "max_tokens": 4096},
    "extreme":   {"model": "claude-opus-4-6",   "provider": "anthropic", "max_tokens": 8192},
})

result = score_prompt("Design a microservice architecture")
# → {"model": "claude-sonnet-4-6", "provider": "anthropic", ...}
```

### Retrain with Your Data

Use the training pipeline to optimize weights on your own prompts:

```bash
# Install training dependencies
pip install scipy numpy scikit-learn datasets

# Train on public datasets (Alpaca + OpenOrca)
python train.py

# Train with custom data
python train.py --dataset your_prompts.jsonl --output my_weights.json
```

Then load custom weights:

```python
from router import _cascade
_cascade.load("my_weights.json")
```

### Build a Personalized Dataset

LLMFit is the built-in dataset factory that creates training data from your own workspace:

```bash
# Extract prompts from your codebase / chat logs
python -m llmfit generate --source workspace --output datasets/raw.jsonl

# Label with rule-based complexity estimation
python -m llmfit label --input datasets/raw.jsonl --mode rule

# Validate dataset quality
python -m llmfit validate --input datasets/labeled.jsonl

# Anonymize PII/secrets before sharing
python -m llmfit.anonymizer --input datasets/raw.jsonl --output datasets/clean.jsonl
```

See `llmfit/` for the full dataset factory toolkit.

---

## Integrations

### Use with OpenClaw

Run as a sidecar service alongside your OpenClaw gateway:

```bash
# Terminal 1: Start the router API
python router.py --serve --port 8080

# In a custom skill, call the router before model selection:
# curl -s -X POST http://localhost:8080/score \
#   -H "Content-Type: application/json" \
#   -d '{"prompt": "'"$PROMPT"'"}' | jq -r '.model'
```

Or import directly in a skill:

```python
import sys; sys.path.insert(0, "/path/to/gateswarm-moa-router")
from router import score_prompt, set_tier_models

result = score_prompt(user_prompt)
# result["model"] → use this for model selection
```

### Use with Pi Agent

```python
# ~/.pi/agent/skills/router_skill.py
from gateswarm.router import score_prompt

def before_model_call(prompt: str, default_model: str) -> str:
    result = score_prompt(prompt)
    return result["model"] if result["confidence"] > 0.7 else default_model
```

### Use with Hermes Agent

```python
# Hermes skill for self-improvement model selection
from router import score_prompt

def select_model_for_task(task_description: str) -> str:
    result = score_prompt(task_description)
    return result["model"]
```

### Use with OpenCode / Codex

```bash
# Pre-route before calling your coding agent
TIER=$(python /path/to/router.py "$PROMPT" --json | jq -r .tier)

case $TIER in
  trivial|light)   MODEL="opencode/minimax-m2.5-free" ;;
  moderate|heavy)  MODEL="opencode/kimi-k2.5-free" ;;
  intensive)       MODEL="zai/glm-5.1" ;;
  extreme)         MODEL="claude-opus-4-6" ;;
esac

opencode --model "$MODEL" --prompt "$PROMPT"
```

### Use with LangChain / LiteLLM / any LLM Framework

```python
from router import score_prompt
import litellm

result = score_prompt(user_prompt)
response = litellm.completion(model=result["model"], messages=[...])
```

---

## Model Recommendations

### Cloud API Models

| Tier | Cost-Optimized | Balanced | Premium |
|------|----------------|----------|---------|
| **trivial** | `glm-4.5-air` (ZAI, FREE) | `gpt-4o-mini` (OpenAI) | `gemini-3-flash` (Google) |
| **light** | `glm-4.7-flash` (ZAI, $0.02/M) | `gpt-4o-mini` (OpenAI) | `gemini-3-flash` (Google) |
| **moderate** | `glm-4.7` (ZAI, $0.10/M) | `qwen3.5-9b` (OpenRouter) | `gpt-4o` (OpenAI) |
| **heavy** | `glm-5.1` (ZAI, $0.13/M) | `qwen3.6-plus` (Bailian) | `claude-sonnet-4.6` (Anthropic) |
| **intensive** | `qwen3.6-plus` (Bailian) | `claude-sonnet-4.6` (Anthropic) | `gpt-5.5` (OpenAI) |
| **extreme** | `qwen3.6-plus` (Bailian) | `claude-opus-4.6` (Anthropic) | `gpt-5.5` (OpenAI) |

### Local Models (Ollama / vLLM / llama.cpp)

| Tier | Fast (4-bit) | Balanced (8-bit) | Best (FP16) |
|------|-------------|------------------|------------|
| **trivial** | `qwen3-0.6b` | `phi-4-mini` | `qwen3-1.7b` |
| **light** | `qwen3-1.7b` | `gemma-3-4b` | `llama-4-scout-17b` |
| **moderate** | `qwen3-4b` | `llama-4-scout-17b` | `qwen3-8b` |
| **heavy** | `qwen3-8b` | `llama-4-maverick-17b` | `deepseek-r1-14b` |
| **intensive** | `deepseek-r1-14b` | `qwen3-14b` | `qwen3-32b` |
| **extreme** | `qwen3-32b` | `deepseek-r1-32b` | `qwen3-72b` |

> **Tip:** Override defaults with `set_tier_models()` to match your provider and budget. See [Customization](#customization).

### Docker

```bash
# Build (full: training + routing)
docker build -t gateswarm-moa-router .

# Build (inference only, minimal)
docker build -f Dockerfile.inference -t gateswarm-moa-router:infer .

# Run the API server
docker run -p 8080:8080 gateswarm-moa-router python router.py --serve --port 8080
```

---

## Project Structure

```
gateswarm-moa-router/
├── router.py                    # Production scorer (standalone, ~450 LOC)
├── train.py                     # Training pipeline (cascade + optimization)
├── v32_cascade_weights.json     # Pre-trained weights (5 classifiers)
├── requirements.txt             # Full deps (training)
├── Dockerfile                   # Full build
├── Dockerfile.inference         # Minimal inference build
│
├── llmfit/                      # Dataset Factory
│   ├── llmfit.py                # Core: extract → label → validate → optimize
│   ├── anonymizer.py            # 35-rule PII/secret redaction
│   ├── self_eval.py             # Self-evaluation + SQLite feedback buffer
│   ├── datasets/
│   │   ├── gpd_generator.py     # 50K synthetic prompt generator
│   │   ├── general-purpose/     # GPD dataset stats
│   │   └── workspace_weights.json
│
├── docs/
│   ├── V3_2_CASCADE_REPORT.md   # Cascade architecture & results
│   └── V3_3_MODEL_ROUTING_STRATEGY.md  # Model routing strategy
│
├── CHANGELOG.md
├── COMPARISON.md                 # Version evolution analysis
├── CONTRIBUTING.md
└── LICENSE                       # MIT
```

---

## Cost Savings

With proper tier routing, you send each prompt to the cheapest capable model:

| Scenario | 10K req/day | 100K req/day | 1M req/day |
|----------|-------------|--------------|------------|
| **Always Opus** | $6.36 | $63.59 | $635.94 |
| **Routed** | $0.24 | $2.35 | $23.54 |
| **Savings** | **96.3%** | **96.3%** | **96.3%** |

Training cost: **~$0.01** per run. No GPU needed — runs on any CPU in under a minute.

---

## Version History

| Version | Date | What changed |
|---------|------|-------------|
| **v0.1.0** | 2026-05-05 | Static hand-tuned weights, 53% accuracy |
| **v0.2.0** | 2026-05-06 | Manual tuning + bonuses, 67% accuracy |
| **v0.3.0** | 2026-05-07 | MSE optimization (scipy), **87.2%** accuracy |
| **v0.3.1** | 2026-05-08 | Multi-dataset + GPD 50K, **99.6%** on 2 tiers |
| **v0.3.2** | 2026-05-08 | Binary cascade, **74.7%** on all 6 tiers |
| **v0.3.5** | 2026-05-11 | Label correction, standalone router, LLMFit toolkit |

See [CHANGELOG.md](./CHANGELOG.md) for details and [COMPARISON.md](./COMPARISON.md) for the full evolution.

---

## License

MIT License — see [LICENSE](LICENSE).

## Citation

```bibtex
@misc{gateswarm-moa-router,
  title={GateSwarm MoA Router: Self-Optimizing Complexity Classification for Model Routing},
  author={Pedro Almeida},
  year={2026},
  url={https://github.com/pealmeida/gateswarm-moa-router}
}
```
