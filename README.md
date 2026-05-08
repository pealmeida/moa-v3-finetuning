# MoA v3 Finetuning — Self-Optimizing Gateway Router

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![RunPod](https://img.shields.io/badge/RunPod-Serverless-orange.svg)](https://runpod.io/)

**A data-driven weight optimization system for Mixture-of-Agents (MoA) Gateway Router complexity classification.**

Automatically tunes the heuristic router that classifies prompt complexity into 6 tiers (trivial → extreme), enabling cost-efficient model routing. Trained on 50K+ prompts, deployed on RunPod Serverless, with 87-99% tier-matching accuracy.

---

## Quick Results

| Version | Dataset | Baseline | Optimized | Improvement | Platform |
|---------|---------|----------|-----------|-------------|----------|
| v1 | 15 manual prompts | — | 53% | — | Local |
| v2 | 15 manual prompts | 53% | 67% | +14pp | Local |
| v2.1 | 30 manual prompts | 67% | 40% | -27pp (overfit) | Local |
| **v3.0** | **10K Alpaca** | **2.3%** | **87.2%** | **+84.9%** | **RunPod Serverless** |
| **v3.1** | **50K GPD** | **19.3%** | **99.6%** | **+80.2%** | **Local / RunPod** |

### Per-Tier Accuracy (v3.1)

| Tier | Accuracy | Use Case |
|------|----------|----------|
| **Trivial** | 100.0% | Greetings, status checks, simple facts |
| **Light** | 98.5% | Summaries, formatting, basic code fixes |
| Moderate | — | *Needs multi-dataset training* |
| Heavy | — | *Needs multi-dataset training* |
| Intensive | — | *Needs multi-dataset training* |
| Extreme | — | *Needs multi-dataset training* |

> **Note:** GPD dataset targets trivial/light prompts. For full-tier coverage, combine with Alpaca (moderate+) or use the merged pipeline.

---

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Dataset    │────▶│   Feature    │────▶│   Optimize   │────▶│   Deploy     │
│   Factory    │     │   Extract    │     │   (scipy)    │     │   Weights    │
│              │     │   (15 feat)  │     │   MSE loss   │     │   Gateway    │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
  • Alpaca HF           word_count,           L-BFGS-B              JSON weights
  • Self-Instruct       sentence_count,       bounds: [0, 0.35]     → MoA Router
  • GPD (50K synth)     has_code,             converges:            → /v3/weights
  • User context        architecture,           31-48 iterations
                        technical_design,
                        ambiguity_score, ...
```

### 15-Feature Vector (v3.1)

| Feature | Type | Description | Weight Range |
|---------|------|-------------|--------------|
| `sentence_count` | int | Number of sentences | 0.00–0.35 |
| `avg_word_length` | float | Average chars per word | 0.00–0.35 |
| `has_question` | bool | Contains `?` | 0.00–0.35 |
| `question_technical` | bool | Technical question | 0.00–0.35 |
| `technical_design` | bool | System design terms | 0.00–0.35 |
| `has_code` | bool | Code blocks/backticks | 0.00–0.35 |
| `architecture` | bool | Architecture keywords | 0.00–0.35 |
| `word_count` | int | Total words | 0.00–0.35 |
| `four_plus` | bool | ≥4 signals active | 0.00–0.35 |
| `has_imperative` | bool | Starts with command verb | 0.00–0.35 |
| `technical_terms` | int | Tech keyword count | 0.00–0.35 |
| `multi_step` | bool | Multiple steps implied | 0.00–0.35 |
| `requires_context` | bool | Needs external context | 0.00–0.35 |
| `domain_specificity` | float | Domain jargon density | 0.00–0.35 |
| `ambiguity_score` | float | Vague language density | 0.00–0.35 |

### Tier Boundaries

| Tier | Score Range | Model Example |
|------|-------------|---------------|
| trivial | 0.00–0.08 | glm-4.5-air (free) |
| light | 0.08–0.18 | glm-4.7-flash ($0.06/M) |
| moderate | 0.18–0.32 | glm-4.7 |
| heavy | 0.32–0.52 | glm-5.1 |
| intensive | 0.52–0.72 | qwen3.6-plus |
| extreme | 0.72–1.00 | qwen3.6-plus / claude-opus |

---

## Project Structure

```
moa-v3-finetuning/
├── README.md                  # This file
├── CHANGELOG.md               # Version history
├── LICENSE                    # MIT License
├── CONTRIBUTING.md            # Contribution guidelines
├── .gitignore
├── .dockerignore
│
├── handler.py                 # v3.0 handler (Alpaca baseline, RunPod compatible)
├── handler_v31.py             # v3.1 handler (GPD + multi-dataset, LLMFit-ready)
├── Dockerfile                 # Standard build
├── Dockerfile.serverless      # RunPod Serverless container
├── Dockerfile.train           # Training container (SSH debug)
├── entrypoint.sh              # Serverless entrypoint
├── requirements.txt           # Python dependencies
│
├── llmfit/                    # LLMFit — Dataset Factory (v3.1)
│   ├── llmfit.py              # Core engine: extract → label → validate → optimize
│   ├── self_eval.py           # Self-evaluation + feedback buffer (SQLite)
│   ├── anonymizer.py          # 35-rule PII/secret/context anonymization
│   ├── handler_v31.py         # v3.1 training handler (copy for RunPod)
│   └── datasets/
│       ├── gpd_generator.py   # Generate 50K synthetic trivial/light prompts
│       ├── general-purpose/
│       │   └── stats.json     # GPD dataset statistics
│       ├── workspace_weights.json  # Workspace-optimized weights
│       └── v31_runpod_alpaca_result.json  # RunPod training results
│
└── docs/
    ├── ARCHITECTURE_V3_1.md   # Full v3.1 architecture spec (38KB)
    └── TRAINING_REPORT_V3_1.md # Training results & anonymization report
```

---

## Usage

### Local Training (CPU, no GPU needed)

```bash
# Install dependencies
pip install scipy numpy datasets runpod requests

# v3.0 — Alpaca baseline (10K prompts)
python handler.py

# v3.1 — GPD dataset (50K synthetic prompts)
python handler_v31.py
```

### RunPod Serverless Deployment

```bash
# Build and push Docker image
docker build -t ghcr.io/pealmeida/moa-v3-finetuning:latest \
  -f Dockerfile.serverless .
docker push ghcr.io/pealmeida/moa-v3-finetuning:latest

# Submit training job (via API)
curl -X POST "https://api.runpod.ai/v2/<endpoint_id>/run" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input":{"datasets":["alpaca"],"max_per":10000}}'
```

### LLMFit — Personalized Dataset Factory

```bash
# Extract workspace dataset
python llmfit/llmfit.py generate \
  --path /your/workspace \
  --output datasets/raw.jsonl

# Label with rule-based complexity estimation
python llmfit/llmfit.py label \
  --input datasets/raw.jsonl \
  --mode rule

# Validate dataset quality
python llmfit/llmfit.py validate \
  --input datasets/labeled.jsonl

# Optimize weights
python llmfit/llmfit.py optimize \
  --input datasets/labeled.jsonl \
  --output weights.json

# Anonymize before sharing
python llmfit/anonymizer.py anonymize \
  --input datasets/labeled.jsonl \
  --output datasets/anonymized.jsonl

# Generate 50K synthetic GPD
python llmfit/datasets/gpd_generator.py \
  --trivial 35000 --light 15000
```

### Python API

```python
from handler_v31 import handler

result = handler({
    "input": {
        "version": "v3.1",
        "datasets": ["gpd", "alpaca"],  # or ["gpd"], ["alpaca"]
        "max_per": 20000,               # max samples per dataset
        "max_iter": 2000,               # scipy optimization iterations
    }
})

print(f"Accuracy: {result['optimized_accuracy']:.1%}")
print(f"Weights: {result['optimized_weights']}")
```

---

## Weight Evolution

### v3.1 Optimized Weights (GPD 50K)

| Feature | v3.0 (Alpaca) | v3.1 (GPD) | Change |
|---------|---------------|------------|--------|
| ambiguity_score | — | **0.6041** | NEW |
| sentence_count | **0.2915** | 0.0000 | ↓↓↓ |
| technical_design | 0.1196 | **0.2071** | ↑ |
| avg_word_length | **0.1890** | 0.0000 | ↓↓↓ |
| architecture | 0.0698 | **0.1208** | ↑ |
| question | **0.1199** | 0.0000 | ↓↓↓ |
| imperative | **0.1141** | **0.0295** | ↓ |
| code | **0.1044** | **0.0267** | ↓ |
| math | **0.1139** | — | REMOVED |
| context | **0.1113** | — | REMOVED |
| constraints | **0.0897** | — | REMOVED |

**Why the difference?** GPD focuses on trivial/light prompts where `ambiguity_score` (vague language) is the primary differentiator. Alpaca is moderate+ where sentence structure and vocabulary dominate. **For best results, merge both datasets.**

---

## Anonymization

All user datasets are anonymized before training or sharing. 35 redaction rules across 4 categories:

| Category | Rules | Examples |
|----------|-------|----------|
| **PII** | 11 | emails → `[EMAIL_REDACTED]`, phones → `[PHONE_REDACTED]` |
| **Secrets** | 10 | API keys → `[API_KEY_REDACTED]`, JWTs → `[JWT_REDACTED]` |
| **Context** | 8 | names → `[PERSON]`, internal hosts → `[INTERNAL_HOST]` |
| **Generalized** | 6 | numbers → `[NUMBER_N]`, URLs → `[URL]` |

See `llmfit/anonymizer.py` for the full rule set.

---

## Performance

| Metric | v3.0 (RunPod) | v3.1 (Local) |
|--------|---------------|--------------|
| Training time | 35s (RTX 4090) | 2s (CPU) |
| Cold start | 428s | N/A |
| Cost per run | ~$0.007 | $0.00 |
| Convergence | 31 iterations | 48 iterations |
| MSE reduction | 98.5% | 99.8% |

### Why CPU is sufficient

The optimization is a **15-parameter L-BFGS-B MSE minimization** over 8K-50K training examples. This is pure numerical optimization — no GPU, no neural network, no embeddings. It runs in seconds on any modern CPU.

GPU is only needed if you extend to:
- Neural network complexity classifiers
- LoRA fine-tuning of LLMs
- Embedding-based prompt representations

---

## Datasets

| Dataset | Source | Samples | Tiers | License |
|---------|--------|---------|-------|---------|
| **Alpaca** | `tatsu-lab/alpaca` (HF) | 52K | moderate+ | Apache 2.0 |
| **Self-Instruct** | `yizhongw/self_instruct` (HF) | 82K | moderate+ | Apache 2.0 |
| **GPD (synthetic)** | Built-in generator | 50K | trivial/light | MIT |
| **User workspace** | LLMFit scanner | Variable | all | Your data |

### Generating the GPD

```bash
python llmfit/datasets/gpd_generator.py \
  --trivial 35000 --light 15000 \
  --output llmfit/datasets \
  --seed 42
```

Generates 50,000 synthetic prompts from 70+ templates covering:
- Trivial: greetings, definitions, conversions, simple facts
- Light: code formatting, summaries, error explanations, file operations

---

## Cost Analysis

### Monthly Projection (after optimization)

| Scenario | 10K req/day | 100K req/day | 1M req/day |
|----------|-------------|--------------|------------|
| **Baseline (always Opus)** | $6.36 | $63.59 | $635.94 |
| **v3.0 routed** | $0.24 | $2.35 | $23.54 |
| **Savings** | **96.3%** | **96.3%** | **96.3%** |

Training cost per run: **$0.007** (35s on RTX 4090 via RunPod Serverless)

---

## Reproduction

To reproduce the v3.0 RunPod Serverless results:

1. Deploy the Docker image to a RunPod Serverless endpoint
2. Submit a job with `{"input": {"datasets": ["alpaca"], "max_per": 10000}}`
3. Results returned in ~460s (cold start + 35s execution)

To reproduce v3.1 locally:

```bash
pip install scipy numpy datasets
python handler_v31.py  # or: python -c "from handler_v31 import handler; print(handler({'input': {'datasets': ['gpd'], 'max_per': 50000}}))"
```

Expected output: 99.6% accuracy on GPD 50K.

---

## Related Projects

- **[MoA Gateway Router](https://github.com/pealmeida/)** — The gateway that uses these weights
- **[Agentic Sovereign Ecosystem](https://github.com/pealmeida/)** — The broader architecture
- **[LLMFit](./llmfit/)** — The dataset factory for personalized training

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Citation

If you use this in your research:

```bibtex
@misc{moa-v3-finetuning,
  title={MoA v3 Finetuning: Self-Optimizing Gateway Router for Complexity Classification},
  author={Pedro Almeida},
  year={2026},
  url={https://github.com/pealmeida/moa-v3-finetuning}
}
```
