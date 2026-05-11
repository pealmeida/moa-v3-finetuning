# GateSwarm MoA Router — Self-Optimizing Gateway Router

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/)
[![RunPod](https://img.shields.io/badge/RunPod-Serverless-orange.svg)](https://runpod.io/)
[![scikit-learn](https://img.shields.io/badge/ml-scikit--learn-yellow.svg)](https://scikit-learn.org/)

**A data-driven weight optimization system for Mixture-of-Agents (MoA) Gateway Router complexity classification.**

Automatically tunes the heuristic router that classifies prompt complexity into 6 tiers (trivial → extreme), enabling cost-efficient model routing. Trained on **75K prompts** across 3 datasets, deployed on RunPod Serverless, with **74.7% tier-matching accuracy** across all 6 tiers using a tier-pair binary cascade.

---

## Quick Results

| Version | Method | Dataset | Baseline | Optimized | Improvement | Platform |
|---------|--------|---------|----------|-----------|-------------|----------|
| v0.1.0 | Static weights | 15 prompts | — | 53% | — | Local |
| v0.2.0 | Manual tuning | 15 prompts | 53% | 67% | +14pp | Local |
| v0.2.1 | Manual tuning | 30 prompts | 67% | 40% | -27pp (overfit) | Local |
| **v0.3.0** | MSE optimization | 10K Alpaca | 2.3% | **87.2%** | **+84.9%** | RunPod |
| **v0.3.1** | MSE + balanced | 50K GPD | 19.3% | **99.6%** | **+80.2%** | Local/RunPod |
| **v0.3.2** | **Binary cascade** | **75K (3 datasets)** | **24.3%** | **74.7%** | **+50.4pp** | **RunPod** |
| **v0.3.5** | **Label correction + LLM judge** | **75K+** | — | **Production** | — | **RunPod** |

### Per-Tier Accuracy (v0.3.2 — All 6 Tiers)

| Tier | Test Samples | Baseline (v0.3.0) | Cascade (v0.3.2) | Δ |
|------|-------------|-----------------|----------------|---|
| **Trivial** | 5,000 | 23.0% | **100.0%** | +77.0pp |
| **Light** | 3,373 | 56.5% | **93.1%** | +36.6pp |
| **Moderate** | 2,007 | 17.0% | **42.4%** | +25.4pp |
| **Heavy** | 1,823 | 13.0% | **38.7%** | +25.7pp |
| **Intensive** | 855 | 0.6% | **19.3%** | +18.7pp |
| **Extreme** | 1,944 | 0.05% | **68.8%** | +68.7pp |
| **Overall** | **15,002** | **24.3%** | **74.7%** | **+50.4pp** |

> **Every single tier improved.** The cascade achieves 93%+ accuracy on trivial and light — the highest-volume tiers that drive most cost savings.

---

## Architecture

### v0.3.0/v0.3.1 — Global Optimization

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
                        ambiguity_score, ...
```

### v0.3.2 — Tier-Pair Binary Cascade

```
Input Prompt
    │
    ▼
┌─────────────────────┐  Yes → TRIVIAL (99.6% acc)
│ Classifier 1:       │
│ Trivial?            │  No
│ (99.2% train acc)   │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐  Yes → LIGHT (93.1% acc)
│ Classifier 2:       │
│ Light?              │  No
│ (85.8% train acc)   │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐  Yes → MODERATE (42.4% acc)
│ Classifier 3:       │
│ Moderate?           │  No
│ (83.9% train acc)   │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐  Yes → HEAVY (38.7% acc)
│ Classifier 4:       │
│ Heavy?              │  No
│ (80.5% train acc)   │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐  Yes → INTENSIVE (19.3% acc)
│ Classifier 5:       │
│ Intensive?          │  No
│ (76.3% train acc)   │
└────────┬────────────┘
         │
         ▼
    EXTREME (68.8% acc)
```

### Why the Cascade Works Better

1. **Balanced training** — Each classifier trains on a 1:1 ratio of positive/negative samples, avoiding class imbalance
2. **Independent feature optimization** — Each classifier finds the features that best separate its tier
3. **Logistic Regression** — sklearn's LogisticRegression handles non-linear boundaries better than linear regression
4. **Cascade ordering** — Easiest tiers first (trivial), hardest last (intensive), with extreme as default

### 15-Feature Vector

| Feature | Type | Description |
|---------|------|-------------|
| `sentence_count` | int | Number of sentences |
| `avg_word_length` | float | Average chars per word |
| `has_question` | bool | Contains `?` |
| `question_technical` | bool | Technical question |
| `technical_design` | bool | System design terms |
| `has_code` | bool | Code blocks/backticks |
| `architecture` | bool | Architecture keywords |
| `word_count` | int | Total words |
| `four_plus` | bool | ≥4 signals active |
| `has_imperative` | bool | Starts with command verb |
| `technical_terms` | int | Tech keyword count |
| `multi_step` | bool | Multiple steps implied |
| `requires_context` | bool | Needs external context |
| `domain_specificity` | float | Domain jargon density |
| `ambiguity_score` | float | Vague language density |

### Tier Boundaries

| Tier | Score Range | Model Example | Cost |
|------|-------------|---------------|------|
| trivial | 0.00–0.08 | glm-4.5-air | FREE |
| light | 0.08–0.18 | glm-4.7-flash | $0.06/M |
| moderate | 0.18–0.32 | glm-4.7 | $0.10/M |
| heavy | 0.32–0.52 | glm-5.1 | $0.13/M |
| intensive | 0.52–0.72 | qwen3.6-plus | $0.26/M |
| extreme | 0.72–1.00 | qwen3.6-plus / claude-opus | $5.00/M |

---

## Project Structure

```
gateswarm-moa-router/
├── README.md                     # This file
├── CHANGELOG.md                  # Version history (v0.1 → v0.3.5)
├── COMPARISON.md                 # Detailed v0.1→v0.2→v0.3 evolution
├── CONTRIBUTING.md               # Contribution guidelines
├── LICENSE                       # MIT License
├── .gitignore
├── .dockerignore
│
├── handler.py                    # Active handler (v0.3.5 label correction)
├── handler_v31.py                # v0.2 handler (GPD + multi-dataset)
├── handler_v31_massive.py        # v0.2 massive per-tier test
├── handler_v32_cascade.py        # v0.3 cascade handler
├── Dockerfile                    # Standard build
├── Dockerfile.serverless         # RunPod Serverless container
├── Dockerfile.train              # Training container (SSH debug)
├── entrypoint.sh                 # Serverless entrypoint
├── requirements.txt              # Python dependencies
│
├── llmfit/                       # LLMFit — Dataset Factory
│   ├── llmfit.py                 # Core engine: extract → label → validate → optimize
│   ├── self_eval.py              # Self-evaluation + feedback buffer (SQLite)
│   ├── anonymizer.py             # 35-rule PII/secret/context anonymization
│   ├── handler_v31.py            # v0.3.1 training handler (copy)
│   ├── handler_v31_massive.py    # v0.3.1 massive test handler (copy)
│   └── datasets/
│       ├── gpd_generator.py      # Generate 50K synthetic trivial/light prompts
│       ├── general-purpose/
│       │   └── stats.json        # GPD dataset statistics
│       ├── v31_runpod_alpaca_result.json
│       ├── v31_gpd_runpod_result.json
│       ├── v31_massive_runpod_result.json
│       ├── v32_cascade_runpod_result.json
│       └── workspace_weights.json
│
└── docs/
    ├── ARCHITECTURE_V3_1.md       # Full v0.3.1 architecture spec (38KB)
    ├── TRAINING_REPORT_V3_1.md    # v0.3.1 training results & anonymization
    ├── PER_TIER_TEST_REPORT.md    # v0.3.1 massive per-tier test analysis
    └── V3_2_CASCADE_REPORT.md     # v0.3.2 cascade final report
```

---

## Usage

### Local Training (CPU, no GPU needed)

```bash
# Install dependencies
pip install scipy numpy scikit-learn datasets runpod requests

# v0.3.0 — Alpaca baseline (10K prompts)
python handler_v31.py

# v0.3.1 — GPD dataset (50K synthetic)
python handler_v31.py

# v0.3.1 massive — 3 datasets (75K)
python handler_v31_massive.py

# v0.3.2 cascade — binary classifiers (recommended)
python handler_v32_cascade.py
```

### RunPod Serverless Deployment

```bash
# Build and push Docker image
docker build -t ghcr.io/pealmeida/gateswarm-moa-router:latest \
  -f Dockerfile.serverless .
docker push ghcr.io/pealmeida/gateswarm-moa-router:latest

# Submit training job
curl -X POST "https://api.runpod.ai/v2/<endpoint_id>/run" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input":{"datasets":["gpd","alpaca","openorca"],"gpd_trivial":25000,"gpd_light":10000,"max_per":20000}}'
```

### LLMFit — Personalized Dataset Factory

```bash
# Extract workspace dataset
python llmfit/llmfit.py generate \
  --path /your/workspace \
  --output datasets/raw.jsonl

# Label with rule-based complexity estimation
python llmfit/llmfit.py label \
  --input datasets/raw.jsonl --mode rule

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
from handler_v32_cascade import handler

result = handler({
    "input": {
        "datasets": ["gpd", "alpaca", "openorca"],
        "gpd_trivial": 25000,
        "gpd_light": 10000,
        "max_per": 20000,
    }
})

print(f"Cascade accuracy: {result['cascade']['accuracy']:.1%}")
print(f"Macro accuracy: {result['cascade']['macro_accuracy']:.1%}")

for tier, data in result['cascade']['tier_accuracy'].items():
    print(f"  {tier}: {data['accuracy']:.1%} ({data['correct']}/{data['total']})")
```

---

## v0.3.2 Cascade — Classifier Details

| Classifier | Train Accuracy | Samples | Top Feature | 2nd Feature | 3rd Feature |
|------------|---------------|---------|-------------|-------------|-------------|
| **Trivial** | 99.2% | 40,000 | ambiguity_score (-18.74) | domain_specificity (+8.24) | has_question (+5.29) |
| **Light** | 85.8% | 26,976 | ambiguity_score (+7.39) | domain_specificity (-7.11) | has_question (-5.84) |
| **Moderate** | 83.9% | 16,054 | domain_specificity (+1.61) | has_question (-1.33) | requires_context (-0.97) |
| **Heavy** | 80.5% | 14,578 | ambiguity_score (+1.01) | architecture (-0.81) | multi_step (-0.81) |
| **Intensive** | 76.3% | 6,834 | ambiguity_score (-1.24) | has_question (-1.01) | requires_context (-0.75) |

### Key Insights

- **`ambiguity_score`** dominates all classifiers — vague/imprecise language is the strongest tier signal
- **`domain_specificity`** separates technical prompts from general ones
- **`has_question`** is positive for trivial (simple questions) but negative for action-oriented prompts
- **`architecture`** is a strong negative for heavy — heavy prompts are technical but not system design

---

## Anonymization

All user datasets are anonymized before training or sharing. **35 redaction rules** across 4 categories:

| Category | Rules | Examples |
|----------|-------|----------|
| **PII** | 11 | emails → `[EMAIL_REDACTED]`, phones → `[PHONE_REDACTED]` |
| **Secrets** | 10 | API keys → `[API_KEY_REDACTED]`, JWTs → `[JWT_REDACTED]` |
| **Context** | 8 | names → `[PERSON]`, internal hosts → `[INTERNAL_HOST]` |
| **Generalized** | 6 | numbers → `[NUMBER_N]`, URLs → `[URL]` |

See `llmfit/anonymizer.py` for the full rule set.

---

## Performance

| Metric | v0.3.0 | v0.3.1 | v0.3.2 |
|--------|------|------|------|
| **Dataset** | 10K Alpaca | 50K GPD | 75K (3 datasets) |
| **Training time** | 35s (RTX 4090) | 2s (CPU) | 58s (RTX 4090) |
| **Cost per run** | ~$0.007 | $0.00 | ~$0.01 |
| **Convergence** | 31 iterations | 48 iterations | 5 classifiers |
| **Overall accuracy** | 87.2% | 99.6%* | 74.7%** |

* v0.3.1 targets trivial/light only (2 tiers)  
** v0.3.2 covers all 6 tiers

### Why CPU is sufficient for v0.3.0/v0.3.1

The optimization is a **15-parameter L-BFGS-B MSE minimization** over 8K-50K training examples. Pure numerical optimization — no GPU needed.

GPU is only needed for v0.3.2 (scikit-learn LogisticRegression on larger balanced datasets).

---

## Datasets

| Dataset | Source | Samples | Tier Coverage | License |
|---------|--------|---------|---------------|---------|
| **Alpaca** | `tatsu-lab/alpaca` (HF) | 52K | light → extreme | Apache 2.0 |
| **OpenOrca** | `Open-Orca/OpenOrca` (HF) | 4M | light → extreme | Apache 2.0 |
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

Generates 50,000 synthetic prompts from 70+ templates:
- **Trivial:** greetings, definitions, conversions, simple facts
- **Light:** code formatting, summaries, error explanations, file operations

---

## Cost Analysis

### Monthly Projection (after optimization)

| Scenario | 10K req/day | 100K req/day | 1M req/day |
|----------|-------------|--------------|------------|
| **Baseline (always Opus)** | $6.36 | $63.59 | $635.94 |
| **v0.3.0 routed** | $0.24 | $2.35 | $23.54 |
| **Savings** | **96.3%** | **96.3%** | **96.3%** |

Training cost per run: **~$0.01** (58s on RTX 4090 via RunPod Serverless)

---

## Reproduction

### v0.3.0 — Alpaca baseline

```bash
# Deploy to RunPod, submit job:
{"input": {"datasets": ["alpaca"], "max_per": 10000}}
# Expected: 87.2% accuracy in ~460s (cold start + 35s execution)
```

### v0.3.1 — GPD (local)

```bash
pip install scipy numpy datasets
python handler_v31.py
# Expected: 99.6% accuracy on GPD 50K (trivial 100%, light 98.5%)
```

### v0.3.2 — Cascade (RunPod recommended)

```bash
pip install scipy numpy scikit-learn datasets runpod requests
python handler_v32_cascade.py
# Expected: 74.7% accuracy on 75K samples, all 6 tiers
```

---

## Version History

See [CHANGELOG.md](./CHANGELOG.md) for the full version history and [COMPARISON.md](./COMPARISON.md) for detailed v0.1→v0.2→v0.3 evolution.

### v0.3.6 — 6-Model Cost-Efficient Routing (Planned)

The next step is **specialized model assignment per tier** — using the cheapest model that can reliably handle each tier's requirements:

| Tier | Model | Cost | Why |
|------|-------|------|-----|
| **Trivial** | glm-4.5-air | FREE | Basic language understanding |
| **Light** | glm-4.7-flash | $0.02/M | Instruction following |
| **Moderate** | qwen3.5-9b | $0.10/M | Code + reasoning |
| **Heavy** | qwen3.6-plus (bailian) | $0.04/M | Deep analysis |
| **Intensive** | claude-sonnet-4.6 | $3.00/M | Architecture, reasoning |
| **Extreme** | claude-opus-4.6 | $5.00/M | Strategic thinking |

**Expected savings:** 80-84% vs always-Opus. See `docs/V3_3_MODEL_ROUTING_STRATEGY.md` for details.

### Chief Scientist Evaluation

A complete evaluation of the training pipeline is available at `docs/CHIEF_SCIENTIST_EVALUATION.md`. Key findings:
- **Engineering quality:** 8.5/10 — clean, reproducible pipelines
- **Label validity:** 2.0/10 — synthetic labels, no ground truth
- **Production readiness:** 6.0/10 — ready for trivial/light only (55% of traffic)
- **Recommended action:** Deploy partially now, validate labels before full rollout

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
@misc{gateswarm-moa-router,
  title={GateSwarm MoA Router: Self-Optimizing Gateway Router for Complexity Classification},
  author={Pedro Almeida},
  year={2026},
  url={https://github.com/pealmeida/gateswarm-moa-router}
}
```
