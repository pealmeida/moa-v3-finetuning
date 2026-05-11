# Changelog

All notable changes to the GateSwarm MoA Router project.

## [0.3.5] — 2026-05-11

### Changed
- **Rebranded** from `moa-v3-finetuning` to `gateswarm-moa-router`
- Version scheme updated: v1.x → v0.1.x, v2.x → v0.2.x, v3.x → v0.3.x
- All Docker image references → `ghcr.io/pealmeida/gateswarm-moa-router`
- Handler version strings updated to v0.3.5

---

## [0.3.4] — 2026-05-10

### Added
- **v3.3 LLM-as-Judge labeling** (`handler_v33_llm_labeling.py`) — Full empirical labeling pipeline with LLM validation
- **Label validation handler** (`handler_v33_label_validation.py`) — Cross-validation between formula and LLM labels
- **Weight export handler** (`handler_v33_weight_export.py`) — Label correction + production weight export
- **Inference handler** (`handler_v33_inference.py`) — Production-ready complexity scoring for RunPod serverless
- **Parallel LLM judge** (`llm_judge_parallel.py`) — Concurrent LLM judging for large datasets
- **Dockerfile.v33**, **Dockerfile.inference**, **Dockerfile.weight-export** — Specialized containers
- **entrypoint_v33.sh** — v3.3 label correction entrypoint
- **Documentation**
  - `docs/V3_3_EMPIRICAL_LABELING_DESIGN.md` — Empirical labeling pipeline design
  - `docs/V3_3_MODEL_ROUTING_STRATEGY.md` — 6-model cost-efficient routing strategy
  - `docs/CHIEF_SCIENTIST_EVALUATION.md` — Independent pipeline evaluation
  - `docs/LABEL_VALIDITY_FIX.md` — Label validity correction approach

### Changed
- `handler.py` → v3.3 label correction handler (active)
- `runpod_handler.py` → wraps v3.3 handler for RunPod SDK
- `Dockerfile.serverless` → updated for v3.3

---

## [0.3.3] — 2026-05-09

### Added
- **Label correction** — Cascade trained on formula labels with LLM-validated correction
- **Production inference** — Standalone complexity scoring endpoint

---

## [0.3.2] — 2026-05-08

### Added
- **Tier-pair binary cascade** (`handler_v32_cascade.py`) — 5 independent binary classifiers for full 6-tier classification
- **scikit-learn integration** — LogisticRegression with LBFGS solver
- **Balanced training** — 1:1 positive/negative sampling per classifier
- **RunPod test results** — 75K samples across 3 datasets, 15K test set, **74.7% overall accuracy**
- **Documentation**
  - `docs/V3_2_CASCADE_REPORT.md` — Full cascade analysis

### Results
| Dataset | Method | Baseline | Optimized | Improvement |
|---------|--------|----------|-----------|-------------|
| 75K (3 datasets) | Cascade | 24.3% | **74.7%** | +50.4pp |

**Per-tier:** trivial 100.0%, light 93.1%, moderate 42.4%, heavy 38.7%, intensive 19.3%, extreme 68.8%

### Changed
- `handler.py` → uses cascade handler
- `Dockerfile.serverless` → updated with sklearn
- `entrypoint.sh` → updated for cascade handler

---

## [0.3.1] — 2026-05-08

### Added
- **Multi-dataset training** — GPD (50K synthetic), Alpaca, workspace data
- **LLMFit Dataset Factory** (`llmfit/`) — Complete pipeline for personalized dataset creation
  - `llmfit.py` — Core engine: workspace scanning, feature extraction, labeling, validation
  - `anonymizer.py` — 35-rule anonymization engine
  - `self_eval.py` — Self-evaluation with SQLite feedback buffer
  - `gpd_generator.py` — 50K synthetic prompt generator
- **General-Purpose Dataset (GPD)** — 50,000 synthetic prompts
- **Documentation**
  - `docs/ARCHITECTURE_V3_1.md` — Full architecture spec
  - `docs/TRAINING_REPORT_V3_1.md` — Training results & anonymization

### Results
| Dataset | Baseline | Optimized | Improvement |
|---------|----------|-----------|-------------|
| GPD 50K | 19.3% | 99.6% | +80.2% |
| Alpaca 10K | 2.3% | 87.2% | +84.9% |

---

## [0.3.0] — 2026-05-07

### Added
- **Full training pipeline** with scipy MSE optimization
- **RunPod Serverless deployment** — Docker + entrypoint for serverless training
- **13-feature complexity vector** — word_count, question, code, imperative, math, multi_step, constraints, context, sentence_count, avg_word_length, architecture, technical_design, four_plus
- **Synthetic complexity labels** — Signal count + length + lexical richness → 6 tiers
- **COMPARISON.md** — Version evolution analysis

### Results
| Dataset | Baseline | Optimized | Improvement |
|---------|----------|-----------|-------------|
| Alpaca 10K | 2.3% | 87.2% | +84.9% |

---

## [0.2.1] — 2026-05-06

### Added
- `architecture` bonus (0.28) for architecture-heavy prompts
- `technical_design` bonus (0.18) for system design terms
- Extended architecture regex keywords
- Length dampener skip for architecture-heavy prompts

### Changed
- Result: 40% on 30 prompts (confirmed overfitting with manual tuning)

---

## [0.2.0] — 2026-05-06

### Added
- Manual weight tuning based on v0.1 misclassification analysis
- `question_technical` bonus (0.12)
- `architecture` bonus (0.15)
- Length dampener for short prompts

### Results
- 67% accuracy on 15 prompts (+14pp over v0.1)
- Dropped to 40% on 30 prompts (overfitting confirmed)

---

## [0.1.0] — 2026-05-05

### Added
- Initial heuristic router with 13 hand-tuned features
- 6-tier complexity classification (trivial → extreme)
- Static weight assignment

### Results
- 53% accuracy on 15 manual prompts
- No training data, no optimization
