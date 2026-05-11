# Changelog

All notable changes to the GateSwarm MoA Router project.

## [0.3.5] — 2026-05-11

### Changed
- **Rebranded** from `moa-v3-finetuning` to `gateswarm-moa-router`
- Version scheme updated: v1.x → v0.1.x, v2.x → v0.2.x, v3.x → v0.3.x
- **Standalone router** — `router.py` works without RunPod SDK or any cloud dependency
- **Cleaned codebase** — removed 30 unused files (stale handlers, RunPod wrappers, internal reports)
- Renamed core files: `handler_v32_cascade.py` → `train.py`, `handler_v33_inference.py` → `router.py`
- Added HTTP API server, CLI, and batch scoring to `router.py`
- Simplified Dockerfiles (removed 4 RunPod-specific, kept 2 generic)
- Updated `requirements.txt` — removed `runpod`, added `scikit-learn`

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

## [0.3.3] — 2026-05-08

### Problem Identified
The Chief Scientist evaluation found a **critical weakness**: formula-based labels have **2.0/10 validity** (synthetic, no ground truth). The labels are circular with features — the optimizer learns to reproduce the formula, not real complexity. Production readiness rated **6.0/10** (safe for trivial/light only).

### Added
- **Chief Scientist evaluation** — Complete independent review of v0.3.0–v0.3.2 pipeline
  - Engineering quality: 8.5/10
  - Label validity: 2.0/10 (critical)
  - Production readiness: 6.0/10
  - 5-phase roadmap for improvement
- **6-model cost-efficient routing strategy** — Specialized model per tier
  - trivial → glm-4.5-air (FREE), extreme → claude-opus-4.6 ($5.00/M)
  - 80–84% savings vs always-Opus
  - 3 optimization strategies: downgrade, confidence-based, escalation
  - Bailian-first provider profile for cost optimization

### Changed
- Identified that pairwise cascade was achieving only **21% accuracy** on real prompts (predicting all "light") — the formula labels were misleading during training
- Pivoted approach: from pure data-driven back to validated heuristic formula

---

## [0.3.4] — 2026-05-09

### Added
- **Label correction pipeline** — Cascade trained on balanced data corrects systematic formula errors
  - Cascade predictions as primary labels: **65.84% accuracy** vs 30.11% formula baseline
  - Key insight: balanced training + binary cascade architecture fixes formula errors even when trained on formula labels
- **LLM-as-Judge labeling** — Ground-truth labels from qwen3.6-plus for empirical validation
  - Stratified sampling: ~300 prompts per tier
  - Batch judging (20 per API call) — ~$5 total cost for 100K samples
  - Cross-validation between formula, cascade, and LLM labels
- **Label validation handler** — Compares formula vs cascade predictions, identifies systematic disagreements
- **Production inference handler** — Real-time complexity scoring with pre-trained cascade weights
  - Returns tier + confidence + score + recommended model/provider
  - Batch support (multiple prompts in one call)
  - Feature extraction included in response
- **Weight export handler** — Label correction + production weight export in one pass
- **Parallel LLM judge** — Concurrent judging for large datasets
- **TurboQuant context compression** — Model switching context optimization

### Changed
- `handler.py` → v3.3 label correction (cascade predictions as primary labels)
- Scoring method: **9-signal heuristic formula** (99% LLM-validated) replaces pure cascade
- Tier profiles: Bailian-first model assignments (glm-4.5-air → qwen3.6-plus → claude-opus)
- Docker: 3 new specialized images (v33, inference, weight-export)

### Results
- Label accuracy improved from 30.11% (formula) to **65.84%** (corrected cascade)
- Heuristic formula validated at **99%** agreement with LLM judge
- Production inference latency: ~12ms per prompt (CPU-only)

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
