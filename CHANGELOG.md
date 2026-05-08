# Changelog

All notable changes to the MoA v3 Finetuning project.

## [Unreleased]

### v3.2 — Tier-Pair Binary Cascade (2026-05-08)

### Added
- **v3.2 cascade handler** (`handler_v32_cascade.py`) — 5 independent binary classifiers in cascade for full 6-tier classification
- **sklearn integration** — LogisticRegression with LBFGS solver for non-linear decision boundaries
- **Balanced training** — 1:1 positive/negative sampling per classifier
- **RunPod test results** — 75K samples across 3 datasets, 15K test set, 74.7% overall accuracy
- **Documentation**
  - `docs/V3_2_CASCADE_REPORT.md` — Full cascade analysis (7.8KB)
  - `docs/PER_TIER_TEST_REPORT.md` — v3.1 massive per-tier test analysis

### Results (v3.2)
| Dataset | Method | Baseline | Optimized | Improvement |
|---------|--------|----------|-----------|-------------|
| 75K (3 datasets) | Cascade | 24.3% | **74.7%** | +50.4pp |

**Per-tier:** trivial 100.0%, light 93.1%, moderate 42.4%, heavy 38.7%, intensive 19.3%, extreme 68.8%

### Changed
- `handler.py` → now uses v3.2 cascade handler
- `Dockerfile.serverless` → updated with sklearn, cascade handler
- `entrypoint.sh` → updated for cascade handler
- `README.md` → updated with v3.2 results and cascade architecture

---

## [3.1.0] — 2026-05-08

### Added
- **v3.1 handler** (`handler_v31.py`) — Multi-dataset training with GPD (50K synthetic), workspace data, and user-uploaded datasets
- **LLMFit Dataset Factory** (`llmfit/`) — Complete pipeline for personalized dataset creation
  - `llmfit.py` — Core engine: workspace scanning, feature extraction (15 features), labeling, validation, weight optimization
  - `anonymizer.py` — 35-rule anonymization engine (PII, secrets, context, generalization)
  - `self_eval.py` — Self-evaluation module with SQLite feedback buffer for online learning
  - `gpd_generator.py` — 50K synthetic trivial/light prompt generator
  - `handler_v31_massive.py` — Massive multi-dataset per-tier test handler
- **General-Purpose Dataset (GPD)** — 50,000 synthetic prompts (35K trivial + 15K light) for light-tier optimization
- **Documentation**
  - `docs/ARCHITECTURE_V3_1.md` — Full v3.1 architecture specification (38KB)
  - `docs/TRAINING_REPORT_V3_1.md` — Training results and anonymization report
- **GitHub templates** — `.gitignore`, `.dockerignore`, `CONTRIBUTING.md`

### Results (v3.1)
| Dataset | Baseline | Optimized | Improvement |
|---------|----------|-----------|-------------|
| GPD 50K | 19.3% | 99.6% | +80.2% |
| Alpaca 10K | 2.3% | 87.2% | +84.9% |
| 75K (3 datasets) | 34.4% | 42.8% | +8.5pp |

---

## [3.0.0] — 2026-05-07

### Added
- **v3.0 handler** (`handler.py`) — Full training pipeline with scipy MSE optimization
- **RunPod Serverless deployment** — Docker image + entrypoint for serverless training
- **13-feature complexity vector** — word_count, question, code, imperative, math, multi_step, constraints, context, sentence_count, avg_word_length, architecture, technical_design, four_plus
- **Synthetic complexity labels** — Signal count + length + lexical richness → 6 tiers
- **COMPARISON.md** — v1 → v2 → v3 analysis with per-tier breakdown

### Results (v3.0)
| Dataset | Baseline | Optimized | Improvement |
|---------|----------|-----------|-------------|
| Alpaca 10K | 2.3% | 87.2% | +84.9% |

### Changed
- Architecture fix in v2.1 (heuristic bonuses) → replaced by data-driven optimization
- Weight optimization: scipy L-BFGS-B converging in 31 iterations

---

## [2.1.0] — 2026-05-06

### Added
- `architecture` bonus (0.28) for architecture-heavy prompts
- `technical_design` bonus (0.18) for system design terms
- Extended architecture regex keywords
- Length dampener skip for architecture-heavy prompts

### Changed
- `architecture` bonus: 0.15 → 0.28
- Result: 40% on 30 prompts (confirmed overfitting with manual tuning)

---

## [2.0.0] — 2026-05-06

### Added
- Manual weight tuning based on v1 misclassification analysis
- `question_technical` bonus (0.12)
- `architecture` bonus (0.15)
- Length dampener for short prompts

### Changed
- `hasImperative`: 0.20 → 0.12
- `hasCode`: 0.24 → 0.18
- `hasQuestion`: 0.02 → 0.15

### Results
- 67% accuracy on 15 prompts (+14pp over v1)
- Dropped to 40% on 30 prompts (overfitting confirmed)

---

## [1.0.0] — 2026-05-05

### Added
- Initial heuristic router with 13 hand-tuned features
- 6-tier complexity classification (trivial → extreme)
- Static weight assignment

### Results
- 53% accuracy on 15 manual prompts
- No training data, no optimization
