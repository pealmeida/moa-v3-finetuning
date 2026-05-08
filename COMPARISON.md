# MoA Router — Heuristic Evolution: v1 → v2 → v3 → v3.1

## Executive Summary

| Version | Dataset | Accuracy | Method | Date | Platform |
|---------|---------|----------|--------|------|----------|
| **v1** | 15 manual prompts | 53% (8/15) | Static hand-tuned weights | 2026-05-05 | Local |
| **v2** | 15 manual prompts | 67% (10/15) | Manual tuning + bonuses | 2026-05-06 | Local |
| **v2.1** | 30 manual prompts | 40% (12/30) | Architecture/tech_design bonuses | 2026-05-06 | Local |
| **v3.0** | 10K Alpaca prompts | **87.2%** (1744/2000) | MSE optimization (scipy L-BFGS-B) | 2026-05-07 | RunPod Serverless |
| **v3.1-GPD** | 15K GPD (trivial+light) | **72.7%** | MSE opt + balanced weights | 2026-05-08 | RunPod Serverless |
| **v3.1-Massive** | 75K (GPD+Alpaca+OpenOrca) | **42.8%** | Global MSE optimization | 2026-05-08 | RunPod Serverless |
| **v3.2-Cascade** | 75K (GPD+Alpaca+OpenOrca) | **74.7%** | Tier-pair binary cascade | 2026-05-08 | RunPod Serverless |

---

## v3.1 — Self-Optimizing Gateway (2026-05-08)

### What Changed from v3.0

| Aspect | v3.0 | v3.1 |
|--------|------|------|
| **Feature vector** | 13 features | 15 features (+ ambiguity_score, domain_specificity) |
| **Datasets** | Alpaca only (moderate+ bias) | GPD (50K trivial/light) + Alpaca + workspace |
| **Feature extraction** | Inline in handler | Modular (`extract_features_v31()`) |
| **Labeling** | Synthetic complexity only | Rule-based + LLM-assisted + user labels |
| **Anonymization** | None | 35-rule engine (PII/secrets/context) |
| **Personalization** | None | Workspace scanning, session mining |
| **Feedback loop** | None | Self-evaluation + SQLite buffer |
| **Platform** | RunPod Serverless | Local CPU + RunPod Serverless |

### v3.1 Results by Dataset

| Dataset | Samples | Baseline | Optimized | Improvement |
|---------|---------|----------|-----------|-------------|
| **GPD (50K)** | 50,000 | 19.3% | **99.6%** | +80.2% |
| **Alpaca (10K)** | 10,000 | 2.3% | **87.2%** | +84.9% |
| **Workspace (2.5K)** | 2,502 | — | — | *Multi-tier* |

### Per-Tier Accuracy (v3.1 GPD)

| Tier | Test Samples | Accuracy |
|------|-------------|----------|
| **Trivial** | 7,050 | **100.0%** |
| **Light** | 2,950 | **98.5%** |

### v3.1 Optimized Weights (GPD 50K)

| Feature | v3.0 (Alpaca) | v3.1 (GPD) |
|---------|---------------|------------|
| ambiguity_score | — | **0.6041** |
| sentence_count | **0.2915** | 0.0000 |
| technical_design | 0.1196 | **0.2071** |
| avg_word_length | **0.1890** | 0.0000 |
| architecture | 0.0698 | **0.1208** |
| has_imperative | **0.1141** | **0.0295** |
| has_code | **0.1044** | **0.0267** |

> **Insight:** GPD weights are dominated by `ambiguity_score` because trivial/light prompts primarily differ by vague language. Alpaca weights favor structural features (sentences, word length). **Merge both datasets for balanced weights.**

---

## v3.0 — MSE Optimization (2026-05-07)

### Pipeline

```
1. LOAD: tatsu-lab/alpaca → 10,000 prompts
2. EXTRACT: 13 features per prompt
3. LABEL: Synthetic complexity (0.0–1.0)
4. MAP: 6 tiers (trivial → extreme)
5. SPLIT: 80/20 → 8,000 train / 2,000 test
6. OPTIMIZE: scipy L-BFGS-B (MSE loss)
7. EVALUATE: Tier-matching accuracy
```

### Per-Tier Breakdown (Alpaca 10K)

| Tier | Test Samples | v2.1 | v3.0 | Δ |
|------|-------------|------|------|---|
| trivial | 0 | — | — | — |
| light | 0 | — | — | — |
| moderate | 1,182 | 40.8% | **88.8%** | +48.0pp |
| heavy | 814 | 34.2% | **67.4%** | +33.2pp |
| intensive | 4 | 25.0% | **100.0%** | +75.0pp |
| extreme | 0 | — | — | — |

### Key Insights

1. **sentence_count is the #1 complexity driver** in Alpaca (0.29 weight)
2. **avg_word_length is #2** — technical vocabulary strongly correlates (0.19 weight)
3. **architecture was massively over-weighted** in v2.1 (0.28 → 0.07)
4. **word_count dropped to zero** — redundant with sentence structure
5. **Alpaca has no trivial/light prompts** — dataset bias confirmed

---

## v2.1 → v3.0 Weight Changes

| Feature | v2.1 | v3.0 | Δ | Interpretation |
|---------|------|------|---|----------------|
| **sentence_count** | 0.03 | **0.29** | ↑↑↑ | Multi-sentence = harder |
| **avg_word_length** | 0.02 | **0.19** | ↑↑↑ | Longer words = technical |
| **question** | 0.02 | **0.12** | ↑↑ | Questions need capability |
| **context** | 0.05 | **0.11** | ↑↑ | Background context = harder |
| **math** | 0.05 | **0.11** | ↑ | Math expressions matter |
| **multi_step** | 0.08 | **0.11** | ↑ | Sequential reasoning |
| **constraints** | 0.06 | **0.09** | ↑ | Limits add complexity |
| **technical_design** | 0.18 | **0.12** | ↓ | Less dominant than assumed |
| **code** | 0.18 | **0.10** | ↓ | Still important |
| **imperative** | 0.12 | **0.11** | ≈ | Stable |
| **architecture** | 0.28 | **0.07** | ↓↓↓ | Rare in Alpaca |
| **word_count** | 0.04 | **0.00** | ↓↓↓ | Redundant |
| **four_plus** | 0.10 | **0.00** | ↓↓↓ | Correlated |

---

## Version History

### v1 — Static Heuristic (2026-05-05)
- 13 hand-tuned features, no training data
- 53% accuracy on 15 prompts
- **Problem:** No systematic tuning, architecture over-weighted

### v2 — Manual Tuning (2026-05-06)
- Analyzed v1 misclassifications, adjusted weights
- 67% on 15 prompts → dropped to 40% on 30 prompts
- **Problem:** Overfitting to tiny sample confirmed

### v2.1 — Architecture Fix (2026-05-06)
- Further manual tuning with architecture bonuses
- 40% on 30 prompts
- **Lesson:** Manual tuning fundamentally doesn't scale

### v3.0 — MSE Optimization (2026-05-07)
- 10K Alpaca prompts, scipy L-BFGS-B, 87.2% accuracy
- **Breakthrough:** Data-driven optimization replaces manual tuning
- Deployed on RunPod Serverless (35s execution)

### v3.1 — Self-Optimizing Gateway (2026-05-08)
- 50K GPD synthetic + workspace data, 99.6% on light tier
- LLMFit Dataset Factory for personalized training
- 35-rule anonymization engine
- Self-evaluation feedback loop
- **Breakthrough:** Users can create their own optimized datasets

### v3.1 Massive Per-Tier Test (2026-05-08)
- 75K samples (GPD 35K + Alpaca 20K + OpenOrca 20K)
- Full 6-tier per-tier accuracy evaluation
- RunPod Serverless: 8.4s execution

**Per-Tier Results:**

| Tier | Samples | Baseline | Optimized | Δ |
|------|---------|----------|-----------|---|
| Trivial | 7,069 | 18.7% | **57.9%** | +39.2pp |
| Light | 5,228 | 49.2% | 30.3% | -18.9pp |
| Moderate | 1,800 | 33.8% | 17.6% | -16.2pp |
| Heavy | 770 | 73.9% | 49.9% | -24.0pp |
| Intensive | 134 | 60.4% | 33.6% | -26.9pp |
| **Overall** | **15,001** | **34.4%** | **42.8%** | **+8.5pp** |

**Findings:** Trivial detection improved massively (+39pp). Light/moderate/heavy separation degraded due to feature overlap. See `docs/PER_TIER_TEST_REPORT.md` for full analysis.

---

## Lessons Learned

1. **Manual tuning doesn't scale** — v1→v2: 53%→67% on 15 prompts, then 40% on 30
2. **Data-driven optimization wins** — v3.0: +84.9pp with 10K prompts
3. **Dataset bias matters** — Alpaca has no trivial/light; GPD fills that gap
4. **Merge datasets for balance** — GPD + Alpaca = full-tier coverage
5. **Anonymization is essential** — User data must be sanitized before training
6. **CPU is sufficient** — 15-parameter optimization runs in seconds without GPU
