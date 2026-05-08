# MoA v3 Fine-Tuning

Serverless training handler for optimizing MoA Gateway Router heuristic weights.

## Results

| Version | Dataset | Accuracy | Method |
|---------|---------|----------|--------|
| v1 | 15 prompts | 53% | Fixed heuristic weights |
| v2 | 15 prompts | 67% | Manual tuning |
| v2.1 | 30 prompts | 40% | Architecture/tech_design bonuses added |
| **v3.0** | **10K Alpaca** | **87.2%** | **scipy MSE optimization** |

See [COMPARISON.md](./COMPARISON.md) for full v1→v2→v3 analysis.

## How It Works

### Pipeline

1. **Load** 10K instruction prompts from `tatsu-lab/alpaca`
2. **Extract** 13 features per prompt (word count, code, question, imperative, math, multi-step, constraints, context, sentences, avg word length, architecture, technical design, richness)
3. **Label** with synthetic complexity score (0.0–1.0) based on signal count, length, and lexical richness
4. **Map** to 6 tiers: trivial (0.00), light (0.08), moderate (0.18), heavy (0.32), intensive (0.52), extreme (0.72)
5. **Optimize** weights via scipy L-BFGS-B minimizing MSE between predicted and actual tier scores
6. **Evaluate** tier-matching accuracy on 2K holdout test set

### v3.0 Optimized Weights

| Feature | v2.1 | v3.0 | Change |
|---------|------|------|--------|
| sentence_count | 0.03 | **0.29** | ↑↑↑ |
| avg_word_length | 0.02 | **0.19** | ↑↑↑ |
| question | 0.02 | **0.12** | ↑↑ |
| technical_design | 0.18 | **0.12** | ↓ |
| imperative | 0.12 | **0.11** | ≈ |
| context | 0.05 | **0.11** | ↑ |
| math | 0.05 | **0.11** | ↑ |
| multi_step | 0.08 | **0.11** | ↑ |
| code | 0.18 | **0.10** | ↓ |
| constraints | 0.06 | **0.09** | ↑ |
| architecture | 0.28 | **0.07** | ↓↓↓ |
| word_count | 0.04 | **0.00** | ↓↓↓ |
| four_plus | 0.10 | **0.00** | ↓↓↓ |

### Per-Tier Accuracy

| Tier | Samples | v2.1 | v3.0 |
|------|---------|------|------|
| moderate | 1,182 | 40.8% | **88.8%** |
| heavy | 814 | 34.2% | **67.4%** |
| intensive | 4 | 25.0% | **100%** |

*No trivial/light prompts in Alpaca — dataset is biased toward moderate+ complexity.*

## Usage

```python
from handler import handler

result = handler({
    "input": {
        "datasets": ["alpaca"],
        "max_per": 10000
    }
})
# → optimized weights, accuracy metrics, tier breakdown
```

## Deploy to RunPod Serverless

```bash
docker build -t ghcr.io/pealmeida/moa-v3-finetuning:latest -f Dockerfile.serverless .
docker push ghcr.io/pealmeida/moa-v3-finetuning:latest
```

Create a serverless endpoint on RunPod using the template, then POST to `/run`.

## Architecture

The handler runs the full training pipeline locally — no GPU needed. It's pure scipy/numpy optimization (L-BFGS-B on a 13-parameter MSE objective over 8K training examples).

### Why Serverless?

The training is CPU-bound and completes in seconds locally. Serverless deployment is useful for:
- **CI/CD integration** — retrain on new datasets automatically
- **API access** — trigger training from the MoA Gateway on demand
- **Scaling** — parallel runs with different datasets/hyperparameters

## Files

| File | Purpose |
|------|---------|
| `handler.py` | Full training pipeline (features → labels → optimize → evaluate) |
| `Dockerfile.serverless` | RunPod serverless container image |
| `COMPARISON.md` | Detailed v1→v2→v3 analysis |

## License

MIT
