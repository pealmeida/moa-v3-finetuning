# MoA v3.3 — Empirical Labeling Pipeline
## Trial-and-Error Architecture with RunPod Serverless

**Date:** 2026-05-08  
**Chief Scientist Design Doc**  
**Status:** Architecture Defined

---

## 1. The Oracle Validation Logic

### Core Principle: Difficulty-to-Solve (DTS) Metric

Instead of synthetic formula labels, we measure the **minimum model capability needed to correctly solve a prompt**. This is empirical ground truth — if GLM-4.5-Air answers correctly, the prompt is trivial. If only Claude Opus succeeds, it's extreme.

### Recursive Evaluation Loop

```
┌─────────────────────────────────────────────────────────────────┐
│                    ORACLE VALIDATION LOOP                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Prompt ──▶ [Model Tier 1: GLM-4.5-Air (FREE)]                 │
│               │                                                  │
│               ├─ PASS? ── YES ──▶ Label = "trivial" (DTS=0)     │
│               │                                                  │
│               └─ FAIL?                                          │
│                    │                                             │
│                    ▼                                             │
│              [Model Tier 2: GLM-4.7-Flash ($0.02/M)]            │
│                    │                                             │
│                    ├─ PASS? ── YES ──▶ Label = "light" (DTS=1)   │
│                    │                                             │
│                    └─ FAIL?                                      │
│                         │                                        │
│                         ▼                                        │
│                   [Model Tier 3: Qwen3.5-9B ($0.10/M)]          │
│                         │                                        │
│                         ├─ PASS? ── YES ──▶ Label = "moderate"   │
│                         │                                        │
│                         └─ FAIL?                                 │
│                              │                                   │
│                              ▼                                   │
│                        [Model Tier 4: Qwen3.6-Plus]             │
│                              │                                   │
│                              ├─ PASS? ── YES ──▶ Label = "heavy" │
│                              │                                   │
│                              └─ FAIL?                            │
│                                   │                              │
│                                   ▼                              │
│                             [Model Tier 5: Claude Sonnet 4.6]    │
│                                   │                              │
│                                   ├─ PASS? ── YES ──▶ intensive  │
│                                   │                              │
│                                   └─ FAIL?                       │
│                                        │                         │
│                                        ▼                         │
│                                  [Model Tier 6: Claude Opus 4.6] │
│                                        │                         │
│                                        ├─ PASS? ── YES ──▶ extreme│
│                                        │                         │
│                                        └─ FAIL?                  │
│                                             │                    │
│                                             ▼                    │
│                                    Label = "extreme" (DTS=5)     │
│                                    (Opus is the ceiling)         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Model Tier Assignment

| Tier Index | Model | Provider | Cost ($/M tokens) | Capability |
|------------|-------|----------|-------------------|------------|
| 0 | `glm-4.5-air` | zai (free) | $0.00 | Basic language understanding |
| 1 | `glm-4.7-flash` | zai | $0.02 | Instruction following |
| 2 | `qwen3.5-9b` | openrouter | $0.10 | Code + reasoning |
| 3 | `qwen3.6-plus` | bailian | $0.04 (Coding Plan) | Deep analysis |
| 4 | `claude-sonnet-4.6` | openrouter | $3.00 | Architecture, multi-step |
| 5 | `claude-opus-4.6` | openrouter | $5.00 | Strategic thinking |

### Oracle Rubric Prompt (JSON Format)

```json
{
  "system_prompt": "You are an impartial technical evaluator. Your job is to determine whether a model's response to a prompt meets professional standards for accuracy, completeness, and instruction following.",
  "evaluation_template": {
    "prompt_text": "{prompt}",
    "model_response": "{response}",
    "criteria": [
      {
        "name": "instruction_following",
        "description": "Does the response address all explicit instructions in the prompt?",
        "weight": 0.35,
        "scoring": {
          "3": "All instructions followed completely",
          "2": "Most instructions followed, minor omissions",
          "1": "Significant instructions missed or partially addressed",
          "0": "Failed to follow core instructions"
        }
      },
      {
        "name": "technical_accuracy",
        "description": "Is the response factually and technically correct? No hallucinations, no wrong code, no incorrect claims.",
        "weight": 0.35,
        "scoring": {
          "3": "Fully accurate, no errors",
          "2": "Minor inaccuracies that don't affect correctness",
          "1": "Significant technical errors but partially correct",
          "0": "Fundamentally incorrect or hallucinated"
        }
      },
      {
        "name": "completeness",
        "description": "Does the response cover all aspects needed for a complete answer? Not just correct but thorough.",
        "weight": 0.15,
        "scoring": {
          "3": "Comprehensive coverage, no gaps",
          "2": "Good coverage with minor gaps",
          "1": "Significant gaps in coverage",
          "0": "Incomplete or superficial"
        }
      },
      {
        "name": "constraint_adherence",
        "description": "If the prompt has constraints (length, format, specific tools), were they respected?",
        "weight": 0.10,
        "scoring": {
          "3": "All constraints met",
          "2": "Most constraints met",
          "1": "Some constraints violated",
          "0": "Ignored constraints entirely",
          "null": "No constraints in prompt"
        }
      },
      {
        "name": "format_quality",
        "description": "Is the response well-structured, readable, and professionally formatted?",
        "weight": 0.05,
        "scoring": {
          "3": "Excellent structure and formatting",
          "2": "Good structure, minor formatting issues",
          "1": "Poor structure, hard to follow",
          "0": "Unreadable or chaotic"
        }
      }
    ],
    "pass_threshold": 0.70,
    "weighted_score_formula": "sum(score_i * weight_i) / 3.0",
    "pass_condition": "weighted_score >= pass_threshold AND technical_accuracy >= 2 AND instruction_following >= 2",
    "output_format": "Respond with ONLY a JSON object: {\"weighted_score\": 0.XX, \"criteria_scores\": {...}, \"pass\": true/false, \"reasoning\": \"brief explanation\"}"
  }
}
```

### Anti-Bias Safeguards

1. **Blind evaluation:** Oracle doesn't know which model tier produced the response
2. **Technical accuracy floor:** Must score ≥2/3 on accuracy regardless of weighted average
3. **Instruction following floor:** Must score ≥2/3 on instruction following regardless of weighted average
4. **Calibration prompts:** Include 50 known-easy and 50 known-hard prompts in each batch to detect evaluator drift
5. **Cross-validation:** Every 100th prompt gets evaluated by a second oracle model

---

## 2. RunPod Serverless Orchestration Architecture

### High-Level Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                        CONTROL PLANE (Local)                        │
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │  Dataset     │───▶│  Batch       │───▶│  Job         │          │
│  │  Loader      │    │  Builder     │    │  Dispatcher  │          │
│  │  (streaming) │    │  (20/batch)  │    │  (async)     │          │
│  └──────────────┘    └──────────────┘    └──────┬───────┘          │
│                                                  │                  │
│                          ┌───────────────────────┼───────────────┐ │
│                          ▼                       ▼               ▼ │
│                    ┌──────────┐           ┌──────────┐    ┌──────────┐│
│                    │ RunPod   │           │ RunPod   │    │ RunPod   ││
│                    │ Worker 1 │           │ Worker 2 │    │ Worker 3 ││
│                    │ (Oracle) │           │ (Oracle) │    │ (Oracle) ││
│                    │          │           │          │    │          ││
│                    │ vLLM     │           │ vLLM     │    │ vLLM     ││
│                    │ Llama-3  │           │ Llama-3  │    │ Llama-3  ││
│                    │ 70B      │           │ 70B      │    │ 70B      ││
│                    └────┬─────┘           └────┬─────┘    └────┬─────┘│
│                         │                     │               │       │
│                         ▼                     ▼               ▼       │
│                    ┌─────────────────────────────────────────┐       │
│                    │          MODEL API CALLS                 │       │
│                    │  (zai, bailian, openrouter - 6 models)   │       │
│                    └─────────────────────────────────────────┘       │
│                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │  Result      │◀───│  Aggregator  │◀───│  Result      │          │
│  │  Merger      │    │  & Validator │    │  Collector   │          │
│  └──────┬───────┘    └──────────────┘    └──────────────┘          │
│         │                                                          │
│         ▼                                                          │
│  ┌──────────────┐                                                   │
│  │  Labeled     │                                                   │
│  │  Dataset     │                                                   │
│  │  (JSONL)     │                                                   │
│  └──────────────┘                                                   │
└────────────────────────────────────────────────────────────────────┘
```

### Python Orchestrator Script

```python
"""
MoA v3.3 — Empirical Labeling Orchestrator
RunPod Serverless with asyncio batch processing
"""
import asyncio
import json
import time
import os
from dataclasses import dataclass, field
from typing import Optional
from collections import Counter
import aiohttp
import runpod

# ── Configuration ──────────────────────────────────────────────────

MODEL_TIERS = [
    {"name": "trivial",   "model": "glm-4.5-air",     "provider": "zai",              "cost_per_m": 0.00},
    {"name": "light",     "model": "glm-4.7-flash",   "provider": "zai",              "cost_per_m": 0.02},
    {"name": "moderate",  "model": "qwen3.5-9b",      "provider": "openrouter",        "cost_per_m": 0.10},
    {"name": "heavy",     "model": "qwen3.6-plus",    "provider": "bailian",           "cost_per_m": 0.04},
    {"name": "intensive", "model": "claude-sonnet-4.6","provider": "openrouter",        "cost_per_m": 3.00},
    {"name": "extreme",   "model": "claude-opus-4.6", "provider": "openrouter",        "cost_per_m": 5.00},
]

ORACLE_MODEL = "meta-llama/llama-3.3-70b-instruct:free"  # Free tier for cost efficiency
ORACLE_BASE_URL = "http://localhost:8000/v1"  # vLLM endpoint on RunPod worker
BATCH_SIZE = 20
MAX_RETRIES = 3
CONCURRENCY = 5  # Parallel model calls

@dataclass
class CostTracker:
    total_api_calls: int = 0
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    estimated_cost_usd: float = 0.0
    tier_costs: dict = field(default_factory=dict)
    
    def record(self, tier_name: str, tokens_in: int, tokens_out: int, cost_per_m: float):
        self.total_api_calls += 1
        self.total_tokens_in += tokens_in
        self.total_tokens_out += tokens_out
        cost = (tokens_in + tokens_out) / 1_000_000 * cost_per_m
        self.estimated_cost_usd += cost
        self.tier_costs[tier_name] = self.tier_costs.get(tier_name, 0) + cost


# ── Model API Calls ────────────────────────────────────────────────

class ModelCaller:
    """Async model caller with retry logic and rate limiting."""
    
    def __init__(self, session: aiohttp.ClientSession):
        self.session = session
        self.semaphore = asyncio.Semaphore(CONCURRENCY)
        
    async def call_model(self, tier: dict, prompt: str, max_tokens: int = 2048) -> dict:
        """Call a model API and return response + token counts."""
        async with self.semaphore:
            provider = tier["provider"]
            model = tier["model"]
            
            # Map provider to endpoint
            if provider == "zai":
                url = "https://api.z.ai/api/coding/paas/v4/chat/completions"
                api_key = os.environ.get("ZAI_API_KEY")
            elif provider == "bailian":
                url = "https://coding-intl.dashscope.aliyuncs.com/v1/chat/completions"
                api_key = os.environ.get("BAILIAN_API_KEY")
            elif provider == "openrouter":
                url = "https://openrouter.ai/api/v1/chat/completions"
                api_key = os.environ.get("OPENROUTER_API_KEY")
            else:
                raise ValueError(f"Unknown provider: {provider}")
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.1,  # Deterministic for evaluation consistency
            }
            
            for attempt in range(MAX_RETRIES):
                try:
                    async with self.session.post(url, headers=headers, json=payload) as resp:
                        if resp.status == 429:
                            wait = min(2 ** attempt, 30)
                            await asyncio.sleep(wait)
                            continue
                        
                        data = await resp.json()
                        return {
                            "success": True,
                            "content": data["choices"][0]["message"]["content"],
                            "tokens_in": data.get("usage", {}).get("prompt_tokens", 0),
                            "tokens_out": data.get("usage", {}).get("completion_tokens", 0),
                        }
                except Exception as e:
                    if attempt == MAX_RETRIES - 1:
                        return {"success": False, "error": str(e)}
                    await asyncio.sleep(2 ** attempt)
            
            return {"success": False, "error": "Max retries exceeded"}


# ── Oracle Evaluation ──────────────────────────────────────────────

ORACLE_SYSTEM_PROMPT = """You are an impartial technical evaluator. Evaluate whether a model's response meets professional standards.

Score each criterion 0-3, then compute the weighted score.
Pass requires weighted_score >= 0.70 AND technical_accuracy >= 2 AND instruction_following >= 2.

Respond with ONLY a JSON object: {"weighted_score": 0.XX, "criteria_scores": {"instruction_following": N, ...}, "pass": true/false, "reasoning": "brief"}"""

ORACLE_USER_TEMPLATE = """Prompt: {prompt}

Model Response:
{response}

Evaluate using the criteria and scoring rubric."""


class OracleEvaluator:
    """Evaluates model responses using the Oracle Rubric."""
    
    def __init__(self, session: aiohttp.ClientSession, oracle_url: str):
        self.session = session
        self.oracle_url = oracle_url
        
    async def evaluate(self, prompt: str, response: str) -> dict:
        """Evaluate a response and return pass/fail decision."""
        user_content = ORACLE_USER_TEMPLATE.format(
            prompt=prompt[:2000],  # Truncate long prompts
            response=response[:4000],
        )
        
        payload = {
            "model": ORACLE_MODEL,
            "messages": [
                {"role": "system", "content": ORACLE_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "max_tokens": 300,
            "temperature": 0.1,
        }
        
        async with self.session.post(
            f"{self.oracle_url}/chat/completions",
            headers={"Content-Type": "application/json"},
            json=payload,
        ) as resp:
            data = await resp.json()
            content = data["choices"][0]["message"]["content"].strip()
            
            # Parse JSON from response
            try:
                import re
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                return {"pass": False, "error": f"Failed to parse: {content[:100]}"}
            except json.JSONDecodeError:
                return {"pass": False, "error": "Invalid JSON"}


# ── Trial-and-Error Pipeline ───────────────────────────────────────

async def evaluate_prompt(prompt: str, caller: ModelCaller, oracle: OracleEvaluator, 
                         cost_tracker: CostTracker) -> dict:
    """
    Recursive evaluation: test prompt against each tier until it passes.
    Returns the tier where it first passes (Difficulty-to-Solve metric).
    """
    for tier_idx, tier in enumerate(MODEL_TIERS):
        # Call the model
        result = await caller.call_model(tier, prompt)
        
        if not result["success"]:
            cost_tracker.record(tier["name"], 0, 0, tier["cost_per_m"])
            continue
        
        # Oracle evaluation
        eval_result = await oracle.evaluate(prompt, result["content"])
        
        # Record cost
        cost_tracker.record(tier["name"], result["tokens_in"], result["tokens_out"], tier["cost_per_m"])
        
        # Check pass condition
        if eval_result.get("pass", False):
            return {
                "prompt": prompt[:500],
                "difficulty_to_solve": tier_idx,
                "assigned_tier": tier["name"],
                "evaluating_tier": tier["name"],
                "oracle_score": eval_result.get("weighted_score", 0),
                "oracle_criteria": eval_result.get("criteria_scores", {}),
                "oracle_reasoning": eval_result.get("reasoning", ""),
                "tokens_in": result["tokens_in"],
                "tokens_out": result["tokens_out"],
            }
        
        # If failed, continue to next tier
    
    # If all tiers fail, assign to extreme
    return {
        "prompt": prompt[:500],
        "difficulty_to_solve": 5,
        "assigned_tier": "extreme",
        "evaluating_tier": "extreme",
        "oracle_score": 0,
        "oracle_reasoning": "All tiers failed",
    }


async def process_batch(batch: list[dict], caller: ModelCaller, oracle: OracleEvaluator,
                       cost_tracker: CostTracker) -> list[dict]:
    """Process a batch of prompts concurrently."""
    tasks = [evaluate_prompt(p["text"], caller, oracle, cost_tracker) for p in batch]
    return await asyncio.gather(*tasks)


# ── Main Orchestrator ──────────────────────────────────────────────

async def main():
    print("=== MoA v3.3 Empirical Labeling Orchestrator ===")
    start_time = time.time()
    cost_tracker = CostTracker()
    
    # Load dataset (streaming from HuggingFace)
    from datasets import load_dataset
    
    prompts = []
    for ds_name, key in [("tatsu-lab/alpaca", "instruction"), ("Open-Orca/OpenOrca", "question")]:
        ds = load_dataset(ds_name, split="train", streaming=True)
        for x in ds:
            txt = x.get(key, "")
            if isinstance(txt, str) and len(txt) > 10:
                prompts.append({"text": txt.strip()})
            if len(prompts) >= 75000:
                break
        print(f"Loaded {len(prompts)} prompts from {ds_name}")
    
    print(f"Total prompts: {len(prompts)}")
    
    # Setup HTTP session
    async with aiohttp.ClientSession() as session:
        caller = ModelCaller(session)
        oracle = OracleEvaluator(session, ORACLE_BASE_URL)
        
        # Process in batches
        results = []
        batch_size = BATCH_SIZE
        total_batches = (len(prompts) + batch_size - 1) // batch_size
        
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i+batch_size]
            batch_num = i // batch_size + 1
            
            batch_results = await process_batch(batch, caller, oracle, cost_tracker)
            results.extend(batch_results)
            
            # Progress update
            if batch_num % 10 == 0:
                elapsed = time.time() - start_time
                rate = len(results) / elapsed
                eta = (len(prompts) - len(results)) / rate if rate > 0 else 0
                
                print(f"Batch {batch_num}/{total_batches}: "
                      f"{len(results)}/{len(prompts)} processed, "
                      f"rate={rate:.1f}/s, "
                      f"ETA={eta/60:.0f}min, "
                      f"cost=${cost_tracker.estimated_cost_usd:.2f}")
            
            # Save intermediate results
            if batch_num % 50 == 0:
                with open(f"labeled_results_batch_{batch_num}.jsonl", "w") as f:
                    for r in results:
                        f.write(json.dumps(r) + "\n")
                print(f"  💾 Saved checkpoint: {len(results)} results")
    
    # Final output
    tier_distribution = Counter(r["assigned_tier"] for r in results)
    dts_distribution = Counter(r["difficulty_to_solve"] for r in results)
    
    print(f"\n=== FINAL RESULTS ===")
    print(f"Total processed: {len(results)}")
    print(f"Time elapsed: {time.time() - start_time:.0f}s")
    print(f"Estimated cost: ${cost_tracker.estimated_cost_usd:.2f}")
    print(f"API calls: {cost_tracker.total_api_calls}")
    print(f"Tokens: {cost_tracker.total_tokens_in} in, {cost_tracker.total_tokens_out} out")
    print(f"\nTier distribution: {dict(tier_distribution)}")
    print(f"DTS distribution: {dict(dts_distribution)}")
    
    # Save final dataset
    with open("v33_labeled_dataset.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    
    # Save metadata
    metadata = {
        "version": "v3.3-empirical-labeling",
        "timestamp": time.time(),
        "total_prompts": len(results),
        "tier_distribution": dict(tier_distribution),
        "dts_distribution": {str(k): v for k, v in dts_distribution.items()},
        "cost": {
            "total_usd": round(cost_tracker.estimated_cost_usd, 4),
            "api_calls": cost_tracker.total_api_calls,
            "tokens_in": cost_tracker.total_tokens_in,
            "tokens_out": cost_tracker.total_tokens_out,
            "per_tier": cost_tracker.tier_costs,
        },
        "model_tiers": MODEL_TIERS,
        "oracle_model": ORACLE_MODEL,
        "batch_size": BATCH_SIZE,
        "concurrency": CONCURRENCY,
    }
    
    with open("v33_labeling_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nDataset saved: v33_labeled_dataset.jsonl")
    print(f"Metadata saved: v33_labeling_metadata.json")


if __name__ == "__main__":
    asyncio.run(main())
```

### Cost Estimation

| Tier | % of Prompts (estimated) | Prompts | Cost per Prompt | Total Cost |
|------|-------------------------|---------|----------------|------------|
| Trivial (GLM-4.5-Air) | 30% | 22,500 | $0.0000 | $0.00 |
| Light (GLM-4.7-Flash) | 25% | 18,750 | $0.0005 | $9.38 |
| Moderate (Qwen3.5-9B) | 20% | 15,000 | $0.0007 | $10.50 |
| Heavy (Qwen3.6-Plus) | 12% | 9,000 | $0.0005 | $4.50 |
| Intensive (Claude Sonnet) | 8% | 6,000 | $0.042 | $252.00 |
| Extreme (Claude Opus) | 5% | 3,750 | $0.070 | $262.50 |
| **Total** | **100%** | **75,000** | | **~$539** |

**Note:** 80%+ of cost comes from intensive/extensive tiers. If we optimize by skipping tiers for obviously complex prompts (using feature-based pre-filter), we can reduce to ~$200.

---

## 3. Statistical Validation

### Inter-Rater Reliability (IRR)

When using multiple Oracle models, calculate IRR using **Cohen's Kappa** for pairwise agreement and **Fleiss' Kappa** for multi-rater scenarios:

```python
from sklearn.metrics import cohen_kappa_score
import numpy as np

def calculate_irr(rater1_labels: list, rater2_labels: list, 
                  tier_order: list = ["trivial", "light", "moderate", "heavy", "intensive", "extreme"]) -> dict:
    """Calculate inter-rater reliability between two oracle evaluators."""
    
    # Convert to numeric
    tier_map = {t: i for i, t in enumerate(tier_order)}
    r1 = [tier_map[l] for l in rater1_labels]
    r2 = [tier_map[l] for l in rater2_labels]
    
    # Unweighted kappa (exact agreement)
    kappa_unweighted = cohen_kappa_score(r1, r2)
    
    # Weighted kappa (adjacent tiers count as partial agreement)
    weights = np.ones((len(tier_order), len(tier_order)))
    for i in range(len(tier_order)):
        for j in range(len(tier_order)):
            weights[i][j] = 1.0 - abs(i - j) / (len(tier_order) - 1)
    kappa_weighted = cohen_kappa_score(r1, r2, weights=weights)
    
    # Exact agreement rate
    exact_agreement = sum(1 for a, b in zip(r1, r2) if a == b) / len(r1)
    
    # Adjacent agreement (off by 1 tier counts as partial)
    adjacent_agreement = sum(1 for a, b in zip(r1, r2) if abs(a - b) <= 1) / len(r1)
    
    return {
        "kappa_unweighted": round(kappa_unweighted, 4),
        "kappa_weighted": round(kappa_weighted, 4),
        "exact_agreement": round(exact_agreement, 4),
        "adjacent_agreement": round(adjacent_agreement, 4),
        "n_samples": len(r1),
    }
```

**Interpretation:**
- Kappa ≥ 0.80: Excellent reliability
- Kappa 0.60-0.79: Good reliability
- Kappa 0.40-0.59: Moderate reliability
- Kappa < 0.40: Poor reliability (investigate Oracle prompt)

### Multi-Oracle Consensus Strategy

```python
def consensus_label(oracle_votes: list[str]) -> str:
    """Get consensus label from multiple oracle evaluations."""
    from collections import Counter
    vote_counts = Counter(oracle_votes)
    most_common, count = vote_counts.most_common(1)[0]
    
    # If majority agrees, use that label
    if count >= len(oracle_votes) // 2 + 1:
        return most_common
    
    # If split, use the middle tier (median)
    tier_order = ["trivial", "light", "moderate", "heavy", "intensive", "extreme"]
    tier_indices = [tier_order.index(v) for v in oracle_votes]
    median_idx = int(np.median(tier_indices))
    return tier_order[median_idx]
```

---

## 4. v3.3 Labeled Dataset Schema

```jsonl
// v33_labeled_dataset.jsonl (one JSON object per line)
{
  "prompt": "Design a distributed payment system...",
  "prompt_hash": "sha256:a3f8c2...",
  "prompt_length": 847,
  "source": "alpaca",
  
  // Primary label (empirical)
  "empirical_label": "intensive",
  "difficulty_to_solve": 4,  // Tier index where first 'Pass' occurred (0-5)
  
  // Evaluation trail
  "evaluation_trail": [
    {"tier": "trivial", "model": "glm-4.5-air", "oracle_score": 0.45, "pass": false},
    {"tier": "light", "model": "glm-4.7-flash", "oracle_score": 0.58, "pass": false},
    {"tier": "moderate", "model": "qwen3.5-9b", "oracle_score": 0.65, "pass": false},
    {"tier": "heavy", "model": "qwen3.6-plus", "oracle_score": 0.72, "pass": false},
    {"tier": "intensive", "model": "claude-sonnet-4.6", "oracle_score": 0.85, "pass": true},
    {"tier": "extreme", "model": "claude-opus-4.6", "oracle_score": null, "pass": null}
  ],
  
  // Features (for cascade training)
  "features": {
    "word_count": 147,
    "sentence_count": 8,
    "avg_word_length": 5.2,
    "has_code": 0.0,
    "has_question": 0.0,
    "has_imperative": 1.0,
    "technical_terms": 12,
    "question_technical": 0.0,
    "architecture": 1.0,
    "technical_design": 1.0,
    "multi_step": 1.0,
    "requires_context": 0.0,
    "domain_specificity": 0.82,
    "ambiguity_score": 0.05,
    "four_plus": 1.0
  },
  
  // Formula label (for comparison)
  "formula_label": "heavy",  // Original synthetic label
  
  // Agreement flag
  "label_agreement": false,  // empirical vs formula differ
  
  // Metadata
  "evaluated_at": "2026-05-08T18:00:00Z",
  "oracle_model": "llama-3.3-70b-instruct:free",
  "calibration_prompt": false,
  "cross_validated": true,
  "irr_score": 0.82
}
```

### Dataset Splits

```
v33_labeled_dataset.jsonl (75K samples)
  ├── v33_labeled_train.jsonl (60K, 80%)
  └── v33_labeled_test.jsonl (15K, 20%)
      ├── v33_labeled_test_stratified.jsonl (per-tier balanced)
      └── v33_labeled_test_calibration.jsonl (100 known-easy/hard prompts)
```

---

## 5. Data Validation Strategy

### Phase 1: Calibration (1K prompts)

1. Manually label 500 prompts (stratified across tiers)
2. Run Oracle on same 500 prompts
3. Calculate agreement rate between human and Oracle
4. If agreement < 0.70, recalibrate Oracle prompt
5. Iterate until Oracle ≥ 0.70 agreement with human

### Phase 2: Production (74K prompts)

1. Run calibrated Oracle on remaining 74K prompts
2. Cross-validate every 100th prompt with second Oracle
3. Monitor IRR in real-time; alert if drops below 0.60
4. Flag low-confidence predictions (oracle_score 0.60-0.75) for manual review

### Phase 3: Cascade Training

1. Use empirical labels as ground truth
2. Train v3.3 cascade on 60K train split
3. Evaluate on 15K test split
4. Compare cascade predictions vs empirical labels
5. Target: ≥ 85% agreement (up from 65.84% in v3.2)

### Quality Gates

| Metric | Target | Action if Failed |
|--------|--------|-----------------|
| Oracle-Human agreement (calibration) | ≥ 0.70 | Recalibrate Oracle prompt |
| Inter-Oracle Kappa | ≥ 0.80 | Investigate Oracle bias |
| Cascade-Empirical agreement | ≥ 0.85 | Retrain with more data |
| Per-tier accuracy | ≥ 0.80 | Collect more samples for weak tier |
| Cost per prompt | ≤ $0.01 | Optimize tier skipping |

---

## 6. Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Oracle model bias | Labels systematically skewed | Cross-validate with 2nd Oracle, calibration prompts |
| API rate limits | Slow processing, high cost | Batch processing, retry logic, rate limiting |
| RunPod worker failure | Job interruption | Checkpoint every 50 batches, resume from last checkpoint |
| Budget overrun | Credits exhausted | Cost tracking per batch, auto-stop at 80% budget |
| Prompt truncation | Long prompts lose context | Truncate to 4K tokens, log truncation events |
| Oracle hallucination | False pass/fail decisions | Anti-bias safeguards, technical accuracy floor |

---

## 7. Rollout Plan

| Week | Phase | Deliverable | Cost Estimate |
|------|-------|-------------|---------------|
| 1 | Calibration | Oracle prompt validated, 1K labeled | $5 |
| 2-3 | Production (trivial+light+moderate) | 45K prompts labeled | $30 |
| 3-4 | Production (heavy+intensive+extreme) | 30K prompts labeled | $500 |
| 5 | Cascade Training | v3.3 cascade trained, evaluated | $0 (CPU) |
| 6 | Validation | Full dataset + quality report | $50 |

**Total estimated cost: $585 for 75K empirically labeled prompts**

---

*Chief Scientist Design Doc v3.3 — 2026-05-08*
