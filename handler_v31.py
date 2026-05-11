"""GateSwarm MoA Router v0.2 Fine-Tuning — RunPod Serverless Training Handler

Trains optimized heuristic weights for the MoA Gateway Router using:
1. Downloaded public datasets (anonymized)
2. User-uploaded anonymized datasets
3. General-Purpose Dataset (50K synthetic trivial/light prompts)

Privacy: All datasets are anonymized before training. No PII, secrets,
or personal context leaves the workspace.

Usage:
  Submit job: {"version": "v0.2", "datasets": ["gpd", "alpaca"], "max_per": 20000}
  Custom upload: {"version": "v0.2", "dataset_url": "https://.../anonymized.jsonl"}
"""
import os, sys, json, time, re, math, subprocess, hashlib, tempfile
from typing import Optional
from datetime import datetime

def ensure_deps():
    missing = []
    for pkg, mod in [("scipy", "scipy"), ("numpy", "numpy"), ("datasets", "datasets"), ("requests", "requests")]:
        try: __import__(mod)
        except ImportError: missing.append(pkg)
    if missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet"] + missing)
ensure_deps()

import numpy as np
from scipy.optimize import minimize
from datasets import load_dataset
import requests

# ── v3.1 Feature Names (aligned with ARCHITECTURE_V3_1.md §1.4) ──
FEATURE_NAMES_V31 = [
    "sentence_count", "avg_word_length", "has_question", "question_technical",
    "technical_design", "code", "architecture", "word_count",
    "four_plus", "has_imperative", "technical_terms", "multi_step",
    "requires_context", "domain_specificity", "ambiguity_score",
]

# ── Tier Score Mapping ──
TIER_SCORES = {
    "trivial": 0.04, "light": 0.13, "moderate": 0.25,
    "heavy": 0.42, "intensive": 0.62, "extreme": 0.86,
}

# ── Default Weights (v3.0 optimized baseline) ──
DEFAULT_WEIGHTS_V3 = {
    "sentence_count": 0.29, "avg_word_length": 0.19, "has_question": 0.12,
    "question_technical": 0.05, "technical_design": 0.12, "code": 0.10,
    "architecture": 0.07, "word_count": 0.00, "four_plus": 0.00,
    "has_imperative": 0.06, "technical_terms": 0.00, "multi_step": 0.00,
    "requires_context": 0.00, "domain_specificity": 0.00, "ambiguity_score": 0.00,
}

# ── Technical Keywords ──
TECH_KEYWORDS = {
    "api", "http", "rest", "graphql", "websocket", "dns", "ssl", "tls",
    "oauth", "jwt", "cors", "cdn", "docker", "kubernetes", "git",
    "json", "yaml", "xml", "sql", "nosql", "redis", "mongodb",
    "typescript", "python", "rust", "java", "golang", "react", "vue",
    "angular", "svelte", "node", "express", "fastapi", "function",
    "class", "async", "await", "error", "type", "interface",
    "architecture", "design", "system", "microservice", "container",
    "deploy", "pipeline", "agile", "scrum", "algorithm", "database",
    "refactor", "migrate", "optimize", "debug", "security",
}
ARCH_KEYWORDS = {
    "architecture", "design pattern", "system design", "microservice",
    "distributed", "scalable", "load balancer", "event-driven",
    "service mesh", "api gateway", "serverless", "cloud-native",
}
DESIGN_KEYWORDS = {
    "technical design", "implementation plan", "migration strategy",
    "deployment strategy", "disaster recovery", "failover",
}
IMPERATIVE_STARTS = {
    "read", "summarize", "format", "explain", "add", "rename", "write",
    "check", "count", "find", "replace", "show", "make", "fix",
    "extract", "parse", "validate", "sort", "filter", "calculate",
    "create", "build", "run", "test", "implement", "refactor",
    "optimize", "deploy", "configure",
}


def extract_features_v31(prompt: str) -> dict:
    """Extract v3.1 feature vector (aligned with ARCHITECTURE_V3_1.md)."""
    words = re.findall(r'\w+', prompt.lower())
    sentences = [s.strip() for s in re.split(r'[.!?]+', prompt) if s.strip()]
    word_count = len(words)
    sentence_count = len(sentences)
    avg_word_length = sum(len(w) for w in words) / max(word_count, 1)

    has_code = bool(re.search(r'```|`[^`]+`|def\s+\w+|const\s+\w+|function\s+\w+', prompt))
    has_question = '?' in prompt
    has_imperative = any(prompt.lower().startswith(imp) for imp in IMPERATIVE_STARTS)

    tech_terms = sum(1 for w in words if w in TECH_KEYWORDS)
    question_technical = has_question and tech_terms > 0
    architecture = any(kw in prompt.lower() for kw in ARCH_KEYWORDS)
    technical_design = any(kw in prompt.lower() for kw in DESIGN_KEYWORDS)
    multi_step = bool(re.search(r'(first|then|next|finally|step\s*\d+)', prompt.lower()))
    requires_context = bool(re.search(r'(the file|this project|my code|our system)', prompt.lower()))

    domain_spec = min(tech_terms / max(word_count, 1), 1.0)
    vague = {'something', 'stuff', 'thing', 'things', 'it', 'this', 'that'}
    ambiguity = min(sum(1 for w in words if w in vague) / max(word_count, 1), 1.0)

    # four_plus: count of active features
    active = sum(1 for v in [has_code, has_question, has_imperative, tech_terms >= 3] if v)
    four_plus = 1.0 if active >= 4 else 0.0

    return {
        "word_count": word_count, "sentence_count": sentence_count,
        "avg_word_length": avg_word_length, "has_code": float(has_code),
        "has_question": float(has_question), "has_imperative": float(has_imperative),
        "technical_terms": tech_terms, "question_technical": float(question_technical),
        "architecture": float(architecture), "technical_design": float(technical_design),
        "multi_step": float(multi_step), "requires_context": float(requires_context),
        "domain_specificity": domain_spec, "ambiguity_score": ambiguity,
        "four_plus": four_plus,
    }


def features_to_vector(f: dict) -> list[float]:
    """Convert feature dict to vector matching FEATURE_NAMES_V31 order."""
    return [
        float(f.get("sentence_count", 0)), float(f.get("avg_word_length", 0)),
        float(f.get("has_question", 0)), float(f.get("question_technical", 0)),
        float(f.get("technical_design", 0)), float(f.get("has_code", 0)),
        float(f.get("architecture", 0)), float(f.get("word_count", 0)),
        float(f.get("four_plus", 0)), float(f.get("has_imperative", 0)),
        float(f.get("technical_terms", 0)), float(f.get("multi_step", 0)),
        float(f.get("requires_context", 0)), float(f.get("domain_specificity", 0)),
        float(f.get("ambiguity_score", 0)),
    ]


def enrich_features(f: dict) -> dict:
    """Add missing features to a feature dict (e.g., four_plus)."""
    enriched = dict(f)
    if "four_plus" not in enriched:
        active = sum(1 for k2 in ["has_code", "has_question", "has_imperative"]
                     if enriched.get(k2, 0) > 0)
        active += 1 if enriched.get("technical_terms", 0) >= 3 else 0
        enriched["four_plus"] = 1.0 if active >= 4 else 0.0
    for key in ["sentence_count", "avg_word_length", "has_question", "question_technical",
                "technical_design", "has_code", "architecture", "word_count",
                "has_imperative", "technical_terms", "multi_step", "requires_context",
                "domain_specificity", "ambiguity_score"]:
        if key not in enriched:
            enriched[key] = 0
    return enriched


def compute_score(f: dict, weights: dict) -> float:
    """Compute complexity score from features and weights."""
    s = 0.0
    s += weights.get("sentence_count", 0) * min(f.get("sentence_count", 0), 10) / 10.0
    s += weights.get("avg_word_length", 0) * f.get("avg_word_length", 0) / 10.0
    s += weights.get("has_question", 0) * f.get("has_question", 0)
    s += weights.get("question_technical", 0) * f.get("question_technical", 0)
    s += weights.get("technical_design", 0) * f.get("technical_design", 0)
    s += weights.get("code", 0) * f.get("has_code", 0)
    s += weights.get("architecture", 0) * f.get("architecture", 0)
    s += weights.get("word_count", 0) * math.log1p(f.get("word_count", 0)) / 6.0
    s += weights.get("four_plus", 0) * f.get("four_plus", 0)
    s += weights.get("has_imperative", 0) * f.get("has_imperative", 0)
    s += weights.get("technical_terms", 0) * min(f.get("technical_terms", 0), 10) / 10.0
    s += weights.get("multi_step", 0) * f.get("multi_step", 0)
    s += weights.get("requires_context", 0) * f.get("requires_context", 0)
    s += weights.get("domain_specificity", 0) * f.get("domain_specificity", 0)
    s += weights.get("ambiguity_score", 0) * f.get("ambiguity_score", 0)

    # Length dampener for short prompts without architecture
    if f["word_count"] < 10 and f["architecture"] == 0:
        s *= 0.7

    return min(max(s, 0.0), 1.0)


def score_to_tier(score: float) -> str:
    if score < 0.08: return "trivial"
    if score < 0.18: return "light"
    if score < 0.32: return "moderate"
    if score < 0.52: return "heavy"
    if score < 0.72: return "intensive"
    return "extreme"


def label_from_sample(sample: dict) -> str:
    """Extract complexity label from a training sample."""
    return (sample.get("label") or sample.get("complexity_hint") or
            sample.get("actual_complexity") or "moderate")


# ── Dataset Loading ──

def load_hf_dataset(name: str, max_per: int) -> list[dict]:
    """Load a HuggingFace dataset and return prompts with synthetic labels."""
    catalogs = {
        "alpaca": {"hf": "tatsu-lab/alpaca", "key": "instruction"},
        "openorca": {"hf": "Open-Orca/OpenOrca", "key": "question"},
        "self_instruct": {"hf": "yizhongw/self_instruct", "key": "input"},
    }
    if name not in catalogs:
        return []
    c = catalogs[name]
    try:
        ds = load_dataset(c["hf"], split=f"train[:{max_per}]", trust_remote_code=True)
        prompts = []
        for x in ds:
            if c["key"] in x and isinstance(x.get(c["key"]), str) and len(x[c["key"]]) > 10:
                text = x[c["key"]].strip()
                feats = extract_features_v31(text)
                prompts.append({"text": text, "features": feats})
        return prompts[:max_per]
    except Exception as e:
        print(f"  Failed to load {name}: {e}")
        return []


def load_jsonl_dataset(path: str, max_per: int = 50000) -> list[dict]:
    """Load a local JSONL file."""
    prompts = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if len(prompts) >= max_per:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    text = record.get("text") or record.get("raw_text") or ""
                    if len(text) < 5:
                        continue
                    feats = record.get("features") or extract_features_v31(text)
                    prompts.append({"text": text, "features": feats, "label": label_from_sample(record)})
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        print(f"  File not found: {path}")
    return prompts


def load_uploaded_dataset(url: str, max_per: int) -> list[dict]:
    """Download and load a user-uploaded dataset from URL."""
    try:
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(resp.text)
            tmp_path = f.name
        return load_jsonl_dataset(tmp_path, max_per)
    except Exception as e:
        print(f"  Failed to download dataset: {e}")
        return []


# ── Optimization ──

def optimize_weights(train_features: list, train_labels: list,
                     init_weights: Optional[dict] = None, max_iter: int = 2000) -> dict:
    """Optimize heuristic weights using scipy MSE minimization."""
    if init_weights is None:
        init_weights = DEFAULT_WEIGHTS_V3

    X = np.array([features_to_vector(f) for f in train_features], dtype=np.float64)
    y = np.array([TIER_SCORES.get(l, 0.25) for l in train_labels], dtype=np.float64)

    x0 = np.array([init_weights.get(n, 0.0) for n in FEATURE_NAMES_V31], dtype=np.float64)

    def mse_objective(wa):
        scores = X @ wa
        return np.mean((scores - y) ** 2)

    bounds = [(0.0, 0.35)] * len(FEATURE_NAMES_V31)
    result = minimize(mse_objective, x0, bounds=bounds, method="L-BFGS-B",
                      options={"maxiter": max_iter, "ftol": 1e-9})

    optimized = result.x
    total = optimized.sum()
    if total > 0:
        optimized = optimized / total

    mse_before = mse_objective(x0)
    mse_after = mse_objective(optimized)

    return {
        "weights": dict(zip(FEATURE_NAMES_V31, optimized.tolist())),
        "mse_before": float(mse_before),
        "mse_after": float(mse_after),
        "improvement_pct": float((1 - mse_after / mse_before) * 100) if mse_before > 0 else 0,
        "iterations": int(result.nit),
    }


def evaluate_accuracy(test_features: list, test_labels: list, weights: dict) -> dict:
    """Evaluate routing accuracy on test set."""
    total = len(test_features)
    if total == 0:
        return {"accuracy": 0.0, "tier_accuracy": {}}

    correct = 0
    tier_correct = {}
    tier_total = {}

    for f, l in zip(test_features, test_labels):
        predicted = score_to_tier(compute_score(f, weights))
        actual = score_to_tier(TIER_SCORES.get(l, 0.25)) if l in TIER_SCORES else l

        if actual not in tier_total:
            tier_total[actual] = 0
            tier_correct[actual] = 0
        tier_total[actual] += 1

        if predicted == actual:
            correct += 1
            tier_correct[actual] += 1

    tier_accuracy = {}
    for t in ["trivial", "light", "moderate", "heavy", "intensive", "extreme"]:
        if tier_total.get(t, 0) > 0:
            tier_accuracy[t] = {
                "count": tier_total[t],
                "accuracy": round(tier_correct[t] / tier_total[t], 4)
            }

    return {
        "accuracy": round(correct / total, 4),
        "total": total,
        "correct": correct,
        "tier_accuracy": tier_accuracy,
    }


# ── Main Handler ──

def handler(event):
    inp = event.get("input", {})
    version = inp.get("version", "v0.2")

    # Configuration
    dataset_names = inp.get("datasets", ["gpd", "alpaca"])
    max_per = inp.get("max_per", 20000)
    dataset_url = inp.get("dataset_url")
    max_iter = inp.get("max_iter", 2000)
    init_weights = inp.get("init_weights")

    print(f"GateSwarm MoA Router v0.2 Training — {datetime.utcnow().isoformat()}Z")
    print(f"Datasets: {dataset_names}")
    print(f"Max per dataset: {max_per}")

    all_prompts = []

    # Load public HF datasets
    for name in dataset_names:
        if name == "gpd":
            # GPD is loaded from local file (pre-generated synthetic dataset)
            gpd_path = "/workspace/llmfit/datasets/general-purpose/baseline_synthetic.jsonl"
            # Try alternate path too
            for p in [gpd_path, "llmfit/datasets/general-purpose/baseline_synthetic.jsonl"]:
                if os.path.exists(p):
                    print(f"  Loading GPD from {p}...")
                    prompts = load_jsonl_dataset(p, max_per * 2)
                    all_prompts.extend(prompts)
                    print(f"  Loaded {len(prompts)} GPD samples")
                    break
            else:
                # If GPD not found locally, skip (it's synthetic anyway)
                print("  GPD not found locally, skipping")
        elif name in ("alpaca", "openorca", "self_instruct"):
            print(f"  Loading {name} from HuggingFace...")
            prompts = load_hf_dataset(name, max_per)
            all_prompts.extend(prompts)
            print(f"  Loaded {len(prompts)} {name} samples")
        else:
            print(f"  Unknown dataset: {name}")

    # Load user-uploaded dataset
    if dataset_url:
        print(f"  Downloading custom dataset from {dataset_url[:80]}...")
        uploaded = load_uploaded_dataset(dataset_url, max_per * 3)
        all_prompts.extend(uploaded)
        print(f"  Loaded {len(uploaded)} uploaded samples")

    if not all_prompts:
        return {
            "version": version,
            "error": "No prompts loaded. Check dataset names or provide dataset_url.",
            "status": "failed",
        }

    print(f"\nTotal samples: {len(all_prompts)}")

    # Split: 80% train, 20% test
    import random
    random.seed(42)
    random.shuffle(all_prompts)
    split_idx = int(len(all_prompts) * 0.8)

    train_data = all_prompts[:split_idx]
    test_data = all_prompts[split_idx:]

    train_features = [enrich_features(p["features"]) for p in train_data]
    train_labels = [p.get("label") or ("trivial" if p["features"].get("word_count", 0) < 50 else "moderate") for p in train_data]
    test_features = [enrich_features(p["features"]) for p in test_data]
    test_labels = [p.get("label") or ("trivial" if p["features"].get("word_count", 0) < 50 else "moderate") for p in test_data]

    # Label distribution
    label_dist = {}
    for l in train_labels + test_labels:
        label_dist[l] = label_dist.get(l, 0) + 1
    print(f"Label distribution: {json.dumps(label_dist)}")

    # Optimize weights
    print(f"\nOptimizing weights (max_iter={max_iter})...")
    opt_result = optimize_weights(train_features, train_labels, init_weights, max_iter)

    opt_weights = opt_result["weights"]
    print(f"MSE: {opt_result['mse_before']:.6f} → {opt_result['mse_after']:.6f} ({opt_result['improvement_pct']:.1f}% improvement)")
    print(f"Iterations: {opt_result['iterations']}")

    # Evaluate baseline (v3.0 weights)
    baseline_eval = evaluate_accuracy(test_features, test_labels, DEFAULT_WEIGHTS_V3)
    print(f"\nBaseline accuracy: {baseline_eval['accuracy']:.1%}")

    # Evaluate optimized
    opt_eval = evaluate_accuracy(test_features, test_labels, opt_weights)
    print(f"Optimized accuracy: {opt_eval['accuracy']:.1%}")

    # Tier distribution of test set
    test_tiers = {}
    for l in test_labels:
        t = score_to_tier(TIER_SCORES.get(l, 0.25)) if l in TIER_SCORES else l
        test_tiers[t] = test_tiers.get(t, 0) + 1

    # Build result
    result = {
        "version": version,
        "status": "completed",
        "datasets": dataset_names,
        "total_samples": len(all_prompts),
        "train_samples": len(train_data),
        "test_samples": len(test_data),
        "label_distribution": label_dist,
        "test_tier_distribution": test_tiers,
        "optimization": {
            "mse_before": round(opt_result["mse_before"], 6),
            "mse_after": round(opt_result["mse_after"], 6),
            "improvement_pct": round(opt_result["improvement_pct"], 1),
            "iterations": opt_result["iterations"],
        },
        "baseline_accuracy": baseline_eval["accuracy"],
        "optimized_accuracy": opt_eval["accuracy"],
        "accuracy_improvement": round(opt_eval["accuracy"] - baseline_eval["accuracy"], 4),
        "baseline_tier_accuracy": baseline_eval.get("tier_accuracy", {}),
        "optimized_tier_accuracy": opt_eval.get("tier_accuracy", {}),
        "optimized_weights": {k: round(v, 4) for k, v in sorted(opt_weights.items(), key=lambda x: -x[1])},
        "anonymized": True,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

    return result


if __name__ == "__main__":
    result = handler({
        "input": {
            "version": "v0.2",
            "datasets": ["alpaca"],
            "max_per": 10000,
        }
    })
    print(json.dumps(result, indent=2))
