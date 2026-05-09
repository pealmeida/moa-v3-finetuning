"""MoA v3.3 — Label Correction + Weight Export

Runs on RunPod serverless to:
1. Load Alpaca + OpenOrca datasets
2. Extract features + formula labels
3. Train cascade on balanced data (corrected labels)
4. Return FULL cascade weights in response (for rebuilding inference image)

Output includes `cascade_weights_json` — drop-in replacement for v32_cascade_weights.json
"""
import json, time, math, random, re, sys, os
from datetime import datetime, timezone
from collections import Counter

def ensure_deps():
    for pkg, mod in [("scikit-learn", "sklearn"), ("numpy", "numpy"), ("datasets", "datasets")]:
        try: __import__(mod)
        except ImportError:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", pkg])
ensure_deps()

import numpy as np
from sklearn.linear_model import LogisticRegression
from datasets import load_dataset
import runpod

# ── Feature Extraction (identical to inference handler) ──
FEATURE_NAMES = [
    "sentence_count", "avg_word_length", "has_question", "question_technical",
    "technical_design", "code", "architecture", "word_count",
    "four_plus", "has_imperative", "technical_terms", "multi_step",
    "requires_context", "domain_specificity", "ambiguity_score",
]
TIERS = ["trivial", "light", "moderate", "heavy", "intensive", "extreme"]

TECH_KEYWORDS = {
    "api", "http", "rest", "graphql", "websocket", "dns", "ssl", "tls",
    "oauth", "jwt", "cors", "cdn", "docker", "kubernetes", "git",
    "json", "yaml", "xml", "sql", "nosql", "redis", "mongodb",
    "typescript", "python", "rust", "java", "react", "vue", "angular",
    "svelte", "node", "express", "fastapi", "function", "class",
    "async", "await", "error", "type", "interface", "architecture",
    "design", "system", "microservice", "container", "deploy", "pipeline",
    "algorithm", "database", "refactor", "optimize", "debug", "security",
}
ARCH_KEYWORDS = {"architecture", "design pattern", "system design", "microservice",
    "distributed", "scalable", "load balancer", "event-driven", "service mesh",
    "api gateway", "serverless", "cloud-native", "infrastructure"}
DESIGN_KEYWORDS = {"technical design", "implementation plan", "migration strategy",
    "deployment strategy", "disaster recovery", "failover"}
IMPERATIVE_STARTS = {
    "read", "summarize", "format", "explain", "add", "rename", "write",
    "check", "count", "find", "replace", "show", "make", "fix",
    "extract", "parse", "validate", "sort", "filter", "calculate",
    "create", "build", "run", "test", "implement", "refactor",
    "optimize", "deploy", "configure",
}


def extract_features(text: str) -> dict:
    words = re.findall(r'\w+', text.lower())
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    wc = len(words)
    sc = len(sentences)
    awl = sum(len(w) for w in words) / max(wc, 1)
    has_code = bool(re.search(r'```|`[^`]+`|def\s+\w+|const\s+\w+|function\s+\w+|import\s+', text))
    has_question = '?' in text
    has_imperative = any(text.lower().startswith(imp) for imp in IMPERATIVE_STARTS)
    tech_terms = sum(1 for w in words if w in TECH_KEYWORDS)
    question_technical = has_question and tech_terms > 0
    architecture = any(kw in text.lower() for kw in ARCH_KEYWORDS)
    technical_design = any(kw in text.lower() for kw in DESIGN_KEYWORDS)
    multi_step = bool(re.search(r'(first|then|next|finally|step\s*\d+)', text.lower()))
    requires_context = bool(re.search(r'(the file|this project|my code|our system|given that|consider)', text.lower()))
    domain_spec = min(tech_terms / max(wc, 1), 1.0)
    vague = {'something', 'stuff', 'thing', 'things', 'it', 'this', 'that'}
    ambiguity = min(sum(1 for w in words if w in vague) / max(wc, 1), 1.0)
    active = sum(1 for v in [has_code, has_question, has_imperative, tech_terms >= 3] if v)
    four_plus = 1.0 if active >= 4 else 0.0
    return {
        "word_count": wc, "sentence_count": sc, "avg_word_length": round(awl, 3),
        "has_code": float(has_code), "has_question": float(has_question),
        "has_imperative": float(has_imperative), "technical_terms": tech_terms,
        "question_technical": float(question_technical),
        "architecture": float(architecture), "technical_design": float(technical_design),
        "multi_step": float(multi_step), "requires_context": float(requires_context),
        "domain_specificity": round(domain_spec, 3), "ambiguity_score": round(ambiguity, 3),
        "four_plus": four_plus,
    }


def features_to_vector(f: dict) -> np.ndarray:
    return np.array([float(f.get(n, 0)) for n in FEATURE_NAMES], dtype=np.float64)


def label_formula(text: str) -> str:
    t = text.lower()
    wc = len(t.split())
    signals = 0
    if '?' in text: signals += 1
    if any(k in t for k in ["code", "function", "def ", "class ", "import ", "``", "fn ", "const "]): signals += 1
    if any(t.startswith(k) for k in ["write ", "create ", "build ", "implement ", "generate ", "fix ", "debug ", "optimize ", "explain ", "analyze ", "describe ", "design "]): signals += 1
    if re.search(r'[0-9]+[\s]*[+\-*/=]', text): signals += 1
    if any(k in t for k in ["first ", "then ", "finally", "step ", "part ", "section ", "also ", "and also"]): signals += 1
    if any(k in t for k in ["must ", "should ", "required ", "only ", "don't", "cannot ", "limit ", "maximum ", "minimum "]): signals += 1
    if any(k in t for k in ["given ", "consider ", "assume ", "suppose ", "based on ", "according to ", "using the ", "from the ", "in the context "]): signals += 1
    if any(k in t for k in ["architecture", "design pattern", "system design", "microservice", "scalable", "distributed"]): signals += 1
    if any(k in t for k in ["technical design", "implementation plan", "migration strategy", "deployment", "pipeline", "schema", "database"]): signals += 1
    has_context = 1.0 if any(k in t for k in ["given ", "consider ", "assume ", "suppose ", "based on ", "according to ", "using the ", "from the ", "in the context "]) else 0.0
    score = signals * 0.15 + math.log1p(wc) * 0.08 + has_context * 0.1
    if score < 0.10: return "trivial"
    if score < 0.20: return "light"
    if score < 0.35: return "moderate"
    if score < 0.50: return "heavy"
    if score < 0.65: return "intensive"
    return "extreme"


def predict_with_confidence(models: dict, features: dict) -> tuple:
    """Predict tier + confidence using trained cascade."""
    x = features_to_vector(features)
    for tier in ["trivial", "light", "moderate", "heavy", "intensive"]:
        if tier not in models:
            continue
        model = models[tier]
        logit = float(x @ model["weights_vec"] + model["intercept"])
        prob = 1.0 / (1.0 + math.exp(-logit))
        if prob > 0.5:
            return tier, round(prob, 4)
    return "extreme", 0.5


def train_cascade(train_data: list[dict]) -> tuple[dict, dict]:
    """Train cascade classifiers and return (models_with_vectors, export_data)."""
    all_features = np.array([features_to_vector(s["features"]) for s in train_data], dtype=np.float64)
    all_labels = np.array([s["formula_label"] for s in train_data])

    models = {}
    export = {}
    results = {}

    for tier in ["trivial", "light", "moderate", "heavy", "intensive"]:
        if tier == "trivial":
            mask_pos = all_labels == tier
            mask_neg = ~mask_pos
        else:
            prev = ["trivial", "light", "moderate", "heavy"][:["trivial", "light", "moderate", "heavy"].index(tier)]
            mask_excluded = np.zeros(len(all_labels), dtype=bool)
            for pt in prev:
                mask_excluded |= all_labels == pt
            remaining = ~mask_excluded
            mask_pos = remaining & (all_labels == tier)
            mask_neg = remaining & (all_labels != tier)

        X_pos = all_features[mask_pos]
        X_neg = all_features[mask_neg]
        n_min = min(len(X_pos), len(X_neg))
        if n_min < 10:
            results[tier] = {"success": False, "reason": f"too few: {n_min}"}
            continue

        rng = np.random.RandomState(42)
        pos_idx = rng.permutation(len(X_pos))[:n_min]
        neg_idx = rng.permutation(len(X_neg))[:n_min]
        X = np.vstack([X_pos[pos_idx], X_neg[neg_idx]])
        y = np.concatenate([np.ones(n_min), np.zeros(n_min)])

        model = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs', random_state=42)
        model.fit(X, y)
        acc = float((model.predict(X) == y).mean())

        weights_dict = dict(zip(FEATURE_NAMES, [round(float(w), 4) for w in model.coef_[0]]))
        intercept = round(float(model.intercept_[0]), 4)

        models[tier] = {
            "weights_vec": np.array(model.coef_[0], dtype=np.float64),
            "intercept": float(model.intercept_[0]),
        }
        export[tier] = {
            "accuracy": round(acc, 4),
            "n_samples": n_min * 2,
            "intercept": intercept,
            "weights": weights_dict,
        }
        results[tier] = {"success": True, "accuracy": round(acc, 4), "n_samples": n_min * 2}

    return models, export, results


def evaluate_cascade(models: dict, test_data: list[dict]) -> dict:
    """Evaluate cascade on test data."""
    correct = 0
    total = len(test_data)
    tier_correct = Counter()
    tier_total = Counter()

    for s in test_data:
        pred, conf = predict_with_confidence(models, s["features"])
        actual = s["formula_label"]
        tier_total[actual] += 1
        if pred == actual:
            correct += 1
            tier_correct[actual] += 1

    per_tier = {}
    for tier in TIERS:
        t = tier_total.get(tier, 0)
        if t > 0:
            per_tier[tier] = {"accuracy": round(tier_correct.get(tier, 0) / t, 4), "samples": t}
        else:
            per_tier[tier] = {"accuracy": 0, "samples": 0}

    return {
        "accuracy": round(correct / max(total, 1), 4),
        "correct": correct,
        "total": total,
        "per_tier": per_tier,
    }


def handler(event):
    """RunPod serverless handler for v3.3 label correction + weight export."""
    t0 = time.time()
    inp = event.get("input", {})
    max_per = inp.get("max_per", 8000)

    print(f"=== MoA v3.3 Label Correction + Weight Export ===")
    print(f"  max_per: {max_per}")

    # 1. Load Alpaca
    print("  Loading Alpaca...")
    ds_alpaca = load_dataset("tatsu-lab/alpaca", split="train", streaming=True)
    alpaca = []
    for i, row in enumerate(ds_alpaca):
        if i >= max_per: break
        text = row.get("instruction", "") or row.get("input", "") or ""
        if not text or len(text) < 3: continue
        feat = extract_features(text)
        label = label_formula(text)
        alpaca.append({"text": text, "features": feat, "formula_label": label, "source": "alpaca"})

    # 2. Load OpenOrca
    print("  Loading OpenOrca...")
    try:
        ds_orca = load_dataset("Open-Orca/OpenOrca", split="train", streaming=True)
        openorca = []
        for i, row in enumerate(ds_orca):
            if i >= max_per: break
            text = row.get("question", "") or ""
            if not text or len(text) < 3: continue
            feat = extract_features(text)
            label = label_formula(text)
            openorca.append({"text": text, "features": feat, "formula_label": label, "source": "openorca"})
    except Exception as e:
        print(f"  OpenOrca load failed: {e}, using Alpaca only")
        openorca = []

    # 3. Combine
    all_samples = alpaca + openorca
    random.seed(42)
    random.shuffle(all_samples)

    label_dist = Counter(s["formula_label"] for s in all_samples)
    print(f"  Total: {len(all_samples)} | Labels: {dict(label_dist)}")

    split_idx = int(len(all_samples) * 0.8)
    train_data = all_samples[:split_idx]
    test_data = all_samples[split_idx:]

    # 4. Train cascade
    print("  Training cascade...")
    models, export, train_results = train_cascade(train_data)

    for tier, r in train_results.items():
        if r.get("success"):
            print(f"    {tier}: acc={r['accuracy']:.1%} n={r['n_samples']}")

    # 5. Evaluate
    print("  Evaluating...")
    eval_result = evaluate_cascade(models, test_data)
    print(f"  Overall: {eval_result['accuracy']:.1%} ({eval_result['correct']}/{eval_result['total']})")
    for tier in TIERS:
        t = eval_result["per_tier"].get(tier, {})
        if t.get("samples", 0) > 0:
            print(f"    {tier}: {t['accuracy']:.1%} ({t['samples']})")

    # 6. Build exportable weights (same format as v32_cascade_weights.json)
    cascade_weights = {
        "version": "v3.3-cascade-corrected",
        "trained": datetime.now(timezone.utc).isoformat() + "Z",
        "dataset": f"{len(all_samples)} (Alpaca{'+OpenOrca' if openorca else ''})",
        "overall_accuracy": eval_result["accuracy"],
        "feature_names": FEATURE_NAMES,
        "classifiers": export,
        "tier_boundaries": [0.08, 0.18, 0.32, 0.52, 0.72],
        "tier_accuracy": {t: eval_result["per_tier"].get(t, {}) for t in TIERS},
        "improvement_notes": {
            "v32_issue": "Light classifier too aggressive — 5/5 test prompts classified as light",
            "v33_fix": "Retrained with balanced per-tier sampling + corrected formula labels",
            "expected": "Better separation of moderate/heavy/intensive/extreme tiers",
        },
    }

    elapsed = round(time.time() - t0, 1)

    return {
        "version": "v3.3-label-correction",
        "status": "completed",
        "elapsed_seconds": elapsed,
        "dataset_stats": {
            "total": len(all_samples),
            "train": len(train_data),
            "test": len(test_data),
            "sources": {"alpaca": len(alpaca), "openorca": len(openorca)},
            "label_distribution": dict(label_dist),
        },
        "cascade_performance": eval_result,
        "cascade_weights_json": cascade_weights,
        "next_step": "Save cascade_weights_json as v33_cascade_weights.json, rebuild inference image",
    }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
