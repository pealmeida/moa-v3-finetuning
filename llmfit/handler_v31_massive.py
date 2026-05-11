"""GateSwarm MoA Router v0.2 — Massive Multi-Dataset Per-Tier Accuracy Test

Trains on 3 combined datasets for FULL 6-tier coverage:
1. GPD (50K synthetic) — trivial + light
2. Alpaca (20K) — moderate + heavy
3. OpenOrca (20K) — heavy + intensive + extreme

Produces per-tier accuracy for ALL 6 tiers.
"""
import os, sys, json, time, re, math, subprocess, tempfile, hashlib, random
from datetime import datetime
from typing import Optional
from collections import Counter

def ensure_deps():
    missing = []
    for pkg, mod in [("scipy", "scipy"), ("numpy", "numpy"), ("datasets", "datasets")]:
        try: __import__(mod)
        except ImportError: missing.append(pkg)
    if missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet"] + missing)
ensure_deps()

import numpy as np
from scipy.optimize import minimize
from datasets import load_dataset

# ── Constants ──
FEATURE_NAMES = [
    "sentence_count", "avg_word_length", "has_question", "question_technical",
    "technical_design", "code", "architecture", "word_count",
    "four_plus", "has_imperative", "technical_terms", "multi_step",
    "requires_context", "domain_specificity", "ambiguity_score",
]

TIER_SCORES = {
    "trivial": 0.04, "light": 0.13, "moderate": 0.25,
    "heavy": 0.42, "intensive": 0.62, "extreme": 0.86,
}
TIER_NAMES = ["trivial", "light", "moderate", "heavy", "intensive", "extreme"]

DEFAULT_WEIGHTS = {
    "sentence_count": 0.29, "avg_word_length": 0.19, "has_question": 0.12,
    "question_technical": 0.05, "technical_design": 0.12, "code": 0.10,
    "architecture": 0.07, "word_count": 0.00, "four_plus": 0.00,
    "has_imperative": 0.06, "technical_terms": 0.00, "multi_step": 0.00,
    "requires_context": 0.00, "domain_specificity": 0.00, "ambiguity_score": 0.00,
}

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
    """Extract v3.1 15-feature vector."""
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
        "word_count": wc, "sentence_count": sc, "avg_word_length": awl,
        "has_code": float(has_code), "has_question": float(has_question),
        "has_imperative": float(has_imperative), "technical_terms": tech_terms,
        "question_technical": float(question_technical),
        "architecture": float(architecture), "technical_design": float(technical_design),
        "multi_step": float(multi_step), "requires_context": float(requires_context),
        "domain_specificity": domain_spec, "ambiguity_score": ambiguity,
        "four_plus": four_plus,
    }


def enrich_features(f: dict) -> dict:
    """Ensure all 15 features present."""
    enriched = dict(f)
    for key in FEATURE_NAMES:
        if key not in enriched:
            enriched[key] = 0
    return enriched


def compute_score(f: dict, weights: dict) -> float:
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
    if f.get("word_count", 0) < 10 and f.get("architecture", 0) == 0:
        s *= 0.7
    return min(max(s, 0.0), 1.0)


def score_to_tier(score: float) -> str:
    if score < 0.08: return "trivial"
    if score < 0.18: return "light"
    if score < 0.32: return "moderate"
    if score < 0.52: return "heavy"
    if score < 0.72: return "intensive"
    return "extreme"


def synthetic_complexity_v3(text: str) -> float:
    """v3.0 synthetic complexity (signal count + length + lexical richness).
    This is independent of the feature weights used by the optimizer."""
    t = text.lower()
    w = t.split()
    wc = len(w)
    sig = 0
    if '?' in t: sig += 1
    if any(k in t for k in ["code", "function", "def ", "class ", "import ", "``"]): sig += 1
    if any(k in t for k in ["write", "create", "build", "implement", "generate", "fix", "debug", "optimize"]): sig += 1
    if re.search(r'\d+[\s]*[+\-*/=]', text): sig += 1
    if any(k in t for k in ["first", "then", "finally", "step", "part", "section"]): sig += 1
    if any(k in t for k in ["must", "should", "required", "only", "don't", "cannot", "limit"]): sig += 1
    if any(k in t for k in ["given", "consider", "assume", "suppose", "based on", "according to"]): sig += 1
    sc_count = len(re.split(r'[.!?]+', text))
    ri = len(set(w)) / wc if wc > 0 else 0
    return min(0.1 * math.log1p(wc) / math.log1p(200) + min(sig * 0.1, 0.9) + min(sc_count * 0.02, 0.1) + ri * 0.05, 1.0)


def label_from_explicit_signals(text: str, feats: dict) -> str:
    """Label based on explicit signal counting.
    More reliable than synthetic complexity for mixed datasets.
    - 0 signals → trivial
    - 1 signal → light
    - 2-3 signals → moderate
    - 4-5 signals → heavy
    - 6+ signals → intensive"""
    t = text.lower()
    signals = 0
    # Signal 1: Question
    if '?' in text: signals += 1
    # Signal 2: Code
    if any(k in t for k in ["code", "function", "def ", "class ", "import ", "``", "fn ", "const "]): signals += 1
    # Signal 3: Imperative action
    if any(t.startswith(k) for k in ["write ", "create ", "build ", "implement ", "generate ", "fix ", "debug ", "optimize ", "explain ", "analyze ", "describe ", "design "]): signals += 1
    # Signal 4: Math
    if re.search(r'\d+[\s]*[+\-*/=]', text): signals += 1
    # Signal 5: Multi-step
    if any(k in t for k in ["first ", "then ", "finally", "step ", "part ", "section ", "also ", "and also"]): signals += 1
    # Signal 6: Constraints
    if any(k in t for k in ["must ", "should ", "required ", "only ", "don't", "cannot ", "limit ", "maximum ", "minimum "]): signals += 1
    # Signal 7: Context dependency
    if any(k in t for k in ["given ", "consider ", "assume ", "suppose ", "based on ", "according to ", "using the ", "from the "]): signals += 1
    # Signal 8: Architecture/system design
    if any(k in t for k in ["architecture", "design pattern", "system design", "microservice", "scalable", "distributed"]): signals += 1
    # Signal 9: Technical design
    if any(k in t for k in ["technical design", "implementation plan", "migration strategy", "deployment", "pipeline", "schema", "database"]): signals += 1
    # Signal 10: Long text (>100 words)
    if len(t.split()) > 100: signals += 1

    if signals == 0: return "trivial"
    if signals == 1: return "light"
    if signals <= 3: return "moderate"
    if signals <= 5: return "heavy"
    return "intensive"



# ── GPD Synthetic Generator (inline for serverless) ──
GPD_TRIVIAL = [
    "What is {concept}?", "Explain {concept} briefly", "Define {concept}",
    "What does {acronym} stand for?", "Is {thing} a type of {category}?",
    "How many {unit} in a {thing}?", "When was {event}?",
    "List the top {n} {items}", "Convert {value} {from_unit} to {to_unit}",
    "What is the capital of {place}?", "How do you say {word} in {language}?",
    "Spell {word}", "What is {number} plus {number2}?",
    "Say hello", "Hi there", "How are you?", "What time is it?",
]
GPD_LIGHT = [
    "Read the file at {path} and tell me what it does",
    "Summarize the following text: {short_text}",
    "Format this code: {code_snippet}",
    "What's wrong with this error: {error_message}",
    "Add a docstring to: {function_signature}",
    "Write a test for: {function_signature}",
    "What does this command do: {shell_cmd}",
    "Fix the typo in: {code_snippet}",
    "Parse this JSON: {json_snippet}",
    "Sort this list: {list_data}",
]

GPD_CONCEPTS = ["API", "REST", "GraphQL", "Docker", "Git", "JSON", "SQL", "Redis", "React", "Python", "TypeScript", "Rust"]
GPD_ACRONYMS = [("API", "Application Programming Interface"), ("HTTP", "Hypertext Transfer Protocol"), ("JSON", "JavaScript Object Notation"), ("SQL", "Structured Query Language"), ("DNS", "Domain Name System")]
GPD_PATHS = ["src/main.py", "config/settings.json", "README.md", ".env", "package.json"]
GPD_ERRORS = ["TypeError: undefined is not a function", "SyntaxError: unexpected token", "ValueError: invalid literal", "KeyError: 'missing_key'", "ModuleNotFoundError: No module named 'requests'"]
GPD_FUNCS = ["def calculate_total(items: list[float]) -> float", "async def fetch_data(url: str) -> dict", "def parse_json(text: str) -> Optional[dict]"]
GPD_CMDS = ["ls -la", "grep -r 'pattern' .", "find . -name '*.py'", "ps aux | grep python", "curl -s https://example.com"]

def _gen_gpd_trivial(rng, idx):
    t = rng.choice(GPD_TRIVIAL)
    kw = {"concept": rng.choice(GPD_CONCEPTS), "acronym": rng.choice(GPD_ACRONYMS)[0],
          "thing": rng.choice(["byte", "kilobyte", "pixel", "vector"]),
          "category": rng.choice(["programming language", "framework", "database"]),
          "unit": rng.choice(["bytes", "seconds", "meters"]),
          "event": rng.choice(["the internet invented", "Python created", "Git released"]),
          "n": str(rng.randint(3, 10)), "items": rng.choice(["databases", "frameworks"]),
          "value": str(rng.randint(1, 1000)), "from_unit": rng.choice(["bytes", "KB"]),
          "to_unit": rng.choice(["KB", "MB"]), "place": rng.choice(["Brazil", "Japan", "Germany"]),
          "language": rng.choice(["Spanish", "French", "German"]),
          "word": rng.choice(["happy", "fast", "beautiful"]),
          "number": str(rng.randint(1, 100)), "number2": str(rng.randint(1, 100))}
    text = t.format(**kw)
    return re.sub(r'\{[^}]+\}', 'something', text)

def _gen_gpd_light(rng, idx):
    t = rng.choice(GPD_LIGHT)
    kw = {"path": rng.choice(GPD_PATHS), "short_text": "Python is a high-level programming language known for readability.",
          "code_snippet": "def hello(): print('hello')", "error_message": rng.choice(GPD_ERRORS),
          "function_signature": rng.choice(GPD_FUNCS), "shell_cmd": rng.choice(GPD_CMDS),
          "json_snippet": '{"key": "value", "count": 42}', "list_data": "[3, 1, 4, 1, 5, 9]"}
    text = t.format(**kw)
    return re.sub(r'\{[^}]+\}', 'data', text)


def generate_gpd_samples(n_trivial: int, n_light: int, seed: int = 42) -> list[dict]:
    """Generate GPD samples inline (no external files needed)."""
    rng = random.Random(seed)
    samples = []
    for i in range(n_trivial):
        text = _gen_gpd_trivial(rng, i)
        samples.append({"text": text, "label": "trivial", "source": "gpd-synthetic"})
    for i in range(n_light):
        text = _gen_gpd_light(rng, i)
        samples.append({"text": text, "label": "light", "source": "gpd-synthetic"})
    return samples


# ── Dataset Loaders ──
def load_hf(name: str, max_per: int) -> list[dict]:
    catalogs = {
        "alpaca": {"hf": "tatsu-lab/alpaca", "key": "instruction"},
        "openorca": {"hf": "Open-Orca/OpenOrca", "key": "question"},
    }
    if name not in catalogs:
        return []
    c = catalogs[name]
    try:
        ds = load_dataset(c["hf"], split=f"train[:{max_per}]")
        prompts = []
        for x in ds:
            txt = x.get(c["key"], "")
            if isinstance(txt, str) and len(txt) > 10:
                txt = txt.strip()
                feats = extract_features(txt)
                # Use synthetic complexity for accurate tier labeling
                lab = label_from_explicit_signals(txt, feats)
                prompts.append({"text": txt, "features": feats, "label": lab, "source": name})
        return prompts[:max_per]
    except Exception as e:
        print(f"  Failed {name}: {e}")
        return []


# ── Optimization ──
def features_to_vector(f: dict) -> list[float]:
    return [float(f.get(n, 0)) for n in FEATURE_NAMES]


def optimize_weights(X: np.ndarray, y: np.ndarray, x0: np.ndarray) -> dict:
    # Balanced weighting: weight each class equally, not each sample
    from collections import Counter as Ctr
    # Map y values to nearest tier for weighting
    tier_map = {}
    for name, score in TIER_SCORES.items():
        tier_map[score] = name
    # Find nearest tier for each y value
    y_tiers = []
    for val in y:
        nearest = min(TIER_SCORES.values(), key=lambda s: abs(s - val))
        y_tiers.append(nearest)
    tier_counts = Ctr(y_tiers)
    n_classes = len(tier_counts)
    class_weights = {t: n_classes / count for t, count in tier_counts.items()}
    sample_weights = np.array([class_weights[t] for t in y_tiers], dtype=np.float64)
    # Normalize weights
    sample_weights = sample_weights / sample_weights.mean()

    def weighted_mse(wa):
        scores = X @ wa
        errors = (scores - y) ** 2
        return np.average(errors, weights=sample_weights)

    bounds = [(0.0, 0.35)] * len(FEATURE_NAMES)
    result = minimize(weighted_mse, x0, bounds=bounds, method="L-BFGS-B", options={"maxiter": 3000, "ftol": 1e-10})

    opt = result.x
    total = opt.sum()
    if total > 0:
        opt = opt / total

    return {
        "weights": dict(zip(FEATURE_NAMES, opt.tolist())),
        "mse_before": float(weighted_mse(x0)),
        "mse_after": float(weighted_mse(opt)),
        "iterations": int(result.nit),
    }


def evaluate_per_tier(test_feats: list, test_labels: list, weights: dict, baseline_weights: dict) -> dict:
    """Full per-tier accuracy evaluation for ALL 6 tiers."""
    total = len(test_feats)

    # Build ground truth: map label to expected tier
    def label_to_tier(lab):
        if lab in TIER_SCORES:
            return score_to_tier(TIER_SCORES[lab])
        # For synthetic labels (trivial/light/moderate/etc from dataset)
        return lab if lab in TIER_NAMES else "moderate"

    def predict_tier(f, w):
        return score_to_tier(compute_score(f, w))

    # Overall accuracy
    baseline_correct = 0
    optimized_correct = 0
    tier_correct_baseline = Counter()
    tier_correct_optimized = Counter()
    tier_total = Counter()

    for f, lab in zip(test_feats, test_labels):
        actual_tier = label_to_tier(lab)
        pred_baseline = predict_tier(f, baseline_weights)
        pred_optimized = predict_tier(f, weights)

        tier_total[actual_tier] += 1

        if pred_baseline == actual_tier:
            baseline_correct += 1
            tier_correct_baseline[actual_tier] += 1
        if pred_optimized == actual_tier:
            optimized_correct += 1
            tier_correct_optimized[actual_tier] += 1

    # Per-tier results
    tier_results = {}
    for tier in TIER_NAMES:
        n = tier_total.get(tier, 0)
        tier_results[tier] = {
            "total": n,
            "baseline_correct": tier_correct_baseline.get(tier, 0),
            "baseline_accuracy": round(tier_correct_baseline.get(tier, 0) / n, 4) if n > 0 else None,
            "optimized_correct": tier_correct_optimized.get(tier, 0),
            "optimized_accuracy": round(tier_correct_optimized.get(tier, 0) / n, 4) if n > 0 else None,
        }

    # Weight comparison
    weight_changes = {}
    for name in FEATURE_NAMES:
        w_old = baseline_weights.get(name, 0)
        w_new = weights.get(name, 0)
        if abs(w_new - w_old) > 0.001:
            delta = "+" if w_new > w_old else ""
            weight_changes[name] = {"old": round(w_old, 4), "new": round(w_new, 4), "delta": f"{delta}{round(w_new - w_old, 4)}"}

    return {
        "total_test": total,
        "baseline_accuracy": round(baseline_correct / total, 4) if total > 0 else 0,
        "optimized_accuracy": round(optimized_correct / total, 4) if total > 0 else 0,
        "improvement": round((optimized_correct - baseline_correct) / total, 4) if total > 0 else 0,
        "tier_accuracy": tier_results,
        "weight_comparison": weight_changes,
    }


# ── Main Handler ──
def handler(event):
    inp = event.get("input", {})
    print(f"GateSwarm MoA Router v0.2 Massive Per-Tier Test — {datetime.utcnow().isoformat()}Z")

    datasets = inp.get("datasets", ["gpd", "alpaca", "openorca"])
    max_per = inp.get("max_per", 20000)
    gpd_trivial = inp.get("gpd_trivial", 25000)
    gpd_light = inp.get("gpd_light", 10000)
    max_iter = inp.get("max_iter", 3000)

    all_samples = []

    # 1. Generate GPD inline
    if "gpd" in datasets:
        print(f"  Generating GPD ({gpd_trivial} trivial + {gpd_light} light)...")
        gpd = generate_gpd_samples(gpd_trivial, gpd_light)
        for s in gpd:
            s["features"] = extract_features(s["text"])
        all_samples.extend(gpd)
        print(f"  ✅ {len(gpd)} GPD samples")

    # 2. Load Alpaca
    if "alpaca" in datasets:
        print(f"  Loading Alpaca (max {max_per})...")
        alpaca = load_hf("alpaca", max_per)
        all_samples.extend(alpaca)
        print(f"  ✅ {len(alpaca)} Alpaca samples")

    # 3. Load OpenOrca
    if "openorca" in datasets:
        print(f"  Loading OpenOrca (max {max_per})...")
        orca = load_hf("openorca", max_per)
        all_samples.extend(orca)
        print(f"  ✅ {len(orca)} OpenOrca samples")

    if not all_samples:
        return {"error": "No prompts loaded", "status": "failed"}

    print(f"\n  Total samples: {len(all_samples)}")

    # Label distribution
    label_dist = Counter(s.get("label", "unknown") for s in all_samples)
    print(f"  Label distribution: {dict(label_dist)}")

    # Stratified split: 80/20
    random.seed(42)
    by_label = {}
    for s in all_samples:
        lab = s.get("label", "unknown")
        by_label.setdefault(lab, []).append(s)

    train_data, test_data = [], []
    for lab, samples in by_label.items():
        random.shuffle(samples)
        split = int(len(samples) * 0.8)
        train_data.extend(samples[:split])
        test_data.extend(samples[split:])

    random.shuffle(train_data)
    random.shuffle(test_data)

    print(f"  Train: {len(train_data)} | Test: {len(test_data)}")

    # Test tier distribution
    test_tier_dist = Counter()
    for s in test_data:
        lab = s.get("label", "moderate")
        if lab in TIER_SCORES:
            test_tier_dist[score_to_tier(TIER_SCORES[lab])] += 1
        else:
            test_tier_dist[lab] += 1
    print(f"  Test tier distribution: {dict(test_tier_dist)}")

    # Build matrices
    train_X = np.array([features_to_vector(enrich_features(s["features"])) for s in train_data], dtype=np.float64)
    train_y = np.array([TIER_SCORES.get(s.get("label", "moderate"), 0.25) for s in train_data], dtype=np.float64)
    test_X = np.array([features_to_vector(enrich_features(s["features"])) for s in test_data], dtype=np.float64)
    test_y = np.array([TIER_SCORES.get(s.get("label", "moderate"), 0.25) for s in test_data], dtype=np.float64)

    # Optimize
    print(f"\n  Optimizing (max_iter={max_iter})...")
    x0 = np.array([DEFAULT_WEIGHTS.get(n, 0) for n in FEATURE_NAMES], dtype=np.float64)
    opt_result = optimize_weights(train_X, train_y, x0)

    opt_weights = opt_result["weights"]
    print(f"  MSE: {opt_result['mse_before']:.6f} → {opt_result['mse_after']:.6f}")
    print(f"  Iterations: {opt_result['iterations']}")

    # Per-tier evaluation
    print(f"\n  Evaluating per-tier accuracy...")
    eval_result = evaluate_per_tier(
        [enrich_features(s["features"]) for s in test_data],
        [s.get("label", "moderate") for s in test_data],
        opt_weights, DEFAULT_WEIGHTS
    )

    # Build output
    result = {
        "version": "v3.1-massive",
        "status": "completed",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "config": {
            "datasets": datasets,
            "max_per": max_per,
            "gpd_trivial": gpd_trivial,
            "gpd_light": gpd_light,
            "max_iter": max_iter,
        },
        "dataset_stats": {
            "total_samples": len(all_samples),
            "train_samples": len(train_data),
            "test_samples": len(test_data),
            "label_distribution": dict(Counter(s.get("label", "unknown") for s in all_samples)),
            "test_tier_distribution": dict(test_tier_dist),
        },
        "optimization": {
            "mse_before": round(opt_result["mse_before"], 6),
            "mse_after": round(opt_result["mse_after"], 6),
            "improvement_pct": round((1 - opt_result["mse_after"] / opt_result["mse_before"]) * 100, 1) if opt_result["mse_before"] > 0 else 0,
            "iterations": opt_result["iterations"],
        },
        "overall": {
            "baseline_accuracy": eval_result["baseline_accuracy"],
            "optimized_accuracy": eval_result["optimized_accuracy"],
            "improvement": eval_result["improvement"],
        },
        "per_tier_accuracy": eval_result["tier_accuracy"],
        "optimized_weights": {k: round(v, 4) for k, v in sorted(opt_weights.items(), key=lambda x: -x[1])},
        "weight_comparison": eval_result["weight_comparison"],
    }

    return result


if __name__ == "__main__":
    result = handler({
        "input": {
            "datasets": ["gpd", "alpaca"],
            "max_per": 10000,
            "gpd_trivial": 25000,
            "gpd_light": 10000,
        }
    })
    print(json.dumps(result, indent=2))
