"""GateSwarm MoA Router v0.3.5 — Label Validation Handler

Compares formula-based synthetic labels against cascade predictions
to identify systematic disagreements and produce a ground-truth
validation report. This addresses the Chief Scientist's critical
finding: labels are synthetic (2.0/10 validity score).

Output: structured report with per-tier disagreement analysis,
systematic error patterns, and recommended samples for LLM-as-judge.
"""
import json, time, math, random, re, subprocess, sys
from datetime import datetime, timezone
from collections import Counter

def ensure_deps():
    for pkg, mod in [("scikit-learn", "sklearn"), ("numpy", "numpy"), ("datasets", "datasets")]:
        try: __import__(mod)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", pkg])
ensure_deps()

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from datasets import load_dataset

# ── Constants ──
FEATURE_NAMES = [
    "sentence_count", "avg_word_length", "has_question", "question_technical",
    "technical_design", "code", "architecture", "word_count",
    "four_plus", "has_imperative", "technical_terms", "multi_step",
    "requires_context", "domain_specificity", "ambiguity_score",
]
TIERS = ["trivial", "light", "moderate", "heavy", "intensive", "extreme"]

# ── Feature Extraction (same as handler_v32_cascade.py) ──
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
IMPERATIVE_STARTS = [
    "read", "summarize", "format", "explain", "add", "rename", "write",
    "check", "count", "find", "replace", "show", "make", "fix",
    "extract", "parse", "validate", "sort", "filter", "calculate",
    "create", "build", "run", "test", "implement", "refactor",
    "optimize", "deploy", "configure",
]


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
        "word_count": wc, "sentence_count": sc, "avg_word_length": awl,
        "has_code": float(has_code), "has_question": float(has_question),
        "has_imperative": float(has_imperative), "technical_terms": tech_terms,
        "question_technical": float(question_technical),
        "architecture": float(architecture), "technical_design": float(technical_design),
        "multi_step": float(multi_step), "requires_context": float(requires_context),
        "domain_specificity": domain_spec, "ambiguity_score": ambiguity,
        "four_plus": four_plus,
    }


def features_to_vector(f: dict) -> list[float]:
    return [float(f.get(n, 0)) for n in FEATURE_NAMES]


def label_formula(text: str) -> str:
    """Original synthetic label formula (Chief Scientist critique target)."""
    t = text.lower()
    words = t.split()
    wc = len(words)
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


def load_dataset_samples(name: str, max_n: int) -> list[dict]:
    """Load prompts from HuggingFace with formula labels + features."""
    catalogs = {
        "alpaca": {"hf": "tatsu-lab/alpaca", "key": "instruction"},
        "openorca": {"hf": "Open-Orca/OpenOrca", "key": "question"},
    }
    if name not in catalogs:
        return []
    c = catalogs[name]
    try:
        ds = load_dataset(c["hf"], split=f"train[:{max_n}]", streaming=True)
        samples = []
        for x in ds:
            txt = x.get(c["key"], "")
            if isinstance(txt, str) and len(txt) > 10:
                txt = txt.strip()
                feats = extract_features(txt)
                formula_label = label_formula(txt)
                samples.append({
                    "text": txt[:200],  # truncate for report
                    "features": feats,
                    "formula_label": formula_label,
                    "source": name,
                })
        return samples[:max_n]
    except Exception as e:
        print(f"  Failed {name}: {e}")
        return []


# ── GPD Synthetic Generator ──
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
GPD_CONCEPTS = ["API", "REST", "GraphQL", "Docker", "Git", "JSON", "SQL", "Redis", "React", "Python"]
GPD_PATHS = ["src/main.py", "config/settings.json", "README.md", ".env", "package.json"]
GPD_ERRORS = ["TypeError: undefined is not a function", "SyntaxError: unexpected token", "ValueError: invalid literal"]
GPD_FUNCS = ["def calculate_total(items: list[float]) -> float", "async def fetch_data(url: str) -> dict"]
GPD_CMDS = ["ls -la", "grep -r 'pattern' .", "find . -name '*.py'", "ps aux | grep python"]


def generate_gpd(n_trivial: int, n_light: int, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    samples = []
    for i in range(n_trivial):
        t = rng.choice(GPD_TRIVIAL)
        kw = {"concept": rng.choice(GPD_CONCEPTS), "acronym": "API",
              "thing": "byte", "category": "framework", "unit": "bytes",
              "event": "Python created", "n": "5", "items": "databases",
              "value": "100", "from_unit": "KB", "to_unit": "MB",
              "place": "Brazil", "language": "Spanish", "word": "happy",
              "number": "42", "number2": "17"}
        text = t.format(**kw)
        text = re.sub(r'\{[^}]+\}', 'something', text)
        samples.append({"text": text, "features": extract_features(text),
                        "formula_label": "trivial", "source": "gpd"})
    for i in range(n_light):
        t = rng.choice(GPD_LIGHT)
        kw = {"path": rng.choice(GPD_PATHS), "short_text": "Python is a high-level language.",
              "code_snippet": "def hello(): print('hello')",
              "error_message": rng.choice(GPD_ERRORS),
              "function_signature": rng.choice(GPD_FUNCS),
              "shell_cmd": rng.choice(GPD_CMDS),
              "json_snippet": '{"key": "value"}', "list_data": "[3, 1, 4, 1, 5, 9]"}
        text = t.format(**kw)
        text = re.sub(r'\{[^}]+\}', 'data', text)
        samples.append({"text": text, "features": extract_features(text),
                        "formula_label": "light", "source": "gpd"})
    return samples


# ── Cascade Training (same as v3.2) ──
class TierCascade:
    def __init__(self):
        self.models = {}

    def train(self, train_data: list[dict]):
        by_tier = {}
        for s in train_data:
            by_tier.setdefault(s["formula_label"], []).append(s)

        all_features = np.array([features_to_vector(s["features"]) for s in train_data], dtype=np.float64)
        all_labels = np.array([s["formula_label"] for s in train_data])
        results = {}

        for tier in ["trivial", "light", "moderate", "heavy", "intensive"]:
            if tier == "trivial":
                mask_pos = all_labels == tier
                mask_neg = ~mask_pos
            else:
                prev_tiers = ["trivial", "light", "moderate", "heavy"][:["trivial", "light", "moderate", "heavy"].index(tier)]
                mask_excluded = np.zeros(len(all_labels), dtype=bool)
                for pt in prev_tiers:
                    mask_excluded |= all_labels == pt
                remaining = ~mask_excluded
                mask_pos = remaining & (all_labels == tier)
                mask_neg = remaining & (all_labels != tier)

            X_pos = all_features[mask_pos]
            X_neg = all_features[mask_neg]
            n_min = min(len(X_pos), len(X_neg))
            if n_min < 10:
                results[tier] = {"success": False, "reason": f"too few samples: {n_min}"}
                continue

            rng = np.random.RandomState(42)
            pos_idx = rng.permutation(len(X_pos))[:n_min]
            neg_idx = rng.permutation(len(X_neg))[:n_min]
            X = np.vstack([X_pos[pos_idx], X_neg[neg_idx]])
            y = np.concatenate([np.ones(n_min), np.zeros(n_min)])

            model = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs', random_state=42)
            model.fit(X, y)
            acc = float((model.predict(X) == y).mean())
            results[tier] = {
                "success": True, "accuracy": round(acc, 4),
                "n_samples": n_min * 2,
                "weights": dict(zip(FEATURE_NAMES, model.coef_[0].tolist())),
                "intercept": float(model.intercept_[0]),
            }
            self.models[tier] = results[tier]
        return results

    def predict(self, features: dict) -> str:
        X = np.array([features_to_vector(features)], dtype=np.float64)
        for tier in ["trivial", "light", "moderate", "heavy", "intensive"]:
            if tier not in self.models:
                continue
            m = self.models[tier]
            w = np.array([m["weights"].get(f, 0) for f in FEATURE_NAMES])
            prob = 1 / (1 + np.exp(-(X @ w + m["intercept"])))
            if prob[0] > 0.5:
                return tier
        return "extreme"


# ── Main Handler ──
def handler(event):
    inp = event.get("input", {})
    print(f"GateSwarm MoA Router v0.3.5 Label Validation — {datetime.now(timezone.utc).isoformat()}")

    max_per = inp.get("max_per", 20000)
    gpd_trivial = inp.get("gpd_trivial", 25000)
    gpd_light = inp.get("gpd_light", 10000)

    # Load data
    all_samples = []
    print("  Generating GPD...")
    gpd = generate_gpd(gpd_trivial, gpd_light)
    all_samples.extend(gpd)

    for ds_name in ["alpaca"]:
        print(f"  Loading {ds_name}...")
        samples = load_dataset_samples(ds_name, max_per)
        all_samples.extend(samples)

    print(f"  Total: {len(all_samples)} samples")

    # Split
    random.seed(42)
    by_label = {}
    for s in all_samples:
        by_label.setdefault(s["formula_label"], []).append(s)

    train_data, test_data = [], []
    for lab, samples in by_label.items():
        random.shuffle(samples)
        split = int(len(samples) * 0.8)
        train_data.extend(samples[:split])
        test_data.extend(samples[split:])
    random.shuffle(train_data)
    random.shuffle(test_data)

    print(f"  Train: {len(train_data)} | Test: {len(test_data)}")

    # Train cascade
    print("  Training cascade...")
    cascade = TierCascade()
    cascade_results = cascade.train(train_data)

    # Evaluate cascade vs formula (baseline is 100% by definition since formula=label)
    # The real analysis: where does the cascade DISAGREE with formula on test data?
    disagreements = []
    tier_agreement = Counter()
    tier_total = Counter()

    for s in test_data:
        formula = s["formula_label"]
        cascade_pred = cascade.predict(s["features"])
        tier_total[formula] += 1

        if formula == cascade_pred:
            tier_agreement[formula] += 1
        else:
            disagreements.append({
                "text": s["text"],
                "formula_label": formula,
                "cascade_label": cascade_pred,
                "source": s["source"],
                "features": {k: round(s["features"].get(k, 0), 3) for k in ["ambiguity_score", "domain_specificity", "has_question", "has_imperative", "architecture", "word_count", "sentence_count"]},
            })

    # Per-tier agreement rates
    tier_agreement_rate = {}
    for tier in TIERS:
        total = tier_total.get(tier, 0)
        agreed = tier_agreement.get(tier, 0)
        tier_agreement_rate[tier] = {
            "total": total,
            "agreed": agreed,
            "disagreed": total - agreed,
            "agreement_rate": round(agreed / total, 4) if total > 0 else None,
        }

    # Disagreement pattern analysis
    pattern_counter = Counter()
    for d in disagreements:
        pattern = f"{d['formula_label']}→{d['cascade_label']}"
        pattern_counter[pattern] += 1

    # Top 10 disagreement examples per pattern
    disagreement_examples = {}
    for pattern in pattern_counter:
        examples = [d for d in disagreements if f"{d['formula_label']}→{d['cascade_label']}" == pattern]
        disagreement_examples[pattern] = examples[:10]

    # Identify high-value samples for LLM-as-judge (max disagreement across tiers)
    llm_candidates = []
    # Pick samples where formula and cascade strongly disagree AND the sample is ambiguous
    for d in disagreements:
        if d["features"].get("ambiguity_score", 0) > 0.3:
            llm_candidates.append(d["text"])
    # Stratified: pick top 50 per disagreement pattern
    llm_sample_by_pattern = {}
    for pattern in pattern_counter:
        samples_for_pattern = [d for d in disagreements if f"{d['formula_label']}→{d['cascade_label']}" == pattern]
        llm_sample_by_pattern[pattern] = [s["text"] for s in samples_for_pattern[:50]]

    # Summary
    total_disagreements = len(disagreements)
    total_test = len(test_data)
    overall_agreement = round((total_test - total_disagreements) / total_test, 4) if total_test > 0 else 0

    result = {
        "version": "v0.3.5-label-validation",
        "status": "completed",
        "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
        "dataset": {
            "total": len(all_samples),
            "train": len(train_data),
            "test": total_test,
        },
        "label_validity": {
            "overall_agreement_rate": overall_agreement,
            "total_disagreements": total_disagreements,
            "interpretation": (
                "HIGH validity" if overall_agreement > 0.9 else
                "MODERATE validity" if overall_agreement > 0.7 else
                "LOW validity — formula labels disagree with learned boundaries"
            ),
            "per_tier_agreement": tier_agreement_rate,
        },
        "disagreement_patterns": dict(pattern_counter),
        "disagreement_examples": {
            k: [{"text": e[:150], "formula": d["formula_label"], "cascade": d["cascade_label"],
                  "features": d["features"]}
                 for e in v[:5]
                 for d in disagreements if e == d["text"]]
            for k, v in list(disagreement_examples.items())[:5]
        },
        "llm_labeling_candidates": {
            "total_ambiguous_disagreements": len(llm_candidates),
            "stratified_samples": {k: v[:20] for k, v in llm_sample_by_pattern.items()},
            "recommendation": f"Label {min(500, len(llm_candidates))} stratified samples with LLM-as-judge for ground truth",
        },
        "cascade_training": {k: {kk: vv for kk, vv in v.items() if kk != "weights"} for k, v in cascade_results.items()},
    }

    # Print summary
    print(f"\n  === LABEL VALIDITY REPORT ===")
    print(f"  Overall agreement: {overall_agreement:.1%}")
    print(f"  Total disagreements: {total_disagreements}/{total_test}")
    print(f"\n  Per-tier agreement:")
    for tier in TIERS:
        ta = tier_agreement_rate.get(tier, {})
        print(f"    {tier}: {ta.get('agreement_rate', 'N/A'):.1%} ({ta.get('agreed', 0)}/{ta.get('total', 0)})")
    print(f"\n  Disagreement patterns:")
    for pattern, count in pattern_counter.most_common(10):
        print(f"    {pattern}: {count}")
    print(f"\n  LLM candidates: {len(llm_candidates)} ambiguous disagreements")

    return result


if __name__ == "__main__":
    result = handler({"input": {"max_per": 20000, "gpd_trivial": 25000, "gpd_light": 10000}})
    print(json.dumps(result, indent=2))
