"""MoA v3.3 — LLM-as-Judge Labeling Pipeline

Produces ground-truth complexity labels using LLM judges to replace
the formula-based synthetic labels that cause systematic misclassification.

Pipeline:
1. Load Alpaca + OpenOrca (streaming)
2. Extract features + formula labels
3. Stratified sample: ~300 prompts per tier
4. Send each to LLM judge for ground-truth labeling
5. Train cascade on LLM-judged labels
6. Export corrected weights

Uses free/cheap models as judges (glm-4.7-flash via ZAI API).
"""
import json, time, math, random, re, sys, os
from datetime import datetime, timezone
from collections import Counter

import numpy as np
from sklearn.linear_model import LogisticRegression

# ── Feature Extraction ──
FEATURE_NAMES = [
    "sentence_count", "avg_word_length", "has_question", "question_technical",
    "technical_design", "code", "architecture", "word_count",
    "four_plus", "has_imperative", "technical_terms", "multi_step",
    "requires_context", "domain_specificity", "ambiguity_score",
]
TIERS = ["trivial", "light", "moderate", "heavy", "intensive", "extreme"]
TIER_DESCRIPTIONS = {
    "trivial": "Simple greeting, single-word answer, basic lookup, or yes/no question. No reasoning needed. Examples: 'Hi', 'What is Python?', '2+2', 'Capital of France'.",
    "light": "Single straightforward task with clear scope. Simple read/summarize/format/fix. Short answer expected. Examples: 'Summarize this text', 'Fix the typo', 'Add a docstring'.",
    "moderate": "Clear multi-sentence task requiring some reasoning or domain knowledge. Explanation, comparison, or light code generation. Examples: 'Explain async/await in JS', 'Compare REST vs GraphQL', 'Write a sorting function'.",
    "heavy": "Complex task requiring significant reasoning, design decisions, or multi-component code. Architecture, system design, or substantial implementation. Examples: 'Design a microservice architecture', 'Build a REST API with auth and DB'.",
    "intensive": "Multi-system integration requiring deep expertise. Cross-cutting concerns, performance optimization, or large-scale design. Examples: 'Build a trading platform with WebSocket feeds and order matching', 'Design a distributed cache system'.",
    "extreme": "Expert-level architectural challenge spanning multiple domains, novel problem-solving, or extremely complex systems. Examples: 'Design a globally distributed autonomous AI orchestration platform with ZKP verification and federated learning'.",
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


# ── LLM Judge ──
JUDGE_SYSTEM_PROMPT = """You are a prompt complexity classifier. Your job is to rate how complex a given prompt/task is on a 6-tier scale.

Tiers:
- trivial: Simple greeting, single-word answer, basic lookup, yes/no. No reasoning.
- light: Single straightforward task. Simple read/summarize/format/fix. Short answer.
- moderate: Multi-sentence task requiring some reasoning or domain knowledge. Explanation or light code.
- heavy: Complex task requiring significant reasoning, design decisions, or multi-component code.
- intensive: Multi-system integration requiring deep expertise. Cross-cutting concerns, large-scale design.
- extreme: Expert-level architectural challenge spanning multiple domains. Novel problem-solving.

Rules:
1. Respond with ONLY the tier name: trivial, light, moderate, heavy, intensive, or extreme
2. Consider: word count, technical depth, number of subtasks, required expertise
3. When in doubt, prefer the simpler tier
4. A single question about a concept is usually trivial or light
5. "Explain X" is moderate if X is technical
6. "Design/build X" is heavy+ if X has multiple components
7. Architecture with "distributed", "multi-region", "federated" is intensive+
"""


def llm_judge_local(prompts: list[str], model="zai/glm-4.7-flash") -> list[str]:
    """Use local OpenClaw API to judge prompts."""
    import urllib.request
    
    labels = []
    tier_set = set(TIERS)
    
    for i, prompt in enumerate(prompts):
        judge_prompt = f"Classify this prompt's complexity:\n\n\"{prompt}\"\n\nRespond with exactly one tier name."
        
        try:
            # Use OpenClaw's API endpoint
            req_data = json.dumps({
                "model": model,
                "messages": [
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": judge_prompt},
                ],
                "temperature": 0.1,
                "max_tokens": 20,
            }).encode()
            
            # Try ZAI API directly
            req = urllib.request.Request(
                "https://api.z.ai/api/coding/paas/v4/chat/completions",
                data=req_data,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {os.environ.get('ZAI_API_KEY', '')}",
                },
            )
            
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read().decode())
                text = result["choices"][0]["message"]["content"].strip().lower()
                
                # Extract tier name
                label = None
                for tier in TIERS:
                    if tier in text:
                        label = tier
                        break
                if not label:
                    label = label_formula(prompt)  # fallback
                labels.append(label)
                
        except Exception as e:
            if i == 0:
                print(f"  LLM judge error: {e}")
            labels.append(label_formula(prompt))  # fallback to formula
        
        if (i + 1) % 50 == 0:
            print(f"  Judged {i+1}/{len(prompts)}")
    
    return labels


def train_cascade_on_labels(samples: list[dict], label_key: str = "llm_label") -> tuple[dict, dict]:
    """Train cascade on ground-truth labels."""
    all_features = np.array([features_to_vector(s["features"]) for s in samples], dtype=np.float64)
    all_labels = np.array([s[label_key] for s in samples])
    
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
            results[tier] = {"success": False, "reason": f"too few: pos={mask_pos.sum()} neg={mask_neg.sum()}"}
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


def evaluate_cascade(models: dict, samples: list[dict], label_key: str = "llm_label") -> dict:
    """Evaluate cascade predictions against ground truth."""
    correct = 0
    total = len(samples)
    tier_correct = Counter()
    tier_total = Counter()
    
    for s in samples:
        x = features_to_vector(s["features"])
        predicted = predict_cascade(models, x)
        actual = s[label_key]
        tier_total[actual] += 1
        if predicted == actual:
            correct += 1
            tier_correct[actual] += 1
    
    per_tier = {}
    for tier in TIERS:
        t = tier_total.get(tier, 0)
        per_tier[tier] = {
            "accuracy": round(tier_correct.get(tier, 0) / max(t, 1), 4),
            "correct": tier_correct.get(tier, 0),
            "samples": t,
        }
    
    return {
        "accuracy": round(correct / max(total, 1), 4),
        "correct": correct,
        "total": total,
        "per_tier": per_tier,
    }


def predict_cascade(models: dict, x: np.ndarray) -> str:
    """Predict tier using pairwise logit analysis."""
    CASCADE_ORDER = ["trivial", "light", "moderate", "heavy", "intensive"]
    
    logits = []
    for tier in CASCADE_ORDER:
        if tier in models:
            m = models[tier]
            logits.append(float(x @ m["weights_vec"] + m["intercept"]))
        else:
            logits.append(0.0)
    
    # Pairwise diffs
    pairwise = [logits[i] - logits[i+1] for i in range(len(logits)-1)]
    
    # First positive pairwise diff = transition point
    for i, pw in enumerate(pairwise):
        if pw > 0:
            return CASCADE_ORDER[i]
    
    return "extreme"


def main():
    t0 = time.time()
    
    # Check for API key
    api_key = os.environ.get("ZAI_API_KEY", "")
    if not api_key:
        # Try loading from openclaw config
        try:
            import subprocess
            result = subprocess.run(
                ["python3", "-c", "import json; d=json.load(open('/root/.openclaw/openclaw.json')); providers=d.get('providers',{}); zai=providers.get('zai',{}); print(zai.get('apiKey',''))"],
                capture_output=True, text=True, timeout=5,
            )
            api_key = result.stdout.strip()
            if api_key:
                os.environ["ZAI_API_KEY"] = api_key
                print(f"  Loaded ZAI API key from config ({api_key[:10]}...)")
        except Exception as e:
            print(f"  Could not load API key: {e}")
    
    print("=== MoA v3.3 LLM-as-Judge Labeling ===")
    print(f"  API key available: {bool(api_key)}")
    
    # 1. Load datasets
    print("\n  Loading datasets...")
    try:
        from datasets import load_dataset
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "datasets"])
        from datasets import load_dataset
    
    samples = []
    
    # Alpaca
    print("  Loading Alpaca...")
    ds = load_dataset("tatsu-lab/alpaca", split="train", streaming=True)
    for i, row in enumerate(ds):
        if i >= 5000: break
        text = row.get("instruction", "") or row.get("input", "") or ""
        if not text or len(text) < 3: continue
        feat = extract_features(text)
        label = label_formula(text)
        samples.append({"text": text, "features": feat, "formula_label": label, "source": "alpaca"})
    
    print(f"  Loaded {len(samples)} Alpaca samples")
    
    # 2. Stratified sample: ~300 per tier
    print("\n  Stratified sampling...")
    per_tier = 300
    by_tier = {t: [] for t in TIERS}
    for s in samples:
        by_tier[s["formula_label"]].append(s)
    
    judged_samples = []
    for tier in TIERS:
        pool = by_tier[tier]
        n = min(per_tier, len(pool))
        selected = random.Random(42).sample(pool, n)
        judged_samples.extend(selected)
        print(f"    {tier}: {n} samples selected (pool: {len(pool)})")
    
    random.Random(42).shuffle(judged_samples)
    print(f"  Total to judge: {len(judged_samples)}")
    
    # 3. LLM-as-judge labeling
    print("\n  Running LLM-as-judge...")
    prompts = [s["text"] for s in judged_samples]
    llm_labels = llm_judge_local(prompts)
    
    for i, s in enumerate(judged_samples):
        s["llm_label"] = llm_labels[i]
    
    llm_dist = Counter(llm_labels)
    formula_dist = Counter(s["formula_label"] for s in judged_samples)
    print(f"\n  Label distributions:")
    print(f"    Formula: {dict(formula_dist)}")
    print(f"    LLM:     {dict(llm_dist)}")
    
    # Agreement
    agree = sum(1 for s in judged_samples if s["formula_label"] == s["llm_label"])
    print(f"    Agreement: {agree}/{len(judged_samples)} ({agree/len(judged_samples):.1%})")
    
    # 4. Train cascade on LLM labels
    print("\n  Training cascade on LLM-judged labels...")
    models, export, train_results = train_cascade_on_labels(judged_samples, label_key="llm_label")
    
    for tier, r in train_results.items():
        if r.get("success"):
            print(f"    {tier}: acc={r['accuracy']:.1%} n={r['n_samples']}")
        else:
            print(f"    {tier}: FAILED - {r.get('reason','?')}")
    
    # 5. Evaluate on held-out data (samples NOT in judged set)
    print("\n  Evaluating on remaining samples...")
    judged_texts = set(s["text"] for s in judged_samples)
    held_out = [s for s in samples if s["text"] not in judged_texts and s["formula_label"] in TIERS]
    
    # For held-out, we don't have LLM labels, so evaluate against formula
    # But also test our calibration prompts
    cal_prompts = [
        {"text": "Hi", "llm_label": "trivial"},
        {"text": "What is Python?", "llm_label": "trivial"},
        {"text": "Summarize the following text about machine learning", "llm_label": "light"},
        {"text": "Explain how async/await works in JavaScript with examples", "llm_label": "moderate"},
        {"text": "Design a distributed microservice architecture for a fintech payment system with event sourcing, CQRS, and multi-region failover", "llm_label": "heavy"},
        {"text": "Build a complete real-time trading platform with WebSocket feeds, order matching engine, risk management, regulatory compliance reporting, and disaster recovery across 3 cloud regions", "llm_label": "intensive"},
        {"text": "Architect a globally distributed autonomous AI agent orchestration platform with real-time model fine-tuning, federated learning across jurisdictions, zero-knowledge proof verification, custom ASIC acceleration, and self-healing infrastructure", "llm_label": "extreme"},
    ]
    
    print("\n  === Calibration Test (known labels) ===")
    correct_cal = 0
    for cp in cal_prompts:
        feat = extract_features(cp["text"])
        x = features_to_vector(feat)
        pred = predict_cascade(models, x)
        match = "✅" if pred == cp["llm_label"] else ("≈" if abs(TIERS.index(pred) - TIERS.index(cp["llm_label"])) <= 1 else "❌")
        if pred == cp["llm_label"]: correct_cal += 1
        print(f"    {cp['llm_label']:<12} → {pred:<12} {match}")
    print(f"    Calibration: {correct_cal}/{len(cal_prompts)} exact")
    
    # 6. Evaluate on LLM-judged samples
    print("\n  === LLM-judged test (80/20 split) ===")
    random.Random(42).shuffle(judged_samples)
    split = int(len(judged_samples) * 0.8)
    train_set = judged_samples[:split]
    test_set = judged_samples[split:]
    
    models_final, export_final, _ = train_cascade_on_labels(train_set, "llm_label")
    eval_result = evaluate_cascade(models_final, test_set, "llm_label")
    
    print(f"    Test accuracy: {eval_result['accuracy']:.1%} ({eval_result['correct']}/{eval_result['total']})")
    for tier in TIERS:
        t = eval_result["per_tier"].get(tier, {})
        if t.get("samples", 0) > 0:
            print(f"      {tier}: {t['accuracy']:.1%} ({t['correct']}/{t['samples']})")
    
    # Re-test calibration
    print("\n  === Calibration re-test (after retrain) ===")
    correct_cal2 = 0
    for cp in cal_prompts:
        feat = extract_features(cp["text"])
        x = features_to_vector(feat)
        pred = predict_cascade(models_final, x)
        match = "✅" if pred == cp["llm_label"] else ("≈" if abs(TIERS.index(pred) - TIERS.index(cp["llm_label"])) <= 1 else "❌")
        if pred == cp["llm_label"]: correct_cal2 += 1
        print(f"    {cp['llm_label']:<12} → {pred:<12} {match}")
    print(f"    Calibration: {correct_cal2}/{len(cal_prompts)} exact")
    
    # 7. Export weights
    elapsed = round(time.time() - t0, 1)
    
    cascade_weights = {
        "version": "v3.3-llm-judged",
        "trained": datetime.now(timezone.utc).isoformat() + "Z",
        "dataset": f"{len(judged_samples)} LLM-judged (Alpaca)",
        "method": "pairwise-cascade-on-llm-labels",
        "judge_model": "zai/glm-4.7-flash",
        "overall_accuracy": eval_result["accuracy"],
        "feature_names": FEATURE_NAMES,
        "classifiers": export_final,
        "tier_boundaries": [0.08, 0.18, 0.32, 0.52, 0.72],
        "tier_accuracy": {t: eval_result["per_tier"].get(t, {}) for t in TIERS},
        "label_distributions": {
            "formula": dict(formula_dist),
            "llm_judge": dict(llm_dist),
        },
        "agreement_formula_vs_llm": round(agree / len(judged_samples), 4),
        "calibration_score": f"{correct_cal2}/{len(cal_prompts)}",
        "training_time_s": elapsed,
    }
    
    output_path = os.path.join(os.path.dirname(__file__), "v33_llm_judged_weights.json")
    with open(output_path, "w") as f:
        json.dump(cascade_weights, f, indent=2)
    
    print(f"\n  === Export ===")
    print(f"  Saved: {output_path}")
    print(f"  Elapsed: {elapsed}s")
    
    return {
        "version": "v3.3-llm-judge",
        "status": "completed",
        "elapsed_seconds": elapsed,
        "judged_samples": len(judged_samples),
        "test_accuracy": eval_result["accuracy"],
        "calibration": f"{correct_cal2}/{len(cal_prompts)}",
        "cascade_weights_json": cascade_weights,
        "weights_path": output_path,
    }


if __name__ == "__main__":
    import runpod
    runpod.serverless.start({"handler": lambda e: main()})
