#!/usr/bin/env python3
"""GateSwarm MoA Router v0.3.5 — LLM-as-Judge Pipeline (Parallel, GPU Pod Optimized)

Run on RunPod pod with SSH. Uses asyncio for 20 concurrent LLM API calls.
Expected: 1,496 samples in ~5-8 min instead of 60+ min.

Usage:
  python3 llm_judge_parallel.py
"""
import json, os, time, math, random, re, sys, asyncio
from datetime import datetime, timezone
from collections import Counter
import numpy as np
from sklearn.linear_model import LogisticRegression

# ── Config ──
ZAI_KEY = os.environ.get("ZAI_API_KEY", "")
ZAI_BASE = os.environ.get("ZAI_BASE_URL", "https://api.z.ai/api/coding/paas/v4")
CONCURRENT = 20  # parallel API calls
PER_TIER = 300

FEATURE_NAMES = [
    "sentence_count", "avg_word_length", "has_question", "question_technical",
    "technical_design", "code", "architecture", "word_count",
    "four_plus", "has_imperative", "technical_terms", "multi_step",
    "requires_context", "domain_specificity", "ambiguity_score",
]
TIERS = ["trivial", "light", "moderate", "heavy", "intensive", "extreme"]

TECH_KEYWORDS = {"api","http","rest","docker","kubernetes","git","json","sql","python","rust","typescript","react","function","class","architecture","design","microservice","database","security","async","await","error","deploy","pipeline"}
ARCH_KEYWORDS = {"architecture","design pattern","system design","microservice","distributed","scalable","load balancer","event-driven","service mesh","api gateway","serverless"}
DESIGN_KEYWORDS = {"technical design","implementation plan","migration strategy","deployment strategy","disaster recovery","failover"}

def extract_features(text):
    words = re.findall(r'\w+', text.lower())
    wc = len(words)
    sc = len([s.strip() for s in re.split(r'[.!?]+', text) if s.strip()])
    awl = sum(len(w) for w in words) / max(wc, 1)
    has_code = float(bool(re.search(r'```|`[^`]+`|def\s+\w+|function\s+\w+', text)))
    has_question = float('?' in text)
    has_imperative = float(any(text.lower().startswith(i) for i in ["read","write","create","build","design","explain","implement","summarize","fix","make","show","check","find","extract","parse","format"]))
    tech_terms = sum(1 for w in words if w in TECH_KEYWORDS)
    architecture = float(any(kw in text.lower() for kw in ARCH_KEYWORDS))
    technical_design = float(any(kw in text.lower() for kw in DESIGN_KEYWORDS))
    multi_step = float(bool(re.search(r'(first|then|next|finally|step)', text.lower())))
    requires_context = float(bool(re.search(r'(the file|this project|my code|given that|consider)', text.lower())))
    domain_spec = min(tech_terms / max(wc, 1), 1.0)
    vague = {'something','stuff','thing','things','it','this','that'}
    ambiguity = min(sum(1 for w in words if w in vague) / max(wc, 1), 1.0)
    active = sum(1 for v in [has_code, has_question, has_imperative, tech_terms >= 3] if v)
    four_plus = 1.0 if active >= 4 else 0.0
    return {
        "word_count": wc, "sentence_count": sc, "avg_word_length": round(awl, 3),
        "has_code": has_code, "has_question": has_question, "has_imperative": has_imperative,
        "technical_terms": tech_terms, "question_technical": float(has_question and tech_terms > 0),
        "architecture": architecture, "technical_design": technical_design,
        "multi_step": multi_step, "requires_context": requires_context,
        "domain_specificity": round(domain_spec, 3), "ambiguity_score": round(ambiguity, 3),
        "four_plus": four_plus,
    }

def features_to_vector(f):
    return np.array([float(f.get(n, 0)) for n in FEATURE_NAMES], dtype=np.float64)

def label_formula(text):
    t = text.lower()
    wc = len(t.split())
    signals = sum([
        '?' in text,
        any(k in t for k in ["code","function","def ","class ","import ","fn ","const "]),
        any(t.startswith(k) for k in ["write ","create ","build ","implement ","generate ","fix ","debug ","optimize ","explain ","analyze ","describe ","design "]),
        bool(re.search(r'[0-9]+[\s]*[+\-*/=]', text)),
        any(k in t for k in ["first ","then ","finally","step ","part ","section ","also "]),
        any(k in t for k in ["must ","should ","required ","only ","cannot ","limit "]),
        any(k in t for k in ["given ","consider ","assume ","suppose ","based on ","according to "]),
        any(k in t for k in ["architecture","design pattern","system design","microservice","scalable","distributed"]),
        any(k in t for k in ["technical design","implementation plan","migration strategy","deployment","pipeline","schema","database"]),
    ])
    has_context = float(any(k in t for k in ["given ","consider ","assume ","suppose ","based on "]))
    score = signals * 0.15 + math.log1p(wc) * 0.08 + has_context * 0.1
    if score < 0.10: return "trivial"
    if score < 0.20: return "light"
    if score < 0.35: return "moderate"
    if score < 0.50: return "heavy"
    if score < 0.65: return "intensive"
    return "extreme"

JUDGE_SYSTEM = """You are a prompt complexity classifier. Rate the prompt on a 6-tier scale.

Tiers:
- trivial: Greeting, yes/no, basic lookup. No reasoning. ('Hi', 'What is Python?', '2+2')
- light: Single simple task. Read/summarize/format/fix. Short answer. ('Fix this typo', 'Summarize text')
- moderate: Multi-sentence task needing reasoning. Explanation, light code. ('Explain async/await', 'Compare REST vs GraphQL')
- heavy: Complex task needing design decisions or multi-component code. ('Design microservice architecture', 'Build REST API with auth')
- intensive: Multi-system integration, deep expertise. ('Build trading platform with WebSocket and order matching')
- extreme: Expert multi-domain architectural challenge. ('Globally distributed AI orchestration with ZKP and federated learning')

Respond with ONLY the tier name: trivial, light, moderate, heavy, intensive, or extreme."""

async def llm_judge_one(session, prompt_text, sem):
    async with sem:
        judge_prompt = f'Classify this prompt complexity:\n\n"{prompt_text}"\n\nRespond with exactly one tier name.'
        payload = {
            "model": "glm-4.7-flash",
            "messages": [
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": judge_prompt},
            ],
            "temperature": 0.1,
            "max_tokens": 20,
        }
        try:
            import aiohttp
            async with session.post(
                f"{ZAI_BASE}/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {ZAI_KEY}"},
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                result = await resp.json()
                text = result["choices"][0]["message"]["content"].strip().lower()
                for tier in TIERS:
                    if tier in text:
                        return tier
                return label_formula(prompt_text)
        except Exception as e:
            return label_formula(prompt_text)

async def judge_all(samples):
    """Judge all samples with CONCURRENT parallel requests."""
    import aiohttp
    sem = asyncio.Semaphore(CONCURRENT)
    done = 0
    total = len(samples)
    
    async def judge_with_progress(session, s):
        nonlocal done
        label = await llm_judge_one(session, s["text"], sem)
        s["llm_label"] = label
        done += 1
        if done % 100 == 0 or done == total:
            dist = Counter(s.get("llm_label", "?") for s in samples[:done])
            print(f"  Judged {done}/{total} | {dict(dist)}")
        return label
    
    async with aiohttp.ClientSession() as session:
        tasks = [judge_with_progress(session, s) for s in samples]
        await asyncio.gather(*tasks)

def train_cascade(samples, label_key="llm_label"):
    all_features = np.array([features_to_vector(s["features"]) for s in samples], dtype=np.float64)
    all_labels = np.array([s[label_key] for s in samples])
    models = {}
    export = {}
    cascade_order = ["trivial", "light", "moderate", "heavy", "intensive"]
    
    for tier in cascade_order:
        if tier == "trivial":
            mask_pos = all_labels == tier
            mask_neg = ~mask_pos
        else:
            prev = cascade_order[:cascade_order.index(tier)]
            mask_excluded = np.zeros(len(all_labels), dtype=bool)
            for pt in prev: mask_excluded |= all_labels == pt
            remaining = ~mask_excluded
            mask_pos = remaining & (all_labels == tier)
            mask_neg = remaining & (all_labels != tier)
        
        X_pos = all_features[mask_pos]
        X_neg = all_features[mask_neg]
        n_min = min(len(X_pos), len(X_neg))
        if n_min < 5:
            print(f"  {tier}: SKIP (pos={mask_pos.sum()} neg={mask_neg.sum()})")
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
        
        models[tier] = {"weights_vec": np.array(model.coef_[0], dtype=np.float64), "intercept": float(model.intercept_[0])}
        export[tier] = {"accuracy": round(acc, 4), "n_samples": n_min * 2, "intercept": round(float(model.intercept_[0]), 4), "weights": weights_dict}
        print(f"  {tier}: acc={acc:.1%} n={n_min*2}")
    
    return models, export

def predict_cascade(models, x):
    cascade_order = ["trivial", "light", "moderate", "heavy", "intensive"]
    logits = []
    for tier in cascade_order:
        if tier in models:
            logits.append(float(x @ models[tier]["weights_vec"] + models[tier]["intercept"]))
        else:
            logits.append(0.0)
    pairwise = [logits[i] - logits[i+1] for i in range(len(logits)-1)]
    for i, pw in enumerate(pairwise):
        if pw > 0:
            return cascade_order[i]
    return "extreme"

def main():
    t0 = time.time()
    print(f"=== GateSwarm MoA Router v0.3.5 LLM-as-Judge (Parallel, {CONCURRENT} concurrent) ===")
    print(f"  ZAI API: {ZAI_BASE}")
    print(f"  Key: {ZAI_KEY[:15]}...")
    
    # 1. Install deps
    print("\n  Installing deps...")
    os.system("pip install --quiet aiohttp datasets scikit-learn numpy 2>/dev/null")
    
    # 2. Load Alpaca (streaming, memory-safe)
    print("  Loading Alpaca...")
    from datasets import load_dataset
    ds = load_dataset("tatsu-lab/alpaca", split="train", streaming=True)
    
    samples_by_tier = {t: [] for t in TIERS}
    total_loaded = 0
    
    for row in ds:
        text = row.get("instruction", "") or row.get("input", "") or ""
        if not text or len(text) < 3: continue
        feat = extract_features(text)
        label = label_formula(text)
        if len(samples_by_tier[label]) < PER_TIER:
            samples_by_tier[label].append({"text": text, "features": feat, "formula_label": label})
        total_loaded += 1
        if total_loaded % 10000 == 0:
            print(f"    Scanned {total_loaded}...")
        if all(len(v) >= PER_TIER for t, v in samples_by_tier.items() if t != "trivial") and len(samples_by_tier["extreme"]) >= 250:
            break
    
    all_samples = []
    for t in TIERS:
        all_samples.extend(samples_by_tier[t])
    random.Random(42).shuffle(all_samples)
    
    dist = Counter(s["formula_label"] for s in all_samples)
    print(f"  Collected {len(all_samples)} samples: {dict(dist)}")
    
    # 3. LLM judge (PARALLEL)
    print(f"\n  Running LLM judge ({CONCURRENT} concurrent)...")
    asyncio.run(judge_all(all_samples))
    
    llm_dist = Counter(s["llm_label"] for s in all_samples)
    formula_dist = Counter(s["formula_label"] for s in all_samples)
    agree = sum(1 for s in all_samples if s["formula_label"] == s["llm_label"])
    print(f"\n  Label distributions:")
    print(f"    Formula: {dict(formula_dist)}")
    print(f"    LLM:     {dict(llm_dist)}")
    print(f"    Agreement: {agree}/{len(all_samples)} ({agree/len(all_samples):.1%})")
    
    # 4. Train/test split
    random.Random(42).shuffle(all_samples)
    split = int(len(all_samples) * 0.8)
    train_set = all_samples[:split]
    test_set = all_samples[split:]
    
    print(f"\n  Training cascade ({len(train_set)} train, {len(test_set)} test)...")
    models, export = train_cascade(train_set)
    
    # Evaluate on test
    print("\n  Evaluating on test set...")
    correct = 0
    tier_correct = Counter()
    tier_total = Counter()
    for s in test_set:
        x = features_to_vector(s["features"])
        pred = predict_cascade(models, x)
        actual = s["llm_label"]
        tier_total[actual] += 1
        if pred == actual:
            correct += 1
            tier_correct[actual] += 1
    
    print(f"  Overall: {correct}/{len(test_set)} ({correct/len(test_set):.1%})")
    for t in TIERS:
        tt = tier_total.get(t, 0)
        if tt > 0:
            print(f"    {t}: {tier_correct.get(t,0)}/{tt} ({tier_correct.get(t,0)/tt:.1%})")
    
    # Calibration
    print("\n  Calibration test:")
    cal = [
        ("trivial", "Hi"),
        ("trivial", "What is Python?"),
        ("light", "Summarize the following text about machine learning"),
        ("moderate", "Explain how async/await works in JavaScript with examples"),
        ("heavy", "Design a distributed microservice architecture for a fintech payment system with event sourcing, CQRS, and multi-region failover"),
        ("intensive", "Build a complete real-time trading platform with WebSocket feeds, order matching engine, risk management, regulatory compliance reporting, and disaster recovery across 3 cloud regions"),
        ("extreme", "Architect a globally distributed autonomous AI agent orchestration platform with real-time model fine-tuning, federated learning across jurisdictions, zero-knowledge proof verification, custom ASIC acceleration, and self-healing infrastructure"),
    ]
    cal_ok = 0
    for exp, prompt in cal:
        feat = extract_features(prompt)
        x = features_to_vector(feat)
        pred = predict_cascade(models, x)
        dist_val = abs(TIERS.index(exp) - TIERS.index(pred))
        ok = "✅" if dist_val == 0 else ("≈" if dist_val == 1 else "❌")
        if dist_val == 0: cal_ok += 1
        print(f"    {exp:<12} → {pred:<12} {ok}")
    print(f"    Calibration: {cal_ok}/{len(cal)} exact")
    
    # Export
    weights = {
        "version": "v0.3.5-llm-judged",
        "trained": datetime.now(timezone.utc).isoformat() + "Z",
        "dataset": f"{len(all_samples)} LLM-judged (Alpaca, glm-4.7-flash)",
        "method": "pairwise-cascade-on-llm-labels",
        "concurrent_judge": CONCURRENT,
        "test_accuracy": round(correct / max(len(test_set), 1), 4),
        "calibration": f"{cal_ok}/{len(cal)}",
        "feature_names": FEATURE_NAMES,
        "classifiers": export,
        "tier_boundaries": [0.08, 0.18, 0.32, 0.52, 0.72],
        "label_distributions": {"formula": dict(formula_dist), "llm_judge": dict(llm_dist)},
        "agreement_formula_vs_llm": round(agree / len(all_samples), 4),
        "training_time_s": round(time.time() - t0, 1),
    }
    
    with open("/workspace/v33_llm_judged_weights.json", "w") as f:
        json.dump(weights, f, indent=2)
    
    print(f"\n  === DONE ===")
    print(f"  Weights: /workspace/v33_llm_judged_weights.json")
    print(f"  Total time: {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
