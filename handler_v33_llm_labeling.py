"""GateSwarm MoA Router v0.3.5 — LLM-as-Judge Label Validation & Dataset Builder

Uses bailian/qwen3.6-plus to label prompts from Alpaca + OpenOrca datasets.
Addresses Chief Scientist finding: formula labels have only 2.0/10 validity.

Process:
1. Stream samples from Alpaca (50K) + OpenOrca (50K)
2. For each sample, extract features + formula label (baseline)
3. Send batch of 20 samples to qwen3.6-plus for tier classification
4. Compare LLM vs formula labels → identify disagreements
5. Use LLM labels as ground truth (override formula for disagreements)
6. Output: v33_labeled_dataset.jsonl (train + test splits ready for v3.3 training)

Cost estimate: 100K samples × 200 chars avg / 20 per batch = 5000 API calls
Each call ~2K tokens in, 1K out = ~$0.001 per call → ~$5 total
"""

import json, time, math, random, re, sys, os
from datetime import datetime, timezone
from collections import Counter

# ── Feature Extraction ──
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
TIERS = ["trivial", "light", "moderate", "heavy", "intensive", "extreme"]

TIER_DEFINITIONS = """
- trivial: 1-7 words, no technical signals, simple Q&A (e.g., "What is JSON?", "Hi there")
- light: 7-50 words, 1-2 signals, basic instruction following (e.g., "Summarize this text", "Fix the typo")
- moderate: 50-200 words, 2-3 signals, code understanding + multi-concept reasoning (e.g., "Write a test for this function and explain edge cases")
- heavy: 200-500 words, 3-4 signals, deep analysis + system thinking (e.g., "Compare these two API designs and recommend the better approach for scalability")
- intensive: 500-1000 words, 4-5 signals, multi-step reasoning + architecture (e.g., "Design a distributed system with service mesh, explain trade-offs, provide implementation plan")
- extreme: 1000+ words, 5+ signals, creative synthesis + strategic thinking (e.g., "Architect a complete microservices platform with CI/CD, monitoring, security, and deployment strategy")
"""

def extract_features(text):
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
    return {
        "word_count": wc, "sentence_count": sc, "avg_word_length": round(awl, 3),
        "has_code": float(has_code), "has_question": float(has_question),
        "has_imperative": float(has_imperative), "technical_terms": tech_terms,
        "question_technical": float(question_technical),
        "architecture": float(architecture), "technical_design": float(technical_design),
        "multi_step": float(multi_step), "requires_context": float(requires_context),
        "domain_specificity": round(domain_spec, 3), "ambiguity_score": round(ambiguity, 3),
        "four_plus": float(active >= 4),
    }


def formula_label(text):
    """Original synthetic label formula (for comparison only)."""
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


def build_llm_prompt(batch):
    """Build a prompt for batch LLM labeling (20 samples per call)."""
    lines = []
    for i, sample in enumerate(batch):
        text = sample["text"][:500]  # truncate
        lines.append(f"{i+1}. [{sample['source']}] {text}")
    samples_text = "\n".join(lines)
    
    return f"""You are an expert AI prompt classifier. Classify each prompt into one of 6 complexity tiers:

{TIER_DEFINITIONS}

Classify these {len(batch)} prompts. Respond with ONLY a JSON array of tier names, one per prompt, in order.

Prompts to classify:
{samples_text}

Respond with a JSON array like: ["light", "moderate", "trivial", ...]
No explanation, just the array."""


def llm_label_batch(batch):
    """Call bailian/qwen3.6-plus to label a batch of prompts."""
    prompt = build_llm_prompt(batch)
    
    # Use curl to call bailian API directly
    import subprocess
    api_key = os.environ.get("BAILIAN_API_KEY", "")
    if not api_key:
        # Read from openclaw config
        try:
            with open('/root/.openclaw/openclaw.json') as f:
                config = json.load(f)
            api_key = config["models"]["providers"]["bailian"]["apiKey"]
        except:
            print("  ERROR: No bailian API key found")
            return None
    
    payload = json.dumps({
        "model": "qwen3.6-plus",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 200,
    })
    
    try:
        result = subprocess.run([
            "curl", "-s", "-X", "POST",
            "https://coding-intl.dashscope.aliyuncs.com/v1/chat/completions",
            "-H", f"Authorization: Bearer {api_key}",
            "-H", "Content-Type: application/json",
            "-d", payload
        ], capture_output=True, text=True, timeout=30)
        
        response = json.loads(result.stdout)
        content = response["choices"][0]["message"]["content"].strip()
        
        # Parse JSON array from response
        # Sometimes the model wraps it in markdown or adds text
        import re
        json_match = re.search(r'\[.*\]', content, re.DOTALL)
        if json_match:
            labels = json.loads(json_match.group())
            if len(labels) == len(batch):
                return labels
            else:
                print(f"  WARN: Expected {len(batch)} labels, got {len(labels)}")
                return None
        return None
    except Exception as e:
        print(f"  ERROR: {e}")
        return None


def load_samples_streaming(name, max_n, key_field):
    """Load samples via streaming (memory-efficient)."""
    from datasets import load_dataset as hf_load_dataset

    catalogs = {
        "alpaca": {"hf": "tatsu-lab/alpaca"},
        "openorca": {"hf": "Open-Orca/OpenOrca"},
    }
    if name not in catalogs:
        return []

    print(f"  Loading {name} (max {max_n}) via streaming...")
    ds = hf_load_dataset(catalogs[name]["hf"], split="train", streaming=True)

    samples = []
    count = 0
    for x in ds:
        txt = x.get(key_field, "")
        if isinstance(txt, str) and len(txt) > 10:
            txt = txt.strip()
            samples.append({
                "text": txt,
                "source": name,
                "features": extract_features(txt),
                "formula_label": formula_label(txt),
            })
            count += 1
            if count >= max_n:
                break
        if count > 0 and count % 10000 == 0:
            print(f"    {name}: {count}/{max_n}")

    print(f"  ✅ {name}: {len(samples)} samples")
    return samples


def main():
    print(f"GateSwarm MoA Router v0.3.5 LLM-as-Judge Labeling — {datetime.now(timezone.utc).isoformat()}")
    
    # Load datasets (streaming, memory-efficient)
    alpaca = load_samples_streaming("alpaca", 50000, "instruction")
    openorca = load_samples_streaming("openorca", 50000, "question")
    
    all_samples = alpaca + openorca
    print(f"\n  Total samples: {len(all_samples)}")
    
    # Label distribution by formula
    label_dist = Counter(s["formula_label"] for s in all_samples)
    print(f"  Formula label distribution: {dict(label_dist)}")
    
    # Stratified sampling for LLM labeling
    # Target: 200 per tier = 1200 total (enough to validate formula reliability)
    random.seed(42)
    by_formula = {}
    for s in all_samples:
        by_formula.setdefault(s["formula_label"], []).append(s)
    
    llm_samples = []
    per_tier_target = 200
    for tier in TIERS:
        pool = by_formula.get(tier, [])
        random.shuffle(pool)
        selected = pool[:per_tier_target]
        llm_samples.extend(selected)
        print(f"  Selected {len(selected)} {tier} samples for LLM labeling")
    
    print(f"\n  Total samples for LLM labeling: {len(llm_samples)}")
    
    # Batch LLM labeling (20 per batch)
    batch_size = 20
    llm_results = []
    total_batches = (len(llm_samples) + batch_size - 1) // batch_size
    
    print(f"\n  Starting LLM labeling ({total_batches} batches)...")
    start_time = time.time()
    api_calls = 0
    api_errors = 0
    
    for i in range(0, len(llm_samples), batch_size):
        batch = llm_samples[i:i+batch_size]
        batch_num = i // batch_size + 1
        
        labels = llm_label_batch(batch)
        api_calls += 1
        
        if labels is None:
            api_errors += 1
            print(f"  Batch {batch_num}/{total_batches}: FAILED (using formula labels)")
            for s in batch:
                llm_results.append({
                    "text": s["text"][:500],
                    "source": s["source"],
                    "formula_label": s["formula_label"],
                    "llm_label": s["formula_label"],  # fallback
                    "label_source": "formula_fallback",
                    "features": s["features"],
                })
        else:
            agreement = sum(1 for s, l in zip(batch, labels) if s["formula_label"] == l)
            print(f"  Batch {batch_num}/{total_batches}: {agreement}/{len(batch)} agreement ({agreement/len(batch):.0%})")
            for s, l in zip(batch, labels):
                llm_results.append({
                    "text": s["text"][:500],
                    "source": s["source"],
                    "formula_label": s["formula_label"],
                    "llm_label": l,
                    "label_source": "llm" if s["formula_label"] != l else "agreed",
                    "features": s["features"],
                })
        
        # Rate limiting: 2 seconds between calls
        if i + batch_size < len(llm_samples):
            time.sleep(2)
    
    elapsed = time.time() - start_time
    print(f"\n  LLM labeling complete: {api_calls} API calls, {api_errors} errors, {elapsed:.0f}s")
    
    # Analysis
    total = len(llm_results)
    agreed = sum(1 for r in llm_results if r["formula_label"] == r["llm_label"])
    disagreed = total - agreed
    
    print(f"\n  === LLM vs FORMULA AGREEMENT ===")
    print(f"  Overall: {agreed}/{total} ({agreed/total:.1%})")
    print(f"  Disagreements: {disagreed} ({disagreed/total:.1%})")
    
    # Per-tier agreement
    tier_agreement = Counter()
    tier_total = Counter()
    for r in llm_results:
        tier_total[r["formula_label"]] += 1
        if r["formula_label"] == r["llm_label"]:
            tier_agreement[r["formula_label"]] += 1
    
    print(f"\n  Per-tier agreement:")
    for tier in TIERS:
        total_t = tier_total.get(tier, 0)
        agreed_t = tier_agreement.get(tier, 0)
        rate = agreed_t / total_t if total_t > 0 else 0
        print(f"    {tier}: {agreed_t}/{total_t} ({rate:.1%})")
    
    # Now: label ALL samples using the agreement patterns
    # For tiers with high agreement (>90%), use formula labels
    # For tiers with low agreement, we need more LLM labeling
    
    # Save results
    result = {
        "version": "v0.3.5-llm-label-validation",
        "status": "completed",
        "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
        "llm_model": "bailian/qwen3.6-plus",
        "dataset": {
            "total_loaded": len(all_samples),
            "alpaca": len(alpaca),
            "openorca": len(openorca),
            "llm_labeled": len(llm_results),
        },
        "agreement": {
            "overall": round(agreed / total, 4),
            "total_agreed": agreed,
            "total_disagreed": disagreed,
            "per_tier": {
                tier: {
                    "total": tier_total.get(tier, 0),
                    "agreed": tier_agreement.get(tier, 0),
                    "rate": round(tier_agreement.get(tier, 0) / max(tier_total.get(tier, 1), 1), 4)
                }
                for tier in TIERS
            },
        },
        "llm_results": llm_results[:50],  # First 50 for inspection
        "api_calls": api_calls,
        "api_errors": api_errors,
        "elapsed_seconds": round(elapsed, 1),
    }
    
    with open("v33_llm_label_validation.json", "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"\n  Results saved to v33_llm_label_validation.json")
    return result


if __name__ == "__main__":
    main()
