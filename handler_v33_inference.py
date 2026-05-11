"""GateSwarm MoA Router v0.3.5 — Production Inference Handler

RunPod Serverless handler for real-time complexity scoring and model routing.
Loads pre-trained cascade weights, extracts features, predicts tier + confidence,
and returns the recommended model/provider for the MoA Gateway Router.

Usage (serverless):
  POST /v2/{ENDPOINT_ID}/runsync
  {"input": {"prompt": "Write a REST API in Python"}}

  Response:
  {
    "tier": "heavy",
    "confidence": 0.82,
    "score": 0.42,
    "model": "glm-5.1",
    "provider": "zai",
    "features": {...},
    "latency_ms": 12
  }
"""
import runpod
import json
import time
import math
import re
import os
from datetime import datetime, timezone

import numpy as np

# ── Constants ──
FEATURE_NAMES = [
    "sentence_count", "avg_word_length", "has_question", "question_technical",
    "technical_design", "code", "architecture", "word_count",
    "four_plus", "has_imperative", "technical_terms", "multi_step",
    "requires_context", "domain_specificity", "ambiguity_score",
]

TIERS = ["trivial", "light", "moderate", "heavy", "intensive", "extreme"]

TIER_MODELS = {
    "trivial":   {"model": "glm-4.5-air",       "provider": "zai",      "max_tokens": 256},
    "light":     {"model": "glm-4.7-flash",      "provider": "zai",      "max_tokens": 512},
    "moderate":  {"model": "glm-4.7",            "provider": "zai",      "max_tokens": 1024},
    "heavy":     {"model": "glm-5.1",            "provider": "zai",      "max_tokens": 2048},
    "intensive": {"model": "qwen3.6-plus",       "provider": "bailian",  "max_tokens": 4096},
    "extreme":   {"model": "qwen3.6-plus",       "provider": "bailian",  "max_tokens": 8192},
}

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
ARCH_KEYWORDS = {
    "architecture", "design pattern", "system design", "microservice",
    "distributed", "scalable", "load balancer", "event-driven", "service mesh",
    "api gateway", "serverless", "cloud-native", "infrastructure",
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


def extract_features(text: str) -> dict:
    """Extract v3.1 15-feature vector from prompt text."""
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


# ── Cascade Classifier ──
class TierCascade:
    """Cascade of 5 binary logistic classifiers for tier prediction."""

    CASCADE_ORDER = ["trivial", "light", "moderate", "heavy", "intensive"]

    def __init__(self):
        self.weights = {}
        self.intercepts = {}
        self._loaded = False

    def load_weights(self, weights_path: str):
        """Load pre-trained cascade weights from JSON."""
        with open(weights_path, "r") as f:
            data = json.load(f)

        classifiers = data.get("classifiers", {})
        for tier in self.CASCADE_ORDER:
            if tier in classifiers:
                c = classifiers[tier]
                self.weights[tier] = np.array(
                    [c["weights"].get(fn, 0.0) for fn in FEATURE_NAMES],
                    dtype=np.float64
                )
                self.intercepts[tier] = float(c.get("intercept", 0.0))
            else:
                print(f"  ⚠️  Missing weights for tier: {tier}")

        self._loaded = len(self.weights) == 5
        if self._loaded:
            print(f"  ✅ Loaded cascade weights ({len(self.weights)} classifiers)")
        else:
            print(f"  ⚠️  Partial load: {len(self.weights)}/5 classifiers")

    def predict(self, features: dict) -> tuple[str, float, float]:
        """
        Predict tier, confidence, and score using pairwise logit analysis.

        Key insight: logits are monotonically increasing with tier order
        because higher-tier classifiers are trained on harder examples.
        Greedy first-match always returns the lowest tier with prob > 0.5,
        which is almost always "light".

        Fix: Use pairwise logit DIFFERENCES between adjacent classifiers.
        Each classifier k is trained as: positive=tier_k, negative=higher tiers.
        So logit_k - logit_{k+1} > 0 means the prompt looks more like tier_k
        than tier_{k+1}. The transition from positive to negative pairwise diff
        identifies the correct tier boundary.

        Returns:
            (tier, confidence, score)
        """
        if not self._loaded:
            return "moderate", 0.5, 0.25

        x = features_to_vector(features)
        TIER_MIDPOINTS = {
            "trivial": 0.04, "light": 0.13, "moderate": 0.25,
            "heavy": 0.42, "intensive": 0.62, "extreme": 0.86,
        }

        # Compute all logits
        logits = []
        for tier in self.CASCADE_ORDER:
            if tier not in self.weights:
                logits.append(0.0)
                continue
            w = self.weights[tier]
            b = self.intercepts[tier]
            logits.append(float(x @ w + b))

        # Pairwise diffs: logit[i] - logit[i+1]
        # positive = looks more like tier i than tier i+1
        pairwise = [logits[i] - logits[i+1] for i in range(len(logits)-1)]

        # Find the FIRST tier where the pairwise diff is POSITIVE
        # This means the classifier for that tier fires more than the next tier
        # → the prompt is at or below this tier's complexity
        predicted_idx = len(self.CASCADE_ORDER)  # default: extreme
        for i, pw in enumerate(pairwise):
            if pw > 0:
                predicted_idx = i
                break

        tier = self.CASCADE_ORDER[predicted_idx] if predicted_idx < len(self.CASCADE_ORDER) else "extreme"

        # Confidence: magnitude of the winning pairwise diff, normalized
        if predicted_idx < len(pairwise):
            confidence = min(1.0 / (1.0 + math.exp(-pairwise[predicted_idx])), 0.99)
        else:
            confidence = 0.5

        # Score: use heuristic as base, nudge by cascade
        base_score = _heuristic_score(features)
        cascade_midpoint = TIER_MIDPOINTS.get(tier, 0.5)
        # Blend: trust heuristic for well-calibrated tiers, cascade for others
        blend = 0.6 * base_score + 0.4 * cascade_midpoint
        score = round(min(max(blend, 0.0), 1.0), 4)

        # Re-derive tier from blended score (more stable)
        tier = score_to_tier(score)

        return tier, round(confidence, 4), score


# ── Global cascade instance (loaded once per worker) ──
cascade = TierCascade()
_weights_loaded = False


def ensure_weights_loaded():
    global _weights_loaded
    if _weights_loaded:
        return

    # Try multiple paths
    candidates = [
        "/workspace/v32_cascade_weights.json",
        "/workspace/weights/v32_cascade_weights.json",
        os.path.join(os.path.dirname(__file__), "v32_cascade_weights.json"),
    ]

    for path in candidates:
        if os.path.exists(path):
            print(f"  Loading weights from: {path}")
            cascade.load_weights(path)
            _weights_loaded = cascade._loaded
            return

    print("  ⚠️  No weights file found — using fallback scoring")
    _weights_loaded = False


def score_to_tier(score: float) -> str:
    """Map continuous score to tier label."""
    if score < 0.08: return "trivial"
    if score < 0.18: return "light"
    if score < 0.32: return "moderate"
    if score < 0.52: return "heavy"
    if score < 0.72: return "intensive"
    return "extreme"


def _heuristic_score(features: dict) -> float:
    """Original v3.2 weighted feature sum — used as base score."""
    f = features
    DEFAULT_WEIGHTS = {
        "sentence_count": 0.29, "avg_word_length": 0.19, "has_question": 0.12,
        "question_technical": 0.05, "technical_design": 0.12, "code": 0.10,
        "architecture": 0.07, "word_count": 0.00, "four_plus": 0.00,
        "has_imperative": 0.06, "technical_terms": 0.00, "multi_step": 0.00,
        "requires_context": 0.00, "domain_specificity": 0.00, "ambiguity_score": 0.00,
    }
    w = DEFAULT_WEIGHTS
    s = 0.0
    s += w.get("sentence_count", 0) * min(f.get("sentence_count", 0), 10) / 10.0
    s += w.get("avg_word_length", 0) * f.get("avg_word_length", 0) / 10.0
    s += w.get("has_question", 0) * f.get("has_question", 0)
    s += w.get("question_technical", 0) * f.get("question_technical", 0)
    s += w.get("technical_design", 0) * f.get("technical_design", 0)
    s += w.get("code", 0) * f.get("has_code", 0)
    s += w.get("architecture", 0) * f.get("architecture", 0)
    s += w.get("word_count", 0) * math.log1p(f.get("word_count", 0)) / 6.0
    s += w.get("four_plus", 0) * f.get("four_plus", 0)
    s += w.get("has_imperative", 0) * f.get("has_imperative", 0)
    s += w.get("technical_terms", 0) * min(f.get("technical_terms", 0), 10) / 10.0
    s += w.get("multi_step", 0) * f.get("multi_step", 0)
    s += w.get("requires_context", 0) * f.get("requires_context", 0)
    s += w.get("domain_specificity", 0) * f.get("domain_specificity", 0)
    s += w.get("ambiguity_score", 0) * f.get("ambiguity_score", 0)
    if f.get("word_count", 0) < 10 and f.get("architecture", 0) == 0:
        s *= 0.7
    return min(max(s, 0.0), 1.0)


def fallback_score(features: dict) -> tuple[str, float, float]:
    """Fallback heuristic scoring if cascade weights aren't loaded."""
    f = features
    wc = f.get("word_count", 0)
    sc = f.get("sentence_count", 0)
    tech = f.get("technical_terms", 0)
    has_code = f.get("has_code", 0)
    has_arch = f.get("architecture", 0)
    has_design = f.get("technical_design", 0)
    multi_step = f.get("multi_step", 0)

    score = 0.0
    score += min(sc, 10) / 10.0 * 0.15
    score += min(wc, 200) / 200.0 * 0.15
    score += min(tech, 10) / 10.0 * 0.20
    score += has_code * 0.15
    score += has_arch * 0.15
    score += has_design * 0.10
    score += multi_step * 0.10

    if wc < 10 and has_arch == 0:
        score *= 0.7

    score = min(max(score, 0.0), 1.0)

    if score < 0.08:   tier = "trivial"
    elif score < 0.18: tier = "light"
    elif score < 0.32: tier = "moderate"
    elif score < 0.52: tier = "heavy"
    elif score < 0.72: tier = "intensive"
    else:               tier = "extreme"

    return tier, round(score, 4), round(score, 4)


# ── RunPod Handler ──
def handler(event):
    """
    RunPod serverless handler for GateSwarm MoA Router v0.3.5 complexity scoring.

    Input:
      {
        "prompt": "Write a REST API in Python",
        "include_features": true (optional, default false),
        "include_routing": true (optional, default true)
      }

    Output:
      {
        "tier": "heavy",
        "confidence": 0.82,
        "score": 0.42,
        "model": "glm-5.1",
        "provider": "zai",
        "max_tokens": 2048,
        "features": {...},        # if include_features
        "latency_ms": 12
      }
    """
    t0 = time.time()
    inp = event.get("input", {})

    # Support both single prompt and batch
    prompts = inp.get("prompts", [])
    single_prompt = inp.get("prompt")
    if single_prompt and not prompts:
        prompts = [single_prompt]

    if not prompts:
        return {
            "error": "No prompt provided",
            "usage": "Send {'prompt': 'your text'} or {'prompts': ['text1', 'text2']}",
        }

    include_features = inp.get("include_features", False)
    include_routing = inp.get("include_routing", True)

    ensure_weights_loaded()

    results = []
    for prompt in prompts:
        ft0 = time.time()

        features = extract_features(prompt)

        if cascade._loaded:
            tier, confidence, score = cascade.predict(features)
            method = "v3.3-cascade"
        else:
            tier, confidence, score = fallback_score(features)
            method = "v3.3-heuristic-fallback"

        ft_ms = round((time.time() - ft0) * 1000, 2)

        result = {
            "tier": tier,
            "confidence": confidence,
            "score": score,
            "method": method,
            "latency_ms": ft_ms,
        }

        if include_routing:
            routing = TIER_MODELS.get(tier, TIER_MODELS["moderate"])
            result["model"] = routing["model"]
            result["provider"] = routing["provider"]
            result["max_tokens"] = routing["max_tokens"]

        if include_features:
            result["features"] = features

        results.append(result)

    total_ms = round((time.time() - t0) * 1000, 2)

    if len(results) == 1:
        output = results[0]
    else:
        output = {
            "results": results,
            "total_prompts": len(prompts),
            "total_latency_ms": total_ms,
        }

    output["version"] = "v0.3.5-inference"
    output["timestamp"] = datetime.now(timezone.utc).isoformat() + "Z"

    return output


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
