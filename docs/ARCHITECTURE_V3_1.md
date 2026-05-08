# Architecture v3.1 — LLMFit: Self-Optimizing Agentic Gateway

**Version:** 3.1  
**Date:** 2026-05-07  
**Status:** Architecture Defined  
**Parent:** ARCHITECTURE.md v1.0 (Core Trinity)  
**Supersedes:** MoA Gateway v3.0 heuristic weights (87.2% accuracy baseline)

---

## Executive Summary

v3.1 introduces **LLMFit** — a closed-loop, RAG-powered system that allows any user to create personalized fine-tuning datasets from their own context, and a **self-improving MoA Gateway Router** that uses LLM self-evaluation to continuously optimize routing accuracy. The system ships with a **General-Purpose Dataset** pre-optimized for trivial/light prompts, enabling automatic accuracy gains on the most frequent task tier.

### Problem Statement

- MoA Gateway v3.0 achieves 87.2% accuracy using synthetic Alpaca data, but these are generic prompts — not the user's actual workload.
- Routing accuracy is limited by how well the heuristic feature weights match the user's real prompt distribution.
- No mechanism exists for users to fine-tune the router on their own data.
- Light/trivial prompts (65%+ of all traffic) dominate cost savings but have the least per-prompt optimization investment.

### v3.1 Solution

1. **LLMFit Dataset Factory** — RAG pipeline that extracts, labels, and packages user context into fine-tuning datasets.
2. **MoA Self-Improvement Loop** — LLMs evaluate their own routing decisions, generating labeled feedback that optimizes heuristic weights.
3. **General-Purpose Dataset** — Pre-built dataset of 50K+ trivial/light prompts, auto-optimized with the user's context for maximum light-tier accuracy.

---

## 1. LLMFit — RAG-Based Personalized Dataset Factory

### 1.1 Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                      LLMFit Dataset Factory                       │
│                                                                   │
│  ┌──────────┐    ┌──────────────┐    ┌───────────────┐           │
│  │  Source   │───▶│  Extractor   │───▶│  Labeller     │           │
│  │  Connectors│   │  (RAG Chunk) │    │  (LLM + Rules)│           │
│  └──────────┘    └──────────────┘    └───────────────┘           │
│       │                                    │                      │
│       ▼                                    ▼                      │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────────┐       │
│  │  Workspace   │    │  Chat Logs   │    │  Tool Outputs │       │
│  │  Files       │    │  (Sessions)  │    │  (Exec/BMAD)  │       │
│  └──────────────┘    └──────────────┘    └───────────────┘       │
│                                                │                  │
│                                                ▼                  │
│                                         ┌───────────────┐        │
│                                         │  Quality Gate │        │
│                                         │  (Validator)  │        │
│                                         └───────────────┘        │
│                                                │                  │
│                                                ▼                  │
│                                         ┌───────────────┐        │
│                                         │  Dataset      │        │
│                                         │  (JSONL)      │        │
│                                         └───────────────┘        │
│                                                │                  │
│                              ┌─────────────────┼────────────┐    │
│                              ▼                 ▼            ▼    │
│                         ┌────────┐      ┌──────────┐  ┌───────┐ │
│                         │ MoA    │      │ Fine-    │  │Export │ │
│                         │ Router │      │ Tune     │  │HF/    │ │
│                         │ Weights│      │ LLM      │  │Local  │ │
│                         └────────┘      └──────────┘  └───────┘ │
└──────────────────────────────────────────────────────────────────┘
```

### 1.2 Source Connectors

The LLMFit factory ingests from three primary sources:

| Connector | Data Source | Format | Privacy |
|-----------|-------------|--------|---------|
| `workspace-scanner` | `/root/.openclaw/workspace/` (files, dirs, code, docs) | File tree + content scan | Local-only, no external access |
| `session-miner` | OpenClaw session transcripts (via `sessions_history`) | Chat JSONL | Requires user consent flag |
| `tool-trace` | Exec/process/browser tool call logs | Tool call JSONL | Strips sensitive args by default |

#### Connector Configuration

```json
{
  "llmfit": {
    "connectors": {
      "workspace-scanner": {
        "enabled": true,
        "paths": ["/root/.openclaw/workspace"],
        "exclude": ["node_modules", ".git", "coverage"],
        "maxFileSizeKb": 512
      },
      "session-miner": {
        "enabled": true,
        "maxSessions": 100,
        "ageDays": 30,
        "consentRequired": true
      },
      "tool-trace": {
        "enabled": true,
        "stripArgs": ["api_key", "password", "token", "secret"],
        "maxEntriesPerSession": 50
      }
    }
  }
}
```

### 1.3 Extractor — RAG Chunking Strategy

The extractor converts raw sources into structured training samples:

```python
@dataclass
class TrainingSample:
    id: str                    # Unique sample ID
    source: str                # Source connector name
    raw_text: str              # Extracted text (max 4K chars)
    context: dict              # Metadata: file path, session key, timestamp
    complexity_hint: str       # Initial complexity estimate: trivial|light|moderate|heavy|intensive|extreme
    intent_type: str           # query|command|analysis|creation|review|debug
    domain: str                # Domain label: coding|ops|research|writing|design|general
    features: dict             # Extracted feature vector (see §1.4)
```

#### Chunking Algorithm

1. **File-based sources:** Split at natural boundaries (function defs, class defs, markdown headers). Target 200-800 token chunks.
2. **Session transcripts:** Extract individual turns. Each user message + agent response = 1 sample. Deduplicate near-identical turns.
3. **Tool traces:** Extract the prompt context + tool name + success/failure. Group related tool calls into a single sample.

```python
def chunk_workspace_file(filepath: str, content: str) -> list[TrainingSample]:
    """Chunk a file into training samples with context preservation."""
    samples = []
    
    if filepath.endswith(('.py', '.ts', '.js')):
        # Split at function/class definitions
        chunks = split_at_def_blocks(content)
    elif filepath.endswith(('.md', '.txt')):
        # Split at headers
        chunks = split_at_headers(content)
    else:
        # Fixed-size sliding window with overlap
        chunks = sliding_window(content, window=4000, overlap=500)
    
    for i, chunk in enumerate(chunks):
        sample = TrainingSample(
            id=f"ws_{hashlib.md5(filepath.encode()).hexdigest()[:8]}_{i:04d}",
            source="workspace-scanner",
            raw_text=truncate(chunk, 4000),
            context={"file": filepath, "chunk": i, "total_chunks": len(chunks)},
            complexity_hint=estimate_complexity(chunk),
            intent_type=classify_intent(chunk),
            domain=classify_domain(chunk),
            features=extract_features(chunk),
        )
        samples.append(sample)
    
    return samples
```

### 1.4 Feature Extraction (LLMFit Standard Vector)

Each sample produces a standardized feature vector compatible with the MoA Gateway's optimizer:

| Feature | Type | Description | Weight Range |
|---------|------|-------------|--------------|
| `word_count` | int | Total word count | 0.00–0.30 |
| `sentence_count` | int | Sentence boundaries | 0.00–0.30 |
| `avg_word_length` | float | Avg characters per word | 0.00–0.20 |
| `has_code` | bool | Contains code blocks/backticks | 0.00–0.15 |
| `has_question` | bool | Contains `?` or interrogative | 0.00–0.15 |
| `has_imperative` | bool | Starts with verb/command | 0.00–0.15 |
| `technical_terms` | int | Count of technical keywords | 0.00–0.15 |
| `question_technical` | bool | Technical question (how/why/architecture) | 0.00–0.15 |
| `architecture` | bool | Architecture/design keywords | 0.00–0.30 |
| `technical_design` | bool | System design terms | 0.00–0.20 |
| `multi_step` | bool | Multiple steps/requirements implied | 0.00–0.15 |
| `requires_context` | bool | Needs external knowledge/files | 0.00–0.15 |
| `code_language` | str | Detected language (py/ts/rs/etc.) | categorical |
| `domain_specificity` | float | Domain jargon density (0-1) | 0.00–0.20 |
| `ambiguity_score` | float | Vague/imprecise language (0-1) | 0.00–0.15 |

### 1.5 Labeller — Dual-Mode Labeling

The labeller assigns ground-truth complexity labels using two complementary modes:

#### Mode A: Rule-Based (Fast, Deterministic)

```python
RULE_LABELS = {
    # Trivial: <50 words, no code, no technical terms, no multi-step
    "trivial": lambda s: s.word_count < 50 and not s.has_code 
                         and s.technical_terms == 0 and not s.multi_step,
    
    # Light: <200 words, simple questions, file reads, summaries
    "light": lambda s: s.word_count < 200 and not s.multi_step
                       and not s.architecture and s.technical_terms <= 2,
    
    # Moderate: 200-500 words, code analysis, light refactoring
    "moderate": lambda s: 50 <= s.word_count <= 500 and (s.has_code or s.technical_terms >= 3),
    
    # Heavy: >500 words, multi-file context, system design hints
    "heavy": lambda s: s.word_count > 300 and (s.multi_step or s.technical_design),
    
    # Intensive: Architecture, complex refactoring, multi-step workflows
    "intensive": lambda s: s.architecture and (s.multi_step or s.requires_context),
    
    # Extreme: Critical decisions, security, deep reasoning
    "extreme": lambda s: s.architecture and s.technical_design and s.multi_step,
}
```

#### Mode B: LLM-Assisted (Slower, Higher Accuracy)

For samples where rule-based labels have low confidence (overlapping rules), an LLM provides a secondary label:

```python
LLM_LABEL_PROMPT = """
Classify the following prompt into exactly ONE complexity tier:

Tiers:
- trivial: Simple questions, greetings, status checks, single-fact queries
- light: Summaries, file reads, formatting, basic explanations
- moderate: Code analysis, light refactoring, error debugging
- heavy: Multi-file analysis, feature implementation, system explanations
- intensive: Architecture design, complex refactoring, workflow automation
- extreme: Critical decisions, security analysis, multi-system integration

Prompt: {text}

Respond with ONLY the tier name (one word).
"""
```

**Label Resolution Strategy:**
- If Mode A and Mode B agree → use that label (high confidence)
- If they disagree → flag for human review OR use Mode B (LLM override)
- If Mode A confidence is high (only one rule matches) → skip Mode B (cost savings)

### 1.6 Quality Gate

Before a dataset is considered valid, it passes through quality validation:

| Check | Threshold | Action |
|-------|-----------|--------|
| Min samples | 100 | Reject dataset |
| Class balance | No class <5% of total | Warn + suggest oversampling |
| Duplicate rate | <10% exact duplicates | Auto-deduplicate |
| Text quality | No samples <5 words or >4000 chars | Filter |
| Label distribution entropy | >1.0 bits | Reject (all-one-class) |
| Feature variance | All features have non-zero variance | Reject (constant features) |

```python
def validate_dataset(samples: list[TrainingSample]) -> ValidationResult:
    """Run all quality gates on a dataset."""
    issues = []
    
    if len(samples) < 100:
        issues.append(f"Too few samples: {len(samples)} < 100")
    
    labels = Counter(s.complexity_hint for s in samples)
    total = len(samples)
    for label, count in labels.items():
        pct = count / total
        if pct < 0.05:
            issues.append(f"Class '{label}' underrepresented: {pct:.1%} < 5%")
    
    # ... more checks
    
    return ValidationResult(
        passed=len(issues) == 0,
        issues=issues,
        sample_count=total,
        class_distribution=dict(labels),
    )
```

### 1.7 Dataset Output Format

```jsonl
{"id": "llmfit_0001", "source": "workspace-scanner", "text": "...", "label": "light", "features": {"word_count": 45, "sentence_count": 3, "has_code": false, ...}, "domain": "coding", "intent": "query", "confidence": 0.92, "created_at": "2026-05-07T23:30:00Z"}
{"id": "llmfit_0002", "source": "session-miner", "text": "...", "label": "moderate", "features": {"word_count": 180, "sentence_count": 8, "has_code": true, ...}, "domain": "ops", "intent": "command", "confidence": 0.87, "created_at": "2026-05-07T23:30:01Z"}
```

### 1.8 Export Targets

| Target | Format | Use Case |
|--------|--------|----------|
| MoA Gateway Weights | JSON weights file | Direct optimizer input for heuristic routing |
| Fine-Tune JSONL | Alpaca/ShareGPT format | LLM fine-tuning (LoRA, QLoRA, full) |
| HuggingFace Dataset | `datasets` library format | Public sharing, community benchmarking |
| Local SQLite | `llmfit.db` | Incremental updates, querying, analytics |

---

## 2. MoA Gateway Router v3.1 — Self-Improving Agentic Accuracy

### 2.1 The Self-Improvement Loop

The core innovation of v3.1 is a **closed feedback loop** where the MoA Gateway continuously improves its routing accuracy using real-time data:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Self-Improvement Loop                         │
│                                                                  │
│  User Prompt                                                     │
│       │                                                          │
│       ▼                                                          │
│  ┌────────────┐    ┌────────────┐    ┌────────────┐             │
│  │  Feature   │───▶│  Route     │───▶│  Execute   │             │
│  │  Extract   │    │  (Heuristic)│   │  (LLM)     │             │
│  └────────────┘    └────────────┘    └────────────┘             │
│                              │                   │               │
│                              │                   ▼               │
│                              │            ┌────────────┐        │
│                              │            │  Evaluate  │        │
│                              │            │  (Self-Rate)│       │
│                              │            └────────────┘        │
│                              │                   │               │
│                              ▼                   ▼               │
│                       ┌──────────────────────────────┐          │
│                       │  Feedback Buffer              │          │
│                       │  (prompt, predicted, actual,  │          │
│                       │   self_rating, cost, latency) │          │
│                       └──────────────────────────────┘          │
│                                          │                       │
│                                          ▼                       │
│                                   ┌────────────┐                │
│                                   │  Optimizer │                │
│                                   │  (Weekly)  │                │
│                                   └────────────┘                │
│                                          │                       │
│                                          ▼                       │
│                                   ┌────────────┐                │
│                                   │  Weights   │                │
│                                   │  (Updated) │                │
│                                   └────────────┘                │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Self-Evaluation Protocol

After each LLM execution, the gateway triggers a lightweight self-evaluation:

```python
@dataclass
class RoutingFeedback:
    prompt_hash: str          # SHA-256 of prompt (privacy-preserving)
    predicted_tier: str       # What the router predicted
    actual_tier: str          # What tier was actually needed
    self_rating: float        # LLM self-assessed fit (0.0-1.0)
    cost_actual: float        # Actual cost incurred
    cost_predicted: float     # Predicted cost for predicted tier
    latency_ms: int           # Actual response time
    tokens_used: int          # Actual token consumption
    was_overkill: bool        # Could a cheaper tier have handled this?
    was_underkill: bool       # Did the tier struggle? (evidenced by retries, long latency)
    timestamp: str
```

#### Self-Evaluation Prompt (runs on L0/L1 model for cost efficiency)

```python
SELF_EVAL_PROMPT = """
Given the following prompt and response, evaluate routing accuracy:

PROMPT: {prompt_summary}
RESPONSE_TIER_USED: {tier_used}
RESPONSE_COST: ${cost}
RESPONSE_LATENCY: {latency}ms
RESPONSE_TOKENS: {tokens}

Rate how well the chosen tier matched the task difficulty:
- If the task was too easy for this tier (overkill): rate low
- If the task was too hard for this tier (underkill): rate low  
- If the tier was appropriate: rate high

Respond with JSON:
{{"self_rating": 0.0-1.0, "was_overkill": bool, "was_underkill": bool, "actual_complexity": "trivial|light|moderate|heavy|intensive|extreme"}}
"""
```

#### Cost-Efficient Self-Eval Routing

| Tier of original response | Self-eval model | Cost per eval |
|---------------------------|-----------------|---------------|
| trivial/light | glm-4.5-air:free | $0.00 |
| moderate | glm-4.7-flash | $0.0001 |
| heavy+ | glm-4.7-flash | $0.0001 |

**Strategy:** Always use the cheapest model for self-evaluation since it's just classifying complexity, not generating content. This adds ~$0.0001 per request to the feedback loop.

### 2.3 Heuristic Weight Optimizer

The optimizer runs on a schedule (default: weekly) and recalculates the best heuristic weights:

```python
def optimize_weights_from_feedback(
    feedback_records: list[RoutingFeedback],
    current_weights: dict[str, float],
    method: str = "scipy_mse",
) -> dict[str, float]:
    """
    Optimize heuristic weights from real routing feedback.
    
    Uses the feedback buffer to:
    1. Build ground truth labels from self-evaluation (actual_complexity)
    2. Re-extract features for each prompt
    3. Run scipy minimize to find weights that minimize misrouting MSE
    4. Return optimized weights
    
    Constraints:
    - All weights >= 0
    - Sum of weights <= 1.0 (normalization)
    - No single weight > 0.35 (prevent overfitting to one feature)
    """
    # Build feature matrix and ground truth
    X = np.array([extract_features(r.prompt_hash) for r in feedback_records])
    y_true = np.array([tier_to_score(r.actual_complexity) for r in feedback_records])
    
    # Define MSE objective
    def mse_objective(weights):
        scores = X @ weights
        return np.mean((scores - y_true) ** 2)
    
    # Optimize with constraints
    bounds = [(0, 0.35)] * len(current_weights)
    result = minimize(
        mse_objective,
        x0=list(current_weights.values()),
        bounds=bounds,
        method="L-BFGS-B",
    )
    
    # Normalize to sum=1
    optimized = result.x
    optimized = optimized / optimized.sum()
    
    return dict(zip(current_weights.keys(), optimized))
```

### 2.4 Accuracy Metrics & Monitoring

The gateway tracks these metrics in real-time:

| Metric | Description | Target |
|--------|-------------|--------|
| **Routing Accuracy** | % of prompts routed to correct tier | >90% |
| **Cost Savings** | % saved vs. baseline (always-extreme) | >80% |
| **Overkill Rate** | % routed to higher tier than needed | <8% |
| **Underkill Rate** | % routed to lower tier than needed | <5% |
| **Feedback Coverage** | % of requests with self-eval data | >95% |
| **Weight Drift** | Change in weights since last optimization | <0.10 per feature |
| **Stale Feedback Age** | Hours since last feedback record | <24h |

### 2.5 MoA Gateway v3.1 API Additions

```
GET    /v3/feedback           # View feedback buffer
POST   /v3/feedback/purge     # Clear feedback buffer (with retention option)
GET    /v3/weights            # Current heuristic weights
POST   /v3/optimize           # Trigger manual weight optimization
GET    /v3/metrics/accuracy   # Real-time routing accuracy
GET    /v3/datasets           # List available datasets
POST   /v3/datasets/generate  # Trigger LLMFit dataset generation
POST   /v3/datasets/import    # Import external dataset (JSONL)
GET    /v3/datasets/:id/stats # Dataset statistics
```

---

## 3. General-Purpose Dataset — Light Prompt Optimization

### 3.1 Purpose

The **General-Purpose Dataset** (GPD) is a pre-built, continuously updated dataset specifically optimized for trivial and light prompts — the highest-volume tier that drives the most cost savings.

**Rationale:** In v3.0, light tier accuracy was 94.9% on synthetic data but drops significantly on real user prompts. The GPD bridges this gap by:

1. Providing a baseline of 50K+ real-world trivial/light prompts
2. Auto-enriching with the user's own context patterns
3. Continuously updating from the self-improvement feedback loop

### 3.2 Dataset Composition

| Source | Samples | Tier Distribution | Update Frequency |
|--------|---------|-------------------|------------------|
| Alpaca (trivial/light subset) | 15,000 | 60% trivial, 40% light | Static |
| Self-Instruct (simple tasks) | 10,000 | 50% trivial, 50% light | Static |
| Synthetic (pattern-generated) | 15,000 | 70% trivial, 30% light | Weekly |
| User context (LLMFit) | Variable | Per-user distribution | Continuous |
| Feedback loop (real routing) | Accumulating | Real distribution | Per-request |

**Total baseline:** 40,000 synthetic + public samples + growing user-specific data.

### 3.3 Synthetic Light Prompt Generator

```python
LIGHT_PROMPT_TEMPLATES = [
    # Trivial patterns
    "What is {concept}?",
    "Explain {concept} briefly",
    "Summarize this: {short_text}",
    "How many {unit} in a {thing}?",
    "What does {acronym} stand for?",
    "Is {thing} a type of {category}?",
    "List the top {n} {items}",
    "Convert {value} to {unit}",
    
    # Light patterns
    "Read the file at {path} and tell me what it does",
    "Format this code: {code_snippet}",
    "What's wrong with this error: {error_message}",
    "Explain this function: {function_code}",
    "Add a docstring to: {function_code}",
    "Rename variables in: {code_snippet}",
    "Write a test for: {function_signature}",
    "Check if {file} exists and return its size",
]
```

### 3.4 Context Auto-Enrichment

The GPD automatically enriches itself from user context:

```python
def enrich_gpd_with_user_context(
    gpd_samples: list[TrainingSample],
    user_workspace: WorkspaceProfile,
    recent_sessions: list[SessionTurn],
) -> list[TrainingSample]:
    """
    Augment the general-purpose dataset with user-specific patterns:
    
    1. Extract domain-specific terminology from workspace files
    2. Mine common prompt patterns from recent sessions
    3. Generate synthetic prompts matching user's style
    4. Add to GPD with user-specific label
    """
    # Extract user's domain vocabulary
    domain_terms = extract_domain_vocabulary(user_workspace)
    
    # Mine frequent prompt patterns
    common_patterns = mine_prompt_patterns(recent_sessions)
    
    # Generate enriched samples
    enriched = []
    for pattern in common_patterns:
        for term in domain_terms[:10]:  # Top 10 terms
            sample = generate_synthetic_from_pattern(pattern, term)
            sample.source = "gpd-enriched"
            sample.confidence = 0.75  # Lower confidence for synthetic
            enriched.append(sample)
    
    return gpd_samples + enriched
```

### 3.5 GPD Optimization Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│              GPD Optimization Pipeline                        │
│                                                               │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐               │
│  │  Baseline│───▶│  Enrich  │───▶│  Split   │               │
│  │  (40K)   │    │  (+user) │    │  80/20   │               │
│  └──────────┘    └──────────┘    └──────────┘               │
│                                        │                      │
│                                        ▼                      │
│                                 ┌──────────────┐             │
│                                 │  Optimize    │             │
│                                 │  Weights     │             │
│                                 │  (light only)│             │
│                                 └──────────────┘             │
│                                        │                      │
│                                        ▼                      │
│                                 ┌──────────────┐             │
│                                 │  Merge with  │             │
│                                 │  Global      │             │
│                                 │  Weights     │             │
│                                 └──────────────┘             │
│                                        │                      │
│                                        ▼                      │
│                                 ┌──────────────┐             │
│                                 │  Deploy to   │             │
│                                 │  Gateway     │             │
│                                 └──────────────┘             │
└─────────────────────────────────────────────────────────────┘
```

### 3.6 GPD File Structure

```
llmfit/datasets/
├── general-purpose/
│   ├── baseline_alpaca_light.jsonl       # 15K Alpaca trivial/light
│   ├── baseline_selfinstruct.jsonl       # 10K Self-Instruct simple
│   ├── baseline_synthetic.jsonl          # 15K pattern-generated
│   ├── user_enriched.jsonl              # User-specific (growing)
│   ├── feedback_accumulated.jsonl       # Real routing feedback
│   ├── combined.jsonl                   # Full merged dataset
│   ├── train.jsonl                      # 80% split
│   ├── test.jsonl                       # 20% split
│   └── stats.json                       # Dataset statistics
├── custom/
│   └── <user-dataset-id>/
│       ├── raw.jsonl                    # Extracted samples
│       ├── labeled.jsonl                # Labeled samples
│       ├── validated.jsonl              # Quality-gated samples
│       └── report.md                    # Generation report
└── README.md                            # Dataset documentation
```

---

## 4. Fine-Tuning Integration

### 4.1 Supported Fine-Tuning Targets

LLMFit can produce datasets for multiple fine-tuning scenarios:

| Target | Dataset Format | Use Case | Effort |
|--------|---------------|----------|--------|
| **MoA Router** | Feature weights JSON | Heuristic weight optimization | Low (scipy) |
| **Small Classifier** | scikit-learn JSONL | Light tier classification model | Low (sklearn) |
| **LoRA Adapter** | Alpaca/ShareGPT JSONL | LLM prompt-complexity classification | Medium (GPU hours) |
| **Full Fine-Tune** | ShareGPT JSONL | Custom routing LLM | High (GPU days) |

### 4.2 Accuracy vs. Precision Tiers

Users can target different accuracy/precision levels based on their needs:

| Level | Method | Expected Accuracy | Cost | Latency |
|-------|--------|-------------------|------|---------|
| **L1 — Heuristic** | scipy weight optimization | 85-92% | $0 (CPU) | <1ms |
| **L2 — Classifier** | sklearn (RF/XGBoost) | 90-95% | $0 (CPU) | <5ms |
| **L3 — LoRA** | QLoRA on 7B model | 93-97% | $2-5 (GPU hrs) | <50ms |
| **L4 — Fine-Tune** | Full fine-tune on 7-13B | 95-98% | $20-50 (GPU days) | <100ms |

**Recommendation:** Start with L1 (heuristic), upgrade to L2 when feedback buffer exceeds 10K samples, consider L3+ only for enterprise workloads.

### 4.3 Fine-Tuning Workflow

```bash
# Step 1: Generate dataset from workspace context
llmfit generate --source workspace --output llmfit/datasets/custom/my-dataset/raw.jsonl

# Step 2: Label the dataset (rule-based + LLM-assisted)
llmfit label --input llmfit/datasets/custom/my-dataset/raw.jsonl \
             --mode hybrid \
             --output llmfit/datasets/custom/my-dataset/labeled.jsonl

# Step 3: Validate quality
llmfit validate --input llmfit/datasets/custom/my-dataset/labeled.jsonl

# Step 4: Optimize router weights (L1 — default)
llmfit optimize --input llmfit/datasets/custom/my-dataset/labeled.jsonl \
                --method scipy_mse \
                --output llmfit/datasets/custom/my-dataset/weights.json

# Step 5: Deploy to gateway
llmfit deploy --weights llmfit/datasets/custom/my-dataset/weights.json \
              --target http://localhost:8900/v3/weights

# Step 6 (optional): Train sklearn classifier (L2)
llmfit train --input llmfit/datasets/custom/my-dataset/labeled.jsonl \
             --model xgboost \
             --output llmfit/datasets/custom/my-dataset/classifier.pkl

# Step 7 (optional): Generate LoRA fine-tuning data (L3)
llmfit export --input llmfit/datasets/custom/my-dataset/labeled.jsonl \
              --format alpaca \
              --output llmfit/datasets/custom/my-dataset/alpaca.json
```

---

## 5. Implementation Plan

### Phase 1: LLMFit Core (Week 1-2)

| Task | Effort | Dependencies |
|------|--------|-------------|
| Workspace scanner connector | 2 days | None |
| RAG chunking engine | 3 days | Workspace scanner |
| Feature extraction pipeline | 2 days | Chunking engine |
| Rule-based labeller | 1 day | Feature extraction |
| Quality gate validator | 1 day | Labeller |
| Dataset output (JSONL) | 1 day | Quality gate |

### Phase 2: MoA Self-Improvement Loop (Week 2-3)

| Task | Effort | Dependencies |
|------|--------|-------------|
| Self-evaluation prompt + integration | 2 days | MoA Gateway v3.0 |
| Feedback buffer (SQLite) | 1 day | Self-evaluation |
| Weight optimizer (scipy) | 2 days | Feedback buffer |
| Gateway v3.1 API endpoints | 2 days | Weight optimizer |
| Accuracy metrics dashboard | 1 day | Gateway API |

### Phase 3: General-Purpose Dataset (Week 3-4)

| Task | Effort | Dependencies |
|------|--------|-------------|
| Download + curate Alpaca light subset | 1 day | None |
| Download + curate Self-Instruct | 1 day | None |
| Synthetic prompt generator | 2 days | Dataset curation |
| Context auto-enrichment | 2 days | LLMFit core |
| GPD optimization pipeline | 2 days | Weight optimizer |
| Integration with MoA Gateway | 1 day | GPD pipeline |

### Phase 4: Fine-Tuning Integration (Week 4-5)

| Task | Effort | Dependencies |
|------|--------|-------------|
| LLM-assisted labeller (Mode B) | 2 days | LLMFit core |
| Sklearn classifier training (L2) | 2 days | Labeled datasets |
| LoRA export format | 1 day | Labeled datasets |
| CLI tooling (`llmfit`) | 3 days | All above |
| Documentation + examples | 2 days | All above |

---

## 6. Security & Privacy

### 6.1 Data Handling Principles

1. **Local-first:** All LLMFit processing happens locally. No data leaves the workspace.
2. **Hash-based IDs:** Sample IDs use SHA-256 hashes, not raw content identifiers.
3. **Opt-in connectors:** Each source connector requires explicit enablement.
4. **Sensitive field stripping:** Tool trace connector strips API keys, passwords, tokens by default.
5. **Export consent:** Dataset export to external targets (HuggingFace) requires explicit user approval.

### 6.2 Privacy-Aware Self-Evaluation

```python
def safe_self_eval_prompt(prompt: str) -> str:
    """Create self-eval prompt without leaking sensitive content."""
    # Replace potential secrets
    sanitized = re.sub(r'(api_key|password|token|secret)[=:]\S+', r'\1=***REDACTED***', prompt)
    # Truncate to first 500 chars
    return sanitized[:500]
```

---

## 7. Migration from v3.0

### 7.1 Backward Compatibility

- v3.0 heuristic weights are automatically imported as the starting point for v3.1 optimization
- Existing MoA Gateway API endpoints remain unchanged; v3.1 adds new `/v3/` endpoints
- Feedback buffer is empty on first run — optimization uses existing v3.0 weights as x0

### 7.2 Migration Steps

```bash
# 1. Export current v3.0 weights
curl http://localhost:8900/v3/weights > weights_v3.0.json

# 2. Install LLMFit
pip install llmfit  # or: cd infra/sovereign-gateway/llmfit && pip install -e .

# 3. Generate initial dataset from existing workspace
llmfit generate --source workspace --output datasets/migration/raw.jsonl

# 4. Merge with v3.0 baseline
llmfit merge --base datasets/general-purpose/combined.jsonl \
             --add datasets/migration/raw.jsonl \
             --output datasets/v3.1/combined.jsonl

# 5. Optimize new weights
llmfit optimize --input datasets/v3.1/combined.jsonl \
                --init weights_v3.0.json \
                --output weights_v3.1.json

# 6. Deploy
llmfit deploy --weights weights_v3.1.json --target http://localhost:8900/v3/weights
```

---

## 8. Expected Accuracy Improvements

| Scenario | v3.0 Accuracy | v3.1 Accuracy (Projected) | Delta |
|----------|---------------|---------------------------|-------|
| Generic prompts (Alpaca) | 87.2% | 90-92% | +3-5% |
| User-specific prompts | ~70%* | 92-95% | +22-25% |
| Light/trivial tier | 94.9% | 97-98% | +2-3% |
| Moderate tier | 81.1% | 90-93% | +9-12% |
| Heavy tier | 67.4% | 82-87% | +15-20% |
| **Overall (mixed workload)** | **87.2%** | **93-95%** | **+6-8%** |

\* Estimated — v3.0 was trained on generic data, not user-specific prompts.

### Accuracy Trajectory

```
Week 0 (deploy):  87.2%  (v3.0 baseline)
Week 1 (LLMFit):  89.5%  (workspace dataset + optimized weights)
Week 2 (feedback): 91.0% (first feedback-driven optimization)
Week 4 (GPD):     93.0%  (general-purpose dataset merged)
Week 8 (mature):  94-95% (accumulated feedback + user context)
```

---

## 9. File Index

| File | Location | Purpose |
|------|----------|---------|
| Architecture spec | `docs/ARCHITECTURE_V3_1.md` | This document |
| LLMFit core | `llmfit/llmfit.py` | Dataset factory engine |
| Workspace scanner | `llmfit/connectors/workspace.py` | Workspace file connector |
| Session miner | `llmfit/connectors/sessions.py` | Session transcript connector |
| Tool tracer | `llmfit/connectors/tools.py` | Tool call log connector |
| Chunker | `llmfit/extractor/chunker.py` | RAG chunking strategies |
| Features | `llmfit/extractor/features.py` | Feature vector extraction |
| Labeller | `llmfit/labeller/labeller.py` | Dual-mode labeling |
| Validator | `llmfit/quality/gate.py` | Dataset quality validation |
| Optimizer | `llmfit/optimizer/weights.py` | scipy weight optimization |
| GPD generator | `llmfit/datasets/gpd_generator.py` | General-purpose dataset |
| Self-eval | `src/self-eval.ts` | MoA self-evaluation integration |
| Feedback buffer | `src/feedback-buffer.ts` | SQLite feedback storage |
| Gateway v3.1 API | `src/gateway-v3.ts` | New API endpoints |
| CLI tool | `llmfit/cli.py` | `llmfit` command-line interface |

---

*Architecture v3.1 — LLMFit: Self-Optimizing Agentic Gateway*  
*Designed for the Agentic Sovereign Ecosystem*  
*2026-05-07*
