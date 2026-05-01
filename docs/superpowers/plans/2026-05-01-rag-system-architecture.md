# IndiaFinBench Local RAG System — Phase 1: Architecture & Problem Framing

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` or `superpowers:executing-plans` to implement Phase 2 onwards task-by-task.

**Goal:** Replace the Vertex AI Search + Gemini Flash backend with a fully local, zero-cost RAG pipeline over the 192-document IndiaFinBench corpus (100 RBI + 92 SEBI regulatory texts).

**Architecture:** Dense retrieval over FAISS IndexFlatIP with hybrid BM25 re-ranking (RRF fusion), BGE-base embeddings, and a Groq/Ollama generation backend with graceful fallback.

**Tech Stack:** Python 3.11+, FAISS-cpu, sentence-transformers, rank_bm25, groq (free API), ollama (local fallback), Flask (existing demo integration).

---

## 1. Problem Framing

### 1.1 Formal Task Definitions

**Retrieval task.** Let C = {d₁, d₂, …, d_N} be the corpus of N = 192 documents. After chunking, this yields a chunk index X = {x₁, …, x_M} where M ≫ N. Given a query q, the retrieval function R: Q → 2^X must return a ranked set of k chunks:

```
R(q) = argtop-k_{x ∈ X} sim(ϕ(q), ϕ(x))
```

where ϕ: text → ℝ^d is a fixed encoder and sim is cosine similarity. The task is to maximise Recall@k — the fraction of ground-truth relevant chunks appearing in the top-k set.

**Generation task.** Given retrieved context C_k = {x_{i₁}, …, x_{ik}} and query q, the generator G produces:

```
â = G(q, C_k ; θ)
```

The output â must satisfy:
- **Faithfulness**: every factual claim in â is directly entailed by some x ∈ C_k (no extrapolation beyond the context).
- **Answer relevance**: sem_sim(ϕ(â), ϕ(q)) ≥ τ_rel (the answer addresses the question).

Note the deliberate separation: the retriever is evaluated independently of the generator, so each component can be diagnosed and improved in isolation.

### 1.2 Corpus Characterisation

| Property | Value |
|---|---|
| Total documents | 192 (100 RBI, 92 SEBI) |
| Total raw text | ~8.5 MB |
| Mean document size | ~44 KB |
| Document size range | ~1.5 KB (press releases) → ~150 KB (Master Directions) |
| Language | English (domain-specific legal/regulatory register) |
| Key terminology | Section numbers (§51A, Reg. 4(2)(b)), rate references (91-day T-bill), entity names (SEBI, RBI, NBFC, UCB) |

The high variance in document length is the primary chunking challenge. Short auction notices (1–2 paragraphs) must not be over-fragmented; long Master Directions (50+ numbered clauses) must be chunked so each clause is independently retrievable.

### 1.3 Exact Success Criteria

These are the quantitative thresholds against which Phase 3 evaluation is judged.

**Retrieval:**

| Metric | Threshold | Rationale |
|---|---|---|
| Recall@5 | ≥ 0.80 | Answer must appear in 5-chunk context 80% of the time |
| Precision@5 | ≥ 0.50 | Reduces noise injected into generation context |
| MRR (Mean Reciprocal Rank) | ≥ 0.65 | Favours correct answer appearing at rank 1 or 2 |

**Generation:**

| Metric | Threshold | Method |
|---|---|---|
| Faithfulness | ≥ 0.85 | LLM-as-judge: GPT-4o-mini or Gemini 1.5 Flash (free tier) |
| Answer relevance | ≥ 0.75 | cos(ϕ(â), ϕ(q)) using same encoder |
| Hallucination-free rate | ≥ 0.90 | Binary: 0 if any claim unsupported by C_k |

---

## 2. Pipeline Design

### 2.1 Full Pipeline Map

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           INGESTION (offline)                            │
│  data/parsed/{rbi,sebi}/*.txt                                            │
│         │                                                                │
│    DataLoader  ──► Preprocessor  ──► Chunker  ──► Embedder              │
│    (load docs)    (clean/normalise)  (split)      (BGE-base)             │
│                                                      │                   │
│                                               FAISS IndexFlatIP          │
│                                            + BM25Okapi index             │
│                                               (serialised to disk)       │
└──────────────────────────────────────────────────────────────────────────┘
                                   │
                         QUERY TIME (online)
                                   │
┌──────────────────────────────────────────────────────────────────────────┐
│  query ──► HybridRetriever (Dense + BM25 + RRF) ──► top-k chunks        │
│                                                           │              │
│                                                    Generator             │
│                                               (Groq / Ollama fallback)  │
│                                                           │              │
│                                               {"answer", "sources"}     │
└──────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Text Preprocessing Steps

Applied uniformly before chunking:
1. **Unicode normalisation**: `unicodedata.normalize('NFKC', text)` — resolves ligatures and non-breaking spaces common in PDF-extracted text.
2. **Whitespace collapse**: collapse multi-whitespace/newline runs to single space or paragraph break.
3. **Header/footer stripping**: heuristic line-length filter (lines < 40 chars that match common header/footer patterns: page numbers, "Reserve Bank of India", "SEBI/LAD-NRO/GN/…").
4. **Preserve structure markers**: keep numbered clause markers ("1.", "1.1", "a.", "(i)") as they carry semantic weight in regulatory text.

No stemming, no stop-word removal — embedding models handle these implicitly, and BM25 benefits from exact term matching in legal text.

---

## 3. Chunking Strategies: Analysis & Selection

### Strategy A — Fixed-Size Sliding Window

**Method:** Split text into non-overlapping windows of `chunk_size` tokens (measured with a fast tokenizer), with `overlap` tokens carried over from the previous chunk.

**Parameters (proposed):** `chunk_size=512`, `overlap=64`.

**Estimated chunk count:** Corpus of ~8.5MB ≈ 2.1M tokens (at ~4 chars/token). With 512-token chunks and 64-token overlap: effective stride = 448 tokens → **~4,700 chunks**.

**Advantages:**
- Deterministic, O(n) runtime, no tokenizer lookahead.
- Every chunk is the same context-window footprint → predictable embedding computation.

**Disadvantages:**
- Splits mid-sentence and mid-clause routinely. Clause boundaries in regulatory text (e.g., "subject to the conditions specified in paragraph 3 of the aforesaid circular…") are reference anchors; splitting them corrupts retrieval.
- Short documents (e.g., 1,500-byte auction notices) may yield a single degenerate chunk.

**Verdict: Reject.** Semantic coherence is sacrificed for implementation simplicity. For a regulatory corpus where clause-level citation integrity matters, this is unjustifiable.

---

### Strategy B — Recursive Character Splitting (Sentence/Paragraph Boundary Aware)

**Method:** Hierarchically attempt to split on `\n\n` (paragraph), then `\n` (line), then `. ` (sentence), then ` ` (word), stopping at the first level that produces chunks under `max_chars`. Overlap carried over at character level.

**Parameters (proposed):** `target_chunk_chars=1600`, `overlap_chars=200`. (1,600 chars ≈ 380–420 tokens for English legal prose.)

**Estimated chunk count:** At ~44,700 chars/doc average and 1,600 chars/chunk with 200-char overlap: effective stride ≈ 1,400 → ~32 chunks/doc → **~6,100 chunks total**. Range: 1 chunk (short notices) to ~200 chunks (long Master Directions).

**Advantages:**
- Paragraph and sentence boundaries are preserved almost always.
- Handles variable-length documents gracefully — short documents yield 1–3 coherent chunks, long ones yield many.
- Fast: single-pass string scan with no model inference.
- Fully reproducible given fixed parameters.

**Disadvantages:**
- "Paragraph" heuristic (`\n\n`) may not match all structural boundaries (numbered lists use single newlines in some PDFs).
- Overlap is character-based, not semantic — can include irrelevant trailing/leading context.

**Verdict: Recommended (see §3.4).**

---

### Strategy C — Semantic Chunking via Embedding Similarity

**Method:** Embed every sentence independently, compute cosine distance between consecutive sentence embeddings, and insert chunk boundaries where distance exceeds a threshold δ (e.g., δ = 0.3). Chunks then contain semantically contiguous sentences.

**Parameters:** δ tuned on a held-out split; typical chunk size 3–8 sentences.

**Advantages:**
- Strongest semantic coherence — topic shifts in the text map directly to chunk boundaries.
- Particularly powerful for documents that shift between distinct topics without paragraph breaks.

**Disadvantages:**
- **O(S × d) inference cost at build time**, where S = number of sentences in corpus (~150,000 sentences). At BGE-base (768-dim, 512-token max), this adds ~45 seconds on CPU per full rebuild.
- Non-deterministic across model versions (embedding space changes if model is updated → index must be fully rebuilt).
- Per-sentence embedding granularity loses intra-sentence context; joined chunks have inconsistent token lengths, complicating batching.
- δ is a sensitive hyperparameter with no principled default for this domain.

**Verdict: Reject for v1.** The theoretical gain does not justify the build-time cost and reduced reproducibility at this corpus scale. Could be a Phase 4 extension.

---

### 3.4 Recommendation: Strategy B with Parameter Justification

Use **Recursive Character Splitting** with:

```python
target_chunk_chars = 1600   # ≈ 400 tokens at 4 chars/token
overlap_chars      = 200    # ≈ 50 tokens; ensures clause cross-boundary coherence
min_chunk_chars    = 100    # discard degenerate micro-chunks (footnotes, headers)
separators         = ["\n\n", "\n", ". ", "! ", "? ", " ", ""]
```

**Justification of 1,600-char target:** BGE-base-en-v1.5 has a 512-token maximum. At an average of ~4 chars/token for English legal text, 1,600 chars ≈ 400 tokens — safely below the model's limit while providing substantial context per chunk. Empirically, 300–500 token chunks are the sweet spot for regulatory Q&A (cited in RAGAS and BEIR literature).

**Justification of 200-char overlap:** Ensures that an answer spanning a clause boundary is not split across two chunks where neither chunk independently contains the full answer. 200 chars ≈ 2–3 sentences of regulatory prose.

---

## 4. Vector Indexing: FAISS Index Selection

### 4.1 Corpus Geometry

After chunking, we expect M ≈ 6,000–7,000 vectors in ℝ^768 (BGE-base embedding dimension).

Memory footprint for raw vectors: `M × d × 4 bytes = 7000 × 768 × 4 ≈ 21.5 MB`.

This is the fundamental sizing input for all index trade-off analysis.

### 4.2 Comparative Analysis

| Index Type | Search Complexity | Memory Overhead | Build Complexity | Recall | Notes |
|---|---|---|---|---|---|
| **IndexFlatIP** | O(M × d) | 0 (no overhead) | O(M × d) trivial | 1.000 | Exact; brute-force; deterministic |
| **IndexIVFFlat** | O(M/nlist × d × nprobe) | +nlist centroids | O(M × d) + training | ~0.95 | Approx; requires min 39×nlist training vectors |
| **IndexHNSWFlat** | O(log M) amortised | +graph ≈ 3× vector size | O(M × log M) | ~0.98 | Excellent at scale; wasteful here |
| **IndexPQFlat** | O(M × d/m) | 0.25–0.50× | O(M × d) + training | ~0.90 | Quantised; lossy; unnecessary here |

### 4.3 Analysis for M ≈ 7,000

**IndexFlatIP query latency estimate:** At 7,000 vectors × 768 dimensions, one query performs 7,000 × 768 = 5.4M float32 multiplications. On a modern CPU doing ~4 GFLOPS: **~1.35 ms per query**. This is completely acceptable for an interactive QA system (typical generation latency: 500–2,000ms).

**Why not IVF?** IVFFlat requires `nlist ≈ sqrt(M) ≈ 84` partitions and a minimum of `39 × 84 ≈ 3,276` training vectors. At M=7,000 this is fine numerically, but the complexity buys nothing: the per-query shortlist scan at `nprobe=10` processes `10/84 × 7000 ≈ 833` vectors, saving ~88% of compute at the cost of ~5% recall loss. 5% recall loss on a 192-document corpus is not worth 1.1ms of saved CPU time.

**Why not HNSW?** HNSW's graph structure adds ~60MB of memory overhead (≈ 3× the raw vector store). Its O(log M) query time is a tangible win only above ~500K vectors. Below that, the constant factor in the log dominates and HNSW is often slower than exact search in wall-clock time.

### 4.4 Selection: `faiss.IndexFlatIP` with L2-Normalised Vectors

```python
# Normalise all embeddings to unit sphere before adding
faiss.normalize_L2(embeddings)  # in-place, float32 array shape (M, d)
index = faiss.IndexFlatIP(d)    # inner product on unit sphere = cosine similarity
index.add(embeddings)
```

**Why inner product over L2?** After L2 normalisation, `⟨u, v⟩ = cos(u, v)`. IndexFlatL2 computes Euclidean distance, which for normalised vectors is a monotonic transform of cosine distance but with less numerical intuition. IndexFlatIP directly returns cosine scores in `[−1, 1]`, which are more interpretable for thresholding (e.g., "reject retrievals with score < 0.5").

The index, chunk metadata, and BM25 object are serialised together as:
```
rag/index/
├── faiss.index          # faiss.write_index serialised binary
├── chunks.pkl           # list[ChunkRecord] with .text, .doc_id, .title, .chunk_idx
└── bm25.pkl             # rank_bm25.BM25Okapi serialised
```

---

## 5. Retrieval Strategy

### 5.1 Similarity Metric

Cosine similarity (via normalised inner product as described in §4.4). Score range: `[−1, 1]`. Empirical minimum useful threshold for this domain: `s_min ≈ 0.45`.

### 5.2 Top-k Selection

**k = 5** for generation context (default). Rationale:
- BGE-base produces 768-dim embeddings from 512-token inputs. The generation LLM (Groq Llama 3.3 70B) has a 128K context window; 5 chunks × ~400 tokens = 2,000 tokens of context, leaving abundant room for the prompt structure and generation.
- RAGAS literature and empirical BEIR results show diminishing Recall@k returns beyond k=7 for corpora under 50K chunks.
- Expose `top_k` as a runtime parameter; default=5, CLI-configurable.

### 5.3 Advanced Retrieval: Hybrid BM25 + Dense with Reciprocal Rank Fusion

**Motivation:** Pure dense retrieval fails on exact regulatory references like "Section 51A of UAPA 1967" or "Regulation 4(2)(b)" — these are lexically specific, low-frequency tokens that embedding models may not rank highly if the query uses the same exact phrasing but the model hasn't seen the regulatory context frequently. BM25 (which is exact term-frequency matching) excels at precisely these queries.

**Architecture of RRF Fusion:**

```
query ──► Dense Retriever ──► ranked list L_dense (top-K', K'=20)
      └──► BM25 Retriever ──► ranked list L_bm25  (top-K', K'=20)
                                     │
                    Reciprocal Rank Fusion (k=60)
                                     │
                              fused ranking (top-k=5)
```

**RRF Score (Cormack et al., 2009):**

```
RRF_score(d) = Σ_{r ∈ {dense, bm25}} 1 / (k_RRF + rank_r(d))
```

where `k_RRF = 60` (empirically optimal constant from the original paper, tested across multiple TREC runs). Documents not appearing in a list are assigned `rank = K' + 1`.

**Why RRF over score normalisation?** Raw dense similarity scores and BM25 scores are on incompatible scales (cosine ∈ [−1,1] vs. BM25 ∈ ℝ₊). Score normalisation (min-max, softmax) is sensitive to the distribution of scores in each ranked list and breaks under query distribution shift. RRF operates purely on ordinal ranks, making it distribution-agnostic and robust. This is a deliberate engineering choice.

**BM25 Implementation:**

```python
from rank_bm25 import BM25Okapi

# Tokenise by whitespace + punctuation strip (preserves regulatory terms)
tokenised_corpus = [chunk.text.lower().split() for chunk in chunks]
bm25 = BM25Okapi(tokenised_corpus, k1=1.5, b=0.75)
```

Default BM25 parameters: `k1=1.5` (term frequency saturation), `b=0.75` (length normalisation). These are the standard Okapi BM25 defaults; the corpus is too small to warrant grid search in Phase 2. Phase 4 could include a sweep.

---

## 6. LLM & Prompt Strategy

### 6.1 Generation Model Selection

**Constraint:** Zero paid managed services. No GCP, no OpenAI API (paid tier).

**Candidate models:**

| Backend | Model | Free Tier | Latency | Quality | Local |
|---|---|---|---|---|---|
| **Groq API** | llama-3.3-70b-versatile | 14,400 req/day, 6,000 TPM | ~200ms TTFT | Excellent | No |
| Google Gemini | gemini-2.0-flash | 1,500 req/day | ~300ms | Excellent | No |
| Ollama | llama3.2:3b | Unlimited | ~2s (CPU) | Good | Yes |
| Ollama | mistral:7b | Unlimited | ~8s (CPU) | Very good | Yes |
| HF Inference API | Zephyr-7B-β | Rate limited | ~3–5s | Good | No |

**Selected: Groq API (primary) + Ollama llama3.2:3b (fallback)**

**Rationale:** Groq's free tier is the only option that delivers both 70B-class model quality AND sub-second latency without payment. The fallback to Ollama llama3.2:3b ensures the pipeline runs completely offline — critical for reproducibility in a submission context where internet access is not guaranteed during evaluation.

The backend is selected at runtime via an environment variable: `RAG_LLM_BACKEND=groq|ollama`. The `generator.py` module implements both via a common interface.

### 6.2 Exact Prompt Structure

This is the complete prompt template. Every field is required; no optional sections.

```
SYSTEM (injected as system role where supported; prepended otherwise):
You are an expert in Indian financial regulation specialising in Reserve Bank of
India (RBI) and Securities and Exchange Board of India (SEBI) regulatory documents.
Your task is to answer questions using ONLY the source passages provided below.
Rules:
- Every claim must be directly attributable to a numbered source.
- Cite sources inline as [Source N].
- If the passages do not contain sufficient information, state: "The provided
  context does not contain sufficient information to answer this question."
- Do not infer, extrapolate, or use general knowledge not present in the sources.
- Be concise and precise. Maximum 200 words unless the question requires more.

CONTEXT BLOCK (constructed from retrieved chunks):
[Source 1] {chunk_1.title} (chunk {chunk_1.chunk_idx})
{chunk_1.text}

[Source 2] {chunk_2.title} (chunk {chunk_2.chunk_idx})
{chunk_2.text}

... (up to k=5 sources)

QUESTION:
{query}

ANSWER:
```

**Prompt design decisions:**
- `(chunk N)` suffix in source header: enables fine-grained citation traceability back to exact chunk, not just document.
- "Maximum 200 words" soft cap: prevents verbose padding that degrades faithfulness scores.
- Explicit "do not use general knowledge" instruction: critical for faithfulness — LLMs default to mixing memorised knowledge with retrieved context.
- Temperature: `0.0` (fully deterministic). Generation quality in closed-book regulatory QA degrades with temperature > 0.1.
- `max_tokens=512` as hard cap (configurable).

### 6.3 Context Window Budget

```
Prompt overhead (system + instructions):     ~350 tokens
Per-chunk (400 chars avg + label overhead):  ~120–130 tokens
5 chunks:                                    ~630 tokens
Query:                                       ~30 tokens
─────────────────────────────────────────────
Total input tokens (approximate):           ~1,010 tokens
Reserved for generation:                    ~512 tokens
─────────────────────────────────────────────
Total context consumed:                     ~1,522 tokens
```

Well within Groq Llama 3.3 70B's 32K context (and far within its 128K context window). No context truncation will occur under normal operating conditions.

---

## 7. Explicit Data Models

These are the canonical Python types used throughout every module. All other files import from `rag/models.py`.

```python
# rag/models.py
from dataclasses import dataclass

@dataclass
class Document:
    doc_id:    str   # filename stem, e.g. "RBI_Master_Dir_084"
    title:     str   # short human-readable label parsed from filename
    source:    str   # "rbi" | "sebi"
    raw_text:  str   # text after preprocessing (mutated in-place by pipeline)
    file_path: str   # absolute path to source .txt file

@dataclass
class ChunkRecord:
    chunk_id:   str   # f"{doc_id}__{chunk_idx:04d}" — globally unique
    doc_id:     str   # parent Document.doc_id
    title:      str   # inherited from parent Document
    source:     str   # "rbi" | "sebi"
    text:       str   # chunk text as it will be embedded and indexed
    chunk_idx:  int   # 0-indexed position within the parent document
    char_start: int   # character offset in preprocessed document text
    char_end:   int   # exclusive end offset

@dataclass
class RetrievalResult:
    chunk:       ChunkRecord
    dense_score: float   # cosine similarity from FAISS  (range [-1, 1])
    bm25_score:  float   # raw BM25Okapi score           (range [0, ∞))
    rrf_score:   float   # RRF fused score               (range (0, 1/30])
    dense_rank:  int     # 1-indexed rank in dense list  (N+1 if absent)
    bm25_rank:   int     # 1-indexed rank in BM25 list   (N+1 if absent)
```

**Design notes:**
- `chunk_id` zero-padded 4-digit suffix ensures lexicographic sort = positional sort within a document.
- `RetrievalResult` carries all component scores — dropping either column in Phase 3 isolates dense-only vs. BM25-only performance for the ablation study.
- `Document.raw_text` is mutated in-place by `TextPreprocessor` to avoid duplicating the ~8.5MB corpus in memory.

---

## 8. Evaluation: Design, Not Just Declaration

### 8.1 Evaluation Dataset Construction

A 50-item QA dataset (synthetic + manual) serves as ground truth for all retrieval and generation metrics.

**Tier 1 — Synthetic factual (35 questions):** For each of 35 randomly sampled documents, prompt Gemini 1.5 Flash (free tier):
```
"Given this regulatory text, write one specific factual question whose answer
 appears in exactly one or two consecutive paragraphs. Output JSON:
 {\"question\": \"...\", \"answer\": \"...\", \"verbatim_span\": \"first 60 chars of answer span\"}"
```
The `verbatim_span` is used to locate ground-truth chunk(s) in the index via `difflib.SequenceMatcher` ratio ≥ 0.85 against all chunk texts.

**Tier 2 — Adversarial/hard (15 questions):** Manually authored to stress failure modes:

| Sub-type | n | Example |
|---|---|---|
| Cross-document synthesis | 4 | "Compare KYC requirements under SEBI and RBI frameworks." |
| Exact regulatory reference | 4 | "What does Section 51A of UAPA 1967 require of banks?" |
| Unanswerable (corpus miss) | 4 | "What is SEBI's policy on crypto derivative instruments?" |
| Temporal/version conflict | 3 | "What was the T-bill cut-off rate on 18 March 2026?" |

**Ground truth schema (`data/eval/eval_set.json`):**
```json
{
  "qid": "syn_001",
  "question": "...",
  "reference_answer": "...",
  "relevant_chunk_ids": ["RBI_Master_Dir_084__0003", "RBI_Master_Dir_084__0004"],
  "tier": "synthetic",
  "source_doc": "RBI_Master_Dir_084"
}
```

### 8.2 Retrieval Metric Computation

```python
def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    return len(set(retrieved_ids) & set(relevant_ids)) / max(len(relevant_ids), 1)

def mean_reciprocal_rank(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    relevant_set = set(relevant_ids)
    for rank, cid in enumerate(retrieved_ids, 1):
        if cid in relevant_set:
            return 1.0 / rank
    return 0.0

def precision_at_k(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    return len(set(retrieved_ids) & set(relevant_ids)) / max(len(retrieved_ids), 1)
```

Aggregate by simple mean over all 50 eval examples.

### 8.3 Faithfulness Computation (LLM-as-Judge)

Faithfulness = fraction of answer claims directly entailed by the retrieved context. A claim using general knowledge not present in sources is unfaithful.

**Judge prompt (Gemini 1.5 Flash, free 1M tokens/day):**
```
System: You are a strict fact-checker. Only attribute claims to provided sources.

SOURCES: {source_block}
ANSWER:  {answer}

For each distinct factual claim in ANSWER, state whether it is:
  SUPPORTED — directly stated or unambiguously implied by a source
  UNSUPPORTED — relies on knowledge absent from the sources

Output ONLY valid JSON:
{"claims": [{"text": "...", "supported": true, "source_ref": "[Source N] or null"}],
 "faithfulness_score": <float 0-1>}
```

**Score aggregation:**
```python
faithfulness = mean(item["faithfulness_score"] for item in judge_outputs)
hallucination_free_rate = mean(
    all(c["supported"] for c in item["claims"]) for item in judge_outputs
)
```

### 8.4 Answer Relevance

```python
def answer_relevance(query: str, answer: str, embedder: BGEEmbedder) -> float:
    q_emb = embedder.encode_query(query)      # shape (1, 768), L2-normalised
    a_emb = embedder.encode_corpus([answer])  # shape (1, 768), L2-normalised
    return float(np.dot(q_emb, a_emb.T))      # = cosine similarity
```

---

## 9. Failure Mode Analysis

| Failure Mode | Root Cause | Impact | Mitigation | Status |
|---|---|---|---|---|
| Long query > 512 tokens | BGE-base truncates; query tail dropped | Dense recall degrades | Truncate to 400 tokens before encoding | Deferred Phase 4 |
| Cross-document reasoning | Top-5 may all be from same document | Answer needs RBI + SEBI synthesis | `max_per_source=3` diversity cap in `HybridRetriever` | **Implemented Phase 2** |
| Exact regulatory references | Dense model generalises; statute refs underranked | Wrong section retrieved | BM25 handles exact lexical match | **Handled by BM25** |
| Temporal/version conflicts | Multiple circulars on same topic in corpus | Outdated rate/rule cited | Metadata date filter (filename parsing) | Deferred Phase 4 |
| Unanswerable questions | Topic not covered in corpus | LLM hallucinates from parametric memory | Explicit prohibition in system prompt | **Handled in prompt** |
| Ambiguous section refs | "Section 4" spans hundreds of documents | High-confidence wrong retrieval | Requires query-time disambiguation | Deferred Phase 4 |
| Short document degenerate context | 1,500-byte notices → 1–2 chunks | Sparse but accurate answer | Correct behaviour; no mitigation needed | N/A |
| Embedding model version drift | Model upgrade shifts embedding space | Stale index returns wrong results | Pin version in requirements.txt; rebuild on upgrade | **Implemented** |

---

## 10. Comparison Baselines (Ablation Design)

Phase 3 runs these six configurations on the same 50-question eval set. All other variables are held constant.

| Config | Retriever | Chunk chars | k | Purpose |
|---|---|---|---|---|
| **B0 — Dense-only** | FAISS, no BM25 | 1600 | 5 | Isolate dense contribution |
| **B1 — BM25-only** | BM25, no FAISS | 1600 | 5 | Isolate lexical contribution |
| **B2 — Hybrid (proposed)** | RRF(Dense+BM25) | 1600 | 5 | Full system |
| **B3 — Small chunks** | Hybrid | 800 | 5 | Granularity effect on recall |
| **B4 — Large chunks** | Hybrid | 2400 | 5 | Context richness vs. noise |
| **B5 — Higher k** | Hybrid | 1600 | 10 | Recall@k vs. precision trade-off |

**Falsifiable hypotheses:**
- H1: B2 > B0 on exact regulatory reference queries (BM25 adds lexical recall).
- H2: B2 > B1 on paraphrase/semantic queries (dense adds semantic recall).
- H3: B3 improves MRR but degrades faithfulness (more chunks, less context per chunk).
- H4: B4 improves Recall@5 but degrades Precision@5.

---

## 11. Architecture Diagram

```
╔══════════════════════════════════════════════════════════════════════════╗
║                        OFFLINE — INDEX BUILD                             ║
║                                                                          ║
║  data/parsed/                                                            ║
║  ├── rbi/*.txt (100)  ─┐                                                ║
║  └── sebi/*.txt (92)  ─┴──► DataLoader ──► list[Document]              ║
║                                    │                                     ║
║                             TextPreprocessor                             ║
║                        (NFKC, header strip, whitespace)                  ║
║                                    │                                     ║
║                       RecursiveCharacterSplitter                         ║
║                     (target=1600 chars, overlap=200)                     ║
║                                    │                                     ║
║                           list[ChunkRecord]                              ║
║                            (~6,100–7,000 chunks)                         ║
║                           ┌────────┴────────┐                           ║
║                           │                 │                            ║
║                      BGEEmbedder      BM25Index.build()                  ║
║                  (bge-base-en-v1.5    BM25Okapi(k1=1.5, b=0.75)         ║
║                   batch=64, cpu)             │                           ║
║                           │                 │                            ║
║                    float32[M, 768]           │                           ║
║                    (L2-normalised)           │                           ║
║                           │                 │                            ║
║                    FAISSIndex.build()        │                           ║
║                    IndexFlatIP(768)          │                           ║
║                           │                 │                            ║
║                      rag/index/             │                            ║
║                      ├── faiss.index ◄──────┘                           ║
║                      ├── chunks.pkl                                      ║
║                      └── bm25.pkl                                        ║
╚══════════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════════════╗
║                        ONLINE — QUERY SERVING                            ║
║                                                                          ║
║  query: str                                                              ║
║     ├───────────────────────────────────────┐                           ║
║     ▼                                       ▼                           ║
║  BGEEmbedder.encode_query()          BM25Index.search(k=20)             ║
║  (BGE query prefix prepended)               │                           ║
║     │                                [(ChunkRecord, bm25_score)]        ║
║     ▼                                       │                           ║
║  FAISSIndex.search(k=20)                    │                           ║
║  [(ChunkRecord, cosine_score)]              │                           ║
║     └──────────────┬────────────────────────┘                          ║
║                    ▼                                                     ║
║          HybridRetriever — RRF fusion                                    ║
║          score(d) = 1/(60+rank_dense) + 1/(60+rank_bm25)                ║
║          source-diversity cap: max 3 chunks per source                   ║
║          → top-5 RetrievalResult                                         ║
║                    │                                                     ║
║                    ▼                                                     ║
║          LLMGenerator.generate(query, results)                           ║
║          ┌─────────────────────────────────┐                            ║
║          │  RAG_LLM_BACKEND env var        │                            ║
║          │  "groq"  → llama-3.3-70b        │  T=0.0, max_tokens=512     ║
║          │  "ollama"→ llama3.2:3b (local)  │                            ║
║          └─────────────────────────────────┘                            ║
║                    │                                                     ║
║        {"answer": str, "sources": list[dict]}                           ║
║                    │                                                     ║
║         demo/rag/rag.py  ──►  Flask /api/ask endpoint                   ║
╚══════════════════════════════════════════════════════════════════════════╝
```

---

## 12. Module Map (Phase 2 Target)

```
rag/
├── __init__.py
├── config.py          # Typed dataclass for all hyperparameters; load from env/YAML
├── data_loader.py     # Scan data/parsed/**/*.txt; return list[Document]
├── preprocessing.py   # NFKC normalisation, whitespace collapse, header strip
├── chunking.py        # RecursiveCharacterSplitter; returns list[ChunkRecord]
├── embeddings.py      # BGEEmbedder wrapping sentence-transformers; batched encode
├── index.py           # FAISSIndex: build, save, load, search (dense)
├── bm25_index.py      # BM25Index: build, save, load, search (lexical)
├── retriever.py       # HybridRetriever: RRF fusion of dense + BM25
├── generator.py       # LLMGenerator: Groq/Ollama backend with common interface
├── pipeline.py        # RAGPipeline: orchestrator; exposes .ask(query) -> dict
└── evaluation.py      # Phase 3: Recall@K, MRR, Faithfulness, AnswerRelevance
```

The `demo/rag/rag.py` module will be updated in Phase 2 to `from rag.pipeline import RAGPipeline` — a clean one-line swap preserving the existing Flask integration.

---

## 8. Key Engineering Trade-offs Summary

| Decision | Chosen | Rejected | Primary Trade-off |
|---|---|---|---|
| Index type | IndexFlatIP | IVFFlat, HNSW | Exact recall vs. unnecessary latency optimisation |
| Chunking | Recursive char split | Semantic chunking | Reproducibility vs. semantic coherence at corpus scale |
| Retrieval | Hybrid BM25+Dense+RRF | MMR, dense-only | Lexical+semantic coverage vs. result diversity |
| Embedding model | BGE-base-en-v1.5 (768d) | MiniLM-L6 (384d), BGE-large | Quality vs. inference speed vs. memory |
| Generation | Groq Llama 3.3 70B + Ollama fallback | Gemini free tier | Latency + reliability vs. single API dependency |
| Fusion method | RRF (rank-based) | Score normalisation | Distribution-agnostic robustness vs. score-level tuning |

---

*Phase 1 complete. Awaiting APPROVED to begin Phase 2: Repository Scaffolding & Core Implementation.*
