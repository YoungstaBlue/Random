# Claude Legal — Architecture & Data-Flow Reference

This document explains how the two source specs fuse into one pipeline, the exact data flow,
and the JSON payload contracts that keep every module decoupled.

## The core idea: Block A nested inside Block B

The two pasted specs are not parallel systems. **Block A (the vector-matrix / e-discovery
engine) is the retrieval substrate that Block B's Module 3 (AI Analysis) and Module 6
(Workload Categorization) depend on.** So instead of two pipelines we build one:

```
Block B outer pipeline
  └── Module 3: AI Analysis & Verification
        └── Block A engine  (ingest → sparse vectorize → LSH index → cosine rank)
```

## End-to-end data flow

`ClaudePipeline.process_case()` (in `claude_legal/pipeline.py`) is the single orchestrator:

1. **Ingest / store** — raw documents load via `ingestion/loaders.py`; `storage/local_db.py`
   persists case files, facts, witness statements, notes (Module 1). `storage/cloud_sync.py`
   and `storage/backup.py` provide off-site redundancy and encrypted offline integrity.
2. **Route** — `routing/router.py` dispatches structured payloads by named route and fires
   `routing/webhooks.py` on every change so the Claude app's visual layers stay live (Module 2).
3. **Categorize + prioritize** — `workload/categorize.py` parses the payload into evidence /
   statutory authority / precedent / procedural rules; `workload/priority.py` ranks tasks by
   procedural-deadline proximity (Module 6).
4. **Semantic search (Block A)** — `analysis/semantic_search.py` runs the four-stage vector
   engine to retrieve doctrine-relevant documents by meaning, not keywords (Module 3):
   - `ingestion/preprocess.py` → tokenize, remove stop-words, lemmatize
   - `vectorize/sparse_vectorizer.py` → TF-IDF into an L2-normalized **`scipy.sparse` CSR** matrix
   - `index/lsh_index.py` → hash rows into LSH buckets for ANN candidate selection
   - `query/similarity.py` → accelerated cosine similarity over just the bucket shortlist
5. **Verify citations** — `analysis/citations.py` extracts citations and classifies
   jurisdiction (federal / state / US territory incl. Puerto Rico) (Module 3/7).
6. **Analyze** — `analysis/llm.py` produces strictly factual analysis via Claude (Module 3),
   or a structured "disabled" note when no API key is set.
7. **Visualize** — `visualization/` emits decision-tree, relational-graph, and mind-map JSON
   for the Claude app's renderers (Module 4).
8. **Format** — `formatting/court_rules.py` builds a compliant caption, signature block, and
   certificate of service; `formatting/legal_dictionary.py` defines Latin terms (Module 5).
9. **Verify compliance** — `workload/verify.py` runs the citation pre-check and flags missing
   filing requirements (e.g. absent certificate of service, missing caption fields) (Module 7).
10. **Version** — `formatting/versioning.py` writes an audit record (version, timestamp,
    author, content hash) (Module 5).
11. **Measure + adapt** — `workload/metrics.py` times every stage and suggests config/routing
    adjustments from observed outcomes (Module 8).
12. **Quantitative strategy (optional)** — when `CaseFinancials` are supplied, `quant/engine.py`
    computes expected value, Bayesian-updated win probability, calculus leverage points (∇f of a
    non-linear damages model), a Monte Carlo award distribution, a game-theoretic settlement Nash
    equilibrium, and cosine-similarity anomalies over the real sparse Block A corpus.

Two cross-cutting guards wrap the flow:

- **Adversarial semantic firewall** (`ingestion/security.py`) runs at ingestion, before step 4,
  quarantining prompt-injection ("poison pill") documents so hostile text never reaches the
  vectorizer or the LLM.
- **Sparse-only invariant** — the corpus stays `scipy.sparse` throughout; the quant anomaly engine
  reuses the same L2-normalized matrix (a sparse dot product = cosine similarity).

The orchestrator returns a single `PipelineResult` aggregating all of the above.

## JSON payload contracts

Modules never import each other's internals; they exchange the pydantic models in
`claude_legal/schemas.py`. Key contracts:

| Model | Produced by | Consumed by |
|-------|-------------|-------------|
| `RawDocument` / `CleanedDocument` | ingestion | vectorize, storage |
| `QueryResult` / `SearchHit` | analysis (Block A) | analysis (LLM), routing, result |
| `CategorizedPayload` | workload.categorize | priority, visualization |
| `PriorityTask` | workload.priority | result |
| `VisualizationBundle` (`TreeNode`, `GraphData`) | visualization | Claude app |
| `CourtFormatConfig` → `FormattedDocument` | formatting | verify |
| `Citation` / `ComplianceReport` | analysis.citations, workload.verify | result |
| `RevisionRecord` | formatting.versioning | storage, result |
| `StageMetric` / `PipelineResult` | pipeline + metrics | caller / Claude app |

Runtime artifacts that don't belong in a JSON contract — the sparse TF-IDF matrix and the
LSH index handle — are deliberately kept out of the schemas and passed by reference within the
`vectorize → index → query` layer only.

## Why this stays pluggable

- One wiring point (`pipeline.py`); modules expose small typed interfaces.
- New services subscribe to lifecycle events (`case.categorized`, `case.searched`,
  `case.visualized`, `case.completed`) on the `PayloadRouter` — no edits to `process_case()`.
- Heavyweight/optional backends are resolved at runtime with graceful fallbacks, so the same
  code runs on a laptop (NumPy) or a GPU box (FAISS + PyTorch) without branching in callers.

## Performance notes (Block A intent)

- **Memory**: sparse CSR keeps a million-document TF-IDF matrix tractable; density on the
  bundled sample is ~0.29 and drops sharply as the vocabulary grows.
- **Speed**: LSH restricts comparison to a small candidate shortlist (`candidates_scanned`
  in `QueryResult`), turning an O(N²) all-pairs scan into near-constant work per query. FAISS
  `IndexLSH` and GPU cosine similarity extend this to production scale.
