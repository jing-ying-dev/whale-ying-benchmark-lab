# Benchmark Run Artifacts & Result Traces

This directory contains verifiable, deterministic evaluation run logs and prediction outputs across models and configurations.

## 📁 Artifact Index

| File | Model & Setting | Metric | Reported Score | Description |
|---|---|:---:|:---:|---|
| `3b-全量100-r1.json` | Qwen2.5-Coder-3B (BM25-only) | Norm. Exact | **47.0%** (47/100) | Run 1 evaluation on 262k FactConsolidation multi-hop. |
| `3b-全量100-r2.json` | Qwen2.5-Coder-3B (BM25-only) | Norm. Exact | **47.0%** (47/100) | Run 2 evaluation demonstrating reproducibility. |
| `3b-全量100-r3.json` | Qwen2.5-Coder-3B (BM25-only) | Norm. Exact | **47.0%** (47/100) | Run 3 evaluation demonstrating reproducibility. |
| `gpt4o-全量100x3.json` | GPT-4o-mini (BM25-only) | Norm. Exact | **51.7%** (51/50/54) | 3-run baseline without dense vector embeddings. |
| `gpt4o-全量100x3-全检索.json` | GPT-4o-mini (Hybrid BM25+Vec+RRF) | Norm. Exact | **49.7%** (49/51/49) | 3-run ablation showing complexity regression (-2.0pp). |

## 🔍 Record Structure

Each JSON entry records:
- `qid`: Benchmark question index
- `question`: Evaluated multi-hop natural language query
- `gold`: Benchmark ground truth entity
- `final`: Model final extracted entity
- `ok`: Boolean verdict under normalized exact match (`norm(final) == norm(gold)`)
- `trace`: Granular step-by-step retrieval hops and intermediate answers
