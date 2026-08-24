# Financial QA Retrieval — SEC 10-K Document Search

> A hybrid 3-stage retrieval system for financial document question answering,
> achieving **3.7× improvement** over BM25 baseline and **74.7% retrieval recall**
> across 111,402 chunks from 400+ S&P 500 filings.

---

## Overview

Financial analysts, auditors, and compliance professionals routinely rely on SEC 10-K
filings to extract critical information regarding risk factors, revenue breakdowns,
accounting policies, and forward-looking statements. These documents often exceed
hundreds of pages, are written in dense legal-financial language, and contain
substantial redundancy across sections.

This project builds an automated **Financial Document Expert** system that retrieves
relevant passages from hundreds of filings in milliseconds, evaluated on the
[FinDER](https://huggingface.co/datasets/Linq-AI-Research/FinDER) benchmark —
5,703 expert-annotated query-document pairs from real SEC 10-K filings.

---

## Results

### Method Comparison

| Method | Recall@5 | Recall@10 | Recall@20 | MRR | nDCG@20 |
|--------|----------|-----------|-----------|-----|---------|
| BM25 (Lexical baseline) | 0.054 | 0.076 | 0.100 | 0.038 | 0.051 |
| Fine-tuned Dense Encoder | 0.206 | 0.282 | 0.371 | 0.138 | 0.185 |
| **Hybrid + Reranker** | **0.282** | **0.339** | **0.394** | **0.224** | **0.260** |

### Candidate Pool Optimization (Baseline k=200 vs Optimized k=500 + FAISS)

| Metric | Baseline (k=200) | Optimized (k=500, FAISS) | Δ Absolute | Δ Relative |
|--------|-----------------|--------------------------|------------|------------|
| **RRF Retrieval Recall** | 0.630 | **0.747** | +0.117 | **+18.6%** |
| MRR | 0.206 | 0.224 | +0.018 | +8.6% |
| Recall@5 | 0.247 | 0.282 | +0.035 | +14.2% |
| Recall@10 | 0.304 | 0.339 | +0.035 | +11.6% |
| Recall@20 | 0.379 | 0.394 | +0.015 | +3.9% |
| nDCG@20 | 0.240 | 0.260 | +0.020 | +8.5% |

> Widening the candidate pool from 200 to 500 per retriever raised RRF Retrieval Recall
> from 63% to 74.7% — the gold passage now appears in the fused candidate pool for
> nearly **three in four queries**, with reranking concentrating correct answers near
> **rank 3** on average among retrieved queries.

---

## Architecture

```
                    ┌─────────────────────────────────────┐
                    │        SEC 10-K HTML Filings         │
                    └──────────────┬──────────────────────┘
                                   │ BeautifulSoup + Regex
                                   ▼
                    ┌─────────────────────────────────────┐
                    │   111,402 Chunks (300 tok, 100 ovlp) │
                    └────────┬────────────────┬───────────┘
                             │                │
                    BM25 Top-500      FAISS Dense Top-500
                    (rank-bm25)    (fine-tuned mpnet-768d)
                             │                │
                             └───────┬────────┘
                                     │ Reciprocal Rank Fusion
                                     ▼
                             Fused Candidates (Top-50)
                                     │
                              Cross-Encoder Reranker
                             (fine-tuned bert-base)
                                     │
                                     ▼
                              Ranked Results
```

**Stage 1 — Document Processing**
- Raw HTML 10-K filings parsed with `BeautifulSoup` + `lxml`
- 7 sections extracted via regex: Business (Item 1), Risk (Item 1A), MD&A (Item 7),
  Accounting (Item 9), Governance (Item 10), Security (Item 12), Legal (Item 3)
- Token-window chunking: **300 tokens, 100-token overlap** → 111,402 chunks with full metadata

**Stage 2 — Hybrid Retrieval (RRF)**
- `BM25Okapi` for lexical search (simple lowercased word tokenization)
- Fine-tuned `all-mpnet-base-v2` (768-dim) via **FAISS IndexFlatIP** for semantic search
- Reciprocal Rank Fusion: `score += 1 / (60 + rank)` applied to both signals

**Stage 3 — Neural Reranking**
- Fine-tuned `bert-base-uncased` cross-encoder with full cross-attention
- Processes top-50 RRF candidates per query
- Trained with binary cross-entropy on FinDER triplets

---

## Repository Structure

```
financial-qa-retrieval/
│
├── notebooks/                             # End-to-end pipeline, run in order
│   ├── 01_BM25_Chunking.ipynb             # HTML parsing + chunking + BM25 index
│   ├── 02_Triplet_Construction.ipynb      # Hard negative mining + triplet building
│   ├── 03_Dense_Retrieval_Training.ipynb  # Bi-encoder fine-tuning
│   ├── 04_CrossEncoder_Finetuning.ipynb   # Cross-encoder fine-tuning
│   ├── 05_Lexical_Dense_Eval.ipynb        # BM25 vs dense evaluation
│   └── 06_Hybrid_Retrieval_Eval.ipynb     # Full hybrid pipeline + checkpointing
│
├── src/
│   ├── preprocessing/
│   │   ├── html_parser.py                 # HTML → clean text → section extraction
│   │   ├── chunker.py                     # Token-window & sentence chunking
│   │   └── triplet_builder.py             # Hard negative mining + triplet construction
│   ├── retrieval/
│   │   ├── bm25_retriever.py              # BM25Retriever class
│   │   ├── dense_retriever.py             # DenseRetriever (FAISS + SentenceTransformer)
│   │   └── hybrid_retriever.py            # RRF fusion + batched cross-encoder scoring
│   ├── training/
│   │   ├── train_dense_encoder.py         # MultipleNegativesRankingLoss training
│   │   └── train_cross_encoder.py         # BCE cross-encoder training
│   └── evaluation/
│       ├── metrics.py                     # Recall@K, MRR, nDCG
│       └── eval_pipeline.py               # Full eval loop with checkpointing + ETA
│
├── data/
│   ├── README.md                          # Data sources + download links
│   └── sample/
│       └── sample_chunks.json             # 100 sample chunks for reference
│
├── results/
│   └── metrics_summary.json               # Final evaluation numbers
│
├── requirements.txt
└── .gitignore
```

---

## Quickstart

### Installation

```bash
git clone https://github.com/<your-username>/financial-qa-retrieval.git
cd financial-qa-retrieval
pip install -r requirements.txt
```

### Download Data & Models

Large files are hosted on Google Drive:

| File | Size | Description |
|------|------|-------------|
| `chunks_index.json` | 229 MB | 111,402 SEC 10-K chunks |
| `finder_triplets_optimized.jsonl` | 26 MB | 3,439 training triplets |
| `fine_tuned_sec_embeddings.pkl` | 195 MB | Pre-computed dense embeddings |
| `finder_dense_encoder_best/` | — | Fine-tuned bi-encoder |
| `cross_encoder_finder_best/` | — | Fine-tuned cross-encoder |

[Download from Google Drive](https://drive.google.com/drive/folders/1-JeFEVFaJe9C-3GK2o9jz3TgshVCJR_c?usp=sharing)

### Run Pipeline (in order)

```bash
jupyter notebook notebooks/01_BM25_Chunking.ipynb
jupyter notebook notebooks/02_Triplet_Construction.ipynb
jupyter notebook notebooks/03_Dense_Retrieval_Training.ipynb
jupyter notebook notebooks/04_CrossEncoder_Finetuning.ipynb
jupyter notebook notebooks/05_Lexical_Dense_Eval.ipynb
jupyter notebook notebooks/06_Hybrid_Retrieval_Eval.ipynb
```

Or run training directly from CLI:

```bash
python -m src.training.train_dense_encoder
python -m src.training.train_cross_encoder
```

---

## Training Details

### Dense Encoder (Bi-Encoder)

| Parameter | Value |
|-----------|-------|
| Base model | `sentence-transformers/all-mpnet-base-v2` |
| Embedding dim | 768 |
| Loss | `MultipleNegativesRankingLoss` |
| Epochs | 5 (early stopping on validation MRR) |
| Batch size | 16 |
| Learning rate | 2e-5 |
| Warmup steps | 100 |
| Max sequence length | 256 tokens |
| Precision | Mixed (FP16) |

### Cross-Encoder (Reranker)

| Parameter | Value |
|-----------|-------|
| Base model | `bert-base-uncased` |
| Loss | Binary cross-entropy |
| Epochs | 3 (early stopping on validation loss) |
| Batch size | 8 |
| Learning rate | 2e-5 |
| Max sequence length | 512 tokens |

### Triplet Construction

- **Positive**: chunk with maximum BM25 token overlap with expert reference text
- **Hard negatives**: BM25 top-30 results from the same company — textually similar but factually incorrect
- **Train/Val split**: `GroupShuffleSplit` by query to prevent query leakage across splits
- **Output**: 3,439 triplets (3 negatives each) → 13,756 total training examples

---

## Key Engineering Decisions

**Why token-window chunking over sentence splitting?**
Financial data contains numerical relationships and context that spans multiple sentences.
A 300-token window with 100-token overlap (33%) ensures relevant information is never
split across boundaries while fitting within model sequence limits.

**Why FAISS over brute-force cosine similarity?**
Full pairwise cosine similarity across ~63,500 embeddings per query scaled poorly
at `dense_k=500`. FAISS IndexFlatIP over L2-normalized vectors is mathematically
equivalent to exact cosine similarity but orders of magnitude faster.

**Why cap cross-encoder at top-50 instead of top-500?**
Cross-encoder reranking runs full BERT cross-attention per query-candidate pair.
Decoupling recall computation (against full 500-candidate pool) from reranking depth
(top-50) isolates retrieval recall ceiling from the reranker's computational budget.

**Why `GroupShuffleSplit` for train/val splitting?**
Standard random splits allow the same query to appear in both train and validation,
inflating metrics. Splitting by query group ensures evaluation reflects genuine
generalization to unseen queries.

---

## Dataset

**FinDER** (Financial Document Expert Retrieval Dataset)

- 5,703 expert-annotated query-document pairs across 8 categories:
  Company Overview (18.95%), Financials (17.36%), Footnotes (16.71%),
  Governance (12.59%), Accounting (8.61%), Risk (8.59%), Legal (8.59%),
  Shareholder Return (8.59%)
- Sourced from real SEC 10-K filings of S&P 500 companies
- Available: [HuggingFace — Linq-AI-Research/FinDER](https://huggingface.co/datasets/Linq-AI-Research/FinDER)

```bibtex
@inproceedings{chen2022finder,
  title={FinDER: Financial Document Expert Retrieval Dataset},
  author={Chen, Zishuo and Gupta, Vivek and Nogueira, Rodrigo},
  booktitle={Proceedings of NAACL},
  year={2022}
}
```

---

## Tech Stack

| Library | Version | Purpose |
|---------|---------|---------|
| `sentence-transformers` | 2.7.0 | Bi-encoder training & inference |
| `transformers` | 4.36.2 | Cross-encoder (BERT) |
| `faiss-cpu` | 1.7.4 | Approximate nearest neighbor search |
| `rank-bm25` | 0.2.2 | Lexical retrieval |
| `torch` | ≥2.0.1 | Model training |
| `beautifulsoup4` | ≥4.12.2 | HTML parsing |
| `pandas` | ≥2.0.3 | Data handling |
| `scikit-learn` | ≥1.3.0 | Train/val splitting |

---

## Author

**Kriti Shahi**
University of Maryland — MS in Applied Machine Learning
[LinkedIn](https://linkedin.com/in/kritishahi)
