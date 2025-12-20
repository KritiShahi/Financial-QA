# Financial-QA

# Financial-QA Retrieval


## Overview

Financial analysts and investors spend hours manually searching through lengthy SEC 10-K filings to answer complex queries. This project addresses this challenge by building an **Financial QA** system that combines:

- **Lexical Search (BM25)**: Fast keyword-based retrieval
- **Dense Retrieval**: Fine-tuned sentence transformers for semantic understanding
- **Neural Reranking**: Cross-encoder models for precise ranking


### Solution

A hybrid three-stage retrieval pipeline:
1. **Document Processing**: Convert raw HTML 10-K filings → clean text → structured chunks
2. **Retrieval**: Use fine-tuned dense encoders to find semantically relevant passages
3. **Reranking**: Apply cross-encoder to refine top-k results for optimal ranking

---

## Key Features

-  **3.7× improvement** over BM25 baseline (10% → 37.1% Recall@20)
-  **37.9% Recall@20** and **0.206 MRR** with hybrid approach
-  **111,402 searchable chunks** with metadata preservation across all major sections of 10-K filings
-  Domain-adapted models trained on **3,700 expert-annotated queries**
-  Reproducible training with validation-based early stopping

---

## Performance

### Comparison of Retrieval Methods

| Method | Recall@5 | Recall@10 | Recall@20 | MRR | nDCG@20 |
|--------|----------|-----------|-----------|-----|---------|
| **BM25 (Lexical)** | 0.054 | 0.076 | 0.100 | 0.038 | 0.051 |
| **Dense Encoder** | 0.206 | 0.282 | 0.371 | 0.138 | 0.185 |
| **Hybrid + Reranker** | **0.247** | **0.304** | **0.379** | **0.206** | **0.240** |

### Key Insights

- **63% retrieval recall** in top-200 results
- **Average rank of 3** for correct answers (when retrieved)
- **67% of answers** in top-5 positions with hybrid approach
- Works across **8 question categories** (Company Overview, Financials, Risk, etc.)

---

### Prerequisites

- Python 3.9 or higher
- CUDA-compatible GPU (recommended for training/encoding)

## Step to run the codes end to end

1. Run all cells of BM25 Chunking Notebook to create chunks from SEC filings
2. Run all the cells of Finder_Triplets_Optimized notebook to create finder triplets
3. To perform fine tuning on sentence transformer and cross encoder, refer Dense Retreival and Cross Encoder Fine Tuning notebook 
4. Evaluate the results of lexical and dense search using Lexical And Dense Eval notebook
5. Evaluate the results of hybrid search and reranker using Hybrid Retreival Eval notebook 

---

## Citation

If you use this work in your research, please cite:

 FinDER dataset:

```bibtex
@inproceedings{chen2022finder,
  title={FinDER: Financial Document Expert Retrieval Dataset},
  author={Chen, Zishuo and Gupta, Vivek and Nogueira, Rodrigo},
  booktitle={Proceedings of NAACL},
  year={2022}
}
```
1. Original Dataset available at https://huggingface.co/datasets/Linq-AI-Research/FinDER
2. Data and Models useful for producing results are present at https://drive.google.com/drive/folders/1-JeFEVFaJe9C-3GK2o9jz3TgshVCJR_c?usp=sharing

---

## Roadmap

### Current Version (v1.0)
- BM25 baseline implementation
- Fine-tuned dense retrieval
- Cross-encoder reranking
- FinDER evaluation

### Future Work (v1.1)
- Incorporate Table Data
- Interactive web UI

