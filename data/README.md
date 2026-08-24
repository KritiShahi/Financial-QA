# Data

## Full Dataset
Due to file size limits, large data files are hosted on Google Drive:

| File | Size | Description |
|------|------|-------------|
| `chunks_index.json` | 229 MB | 111,402 token-window chunks from 400+ S&P 500 10-K filings |
| `finder_triplets_optimized.jsonl` | 26 MB | 3,439 training triplets (query, positive, 3 hard negatives) |
| `finder_augmented.csv` | 17 MB | FinDER dataset with company metadata |
| `fine_tuned_sec_embeddings.pkl` | 195 MB | Pre-computed dense embeddings for all chunks |

Download: [Google Drive](https://drive.google.com/drive/folders/1-JeFEVFaJe9C-3GK2o9jz3TgshVCJR_c?usp=sharing)

## Sample Data
`sample/sample_chunks.json` — first 100 chunks for reference and testing.

## Source
- SEC 10-K filings: [SEC EDGAR](https://www.sec.gov/cgi-bin/browse-edgar)
- FinDER dataset: [HuggingFace](https://huggingface.co/datasets/Linq-AI-Research/FinDER)
