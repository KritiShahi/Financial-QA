import re
import numpy as np
import torch
import faiss
from collections import defaultdict
from typing import List, Dict, Tuple
from torch.utils.data import DataLoader
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder


def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip())


def tokenize(s: str) -> List[str]:
    return normalize(s).lower().split()


def build_bm25(corpus_texts: List[str]) -> BM25Okapi:
    return BM25Okapi([tokenize(t) for t in corpus_texts])


def build_faiss_index(embeddings: torch.Tensor) -> faiss.IndexFlatIP:
    corpus_np = embeddings.cpu().numpy().astype("float32")
    faiss.normalize_L2(corpus_np)
    index = faiss.IndexFlatIP(corpus_np.shape[1])
    index.add(corpus_np)
    print(f"FAISS index built: {index.ntotal} vectors of dim {corpus_np.shape[1]}")
    return index, corpus_np


def encode_queries(model: SentenceTransformer, queries: List[str],
                   batch_size: int = 128) -> np.ndarray:
    embeddings = model.encode(
        queries,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype("float32")
    faiss.normalize_L2(embeddings)
    return embeddings


def rrf_retrieve_faiss(q_emb_np: np.ndarray, query_text: str,
                       faiss_index: faiss.IndexFlatIP, bm25: BM25Okapi,
                       corpus_texts: List[str],
                       dense_k: int = 500, bm25_k: int = 500,
                       rrf_k: int = 60, rerank_top: int = 50) -> Tuple[List[str], List[str]]:
    # Dense retrieval via FAISS
    _, dense_idx = faiss_index.search(q_emb_np, dense_k)
    dense_idx = dense_idx[0].tolist()

    # Lexical retrieval via BM25
    bm25_scores = bm25.get_scores(tokenize(query_text))
    bm25_idx    = sorted(range(len(bm25_scores)),
                         key=lambda i: bm25_scores[i], reverse=True)[:bm25_k]

    # Reciprocal Rank Fusion
    rrf_scores: Dict[int, float] = defaultdict(float)
    for rank, idx in enumerate(dense_idx, start=1):
        rrf_scores[idx] += 1.0 / (rrf_k + rank)
    for rank, idx in enumerate(bm25_idx, start=1):
        rrf_scores[idx] += 1.0 / (rrf_k + rank)

    fused_idx      = sorted(rrf_scores, key=lambda i: rrf_scores[i], reverse=True)
    all_candidates = [corpus_texts[i] for i in fused_idx]
    rerank_cands   = [corpus_texts[i] for i in fused_idx[:rerank_top]]

    return all_candidates, rerank_cands


def batched_ce_predict(reranker: CrossEncoder, pairs: List[List[str]],
                       batch_size: int = 256) -> List[float]:
    scores = []
    reranker.model.eval()
    with torch.no_grad():
        for batch in DataLoader(pairs, batch_size=batch_size,
                                collate_fn=lambda x: x):
            scores.extend(reranker.predict(batch))
    return scores
