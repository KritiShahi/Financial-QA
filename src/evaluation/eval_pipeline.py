import json
import math
import os
import time
from typing import Dict, List

import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from src.retrieval.hybrid_retriever import rrf_retrieve_faiss, batched_ce_predict

CHECKPOINT_FILE = "eval_checkpoint.json"


def save_checkpoint(i: int, mrr: float, recall: Dict, precision: Dict,
                    ndcg: Dict, retrieval_success: int, mrr_cond: float) -> None:
    checkpoint = {
        "last_index":        i,
        "mrr":               mrr,
        "recall":            recall,
        "precision":         precision,
        "ndcg":              ndcg,
        "retrieval_success": retrieval_success,
        "mrr_cond":          mrr_cond,
    }
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(checkpoint, f)


def load_checkpoint() -> Dict | None:
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            print("✅ Checkpoint found — resuming from saved state...")
            return json.load(f)
    return None


def load_finder_triplets(path: str) -> List[Dict]:
    triplets = []
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            triplets.append({
                "query":    item["query"],
                "positive": item["positive"]["text"],
                "negatives": [n["text"] for n in item["negatives"]],
            })
    return triplets


def corpus_level_rrf_rerank_eval(
    reranker: CrossEncoder,
    triplets: List[Dict],
    query_embeddings: np.ndarray,
    faiss_index: faiss.IndexFlatIP,
    bm25: BM25Okapi,
    corpus_texts: List[str],
    ks: List[int] = [5, 10, 20],
    dense_k: int = 500,
    bm25_k: int = 500,
    rrf_k: int = 60,
    rerank_top: int = 50,
    checkpoint_every: int = 50,
) -> Dict:
    N = len(triplets)

    # Load checkpoint if exists
    checkpoint = load_checkpoint()
    if checkpoint:
        start_i           = checkpoint["last_index"] + 1
        mrr               = checkpoint["mrr"]
        recall            = {int(k): v for k, v in checkpoint["recall"].items()}
        precision         = {int(k): v for k, v in checkpoint["precision"].items()}
        ndcg              = {int(k): v for k, v in checkpoint["ndcg"].items()}
        retrieval_success = checkpoint["retrieval_success"]
        mrr_cond          = checkpoint["mrr_cond"]
        print(f"Resuming from query {start_i}/{N}")
    else:
        start_i           = 0
        mrr               = 0.0
        recall            = {k: 0.0 for k in ks}
        precision         = {k: 0.0 for k in ks}
        ndcg              = {k: 0.0 for k in ks}
        retrieval_success = 0
        mrr_cond          = 0.0
        print("Starting fresh evaluation...")

    start_time = time.time()

    for i in range(start_i, N):

        # Progress + ETA every 10 queries
        if i % 10 == 0 and i > start_i:
            elapsed        = time.time() - start_time
            avg_per_q      = elapsed / (i - start_i)
            remaining      = avg_per_q * (N - i)
            print(f"[{i}/{N}] | "
                  f"Elapsed: {int(elapsed//60)}m {int(elapsed%60)}s | "
                  f"ETA: {int(remaining//60)}m {int(remaining%60)}s | "
                  f"Avg: {avg_per_q:.2f}s/query")

        # Save checkpoint every N queries
        if i % checkpoint_every == 0 and i > start_i:
            save_checkpoint(i, mrr, recall, precision, ndcg,
                            retrieval_success, mrr_cond)
            print(f"💾 Checkpoint saved at query {i}")

        t          = triplets[i]
        query      = t["query"]
        gold_text  = t["positive"]
        q_emb_np   = query_embeddings[i:i+1]

        all_candidates, rerank_candidates = rrf_retrieve_faiss(
            q_emb_np, query, faiss_index, bm25, corpus_texts,
            dense_k=dense_k, bm25_k=bm25_k,
            rrf_k=rrf_k, rerank_top=rerank_top,
        )

        if gold_text not in all_candidates[:max(dense_k, bm25_k)]:
            continue

        retrieval_success += 1

        ce_scores    = batched_ce_predict(reranker,
                                          [[query, c] for c in rerank_candidates],
                                          batch_size=256)
        reranked     = sorted(zip(rerank_candidates, ce_scores), key=lambda x: -x[1])
        ranked_texts = [c for c, _ in reranked]

        if gold_text not in ranked_texts:
            continue

        rank      = ranked_texts.index(gold_text) + 1
        mrr      += 1 / rank
        mrr_cond += 1 / rank

        for k in ks:
            if rank <= k:
                recall[k]    += 1
                precision[k] += 1 / k
                ndcg[k]      += 1 / math.log2(rank + 1)

    # Cleanup checkpoint on completion
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        print("✅ Evaluation complete — checkpoint deleted.")

    results = {
        "Num_Queries":          N,
        "RRF_Retrieval_Recall": round(retrieval_success / N, 4),
        "MRR":                  round(mrr / N, 4),
        "MRR_given_retrieval":  round(mrr_cond / max(retrieval_success, 1), 4),
    }
    for k in ks:
        results[f"Recall@{k}"]    = round(recall[k] / N, 4)
        results[f"Precision@{k}"] = round(precision[k] / N, 4)
        results[f"nDCG@{k}"]      = round(ndcg[k] / N, 4)

    return results
