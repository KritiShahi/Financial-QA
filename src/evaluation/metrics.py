import math
import numpy as np
import torch
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


def evaluate_ranking_metrics(model: SentenceTransformer, triplets: List[Dict],
                              ks: List[int] = [5, 10, 20]) -> Dict:
    metrics = {f"Recall@{k}":    0.0 for k in ks}
    metrics.update({f"Precision@{k}": 0.0 for k in ks})
    metrics.update({f"nDCG@{k}":      0.0 for k in ks})
    metrics["MRR"] = 0.0

    for t in triplets:
        candidates = [t["positive"]] + t["negatives"]

        q_emb = model.encode(t["query"],   convert_to_tensor=True, normalize_embeddings=True)
        d_emb = model.encode(candidates,   convert_to_tensor=True, normalize_embeddings=True)

        ranking = torch.argsort(cos_sim(q_emb, d_emb)[0], descending=True).cpu().tolist()
        rank    = ranking.index(0) + 1

        metrics["MRR"] += 1 / rank

        for k in ks:
            if 0 in ranking[:k]:
                metrics[f"Recall@{k}"]    += 1
                metrics[f"Precision@{k}"] += 1 / k
                metrics[f"nDCG@{k}"]      += 1 / np.log2(rank + 1)

    n = len(triplets)
    return {k: round(v / n, 4) for k, v in metrics.items()}


def compute_corpus_metrics(ranked_results: List[Dict], gold_doc_ids: List[str],
                            ks: List[int] = [5, 10, 20]) -> Dict:
    metrics = {f"Recall@{k}": 0.0 for k in ks}
    metrics["MRR"] = 0.0
    N = len(ranked_results)

    for results, gold_id in zip(ranked_results, gold_doc_ids):
        result_ids = [r["doc_id"] for r in results]

        if gold_id in result_ids:
            rank = result_ids.index(gold_id) + 1
            metrics["MRR"] += 1 / rank
            for k in ks:
                if rank <= k:
                    metrics[f"Recall@{k}"] += 1

    return {k: round(v / N, 4) for k, v in metrics.items()}
