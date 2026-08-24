import json
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.util import cos_sim


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


def split_triplets(triplets: List[Dict], test_size: float = 0.2):
    queries = [t["query"] for t in triplets]
    gss = GroupShuffleSplit(test_size=test_size, n_splits=1, random_state=42)
    train_idx, val_idx = next(gss.split(triplets, groups=queries))
    return [triplets[i] for i in train_idx], [triplets[i] for i in val_idx]


def triplets_to_pairs(triplets: List[Dict]) -> List[InputExample]:
    return [InputExample(texts=[t["query"], t["positive"]]) for t in triplets]


def evaluate_ranking_metrics(model: SentenceTransformer, triplets: List[Dict],
                              ks: List[int] = [5, 10, 20]) -> Dict:
    metrics = {f"Recall@{k}": 0 for k in ks}
    metrics.update({f"nDCG@{k}": 0 for k in ks})
    metrics["MRR"] = 0

    for t in triplets:
        candidates = [t["positive"]] + t["negatives"]
        q_emb = model.encode(t["query"],    convert_to_tensor=True, normalize_embeddings=True)
        d_emb = model.encode(candidates,    convert_to_tensor=True, normalize_embeddings=True)

        ranking = torch.argsort(cos_sim(q_emb, d_emb)[0], descending=True).cpu().tolist()
        rank    = ranking.index(0) + 1

        metrics["MRR"] += 1 / rank
        for k in ks:
            if 0 in ranking[:k]:
                metrics[f"Recall@{k}"] += 1
                metrics[f"nDCG@{k}"]   += 1 / np.log2(rank + 1)

    n = len(triplets)
    return {k: v / n for k, v in metrics.items()}


def train(triplets_path: str, output_dir: str = "models/finder_dense_encoder_best",
          base_model: str = "sentence-transformers/all-mpnet-base-v2",
          epochs: int = 5, batch_size: int = 16, lr: float = 2e-5,
          max_seq_length: int = 256) -> None:

    all_triplets = load_finder_triplets(triplets_path)
    train_triplets, val_triplets = split_triplets(all_triplets)
    train_examples = triplets_to_pairs(train_triplets)

    model = SentenceTransformer(base_model)
    model.max_seq_length = max_seq_length

    train_loader = DataLoader(train_examples, shuffle=True,
                              batch_size=batch_size, drop_last=True)
    train_loss   = losses.MultipleNegativesRankingLoss(model)

    best_mrr = 0.0
    print(f"Training for {epochs} epochs on {len(train_examples)} examples...")

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")

        model.fit(
            train_objectives=[(train_loader, train_loss)],
            epochs=1,
            warmup_steps=100,
            optimizer_params={"lr": lr},
            use_amp=True,
            show_progress_bar=True,
        )

        val_metrics = evaluate_ranking_metrics(model, val_triplets)
        print("Validation metrics:")
        for k, v in val_metrics.items():
            print(f"  {k}: {v:.4f}")

        if val_metrics["MRR"] > best_mrr:
            best_mrr = val_metrics["MRR"]
            model.save(output_dir)
            print(f"  ✅ Saved new best model (MRR={best_mrr:.4f})")

    print(f"\nTraining complete. Best MRR: {best_mrr:.4f}")


if __name__ == "__main__":
    train("finder_triplets_optimized.jsonl")
