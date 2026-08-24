import json
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader
from sentence_transformers import InputExample, CrossEncoder


def load_finder_triplets(path: str) -> List[InputExample]:
    examples = []
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            query = item["query"]
            examples.append(InputExample(texts=[query, item["positive"]["text"]], label=1))
            for neg in item["negatives"]:
                examples.append(InputExample(texts=[query, neg["text"]], label=0))
    return examples


def split_examples(examples: List[InputExample], test_size: float = 0.1):
    queries = [ex.texts[0] for ex in examples]
    gss = GroupShuffleSplit(test_size=test_size, n_splits=1, random_state=42)
    train_idx, val_idx = next(gss.split(examples, groups=queries))
    return [examples[i] for i in train_idx], [examples[i] for i in val_idx]


def make_collate_fn(model: CrossEncoder):
    def collate_fn(batch):
        queries   = [ex.texts[0] for ex in batch]
        passages  = [ex.texts[1] for ex in batch]
        labels    = torch.tensor([ex.label for ex in batch], dtype=torch.float)
        tokenized = model.tokenizer(
            queries, passages,
            padding=True, truncation="longest_first",
            max_length=model.max_length, return_tensors="pt"
        )
        return tokenized, labels
    return collate_fn


def compute_val_loss(model: CrossEncoder, dataloader: DataLoader) -> float:
    model.model.eval()
    losses = []
    device = model.device
    with torch.no_grad():
        for features, labels in dataloader:
            features = {k: v.to(device) for k, v in features.items()}
            labels   = labels.float().to(device)
            logits   = model.model(**features).logits.squeeze(-1)
            losses.append(F.binary_cross_entropy_with_logits(logits, labels).item())
    return sum(losses) / len(losses)


def compute_recall_at_k(model: CrossEncoder, examples: List[InputExample],
                         k_list: List[int] = [5, 10, 20]) -> Dict:
    grouped = {}
    for ex in examples:
        grouped.setdefault(ex.texts[0], []).append((ex.texts[1], ex.label))

    recalls = {k: [] for k in k_list}
    for q, pairs in grouped.items():
        passages, labels = zip(*pairs)
        scores  = model.predict([[q, p] for p in passages])
        ranked  = sorted(zip(labels, scores), key=lambda x: -x[1])
        ranked_labels = [lbl for lbl, _ in ranked]
        for k in k_list:
            recalls[k].append(1 if 1 in ranked_labels[:k] else 0)

    return {k: float(np.mean(v)) for k, v in recalls.items()}


def train(triplets_path: str, output_dir: str = "cross_encoder_finder_best",
          base_model: str = "bert-base-uncased", epochs: int = 3,
          batch_size: int = 8, lr: float = 2e-5, max_length: int = 512) -> None:

    examples = load_finder_triplets(triplets_path)
    train_samples, val_samples = split_examples(examples)

    model  = CrossEncoder(base_model, num_labels=1, max_length=max_length)
    device = model.device

    collate_fn       = make_collate_fn(model)
    train_dataloader = DataLoader(train_samples, shuffle=True,
                                  batch_size=batch_size, collate_fn=collate_fn)
    val_dataloader   = DataLoader(val_samples, shuffle=False,
                                  batch_size=batch_size, collate_fn=collate_fn)

    optimizer     = torch.optim.AdamW(model.model.parameters(), lr=lr)
    best_val_loss = float("inf")

    print(f"Training cross-encoder for {epochs} epochs...")

    for epoch in range(epochs):
        model.model.train()
        total_loss, count = 0.0, 0

        for features, labels in train_dataloader:
            features = {k: v.to(device) for k, v in features.items()}
            labels   = labels.float().to(device)
            optimizer.zero_grad()
            logits = model.model(**features).logits.squeeze(-1)
            loss   = F.binary_cross_entropy_with_logits(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            count += 1

        avg_train = total_loss / count
        avg_val   = compute_val_loss(model, val_dataloader)
        print(f"Epoch {epoch+1}/{epochs} — Train: {avg_train:.4f} | Val: {avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            model.save(output_dir)
            print(f"  ✅ Saved best checkpoint (val_loss={best_val_loss:.4f})")

    recall = compute_recall_at_k(model, val_samples)
    print("\nFinal Recall on validation set:")
    for k, v in recall.items():
        print(f"  Recall@{k}: {v:.4f}")


if __name__ == "__main__":
    train("finder_triplets_optimized.jsonl")
