import pickle
import torch
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


class DenseRetriever:
    def __init__(self, model_path: str, max_seq_length: int = 256):
        self.model = SentenceTransformer(model_path)
        self.model.max_seq_length = max_seq_length
        self.corpus_embeddings = None
        self.corpus_docs = None

    def encode_corpus(self, docs: List[Dict], batch_size: int = 128,
                      show_progress: bool = True) -> None:
        self.corpus_docs = docs
        texts = [d["text"] for d in docs]
        self.corpus_embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
            convert_to_tensor=True,
        )

    def save_embeddings(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.corpus_embeddings, f)
        print(f"Saved embeddings → {path}")

    def load_embeddings(self, path: str, docs: List[Dict]) -> None:
        with open(path, "rb") as f:
            self.corpus_embeddings = pickle.load(f)
        self.corpus_docs = docs
        print(f"Loaded embeddings from {path}")

    def search(self, query: str, top_k: int = 20) -> List[Dict]:
        q_emb   = self.model.encode(query, normalize_embeddings=True,
                                    convert_to_tensor=True)
        scores  = cos_sim(q_emb, self.corpus_embeddings)[0]
        top_idx = torch.topk(scores, top_k).indices.tolist()

        return [
            {
                "rank":    rank,
                "doc_id":  self.corpus_docs[idx]["doc_id"],
                "score":   float(scores[idx]),
                "text":    self.corpus_docs[idx]["text"],
                "filename": self.corpus_docs[idx]["filename"],
                "section": self.corpus_docs[idx]["section"],
            }
            for rank, idx in enumerate(top_idx, start=1)
        ]
