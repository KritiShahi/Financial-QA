import re
from typing import List, Dict, Callable
from rank_bm25 import BM25Okapi


def simple_tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


class BM25Retriever:
    def __init__(self, chunk_docs: List[Dict], tokenizer: Callable = simple_tokenize):
        self.docs = chunk_docs
        self.tokenizer = tokenizer
        self.corpus_tokens = [self.tokenizer(d["text"]) for d in self.docs]
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        q_tokens = self.tokenizer(query)
        scores = self.bm25.get_scores(q_tokens)
        ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for rank, idx in enumerate(ranked_idx, start=1):
            d = self.docs[idx]
            results.append({
                "rank":         rank,
                "doc_id":       d["doc_id"],
                "filename":     d["filename"],
                "section":      d["section"],
                "chunk_id":     d["chunk_id"],
                "score":        float(scores[idx]),
                "text_snippet": d["text"][:600],
            })
        return results
