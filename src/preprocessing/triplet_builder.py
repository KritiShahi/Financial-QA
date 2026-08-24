import json
import random
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd
from rank_bm25 import BM25Okapi

random.seed(42)

NUM_NEGATIVES = 3
NEG_TOP_K = 30


def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip())


def tokenize(s: str) -> List[str]:
    return normalize(s).lower().split()


def parse_refs(x) -> List[str]:
    try:
        return json.loads(str(x).replace("'", '"'))
    except Exception:
        return []


def load_finder_dataset(csv_path: str) -> tuple:
    df = pd.read_csv(csv_path)
    df["text_x"] = df["text_x"].apply(normalize)
    df["company_name"] = df["company_name"].astype(str)
    df["references"] = df["references"].apply(parse_refs)
    queries = df.to_dict(orient="records")
    companies = set(df["company_name"].dropna().unique())
    return queries, companies


def build_company_indices(chunks: List[Dict], finder_companies: set) -> tuple:
    filtered = [ch for ch in chunks if ch["filename"] in finder_companies]

    company_to_chunks: Dict[str, List[Dict]] = {}
    for ch in filtered:
        company_to_chunks.setdefault(ch["filename"], []).append(ch)

    company_to_bm25 = {
        comp: BM25Okapi([tokenize(normalize(ch["text"])) for ch in clist])
        for comp, clist in company_to_chunks.items()
    }
    company_to_ids = {
        comp: [ch["doc_id"] for ch in clist]
        for comp, clist in company_to_chunks.items()
    }
    company_chunk_lookup = {
        comp: {ch["doc_id"]: ch for ch in clist}
        for comp, clist in company_to_chunks.items()
    }

    return company_to_chunks, company_to_bm25, company_to_ids, company_chunk_lookup


def choose_positive_chunk(query_obj: Dict, company: str,
                           company_to_chunks: Dict) -> Dict | None:
    refs = query_obj.get("references", [])
    ref_tokens = set(tokenize(" ".join(refs)))
    best_chunk, best_score = None, -1

    for ch in company_to_chunks[company]:
        score = len(set(tokenize(ch["text"])) & ref_tokens)
        if score > best_score:
            best_score = score
            best_chunk = ch

    return best_chunk


def sample_negatives(query_text: str, company: str, pos_doc_id: str,
                     company_to_bm25: Dict, company_to_ids: Dict,
                     company_chunk_lookup: Dict) -> List[Dict]:
    bm25      = company_to_bm25[company]
    chunk_ids = company_to_ids[company]
    scores    = bm25.get_scores(tokenize(query_text))
    ranked    = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    neg_ids = [
        chunk_ids[idx] for idx in ranked[:NEG_TOP_K]
        if chunk_ids[idx] != pos_doc_id
    ][:NUM_NEGATIVES]

    # fallback to other companies if not enough negatives
    if len(neg_ids) < NUM_NEGATIVES:
        other_ids = [
            cid for comp, ids in company_to_ids.items()
            if comp != company for cid in ids
        ]
        random.shuffle(other_ids)
        neg_ids.extend(other_ids[:NUM_NEGATIVES - len(neg_ids)])

    negatives = []
    for nid in neg_ids:
        comp = nid.split("::")[0]
        ch   = company_chunk_lookup[comp][nid]
        negatives.append({"chunk_id": nid, "text": ch["text"], "company_name": comp})

    return negatives


def build_triplets(queries: List[Dict], company_to_chunks: Dict,
                   company_to_bm25: Dict, company_to_ids: Dict,
                   company_chunk_lookup: Dict) -> List[Dict]:
    triplets = []

    for q in queries:
        company = q["company_name"]
        if company not in company_to_chunks:
            print(f"[SKIP] No chunks for company {company}, query {q['_id']}")
            continue

        pos = choose_positive_chunk(q, company, company_to_chunks)
        if pos is None:
            print(f"[WARN] No positive for query {q['_id']}")
            continue

        negatives = sample_negatives(
            q["text_x"], company, pos["doc_id"],
            company_to_bm25, company_to_ids, company_chunk_lookup
        )

        triplets.append({
            "query_id":    q["_id"],
            "query":       q["text_x"],
            "company_name": company,
            "positive":    {"chunk_id": pos["doc_id"], "text": pos["text"], "company_name": company},
            "negatives":   negatives,
        })

    return triplets


def save_triplets(triplets: List[Dict], out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        for t in triplets:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"Saved {len(triplets)} triplets → {out_path}")
