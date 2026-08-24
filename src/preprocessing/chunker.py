import re
from typing import List, Dict, Tuple


def simple_tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


def chunk_by_token_window(
    text: str, window_size: int = 300, overlap: int = 100
) -> List[Tuple[int, int, int, str]]:
    tokens = re.findall(r"\w+|\S", text)
    if not tokens:
        return []

    chunks = []
    start = 0
    chunk_id = 0

    while start < len(tokens):
        end = min(start + window_size, len(tokens))
        chunk_text = " ".join(tokens[start:end])
        chunks.append((chunk_id, start, end, chunk_text))
        chunk_id += 1
        if end == len(tokens):
            break
        start = end - overlap

    return chunks


def chunk_by_sentence_overlap(
    text: str, max_sentences: int = 6, overlap: int = 1
) -> List[Tuple[int, int, int, str]]:
    sentences = re.split(r'(?<=[\.\?\!])\s+', text.strip())
    chunks = []
    i = 0
    chunk_id = 0

    while i < len(sentences):
        chunk_sentences = sentences[i:i + max_sentences]
        chunk_text = " ".join(chunk_sentences).strip()
        chunks.append((chunk_id, i, i + len(chunk_sentences), chunk_text))
        chunk_id += 1
        i += max_sentences - overlap

    return chunks


def build_chunks_from_section_texts(
    section_texts: Dict[str, Dict[str, str]],
    chunk_method: str = "token",
    window_size: int = 300,
    overlap: int = 100,
    max_sentences: int = 6,
    sentence_overlap: int = 1,
) -> List[Dict]:
    all_chunks = []

    for filename, sections in section_texts.items():
        for section_name, text in sections.items():
            if not text or not text.strip():
                continue

            if chunk_method == "sentence":
                raw_chunks = chunk_by_sentence_overlap(
                    text, max_sentences=max_sentences, overlap=sentence_overlap
                )
                for cid, sidx, eidx, chunk_text in raw_chunks:
                    all_chunks.append({
                        "doc_id":     f"{filename}::sec_{section_name}::chunk{cid}",
                        "filename":   filename,
                        "section":    section_name,
                        "chunk_id":   cid,
                        "start_sent": sidx,
                        "end_sent":   eidx,
                        "text":       chunk_text,
                    })
            else:
                raw_chunks = chunk_by_token_window(
                    text, window_size=window_size, overlap=overlap
                )
                for cid, start_tok, end_tok, chunk_text in raw_chunks:
                    all_chunks.append({
                        "doc_id":      f"{filename}::sec_{section_name}::chunk{cid}",
                        "filename":    filename,
                        "section":     section_name,
                        "chunk_id":    cid,
                        "start_token": start_tok,
                        "end_token":   end_tok,
                        "text":        chunk_text,
                    })

    return all_chunks
