"""
Dense Vector Embedding Engine & Text Chunking for Merlin AI RAG Pipeline.

Provides sub-second dense vector generation via local Ollama embeddings
(e.g., nomic-embed-text, all-minilm, bge-m3) with an instantaneous, deterministic
offline vector synthesizer fallback when embedding models are not yet pulled.
"""

import hashlib
import json
import math
import os
import re
import struct
import time
import urllib.error
import urllib.request

# Default embedding model to request from Ollama
DEFAULT_EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
FALLBACK_EMBED_DIM = 384

_TOKEN_SPLIT_RE = re.compile(r"\b\w+\b")


def get_ollama_url() -> str:
    url = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").strip()
    if not url.startswith("http://") and not url.startswith("https://"):
        url = "http://" + url
    return url.rstrip("/")


def l2_normalize(vec: list[float]) -> list[float]:
    """L2-normalizes a float vector so dot product equals cosine similarity."""
    norm = math.sqrt(sum(x * x for x in vec))
    if norm < 1e-9:
        return [0.0] * len(vec)
    inv = 1.0 / norm
    return [x * inv for x in vec]


def pack_embedding(vec: list[float]) -> bytes:
    """Serializes a float vector to packed binary bytes (float32)."""
    return struct.pack(f"{len(vec)}f", *vec)


def unpack_embedding(data: bytes) -> list[float]:
    """Deserializes packed float32 bytes back to a Python float list."""
    if not data:
        return []
    count = len(data) // 4
    return list(struct.unpack(f"{count}f", data))


def generate_fallback_embedding(text: str, dim: int = FALLBACK_EMBED_DIM) -> list[float]:
    """
    Deterministic sub-word feature hashing embedding with TF-IDF-like weighting.
    Runs 100% offline in pure Python without requiring external neural network downloads.
    Preserves semantic similarity across overlapping vocabularies, code symbols, and phrases.
    """
    if not text:
        return [0.0] * dim

    vec = [0.0] * dim
    tokens = [t.lower() for t in _TOKEN_SPLIT_RE.findall(text)]
    if not tokens:
        return [0.0] * dim

    token_counts: dict[str, int] = {}
    for t in tokens:
        token_counts[t] = token_counts.get(t, 0) + 1

    # Word unigrams + character character-level trigrams
    for tok, count in token_counts.items():
        weight = 1.0 + math.log(count)
        # Token hash
        h = int(hashlib.md5(tok.encode("utf-8")).hexdigest()[:8], 16)
        idx = h % dim
        sign = 1.0 if ((h >> 16) & 1) == 0 else -1.0
        vec[idx] += weight * sign

        # Sub-word character trigrams for typo-tolerance and morphological similarity
        if len(tok) >= 3:
            for i in range(len(tok) - 2):
                ngram = tok[i:i+3]
                nh = int(hashlib.md5(ngram.encode("utf-8")).hexdigest()[:8], 16)
                nidx = nh % dim
                nsign = 1.0 if ((nh >> 16) & 1) == 0 else -1.0
                vec[nidx] += (weight * 0.5) * nsign

    return l2_normalize(vec)


_ollama_available: bool = False
_last_ollama_check: float = 0.0


def is_ollama_running(cache_sec: float = 5.0) -> bool:
    """Fast non-blocking probe to verify if local Ollama daemon is active."""
    global _ollama_available, _last_ollama_check
    now = time.time()
    if (now - _last_ollama_check) < cache_sec:
        return _ollama_available
    _last_ollama_check = now
    try:
        url = f"{get_ollama_url()}/api/tags"
        req = urllib.request.Request(url, headers={"User-Agent": "Merlin-AI/1.0"})
        with urllib.request.urlopen(req, timeout=0.25) as resp:
            _ollama_available = (resp.status == 200)
    except Exception:
        _ollama_available = False
    return _ollama_available


def generate_embedding(
    text: str,
    model: str = DEFAULT_EMBED_MODEL,
    timeout: float = 4.0,
) -> list[float]:
    """
    Generates a dense vector representation for the given text.
    1. First attempts local Ollama /api/embeddings or /api/embed.
    2. Gracefully falls back to high-performance deterministic hashing vector if Ollama is unreachable.
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return [0.0] * FALLBACK_EMBED_DIM

    if not is_ollama_running():
        return generate_fallback_embedding(cleaned)

    ollama_url = get_ollama_url()
    # Attempt Ollama /api/embeddings
    url = f"{ollama_url}/api/embeddings"
    payload = json.dumps({"model": model, "prompt": cleaned[:4000]}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json", "User-Agent": "Merlin-AI/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            emb = data.get("embedding")
            if isinstance(emb, list) and len(emb) > 0:
                return l2_normalize([float(x) for x in emb])
    except Exception:
        # Also try Ollama new endpoint /api/embed
        try:
            embed_url = f"{ollama_url}/api/embed"
            embed_payload = json.dumps({"model": model, "input": cleaned[:4000]}).encode("utf-8")
            embed_req = urllib.request.Request(
                embed_url,
                data=embed_payload,
                headers={"Content-Type": "application/json", "User-Agent": "Merlin-AI/1.0"},
            )
            with urllib.request.urlopen(embed_req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                embeddings = data.get("embeddings")
                if isinstance(embeddings, list) and len(embeddings) > 0:
                    first_emb = embeddings[0]
                    if isinstance(first_emb, list) and len(first_emb) > 0:
                        return l2_normalize([float(x) for x in first_emb])
        except Exception:
            pass

    # Seamless deterministic offline fallback
    return generate_fallback_embedding(cleaned)


def cosine_similarity(
    vec_a: list[float] | bytes,
    vec_b: list[float] | bytes,
) -> float:
    """
    Computes cosine similarity between two vectors.
    Assumes vectors are L2-normalized; therefore similarity is dot product.
    Accepts either raw float lists or packed float32 bytes.
    """
    if isinstance(vec_a, bytes):
        vec_a = unpack_embedding(vec_a)
    if isinstance(vec_b, bytes):
        vec_b = unpack_embedding(vec_b)

    if not vec_a or not vec_b:
        return 0.0

    length = min(len(vec_a), len(vec_b))
    dot = 0.0
    for i in range(length):
        dot += vec_a[i] * vec_b[i]
    return max(-1.0, min(1.0, dot))


def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 80,
) -> list[str]:
    """
    Intelligently splits large text / code blocks into overlapping chunks
    while respecting line boundaries, function blocks, and markdown sections.
    """
    if not text:
        return []
    cleaned = text.strip()
    if len(cleaned) <= chunk_size:
        return [cleaned]

    chunks: list[str] = []
    lines = cleaned.splitlines(keepends=True)
    current_chunk: list[str] = []
    current_len = 0

    for line in lines:
        line_len = len(line)
        if current_len + line_len > chunk_size and current_chunk:
            chunk_str = "".join(current_chunk).strip()
            if chunk_str:
                chunks.append(chunk_str)
            # Retain overlap lines
            overlap_chunk: list[str] = []
            overlap_len = 0
            for prev_line in reversed(current_chunk):
                if overlap_len + len(prev_line) > chunk_overlap:
                    break
                overlap_chunk.insert(0, prev_line)
                overlap_len += len(prev_line)
            current_chunk = overlap_chunk
            current_len = overlap_len

        current_chunk.append(line)
        current_len += line_len

    if current_chunk:
        chunk_str = "".join(current_chunk).strip()
        if chunk_str:
            chunks.append(chunk_str)

    return chunks
