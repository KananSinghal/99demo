"""Bi-encoder embeddings.

Two backends, same interface:

  hash  (default)  A deterministic hashed character-n-gram + token embedding with
                   IDF-style weighting. No downloads, no GPU, no model weights -
                   which means it runs inside an air-gapped refinery node and
                   produces byte-identical vectors on every machine. Good enough
                   to carry candidate generation, which is a recall problem.

  st               sentence-transformers with an open-weight model (bge-small by
                   default). Used where the team has installed it and cached the
                   weights. Never calls an external API.

Candidate generation is recall-first: a pair the blocker drops can never be matched
however good the reranker is, so this stage is tuned for recall and the precision
work happens downstream in the reranker, the constraint solver and the graph.
"""

from __future__ import annotations

import hashlib
import math
import struct
from typing import Dict, Iterable, List, Optional, Sequence

from .. import config
from . import text

try:  # numpy is a nice-to-have, never a requirement
    import numpy as _np
except Exception:  # pragma: no cover
    _np = None

_ST_MODEL = None
_IDF: Dict[str, float] = {}
_N_DOCS = 0


# ------------------------------------------------------------------ IDF corpus

def fit_idf(corpus: Iterable[str]) -> int:
    """Learn token document frequencies so common words (BOLT, PIPE) stop dominating
    the vector and the discriminative tokens (M12X1.75, 6205) carry it."""
    global _IDF, _N_DOCS
    df: Dict[str, int] = {}
    n = 0
    for doc in corpus:
        n += 1
        for tok in set(_features(doc)):
            df[tok] = df.get(tok, 0) + 1
    _N_DOCS = max(n, 1)
    _IDF = {tok: math.log((_N_DOCS + 1) / (cnt + 0.5)) + 1.0 for tok, cnt in df.items()}
    return len(_IDF)


def idf_state() -> dict:
    return {"documents": _N_DOCS, "vocabulary": len(_IDF)}


def load_idf(state: dict) -> None:
    global _IDF, _N_DOCS
    _IDF = {k: float(v) for k, v in (state.get("idf") or {}).items()}
    _N_DOCS = int(state.get("documents") or 1)


def dump_idf() -> dict:
    return {"documents": _N_DOCS, "idf": _IDF}


# ------------------------------------------------------------------ hash backend

def _features(value: str) -> List[str]:
    norm = text.normalise(value)
    toks = [t for t in norm.split(" ") if t]
    feats: List[str] = []
    feats.extend(f"w:{t}" for t in toks)
    for i in range(len(toks) - 1):
        feats.append(f"b:{toks[i]}_{toks[i+1]}")
    feats.extend(f"c:{g}" for g in text.char_ngrams(norm, 3, 4))
    return feats


def _bucket(feature: str, dim: int) -> int:
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
    return struct.unpack("<Q", digest)[0] % dim


def _sign(feature: str) -> float:
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=1, salt=b"sign").digest()
    return 1.0 if digest[0] & 1 else -1.0


def _hash_embed(value: str, dim: int) -> List[float]:
    vec = [0.0] * dim
    for feat in _features(value):
        weight = _IDF.get(feat, 1.6 if feat.startswith("w:") else 0.6)
        vec[_bucket(feat, dim)] += _sign(feat) * weight
    norm = math.sqrt(sum(v * v for v in vec))
    if norm <= 0:
        return vec
    return [v / norm for v in vec]


# ------------------------------------------------------------------ st backend

def _st():
    global _ST_MODEL
    if _ST_MODEL is None:
        from sentence_transformers import SentenceTransformer  # noqa: WPS433
        _ST_MODEL = SentenceTransformer(config.EMBEDDING_MODEL)
    return _ST_MODEL


def backend() -> str:
    if config.EMBEDDING_BACKEND == "st":
        try:
            _st()
            return "st"
        except Exception:
            return "hash"
    return "hash"


def dim() -> int:
    if backend() == "st":
        try:
            return int(_st().get_sentence_embedding_dimension())
        except Exception:
            pass
    return config.EMBEDDING_DIM


# ------------------------------------------------------------------ public API

def encode(value: str) -> List[float]:
    return encode_many([value])[0]


def encode_many(values: Sequence[str], batch_size: int = 256) -> List[List[float]]:
    if backend() == "st":
        try:
            model = _st()
            out = model.encode(
                [text.normalise(v) for v in values],
                batch_size=batch_size, normalize_embeddings=True,
                show_progress_bar=False,
            )
            return [list(map(float, row)) for row in out]
        except Exception:
            pass
    d = config.EMBEDDING_DIM
    return [_hash_embed(v, d) for v in values]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if _np is not None:
        va, vb = _np.asarray(a, dtype=float), _np.asarray(b, dtype=float)
        na, nb = float(_np.linalg.norm(va)), float(_np.linalg.norm(vb))
        if na == 0 or nb == 0:
            return 0.0
        return float(va.dot(vb) / (na * nb))
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def pack(vector: Sequence[float]) -> bytes:
    return struct.pack(f"<{len(vector)}f", *[float(v) for v in vector])


def unpack(blob: bytes) -> List[float]:
    if not blob:
        return []
    return list(struct.unpack(f"<{len(blob)//4}f", blob))


# ------------------------------------------------------------------ ANN index

class AnnIndex:
    """Exact cosine search, vectorised through numpy when present.

    For a national run this is where pgvector/HNSW or FAISS goes - the interface is
    deliberately the same three methods so swapping it is a one-file change. At the
    demo corpus size an exact search is both faster and free of recall loss, and
    quoting recall@blocking honestly matters more than quoting index speed.
    """

    def __init__(self, dim_: Optional[int] = None):
        self.dim = dim_ or dim()
        self.ids: List[int] = []
        self._rows: List[List[float]] = []
        self._matrix = None
        self._groups: List[str] = []

    def add(self, item_id: int, vector: Sequence[float], group: str = "") -> None:
        self.ids.append(item_id)
        self._rows.append(list(vector))
        self._groups.append(group)
        self._matrix = None

    def build(self) -> None:
        if _np is not None and self._rows:
            mat = _np.asarray(self._rows, dtype="float32")
            norms = _np.linalg.norm(mat, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self._matrix = mat / norms
        else:
            self._matrix = None

    def __len__(self) -> int:
        return len(self.ids)

    def query(self, vector: Sequence[float], top_k: int = 50, exclude_group: str = "") -> List[tuple]:
        """Return [(item_id, score)] sorted by descending cosine.

        exclude_group suppresses same-CPSE hits: harmonisation is a cross-organisation
        problem, and a CPSE's internal duplicates are that CPSE's own business.
        """
        if not self.ids:
            return []
        if self._matrix is None and _np is not None:
            self.build()
        if self._matrix is not None:
            q = _np.asarray(vector, dtype="float32")
            n = float(_np.linalg.norm(q))
            if n == 0:
                return []
            scores = self._matrix.dot(q / n)
            if exclude_group:
                mask = _np.asarray([g == exclude_group for g in self._groups])
                scores = _np.where(mask, -2.0, scores)
            k = min(top_k, len(self.ids))
            idx = _np.argpartition(-scores, k - 1)[:k] if k < len(self.ids) else _np.arange(len(self.ids))
            pairs = [(self.ids[int(i)], float(scores[int(i)])) for i in idx if scores[int(i)] > -1.5]
            pairs.sort(key=lambda p: -p[1])
            return pairs[:top_k]
        scored = []
        for item_id, row, group in zip(self.ids, self._rows, self._groups):
            if exclude_group and group == exclude_group:
                continue
            scored.append((item_id, cosine(vector, row)))
        scored.sort(key=lambda p: -p[1])
        return scored[:top_k]
