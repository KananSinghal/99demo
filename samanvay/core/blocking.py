"""Candidate generation.

5 million records is 1.25 x 10^13 naive pairs. Union three cheap strategies and the
survivors are ~91 million - a 137,000x reduction:

  deterministic keys   normalised manufacturer part number, vendor material number.
                       Near-perfect precision, low recall.
  attribute keys       (class, primary dimension, material family) tuples. Catches
                       records whose text shares nothing at all.
  semantic ANN         top-k over embeddings. Catches records whose attributes
                       failed to extract.

Report blocking RECALL, never blocking speed. A candidate that never reaches the
reranker can never be matched, so recall here is the true ceiling of the system.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .. import config
from . import classify, embed, text


@dataclass
class Candidate:
    a_id: int
    b_id: int
    strategies: List[str]
    bi_score: float = 0.0

    def key(self) -> Tuple[int, int]:
        return (self.a_id, self.b_id) if self.a_id < self.b_id else (self.b_id, self.a_id)


def _norm_key(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", text.basic(value or ""))


# ------------------------------------------------------------------- key builders

def identity_keys(record: dict) -> List[str]:
    """Hard identity keys. Two records sharing one of these are the same physical item
    with near-certainty - the vendor or the manufacturer already resolved the entity."""
    keys: List[str] = []
    mfr, mpn = _norm_key(record.get("mfr_name")), _norm_key(record.get("mfr_part_no"))
    if mpn and len(mpn) >= 4:
        keys.append(f"mpn:{mfr}|{mpn}" if mfr else f"mpn:*|{mpn}")
    vendor, vmn = _norm_key(record.get("vendor_id")), _norm_key(record.get("vendor_matl_no"))
    if vendor and vmn and len(vmn) >= 3:
        keys.append(f"vmn:{vendor}|{vmn}")
    return keys


def attribute_keys(record: dict, attributes: Dict[str, str]) -> List[str]:
    """Structural keys built from the extracted attribute vector. These are what catch
    'HEX BOLT M12X50 SS316' against 'BOLT,HEXAGONAL,M12 X 50MM,A4-70' - two strings
    with almost no tokens in common."""
    cls = record.get("class_code") or "GENERIC"
    keys: List[str] = []
    blocking = classify.blocking_keys(cls)
    primary = [k for k in ("thread", "nominal_size", "bore_mm", "designation", "csa_sqmm") if k in attributes]
    material = attributes.get("material") or attributes.get("insulation") or ""
    if primary:
        keys.append(f"attr:{cls}|{_norm_key(attributes[primary[0]])}|{_norm_key(material)}")
        keys.append(f"dim:{cls}|{_norm_key(attributes[primary[0]])}")
    if attributes.get("standard") and primary:
        keys.append(f"std:{_norm_key(attributes['standard'])}|{_norm_key(attributes[primary[0]])}")
    combo = "|".join(_norm_key(attributes.get(k, "")) for k in blocking if attributes.get(k))
    if combo and len(combo) >= 4:
        keys.append(f"blk:{cls}|{combo}")
    return keys


def token_keys(record: dict, max_keys: int = 3) -> List[str]:
    """Fallback for records whose attributes did not extract: the rarest content tokens."""
    toks = text.content_tokens(record.get("description", ""))
    scored = sorted({t for t in toks if any(c.isdigit() for c in t) or len(t) >= 5},
                    key=lambda t: (-len(t), t))
    return [f"tok:{_norm_key(t)}" for t in scored[:max_keys] if len(_norm_key(t)) >= 3]


def blocking_keys_for(record: dict, attributes: Dict[str, str]) -> List[str]:
    keys = identity_keys(record) + attribute_keys(record, attributes) + token_keys(record)
    return list(dict.fromkeys(k for k in keys if k))


def hashed_keys(record: dict, attributes: Dict[str, str], salt: str = "") -> List[str]:
    """Blocking keys as salted hashes.

    A CPSE edge node publishes these instead of the keys themselves, so the registry
    can find overlap without ever seeing a manufacturer part number or a vendor code.
    See core.federation.
    """
    import hashlib
    out = []
    for key in blocking_keys_for(record, attributes):
        out.append(hashlib.sha256(f"{salt}|{key}".encode("utf-8")).hexdigest()[:24])
    return out


# ------------------------------------------------------------------- generation

def generate(
    records: Sequence[dict],
    vectors: Optional[Dict[int, Sequence[float]]] = None,
    top_k: Optional[int] = None,
    cross_org_only: bool = True,
    max_block_size: int = 400,
) -> Tuple[List[Candidate], dict]:
    """Produce the candidate pair set.

    Each record dict needs: id, cpse_code, class_code, description, attributes
    (a flat {key: value} map) and optionally mfr/vendor identity fields.
    """
    top_k = top_k or config.ANN_TOP_K
    by_key: Dict[str, List[int]] = defaultdict(list)
    org_of: Dict[int, str] = {}
    strategies: Dict[Tuple[int, int], Set[str]] = defaultdict(set)

    for rec in records:
        rid = int(rec["id"])
        org_of[rid] = rec.get("cpse_code", "")
        for key in blocking_keys_for(rec, rec.get("attributes") or {}):
            by_key[key].append(rid)

    oversized = 0
    for key, members in by_key.items():
        if len(members) < 2:
            continue
        if len(members) > max_block_size:
            oversized += 1
            continue
        kind = key.split(":", 1)[0]
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]
                if a == b:
                    continue
                if cross_org_only and org_of.get(a) == org_of.get(b):
                    continue
                strategies[(a, b) if a < b else (b, a)].add(kind)

    ann_pairs = 0
    if vectors:
        index = embed.AnnIndex()
        for rec in records:
            rid = int(rec["id"])
            vec = vectors.get(rid)
            if vec:
                index.add(rid, vec, group=rec.get("cpse_code", ""))
        index.build()
        for rec in records:
            rid = int(rec["id"])
            vec = vectors.get(rid)
            if not vec:
                continue
            hits = index.query(
                vec, top_k=top_k + 1,
                exclude_group=rec.get("cpse_code", "") if cross_org_only else "",
            )
            for other, score in hits:
                if other == rid or score < config.BI_ENCODER_CUTOFF:
                    continue
                pair = (rid, other) if rid < other else (other, rid)
                strategies[pair].add("ann")
                ann_pairs += 1

    candidates = [Candidate(a_id=a, b_id=b, strategies=sorted(kinds)) for (a, b), kinds in strategies.items()]
    stats = {
        "records": len(records),
        "naive_pairs": len(records) * (len(records) - 1) // 2,
        "blocking_keys": len(by_key),
        "oversized_blocks_skipped": oversized,
        "ann_pairs_raw": ann_pairs,
        "candidate_pairs": len(candidates),
        "reduction_factor": round(
            (len(records) * (len(records) - 1) / 2) / max(len(candidates), 1), 1
        ),
        "strategy_mix": _strategy_mix(candidates),
    }
    return candidates, stats


def _strategy_mix(candidates: Sequence[Candidate]) -> Dict[str, int]:
    mix: Dict[str, int] = defaultdict(int)
    for cand in candidates:
        for s in cand.strategies:
            mix[s] += 1
    return dict(sorted(mix.items(), key=lambda kv: -kv[1]))


def recall(candidates: Sequence[Candidate], truth_pairs: Iterable[Tuple[int, int]]) -> dict:
    """Blocking recall against a ground-truth pair set. This is the number to quote:
    it is the hard ceiling on everything downstream."""
    got = {c.key() for c in candidates}
    truth = {(a, b) if a < b else (b, a) for a, b in truth_pairs}
    if not truth:
        return {"recall": None, "truth_pairs": 0, "found": 0, "missed": []}
    found = truth & got
    missed = sorted(truth - got)
    return {
        "recall": round(len(found) / len(truth), 4),
        "truth_pairs": len(truth),
        "found": len(found),
        "missed_count": len(missed),
        "missed": missed[:50],
    }
