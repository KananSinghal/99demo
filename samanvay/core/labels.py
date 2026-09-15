"""Distant supervision: the training labels are already in the ERP.

"Where is your labelled data?" kills more of these projects than any other question.
The answer is that the vendor has been doing entity resolution for thirty years and
nobody read the answer off their order book.

  vendor material number   EINA-IDNLF + EINA-LIFNR. If two CPSEs' materials each
                           carry a purchasing info record against the SAME vendor
                           with the SAME vendor material number, they are the same
                           physical item. This is the strongest signal in the system
                           and it costs nothing.
  manufacturer part number MARA-MFRPN + MARA-MFRNR against a normalised maker name.
  classification vectors   AUSP / KSSK / CABN. Full characteristic-vector matches
                           within a class teach lexical robustness.

Hard negatives are mined, not written: same-block pairs with high similarity that
differ in EXACTLY ONE extracted attribute. That is 6205 against 6206, M12x50 against
M12x60, 150# against 300# - thousands of them, free, and precisely where a naive
model fails.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from . import blocking, classify, features, text


@dataclass
class Label:
    a_id: int
    b_id: int
    label: int             # 1 positive, 0 negative
    source: str            # vmn | mpn | classification | hard_negative | steward
    reason: str = ""
    weight: float = 1.0

    def key(self) -> Tuple[int, int]:
        return (self.a_id, self.b_id) if self.a_id < self.b_id else (self.b_id, self.a_id)


def _norm(value: Optional[str]) -> str:
    return re.sub(r"[^A-Z0-9]", "", text.basic(value or ""))


# ------------------------------------------------------------------- positives

def mine_positives(records: Sequence[dict], cross_org_only: bool = True) -> List[Label]:
    by_vmn: Dict[str, List[int]] = defaultdict(list)
    by_mpn: Dict[str, List[int]] = defaultdict(list)
    by_clfn: Dict[str, List[int]] = defaultdict(list)
    org: Dict[int, str] = {}

    for rec in records:
        rid = int(rec["id"])
        org[rid] = rec.get("cpse_code", "")
        vendor, vmn = _norm(rec.get("vendor_id")), _norm(rec.get("vendor_matl_no"))
        if vendor and vmn and len(vmn) >= 3:
            by_vmn[f"{vendor}|{vmn}"].append(rid)
        mfr, mpn = _norm(rec.get("mfr_name")), _norm(rec.get("mfr_part_no"))
        if mpn and len(mpn) >= 4:
            by_mpn[f"{mfr}|{mpn}" if mfr else f"*|{mpn}"].append(rid)
        attrs = rec.get("attributes") or {}
        cls = rec.get("class_code") or "GENERIC"
        mand = classify.mandatory_keys(cls)
        if mand and all(attrs.get(k) for k in mand):
            vec = "|".join(f"{k}={_norm(str(attrs[k]))}" for k in sorted(attrs) if attrs[k])
            if len(vec) > 12:
                by_clfn[f"{cls}|{vec}"].append(rid)

    out: List[Label] = []
    seen: Set[Tuple[int, int]] = set()

    def emit(groups: Dict[str, List[int]], source: str, reason: str, weight: float,
             require_cross_org: bool) -> None:
        for key, members in groups.items():
            if len(members) < 2 or len(members) > 60:
                continue
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    a, b = members[i], members[j]
                    if require_cross_org and cross_org_only and org.get(a) == org.get(b):
                        continue
                    pair = (a, b) if a < b else (b, a)
                    if pair in seen:
                        continue
                    seen.add(pair)
                    out.append(Label(pair[0], pair[1], 1, source, reason, weight))

    emit(by_vmn, "vmn",
         "same vendor and same vendor material number on both purchasing info records - "
         "the vendor already resolved this entity", 1.0, True)
    emit(by_mpn, "mpn",
         "same manufacturer part number against the same normalised manufacturer", 1.0, True)
    emit(by_clfn, "classification",
         "identical classification characteristic vector within the class", 0.6, False)
    return out


# ------------------------------------------------------------------- negatives

def mine_hard_negatives(
    records: Sequence[dict],
    candidates: Sequence,
    positives: Sequence[Label],
    max_negatives: int = 4000,
) -> List[Label]:
    """Same-block, high-similarity pairs differing in exactly one extracted attribute.

    These are the pairs an embedding-only system gets wrong, so they are the only
    negatives worth training on. Random negatives teach a model nothing.
    """
    by_id = {int(r["id"]): r for r in records}
    positive_keys = {p.key() for p in positives}
    out: List[Label] = []

    for cand in candidates:
        key = cand.key() if hasattr(cand, "key") else (
            (cand[0], cand[1]) if cand[0] < cand[1] else (cand[1], cand[0])
        )
        if key in positive_keys:
            continue
        a, b = by_id.get(key[0]), by_id.get(key[1])
        if not a or not b:
            continue
        if a.get("class_code") != b.get("class_code"):
            continue
        cmp = features.compare_attributes(a, b)
        conflicts = cmp["conflicts"]
        if len(conflicts) != 1:
            continue
        row = conflicts[0]
        shared = cmp["exact"] + cmp["kg_equiv"]
        if shared < 2:
            continue
        out.append(Label(
            key[0], key[1], 0, "hard_negative",
            f"differs in exactly one attribute - {row['label']}: {row['a']} vs {row['b']}",
            1.0,
        ))
        if len(out) >= max_negatives:
            break
    return out


def mine_easy_negatives(records: Sequence[dict], per_record: int = 1, seed: int = 7) -> List[Label]:
    """A thin band of obviously-different pairs, so the model also learns the easy
    boundary. Deliberately few - hard negatives do the real work."""
    import random
    rng = random.Random(seed)
    by_class: Dict[str, List[int]] = defaultdict(list)
    for rec in records:
        by_class[rec.get("class_code") or "GENERIC"].append(int(rec["id"]))
    classes = [c for c, ids in by_class.items() if ids]
    out: List[Label] = []
    if len(classes) < 2:
        return out
    for rec in records:
        for _ in range(per_record):
            other_cls = rng.choice([c for c in classes if c != (rec.get("class_code") or "GENERIC")])
            other = rng.choice(by_class[other_cls])
            a, b = int(rec["id"]), other
            if a == b:
                continue
            out.append(Label(min(a, b), max(a, b), 0, "easy_negative",
                             "different material class", 0.4))
    return out


# --------------------------------------------------------------- training frame

def build_training_set(
    records: Sequence[dict],
    labels: Sequence[Label],
    cosines: Optional[Dict[Tuple[int, int], float]] = None,
) -> Tuple[List[List[float]], List[int], List[str]]:
    """Turn labelled pairs into (X, y, class_codes) for the scorer and the calibrator."""
    by_id = {int(r["id"]): r for r in records}
    X: List[List[float]] = []
    y: List[int] = []
    classes: List[str] = []
    for lab in labels:
        a, b = by_id.get(lab.a_id), by_id.get(lab.b_id)
        if not a or not b:
            continue
        cos = (cosines or {}).get(lab.key(), 0.0)
        cmp = features.compare_attributes(a, b)
        X.append(features.vector(a, b, cos, cmp))
        y.append(int(lab.label))
        classes.append(a.get("class_code") or "GENERIC")
    return X, y, classes


def summarise(labels: Sequence[Label]) -> dict:
    by_source: Dict[str, int] = defaultdict(int)
    for lab in labels:
        by_source[lab.source] += 1
    return {
        "total": len(labels),
        "positive": len([l for l in labels if l.label == 1]),
        "negative": len([l for l in labels if l.label == 0]),
        "by_source": dict(sorted(by_source.items(), key=lambda kv: -kv[1])),
    }


# ------------------------------------------------------------- active learning

def expected_value_of_information(
    proposal: dict,
    cluster_size: int = 1,
    annual_spend: float = 0.0,
) -> float:
    """Queue rank = uncertainty x cluster size x spend at risk.

    Ordering the queue by uncertainty alone optimises the model. Ordering it by
    uncertainty x money optimises the programme, which is what the steward's time is
    actually being spent on. The top few thousand decisions then cover most of the
    value at risk, rather than the system asking for 240,000 of them.
    """
    score = float(proposal.get("score") or 0.5)
    uncertainty = 1.0 - abs(score - 0.5) * 2.0          # peaks at 0.5, zero at 0/1
    size_factor = 1.0 + (max(cluster_size, 1) - 1) ** 0.5
    spend_factor = 1.0 + (max(annual_spend, 0.0) ** 0.5) / 100.0
    return round(uncertainty * size_factor * spend_factor, 6)
