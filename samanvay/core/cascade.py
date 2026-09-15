"""The nine-stage cascade.

Recall-first funnel, then a precision-first gate. One embedding model over everything
loses at both ends: it misses matches that do not look alike, and it merges things
that do.

  00  ingest & profile          completeness scoring, quarantine, engineered-item exclusion
  01  normalise                 mined abbreviation lexicon, UoM algebra, pack size
  02  extract attributes        class-conditioned, grammar-first, schema-constrained
  03  block + ANN recall        multi-key union HNSW top-k, cross-CPSE only
  04  bi-encoder score          recall-first cutoff
  05  cross-encoder rerank      attribute-serialised pair input
  06  constraint solver + KG    hard conflicts, standards equivalence closure
  07  conformal decision        accept / review / reject with a per-class error bound
  08  correlation clustering    transitivity repair under steward constraints
  09  mint + queue              NMC minting, golden records, value-ranked review queue
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .. import config
from . import (blocking, classify, cluster, conformal, constraints, embed, evidence,
               extract, features, labels as labels_mod, lexicon, nmc as nmc_mod,
               render, scorer as scorer_mod, text, uom)


@dataclass
class StageTiming:
    stage: str
    seconds: float
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"stage": self.stage, "seconds": round(self.seconds, 3), **self.detail}


@dataclass
class CascadeResult:
    canonical: List[dict] = field(default_factory=list)
    proposals: List[dict] = field(default_factory=list)
    clusters: List[List[int]] = field(default_factory=list)
    substitutions: List[dict] = field(default_factory=list)
    quarantined: List[dict] = field(default_factory=list)
    excluded: List[dict] = field(default_factory=list)
    timings: List[StageTiming] = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    scorer: Optional[scorer_mod.LogisticScorer] = None
    calibrator: Optional[conformal.Calibrator] = None
    training: dict = field(default_factory=dict)

    def funnel(self) -> List[dict]:
        return self.stats.get("funnel", [])

    def to_dict(self) -> dict:
        return {
            "canonical": len(self.canonical),
            "proposals": len(self.proposals),
            "clusters": len(self.clusters),
            "substitutions": len(self.substitutions),
            "quarantined": len(self.quarantined),
            "excluded": len(self.excluded),
            "timings": [t.to_dict() for t in self.timings],
            "stats": self.stats,
            "training": self.training,
        }


class _Clock:
    def __init__(self, sink: List[StageTiming]):
        self.sink = sink
        self._t0 = time.perf_counter()

    def mark(self, stage: str, **detail) -> None:
        now = time.perf_counter()
        self.sink.append(StageTiming(stage=stage, seconds=now - self._t0, detail=detail))
        self._t0 = now


# --------------------------------------------------------------- stages 00 - 02

def canonicalise(record: dict, mine_lexicon: bool = False) -> dict:
    """Stages 00-02 for a single source record. Used by the batch cascade and by the
    real-time create-time duplicate check, which is the same code with a corpus of one."""
    description = record.get("description") or ""
    extra = " ".join(str(record.get(k) or "") for k in ("long_text", "mfr_part_no", "spec_text"))

    cls = classify.classify(description, extra)
    completeness = text.completeness(description)
    ex = extract.extract(description, cls.class_code, extra)

    unit_price_base, uom_res = uom.base_unit_price(
        record.get("unit_price"), record.get("uom") or "", description
    )
    attrs = ex.values()

    canonical = {
        "id": record.get("id"),
        "source_id": record.get("id"),
        "cpse_code": record.get("cpse_code"),
        "cpse_name": record.get("cpse_name"),
        "plant": record.get("plant"),
        "matnr": record.get("matnr"),
        "description": description,
        "class_code": cls.class_code,
        "class_confidence": cls.confidence,
        "item_name": cls.item_name,
        "standardisable": cls.standardisable,
        "classification_reason": cls.reason,
        "normalised_text": text.normalise(description),
        "attributes": attrs,
        "numerics": ex.numerics(),
        "extraction": ex.to_dict(),
        "coverage": extract.coverage(ex),
        "completeness": completeness,
        "missing_mandatory": ex.missing_mandatory,
        "residue": ex.residue,
        "golden_description": render.golden_description(cls.class_code, attrs),
        "short_description": render.short_description(cls.class_code, attrs),
        "facets": classify.facets(cls.class_code),
        "uom": record.get("uom"),
        "uom_resolution": uom_res.to_dict(),
        "unit_price": record.get("unit_price"),
        "unit_price_base": unit_price_base,
        "stock_qty": record.get("stock_qty"),
        "annual_demand": record.get("annual_demand"),
        "last_movement_days": record.get("last_movement_days"),
        "mfr_name": record.get("mfr_name"),
        "mfr_part_no": record.get("mfr_part_no"),
        "vendor_id": record.get("vendor_id"),
        "vendor_matl_no": record.get("vendor_matl_no"),
    }
    canonical["blocking_keys"] = blocking.blocking_keys_for(canonical, attrs)
    canonical["identity_keys"] = blocking.identity_keys(canonical)
    return canonical


QUARANTINE_COMPLETENESS = 0.18


def run(
    records: Sequence[dict],
    scorer: Optional[scorer_mod.LogisticScorer] = None,
    calibrator: Optional[conformal.Calibrator] = None,
    train: bool = True,
    mine_lexicon: bool = True,
    serial_start: Optional[int] = None,
    progress: Optional[Callable[[str, dict], None]] = None,
) -> CascadeResult:
    """Run the full cascade over a corpus of source material records."""
    result = CascadeResult()
    clock = _Clock(result.timings)
    emit = progress or (lambda stage, info: None)
    funnel: List[dict] = []

    def step(stage: str, label: str, count, detail: str = "") -> None:
        funnel.append({"stage": stage, "label": label, "count": count, "detail": detail})
        emit(stage, {"label": label, "count": count, "detail": detail})

    n = len(records)
    step("00", "Ingest & profile", n, "completeness scoring, quarantine, engineered exclusion")
    step("--", "Naive pair space", n * (n - 1) // 2, "never computed; shown to justify blocking")

    # ---------------------------------------------------------- stage 01 lexicon
    if mine_lexicon and n >= 8:
        mined = lexicon.mine_and_persist(r.get("description", "") for r in records)
        clock.mark("01a_mine_lexicon", mined_pairs=len(mined))
        step("01", "Normalise", n, f"{len(mined)} abbreviation pairs mined from this corpus")
    else:
        clock.mark("01a_mine_lexicon", mined_pairs=0)
        step("01", "Normalise", n, "seed lexicon only (corpus too small to mine)")

    # ------------------------------------------------------- stages 00 - 02 body
    canonical: List[dict] = []
    for rec in records:
        canon = canonicalise(rec)
        if not canon["standardisable"]:
            result.excluded.append({
                "id": canon["id"], "matnr": canon.get("matnr"),
                "description": canon["description"],
                "reason": canon["classification_reason"] or "engineered to order",
            })
            continue
        if canon["completeness"] < QUARANTINE_COMPLETENESS:
            result.quarantined.append({
                "id": canon["id"], "matnr": canon.get("matnr"),
                "description": canon["description"],
                "completeness": canon["completeness"],
                "reason": "insufficient information to resolve - routed to enrichment, not guessed at",
            })
            continue
        canonical.append(canon)
    result.canonical = canonical
    clock.mark("02_extract", canonical=len(canonical), quarantined=len(result.quarantined),
               excluded=len(result.excluded))
    step("02", "Extract attributes", len(canonical),
         f"{len(result.quarantined)} quarantined, {len(result.excluded)} engineered-to-order excluded")

    if not canonical:
        result.stats = {"funnel": funnel, "records": n}
        return result

    # -------------------------------------------------------------- stage 03/04
    embed.fit_idf(c["normalised_text"] for c in canonical)
    vectors_list = embed.encode_many([c["normalised_text"] for c in canonical])
    vectors: Dict[int, List[float]] = {int(c["id"]): v for c, v in zip(canonical, vectors_list)}
    for c, v in zip(canonical, vectors_list):
        c["embedding"] = v
    clock.mark("03a_embed", backend=embed.backend(), dim=len(vectors_list[0]) if vectors_list else 0)

    candidates, block_stats = blocking.generate(canonical, vectors)
    clock.mark("03b_block", **{k: v for k, v in block_stats.items() if k != "strategy_mix"})
    step("03", "Block + ANN recall", block_stats["candidate_pairs"],
         f"reduction {block_stats['reduction_factor']:,.0f}x · strategies "
         + ", ".join(f"{k}={v}" for k, v in list(block_stats["strategy_mix"].items())[:4]))

    by_id = {int(c["id"]): c for c in canonical}
    cosines: Dict[Tuple[int, int], float] = {}
    survivors: List[blocking.Candidate] = []
    # The cutoff prunes the SEMANTIC tail only. A pair produced by a structural
    # strategy - identity key, attribute tuple, standard + size - always reaches the
    # reranker whatever its text similarity, because catching records whose text
    # shares nothing is the entire reason attribute blocking exists. Applying the
    # cosine cutoff to those pairs throws away exactly the matches the system is for.
    structural = {"mpn", "vmn", "attr", "blk", "dim", "std"}
    dropped_semantic = 0
    for cand in candidates:
        a, b = by_id.get(cand.a_id), by_id.get(cand.b_id)
        if not a or not b:
            continue
        cos = embed.cosine(a["embedding"], b["embedding"])
        cand.bi_score = cos
        cosines[cand.key()] = cos
        is_structural = bool(structural & set(cand.strategies))
        has_identity = bool(set(a["identity_keys"]) & set(b["identity_keys"]))
        if is_structural or has_identity or cos >= config.BI_ENCODER_CUTOFF:
            survivors.append(cand)
        else:
            dropped_semantic += 1
    clock.mark("04_bi_encoder", scored=len(candidates), survivors=len(survivors),
               dropped_semantic_only=dropped_semantic)
    step("04", "Bi-encoder score", len(survivors),
         f"cutoff tau={config.BI_ENCODER_CUTOFF} applied to semantic-only pairs; "
         f"{dropped_semantic:,} dropped, structural pairs always retained")

    # ------------------------------------------------------- distant supervision
    training: dict = {}
    model = scorer or scorer_mod.LogisticScorer()
    cal = calibrator or conformal.Calibrator()

    if train:
        positives = labels_mod.mine_positives(canonical)
        hard_negs = labels_mod.mine_hard_negatives(canonical, survivors, positives)
        easy_negs = labels_mod.mine_easy_negatives(canonical)
        all_labels = list(positives) + list(hard_negs) + list(easy_negs)
        X, y, classes = labels_mod.build_training_set(canonical, all_labels, cosines)
        if X and len(set(y)) > 1:
            fit = model.fit(X, y)
            scores = model.score_many(X)
            cal.fit(list(zip(classes, scores, y)))
            training = {
                "labels": labels_mod.summarise(all_labels),
                "fit": fit,
                "calibration": cal.summary(),
                "weights": model.weight_table(),
            }
        else:
            training = {"labels": labels_mod.summarise(all_labels),
                        "note": "not enough labelled variety to train; priors retained"}
        clock.mark("04b_distant_supervision", **{k: v for k, v in
                                                 (training.get("labels") or {}).items()
                                                 if not isinstance(v, dict)})
    result.scorer = model
    result.calibrator = cal
    result.training = training

    # ------------------------------------------------------------- stages 05-07
    proposals: List[dict] = []
    substitutions: List[dict] = []
    tier_counts = {conformal.AUTO_ACCEPT: 0, conformal.REVIEW: 0, conformal.AUTO_REJECT: 0}
    relation_counts: Dict[str, int] = {}

    cross_pairs = [(by_id[c.a_id], by_id[c.b_id]) for c in survivors]
    cross_override = scorer_mod.cross_scores(cross_pairs) if cross_pairs else None

    for idx, cand in enumerate(survivors):
        a, b = by_id[cand.a_id], by_id[cand.b_id]
        cmp = features.compare_attributes(a, b)
        rel = constraints.solve(a, b, cmp)
        fv = features.vector(a, b, cand.bi_score, cmp)
        score = cross_override[idx] if cross_override else model.score(fv)

        has_graph_path = cmp["kg_equiv"] > 0 or rel.relation == constraints.IDENTICAL
        decision = cal.decide(
            a["class_code"], score, rel.relation,
            has_identity_key=rel.identity_key,
            has_graph_path=has_graph_path,
            blocked=bool(rel.conflicts) and rel.relation == constraints.DISTINCT,
        )

        relation_counts[rel.relation] = relation_counts.get(rel.relation, 0) + 1
        tier_counts[decision.tier] += 1

        annual_spend = float(a.get("annual_demand") or 0) * float(a.get("unit_price_base") or 0) \
            + float(b.get("annual_demand") or 0) * float(b.get("unit_price_base") or 0)
        priority = labels_mod.expected_value_of_information(
            {"score": score}, cluster_size=2, annual_spend=annual_spend
        )

        proposal = {
            "a_id": a["id"], "b_id": b["id"],
            "a_cpse": a["cpse_code"], "b_cpse": b["cpse_code"],
            "class_code": a["class_code"],
            "relation": rel.relation,
            "direction": rel.direction,
            "score": round(float(score), 6),
            "bi_score": round(float(cand.bi_score), 6),
            "tier": decision.tier,
            "strategies": cand.strategies,
            "identity_key": rel.identity_key,
            "proof_strength": rel.proof_strength,
            "blocked_by": rel.blocked_by,
            "priority": priority,
            "annual_spend": round(annual_spend, 2),
            "relation_detail": rel.to_dict(),
            "decision": decision.to_dict(),
            "feature_vector": fv,
            "evidence": evidence.build(
                a, b, score, rel, decision, cmp,
                weights=model.weights, feature_vector=fv,
            ),
        }
        proposal["narrative"] = evidence.narrative(proposal["evidence"])
        proposals.append(proposal)

        if rel.relation == constraints.SUBSTITUTABLE:
            src, dst = (a, b) if rel.direction != "b->a" else (b, a)
            substitutions.append({
                "from_id": src["id"], "to_id": dst["id"],
                "from_cpse": src["cpse_code"], "to_cpse": dst["cpse_code"],
                "class_code": a["class_code"],
                "score": round(float(score), 6),
                "reasons": rel.reasons,
                "caveats": rel.caveats,
                "citations": rel.citations,
            })

    result.proposals = proposals
    result.substitutions = substitutions
    clock.mark("05_07_rerank_solve_decide", pairs=len(proposals),
               backend=scorer_mod.backend(), **tier_counts)
    step("05", "Cross-encoder rerank", len(proposals), f"backend: {scorer_mod.backend()}")
    step("06", "Constraint solver + KG", len(proposals),
         " · ".join(f"{k}={v}" for k, v in sorted(relation_counts.items(), key=lambda kv: -kv[1])))
    step("07", "Conformal decision",
         tier_counts[conformal.AUTO_ACCEPT] + tier_counts[conformal.REVIEW],
         f"auto-accept={tier_counts[conformal.AUTO_ACCEPT]} · review={tier_counts[conformal.REVIEW]}"
         f" · auto-reject={tier_counts[conformal.AUTO_REJECT]}")

    # ---------------------------------------------------------------- stage 08
    edges: List[cluster.Edge] = []
    cannot_link: List[Tuple[int, int]] = []
    for p in proposals:
        if p["relation"] in (constraints.IDENTICAL, constraints.EQUIVALENT) and \
                p["tier"] in (conformal.AUTO_ACCEPT,):
            edges.append(cluster.Edge(int(p["a_id"]), int(p["b_id"]), float(p["score"]), p["relation"]))
        elif p["relation"] == constraints.DISTINCT:
            cannot_link.append((int(p["a_id"]), int(p["b_id"])))

    cl = cluster.correlation_cluster(
        [int(c["id"]) for c in canonical], edges, cannot_link=cannot_link
    )
    baseline = cluster.union_find_baseline([int(c["id"]) for c in canonical], edges)
    result.clusters = cl.clusters
    clock.mark("08_cluster", **cl.to_dict())
    step("08", "Correlation clustering", len(cl.clusters),
         f"largest={cl.max_cluster_size} (union-find baseline would be {baseline.max_cluster_size})")

    # ---------------------------------------------------------------- stage 09
    review_queue = sorted(
        [p for p in proposals if p["tier"] == conformal.REVIEW],
        key=lambda p: -p["priority"],
    )
    step("09", "Steward queue", len(review_queue), "ranked by expected value of information")

    absorbed = len(canonical) - len(cl.clusters)
    result.stats = {
        "funnel": funnel,
        "records_in": n,
        "canonical": len(canonical),
        "quarantined": len(result.quarantined),
        "excluded": len(result.excluded),
        "blocking": block_stats,
        "pairs_scored": len(proposals),
        "tiers": tier_counts,
        "relations": relation_counts,
        "clusters": cl.to_dict(),
        "union_find_largest_cluster": baseline.max_cluster_size,
        "records_absorbed": absorbed,
        "duplicate_rate": round(absorbed / len(canonical), 4) if canonical else 0.0,
        "review_queue": len(review_queue),
        "substitutions": len(substitutions),
        "embedding_backend": embed.backend(),
        "rerank_backend": scorer_mod.backend(),
        "total_seconds": round(sum(t.seconds for t in result.timings), 3),
    }
    return result


def value_at_risk_curve(proposals: Sequence[dict], buckets: int = 20) -> List[dict]:
    """Share of at-risk spend resolved per N steward decisions - the operational ROI
    of the human in the loop, and the answer to "you don't review 240,000 pairs"."""
    queue = sorted(
        [p for p in proposals if p["tier"] == conformal.REVIEW],
        key=lambda p: -p["priority"],
    )
    total = sum(p["annual_spend"] for p in queue) or 1.0
    out: List[dict] = []
    running = 0.0
    if not queue:
        return out
    step = max(1, len(queue) // buckets)
    for i, p in enumerate(queue, 1):
        running += p["annual_spend"]
        if i % step == 0 or i == len(queue):
            out.append({
                "decisions": i,
                "share_of_queue": round(i / len(queue), 4),
                "spend_resolved": round(running, 2),
                "share_of_spend": round(running / total, 4),
            })
    return out
