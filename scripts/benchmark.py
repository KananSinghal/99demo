#!/usr/bin/env python3
"""Benchmark the nine-stage cascade against the synthetic ground truth.

Every number this script prints is measured against `truth_groups` /
`truth_pairs` that scripts/seed.py plants deliberately -- which physical items
were split into records across which CPSEs -- not asserted. This is the
judge-facing evidence behind the claims in the blueprint: blocking recall, the
auto-accept precision/recall slice, cluster purity (the mega-cluster check),
and the six adversarial pairs used throughout the design writeup and the test
suite.

No database and no HTTP server are involved: the corpus is loaded (or
generated) straight into the in-memory cascade, so this runs in a few seconds
and is safe to run live in front of judges.

    python3 scripts/benchmark.py                  # default: 200 groups
    python3 scripts/benchmark.py --groups 400
    python3 scripts/benchmark.py --strict          # non-zero exit on a real regression

Zero dependencies beyond the stdlib and this package, same as the rest of the
build.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from samanvay.core import blocking, cascade, cluster, conformal, constraints, embed  # noqa: E402


# --------------------------------------------------------------------- corpus

def load_corpus(groups: int, seed: int, path: Path) -> dict:
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        print(f"[corpus] loaded {path} ({len(payload['records'])} records, cached)")
        return payload
    sys.path.insert(0, str(ROOT / "scripts"))
    from seed import generate  # type: ignore
    payload = generate(groups, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"[corpus] generated {len(payload['records'])} records -> {path}")
    return payload


# -------------------------------------------------------------------- metrics

def pairwise_from_clusters(clusters: Sequence[Sequence[int]]) -> Set[Tuple[int, int]]:
    pairs: Set[Tuple[int, int]] = set()
    for members in clusters:
        m = sorted(members)
        for i in range(len(m)):
            for j in range(i + 1, len(m)):
                pairs.add((m[i], m[j]))
    return pairs


def prf(predicted: Set[Tuple[int, int]], truth: Set[Tuple[int, int]]) -> dict:
    tp = len(predicted & truth)
    fp = len(predicted - truth)
    fn = len(truth - predicted)
    precision = (tp / (tp + fp)) if (tp + fp) else None
    recall = (tp / (tp + fn)) if (tp + fn) else None
    f1 = (2 * precision * recall / (precision + recall)) if precision and recall else None
    return {
        "precision": round(precision, 4) if precision is not None else None,
        "recall": round(recall, 4) if recall is not None else None,
        "f1": round(f1, 4) if f1 is not None else None,
        "true_positive_pairs": tp, "false_positive_pairs": fp, "false_negative_pairs": fn,
    }


def relation_recall(proposals: List[dict], truth_pairs: List[Tuple[int, int]]) -> dict:
    """Of the true duplicate pairs, what fraction did the solver TYPE correctly
    (IDENTICAL/EQUIVALENT), regardless of which tier a steward ends up in? This
    is the honest 'did we find it' number -- separate from 'did we auto-merge
    it with no human at all', which is the stricter number below."""
    by_pair: Dict[Tuple[int, int], dict] = {}
    for p in proposals:
        a, b = int(p["a_id"]), int(p["b_id"])
        by_pair[(a, b) if a < b else (b, a)] = p

    buckets = {"typed_mergeable": 0, "typed_other_relation": 0, "not_a_candidate": 0}
    tier_of_true_positive = {conformal.AUTO_ACCEPT: 0, conformal.REVIEW: 0, conformal.AUTO_REJECT: 0}
    for a, b in truth_pairs:
        key = (a, b) if a < b else (b, a)
        p = by_pair.get(key)
        if p is None:
            buckets["not_a_candidate"] += 1
            continue
        if p["relation"] in constraints.MERGEABLE:
            buckets["typed_mergeable"] += 1
            tier_of_true_positive[p["tier"]] += 1
        else:
            buckets["typed_other_relation"] += 1
    total = len(truth_pairs) or 1
    return {
        "total_true_pairs": len(truth_pairs),
        "typed_mergeable_pct": round(100 * buckets["typed_mergeable"] / total, 2),
        "typed_other_relation_pct": round(100 * buckets["typed_other_relation"] / total, 2),
        "not_a_candidate_pct": round(100 * buckets["not_a_candidate"] / total, 2),
        "counts": buckets,
        "tier_of_correctly_typed_true_pairs": tier_of_true_positive,
    }


# --------------------------------------------------------------- named cases

ADVERSARIAL_CASES = [
    ("Metric vs unified thread, same nominal diameter",
     "HEX BOLT M12X60 SS316 FULL THD", 'HEX BOLT 1/2"-13UNC X 2" SS316 FULL THD',
     constraints.DISTINCT),
    ("Same bolt, three different house styles",
     "HEX BOLT M12X60 SS316 FULL THD", "BOLT,HEXAGONAL,M12 X 60MM,SS316,ISO4017",
     constraints.IDENTICAL),
    ("Bearing 6205 vs 6206 (one bore-size digit apart)",
     "BEARING 6205-2RS", "BEARING 6206-2RS", constraints.DISTINCT),
    ("Cable armour: SWA vs generic ARMOURED",
     "CABLE XLPE 1.1KV 3C X 95 SQMM CU SWA", "CABLE XLPE 1.1KV 3C X 95 SQMM CU ARMOURED",
     constraints.EQUIVALENT),
    ("Cable armour: SWA vs UNARMOURED",
     "CABLE XLPE 1.1KV 3C X 95 SQMM CU SWA", "CABLE XLPE 1.1KV 3C X 95 SQMM CU UNARMOURED",
     constraints.DISTINCT),
    ("NACE MR0175 stated on only one of two otherwise-identical valves",
     'GATE VALVE 6" 150# WCB FLGD RF HW NACE MR0175', 'GATE VALVE 6" 150# WCB FLGD RF HW',
     constraints.UNDETERMINED),
]


def run_named_cases() -> List[dict]:
    from samanvay.core import features
    from samanvay.core.cascade import canonicalise
    rows = []
    for name, desc_a, desc_b, expected in ADVERSARIAL_CASES:
        a = canonicalise({"id": 1, "cpse_code": "X", "description": desc_a, "uom": "NO", "unit_price": 100.0})
        b = canonicalise({"id": 2, "cpse_code": "Y", "description": desc_b, "uom": "NO", "unit_price": 100.0})
        rel = constraints.solve(a, b, features.compare_attributes(a, b))
        rows.append({
            "case": name, "expected": expected, "actual": rel.relation,
            "pass": rel.relation == expected,
            "reason": rel.reasons[0] if rel.reasons else "",
        })
    return rows


# ------------------------------------------------------------------- report

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--groups", type=int, default=200, help="physical items in the synthetic corpus")
    ap.add_argument("--seed", type=int, default=20260914)
    ap.add_argument("--corpus", default=str(ROOT / "var" / "bench_corpus.json"),
                     help="corpus cache path; deleted (or pass a new path) to force regeneration")
    ap.add_argument("--fresh", action="store_true", help="ignore any cached corpus and regenerate")
    ap.add_argument("--out", default=str(ROOT / "var" / "benchmark.json"))
    ap.add_argument("--strict", action="store_true",
                     help="exit non-zero if a hard correctness invariant is violated")
    args = ap.parse_args()

    corpus_path = Path(args.corpus)
    if args.fresh and corpus_path.exists():
        corpus_path.unlink()

    t0 = time.perf_counter()
    payload = load_corpus(args.groups, args.seed, corpus_path)
    records = payload["records"]
    truth_groups: Dict[str, List[int]] = {k: list(v) for k, v in payload["truth_groups"].items()}
    truth_pairs = [tuple(sorted(p)) for p in payload["truth_pairs"]]
    truth_pair_set = set(truth_pairs)
    truth_label: Dict[int, str] = {rid: gid for gid, members in truth_groups.items() for rid in members}

    print(f"\n{'=' * 72}\nSAMANVAY BENCHMARK -- {len(records)} records, "
          f"{len(truth_groups)} true groups, {len(truth_pairs)} true pairs\n{'=' * 72}")

    # ---- stages 00-03 standalone, to measure blocking recall (the ceiling)
    canonical = []
    for rec in records:
        c = cascade.canonicalise(rec)
        if c["standardisable"] and c["completeness"] >= cascade.QUARANTINE_COMPLETENESS:
            canonical.append(c)
    embed.fit_idf(c["normalised_text"] for c in canonical)
    vecs = embed.encode_many([c["normalised_text"] for c in canonical])
    vectors = {int(c["id"]): v for c, v in zip(canonical, vecs)}
    candidates, block_stats = blocking.generate(canonical, vectors)
    block_recall = blocking.recall(candidates, truth_pairs)

    print(f"\n[1] BLOCKING -- the hard ceiling on everything downstream")
    print(f"    naive pair space   : {block_stats['naive_pairs']:,}")
    print(f"    candidate pairs    : {block_stats['candidate_pairs']:,}  "
          f"(reduction {block_stats['reduction_factor']:,.0f}x)")
    print(f"    blocking recall    : {block_recall['recall']}  "
          f"({block_recall['found']}/{block_recall['truth_pairs']} true pairs reachable)")
    print(f"    strategy mix       : {block_stats['strategy_mix']}")

    # ---- full end-to-end cascade
    result = cascade.run(records, train=True, mine_lexicon=True)
    total_seconds = time.perf_counter() - t0

    print(f"\n[2] FUNNEL")
    for row in result.stats["funnel"]:
        print(f"    {row['stage']:>3} {row['label']:<26} {row['count']:>8,}   {row['detail']}")

    print(f"\n[3] TIERS (all scored pairs, not just true positives)")
    tiers = result.stats["tiers"]
    for k in (conformal.AUTO_ACCEPT, conformal.REVIEW, conformal.AUTO_REJECT):
        print(f"    {k:<12} {tiers.get(k, 0):>8,}")

    # ---- relation recall (tier-agnostic: did we type it right at all?)
    rr = relation_recall(result.proposals, truth_pairs)
    print(f"\n[4] RELATION RECALL against {rr['total_true_pairs']} true duplicate pairs")
    print(f"    typed mergeable (IDENTICAL/EQUIVALENT) : {rr['typed_mergeable_pct']:>6.2f}%  "
          f"of which by tier: {rr['tier_of_correctly_typed_true_pairs']}")
    print(f"    typed some other relation               : {rr['typed_other_relation_pct']:>6.2f}%  "
          f"(label noise / a real attribute genuinely differs)")
    print(f"    never became a candidate (blocking miss): {rr['not_a_candidate_pct']:>6.2f}%")

    # ---- auto-accept pairwise precision/recall/F1 (the fully unattended slice)
    predicted_pairs = pairwise_from_clusters(result.clusters)
    auto_prf = prf(predicted_pairs, truth_pair_set)
    print(f"\n[5] AUTO-ACCEPT PAIRWISE (fully unattended, zero human review)")
    print(f"    precision: {auto_prf['precision']}   recall: {auto_prf['recall']}   f1: {auto_prf['f1']}")
    print(f"    tp={auto_prf['true_positive_pairs']} fp={auto_prf['false_positive_pairs']} "
          f"fn={auto_prf['false_negative_pairs']}")
    if auto_prf["false_positive_pairs"]:
        print(f"    ** {auto_prf['false_positive_pairs']} false-positive auto-merge(s) -- "
              f"see var/benchmark.json for the exact pairs **")

    # ---- cluster purity (the mega-cluster check correlation clustering exists for)
    purity = cluster.purity(cluster.ClusterResult(clusters=result.clusters), truth_label)
    baseline_largest = result.stats["union_find_largest_cluster"]
    print(f"\n[6] CLUSTER PURITY (catches the mega-cluster failure pairwise F1 hides)")
    print(f"    purity: {purity['purity']}   impure clusters: {purity['impure_clusters']}"
          f" / {purity['clusters_scored']}")
    print(f"    largest cluster -- correlation: {result.stats['clusters']['largest_cluster']}"
          f"   vs union-find baseline: {baseline_largest}")

    # ---- quarantine / exclusion sanity against the corpus's own _expect tags
    expected_quarantine = {r["id"] for r in records if r.get("_expect") == "quarantine"}
    expected_excluded = {r["id"] for r in records if r.get("_expect") == "excluded"}
    got_quarantine = {r["id"] for r in result.quarantined}
    got_excluded = {r["id"] for r in result.excluded}
    print(f"\n[7] QUARANTINE / EXCLUSION SANITY")
    print(f"    quarantined as expected: {len(expected_quarantine & got_quarantine)}/{len(expected_quarantine)}")
    print(f"    excluded as expected   : {len(expected_excluded & got_excluded)}/{len(expected_excluded)}")

    # ---- named adversarial cases (the ones in the blueprint / judge Q&A / tests)
    named = run_named_cases()
    print(f"\n[8] NAMED ADVERSARIAL CASES")
    all_named_pass = True
    for row in named:
        mark = "PASS" if row["pass"] else "FAIL"
        all_named_pass = all_named_pass and row["pass"]
        print(f"    [{mark}] {row['case']}")
        print(f"           expected={row['expected']:<12} actual={row['actual']:<12} {row['reason']}")

    print(f"\n[9] TIMING")
    for t in result.timings:
        print(f"    {t.stage:<22} {t.seconds:>7.3f}s")
    print(f"    {'total (incl. blocking pass)':<22} {total_seconds:>7.3f}s")

    # --------------------------------------------------------------- verdict
    hard_invariants = {
        "no_false_positive_auto_merges": auto_prf["false_positive_pairs"] == 0,
        "no_cross_group_contamination": purity["purity"] is None or purity["purity"] >= 0.999,
        "all_named_adversarial_cases_pass": all_named_pass,
        "mega_cluster_capped_below_baseline": result.stats["clusters"]["largest_cluster"] <= max(baseline_largest, 1),
    }
    print(f"\n{'=' * 72}\nHARD INVARIANTS")
    ok = True
    for name, passed in hard_invariants.items():
        ok = ok and passed
        print(f"    [{'OK' if passed else 'BROKEN'}] {name}")
    print(f"{'=' * 72}\n")

    report = {
        "corpus": {"records": len(records), "true_groups": len(truth_groups), "true_pairs": len(truth_pairs)},
        "blocking": {**block_stats, "recall": block_recall},
        "funnel": result.stats["funnel"],
        "tiers": tiers,
        "relation_recall": rr,
        "auto_accept_pairwise": auto_prf,
        "cluster_purity": purity,
        "union_find_largest_cluster": baseline_largest,
        "quarantine_sanity": {"expected": len(expected_quarantine), "matched": len(expected_quarantine & got_quarantine)},
        "exclusion_sanity": {"expected": len(expected_excluded), "matched": len(expected_excluded & got_excluded)},
        "named_adversarial_cases": named,
        "timing_seconds": {t.stage: round(t.seconds, 4) for t in result.timings},
        "total_seconds": round(total_seconds, 3),
        "hard_invariants": hard_invariants,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"full report written to {out_path}")

    if args.strict and not ok:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
