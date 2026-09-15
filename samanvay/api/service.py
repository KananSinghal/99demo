"""The service layer: cascade + persistence + governance workflow.

The governance rules are enforced HERE, not in the UI, because a rule a client can
skip is not a rule:

  * the proposal engine can only create PROPOSED events - it cannot endorse, merge,
    mint, or write to any ERP
  * a cross-CPSE merge requires an endorsement from a named steward in EACH
    organisation (dual key). No CPSE's master data is changed on another CPSE's
    authority, or on an algorithm's
  * every state change is an append-only ledger event with a named actor and a reason
  * a merge adds a LINK; the source record is never rewritten, which is why un-merge
    is a replay rather than a recovery
"""

from __future__ import annotations

import json
import sqlite3
import threading
from typing import Dict, List, Optional, Sequence, Tuple

from .. import config
from ..core import (cascade, conformal, constraints, economics, evidence, extract,
                    features, federation, kg, labels as labels_mod, nmc as nmc_mod,
                    render, scorer as scorer_mod, sectors, uom)
from ..store import Repo, connect, init_db

_LOCAL = threading.local()
_RUN_LOCK = threading.Lock()


def repo() -> Repo:
    """One connection per thread. sqlite3 connections are not thread-safe."""
    conn = getattr(_LOCAL, "conn", None)
    if conn is None:
        conn = connect()
        init_db(conn)
        _LOCAL.conn = conn
    return Repo(conn)


# ============================================================== pipeline

def run_pipeline(train: bool = True, mine_lexicon: bool = True) -> dict:
    """Run the full cascade over everything ingested, and persist the result."""
    with _RUN_LOCK:
        r = repo()
        sources = r.source_records()
        if not sources:
            return {"error": "nothing ingested yet - POST /api/ingest first"}

        model = scorer_mod.LogisticScorer.load()
        cal = conformal.Calibrator.load()
        result = cascade.run(sources, scorer=model, calibrator=cal,
                             train=train, mine_lexicon=mine_lexicon)

        if result.scorer:
            result.scorer.save()
        if result.calibrator:
            result.calibrator.save()

        # Persist canonical records, keeping a source_id -> canonical_id map so the
        # proposals (which reference source ids) land on the right rows.
        canonical_by_source: Dict[int, int] = {}
        for canon in result.canonical:
            cid = r.save_canonical(canon, status="active")
            canonical_by_source[int(canon["id"])] = cid
        for q in result.quarantined:
            src = next((s for s in sources if s["id"] == q["id"]), None)
            if src:
                canon = cascade.canonicalise(src)
                canonical_by_source[int(q["id"])] = r.save_canonical(
                    canon, status="quarantined", reason=q["reason"])
        for e in result.excluded:
            src = next((s for s in sources if s["id"] == e["id"]), None)
            if src:
                canon = cascade.canonicalise(src)
                canonical_by_source[int(e["id"])] = r.save_canonical(
                    canon, status="excluded", reason=e["reason"])

        for p in result.proposals:
            r.save_proposal(p, canonical_by_source)
        for s in result.substitutions:
            r.save_substitution(s, canonical_by_source)

        # Auto-accept tier: mint and link. Still fully reversible - see unmerge().
        minted = _materialise_clusters(r, result, canonical_by_source)

        run_id = r.record_run(len(sources), result.stats, result.training, result.stats["funnel"])
        return {
            "run_id": run_id,
            "funnel": result.stats["funnel"],
            "stats": result.stats,
            "training": result.training,
            "minted": minted,
            "timings": [t.to_dict() for t in result.timings],
        }


def _materialise_clusters(r: Repo, result, canonical_by_source: Dict[int, int]) -> dict:
    """Turn auto-accepted clusters into NMCs. Every CPSE material gets an NMC, even a
    singleton - the registry is a national catalogue, not a duplicate list."""
    minted, linked = 0, 0
    by_source = {int(c["id"]): c for c in result.canonical}
    for members in result.clusters:
        canon_ids = [canonical_by_source[m] for m in members if m in canonical_by_source]
        if not canon_ids:
            continue
        existing = r.nmc_for_canonical(canon_ids[0])
        if existing:
            r.link_members(existing["id"], canon_ids, actor="engine",
                           reason="cluster membership refreshed by the cascade")
            linked += len(canon_ids)
            continue
        # The cluster's golden record is the member with the most complete attributes.
        best = max((by_source[m] for m in members if m in by_source),
                   key=lambda c: (c.get("coverage") or 0, c.get("completeness") or 0),
                   default=None)
        if best is None:
            continue
        rec = r.mint_nmc(best["class_code"], best["golden_description"],
                         best["attributes"], actor="registrar")
        r.link_members(rec["id"], canon_ids, actor="engine",
                       reason="auto-accepted cluster, within the class error bound")
        minted += 1
        linked += len(canon_ids)
    return {"nmc_minted": minted, "members_linked": linked}


# ============================================================== governance

def _proposal_or_error(r: Repo, proposal_id: int) -> Tuple[Optional[dict], Optional[dict]]:
    p = r.proposal(proposal_id)
    if not p:
        return None, {"error": f"proposal {proposal_id} not found"}
    return p, None


def endorse(proposal_id: int, actor: str, actor_cpse: str, reason: str = "",
            actor_role: str = "steward") -> dict:
    """A steward endorses a proposal FOR THEIR OWN CPSE.

    Dual-key: a merge that touches two CPSEs' codes requires an endorsement from a
    steward in each. This is the organisational keystone - it is what makes the
    system adoptable across organisations that do not otherwise share data.
    """
    r = repo()
    p, err = _proposal_or_error(r, proposal_id)
    if err:
        return err
    a = r.canonical(int(p["a_id"]))
    b = r.canonical(int(p["b_id"]))
    if not a or not b:
        return {"error": "proposal references a record that no longer exists"}

    owning = {a["cpse_code"], b["cpse_code"]}
    if actor_cpse and actor_cpse not in owning and actor_role != "national_registrar":
        return {
            "error": "a steward can only endorse proposals affecting their own CPSE's codes",
            "your_cpse": actor_cpse, "proposal_cpses": sorted(owning),
        }

    endorsements = list(p.get("endorsements") or [])
    if any(e.get("cpse") == actor_cpse for e in endorsements):
        return {"error": f"{actor_cpse} has already endorsed this proposal",
                "endorsements": endorsements}

    endorsements.append({"actor": actor, "cpse": actor_cpse, "role": actor_role, "reason": reason})
    r.conn.execute("UPDATE proposal SET endorsements_json = ? WHERE id = ?",
                   (json.dumps(endorsements), proposal_id))
    r.record_decision(proposal_id, actor, "endorse", reason,
                      a_id=int(p["a_id"]), b_id=int(p["b_id"]),
                      actor_cpse=actor_cpse, actor_role=actor_role)
    r.ledger.append("ENDORSED",
                    {"proposal": proposal_id, "a": p["a_id"], "b": p["b_id"],
                     "relation": p["relation"], "cpse": actor_cpse},
                    actor=actor, actor_cpse=actor_cpse, actor_role=actor_role,
                    subject=f"proposal:{proposal_id}", reason=reason or "steward endorsement")
    r.add_constraint(int(p["a_id"]), int(p["b_id"]), "must", actor, reason)

    endorsed_cpses = {e.get("cpse") for e in endorsements if e.get("cpse")}
    needed = owning - endorsed_cpses
    if needed:
        r.conn.execute("UPDATE proposal SET status='endorsed' WHERE id=?", (proposal_id,))
        return {
            "status": "awaiting_second_endorsement",
            "endorsements": endorsements,
            "awaiting_from": sorted(needed),
            "rule": "a cross-CPSE merge requires an endorsement from a steward in each organisation",
        }

    merged = _merge(r, p, a, b, actor)
    return {"status": "merged", "endorsements": endorsements, **merged}


def _merge(r: Repo, p: dict, a: dict, b: dict, actor: str) -> dict:
    existing = r.nmc_for_canonical(int(p["a_id"])) or r.nmc_for_canonical(int(p["b_id"]))
    if existing:
        rec_id = int(existing["id"])
        code = existing["code"]
    else:
        best = a if (a.get("coverage") or 0) >= (b.get("coverage") or 0) else b
        rec = r.mint_nmc(best["class_code"], best["golden_description"],
                         best["attributes"], actor=actor)
        rec_id, code = rec["id"], rec["code"]
    role = "variant" if p["relation"] == constraints.VARIANT_OF else "member"
    r.link_members(rec_id, [int(p["a_id"]), int(p["b_id"])], actor=actor, role=role,
                   reason=f"dual endorsement reached for a {p['relation']} relation")
    r.conn.execute("UPDATE proposal SET status='merged' WHERE id=?", (p["id"],))
    crs = _queue_writebacks(r, code, [a, b], p["id"], actor)
    return {"nmc": code, "members": [int(p["a_id"]), int(p["b_id"])],
            "change_requests": crs,
            "note": "the CPSE material numbers were not changed; the NMC is an additive cross-reference"}


def reject(proposal_id: int, actor: str, actor_cpse: str = "", reason: str = "",
           distinct: bool = False) -> dict:
    r = repo()
    p, err = _proposal_or_error(r, proposal_id)
    if err:
        return err
    decision = "distinct" if distinct else "reject"
    r.conn.execute("UPDATE proposal SET status='rejected' WHERE id=?", (proposal_id,))
    r.record_decision(proposal_id, actor, decision, reason,
                      a_id=int(p["a_id"]), b_id=int(p["b_id"]), actor_cpse=actor_cpse)
    if distinct:
        r.add_constraint(int(p["a_id"]), int(p["b_id"]), "cannot", actor, reason)
    r.ledger.append("REJECTED",
                    {"proposal": proposal_id, "a": p["a_id"], "b": p["b_id"],
                     "relation": p["relation"], "marked_distinct": bool(distinct)},
                    actor=actor, actor_cpse=actor_cpse, subject=f"proposal:{proposal_id}",
                    reason=reason or "steward rejection")
    return {"status": "rejected", "marked_distinct": bool(distinct),
            "note": ("recorded as DISTINCT and never proposed again"
                     if distinct else "rejected for this run")}


def unmerge(nmc_code: str, canonical_ids: Sequence[int], actor: str, reason: str) -> dict:
    """The reversibility proof. Merge, un-merge, and the projection is byte-identical
    to the pre-merge state - because the source records were never rewritten."""
    r = repo()
    before = r.ledger.replay()["membership"].get(nmc_code, [])
    out = r.unlink_members(nmc_code, canonical_ids, actor, reason)
    if "error" in out:
        return out
    after = r.ledger.replay()["membership"].get(nmc_code, [])
    verify = r.ledger.verify()
    return {
        **out,
        "membership_before": before,
        "membership_after": after,
        "chain_valid": verify["valid"],
        "chain_statement": verify["statement"],
        "projection_matches_tables": r.ledger.projection_matches_tables()["matches"],
    }


def approve_substitution(sub_id: int, actor: str, reason: str = "") -> dict:
    """A directed substitution is an engineering judgment, not a model output. The
    system proposes and evidences it; an engineer approves it."""
    r = repo()
    row = r.conn.execute("SELECT * FROM substitution WHERE id=?", (sub_id,)).fetchone()
    if not row:
        return {"error": f"substitution {sub_id} not found"}
    r.conn.execute(
        "UPDATE substitution SET status='approved', approved_by=?, approved_at=datetime('now') WHERE id=?",
        (actor, sub_id),
    )
    r.ledger.append("SUBSTITUTION_APPROVED",
                    {"substitution": sub_id, "from": row["from_id"], "to": row["to_id"]},
                    actor=actor, actor_role="domain_engineer",
                    subject=f"substitution:{sub_id}",
                    reason=reason or "directional substitution approved by a domain engineer")
    return {"status": "approved", "substitution": sub_id}


def _queue_writebacks(r: Repo, nmc_code: str, records: Sequence[dict],
                      proposal_id: Optional[int], actor: str) -> List[int]:
    from ..connectors.sap import build_change_request
    out = []
    for rec in records:
        payload = build_change_request(nmc_code, rec)
        out.append(r.queue_change_request(
            nmc_code, rec["cpse_code"], rec["matnr"], "mdg_change_request",
            payload, proposal_id=proposal_id, actor=actor,
        ))
    return out


# ============================================================== create-time check

def duplicate_check(record: dict, top_k: int = 5) -> dict:
    """The sub-300ms check a CPSE wires into its material-creation path.

    Cleaning without prevention means the mess returns within three years. This is
    the same cascade with a corpus of one, which is why it is the cheapest thing in
    the system to build and the one that stops the problem recurring.
    """
    import time
    t0 = time.perf_counter()
    r = repo()
    record = dict(record)
    record.setdefault("id", -1)
    canon = cascade.canonicalise(record)

    if not canon["standardisable"]:
        return {"verdict": "not_standardisable", "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2),
                "reason": canon["classification_reason"],
                "guidance": "engineered-to-order items are excluded by design, not harmonised"}

    pool = r.canonical_all("active")
    if not pool:
        return {"verdict": "no_registry", "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2),
                "matches": []}

    from ..core import blocking, embed
    embed.fit_idf([c["normalised_text"] for c in pool] + [canon["normalised_text"]])
    qvec = embed.encode(canon["normalised_text"])
    keys = set(canon["blocking_keys"])

    scored: List[Tuple[float, dict]] = []
    for other in pool:
        shares_key = bool(keys & set(other.get("blocking_keys") or []))
        vec = other.get("embedding") or []
        cos = embed.cosine(qvec, vec) if vec else 0.0
        if not shares_key and cos < config.BI_ENCODER_CUTOFF:
            continue
        scored.append((cos, other))
    scored.sort(key=lambda t: -t[0])

    model = scorer_mod.LogisticScorer.load()
    cal = conformal.Calibrator.load()
    matches: List[dict] = []
    for cos, other in scored[:40]:
        cmp = features.compare_attributes(canon, other)
        rel = constraints.solve(canon, other, cmp)
        if rel.relation == constraints.DISTINCT:
            continue
        fv = features.vector(canon, other, cos, cmp)
        score = model.score(fv)
        decision = cal.decide(canon["class_code"], score, rel.relation,
                              has_identity_key=rel.identity_key,
                              has_graph_path=cmp["kg_equiv"] > 0)
        nmc_rec = r.nmc_for_canonical(int(other["id"]))
        matches.append({
            "canonical_id": other["id"],
            "nmc": nmc_rec["code"] if nmc_rec else None,
            "cpse_code": other["cpse_code"],
            "plant": other.get("plant"),
            "matnr": other["matnr"],
            "description": other["description"],
            "golden_description": other["golden_description"],
            "stock_qty": other.get("stock_qty"),
            "relation": rel.relation,
            "direction": rel.direction,
            "score": round(score, 4),
            "tier": decision.tier,
            "reasons": rel.reasons,
            "evidence": evidence.build(canon, other, score, rel, decision, cmp),
        })
        if len(matches) >= top_k:
            break

    matches.sort(key=lambda m: -m["score"])
    top = matches[0] if matches else None
    elapsed = round((time.perf_counter() - t0) * 1000, 2)

    if top and top["score"] >= 0.75:
        message = (
            f"This is {top['score']*100:.0f}% likely to be "
            f"{top['nmc'] or 'an existing item'} - already stocked as material "
            f"{top['matnr']} at {top['cpse_code']} {top.get('plant') or ''}"
            + (f", {top['stock_qty']:g} on hand" if top.get("stock_qty") else "")
        )
        verdict = "likely_duplicate"
    else:
        message = "No close match in the national registry - this looks like a new item."
        verdict = "likely_new"

    return {
        "verdict": verdict,
        "message": message.strip(),
        "elapsed_ms": elapsed,
        "canonical": {
            "class_code": canon["class_code"],
            "attributes": canon["attributes"],
            "golden_description": canon["golden_description"],
            "missing_mandatory": canon["missing_mandatory"],
            "completeness": canon["completeness"],
        },
        "options": [
            "use the existing code",
            "create a variant under the same national code",
            "create new with a recorded reason",
        ],
        "matches": matches,
    }


# ============================================================== analytics

def analytics() -> dict:
    r = repo()
    counts = r.counts()
    run = r.latest_run() or {}
    stats = run.get("stats") or {}

    by_class = [dict(x) for x in r.conn.execute(
        """SELECT class_code, COUNT(*) AS records,
                  ROUND(AVG(coverage), 4) AS avg_coverage,
                  ROUND(AVG(completeness), 4) AS avg_completeness
           FROM canonical_record WHERE status='active'
           GROUP BY class_code ORDER BY records DESC"""
    ).fetchall()]

    dupes = [dict(x) for x in r.conn.execute(
        """SELECT c.class_code,
                  COUNT(*) AS members,
                  COUNT(DISTINCT m.nmc_id) AS codes
           FROM nmc_member m JOIN canonical_record c ON c.id = m.canonical_id
           WHERE m.active = 1 GROUP BY c.class_code ORDER BY members DESC"""
    ).fetchall()]
    for row in dupes:
        row["duplicate_rate"] = round(1 - (row["codes"] / row["members"]), 4) if row["members"] else 0.0

    by_cpse = [dict(x) for x in r.conn.execute(
        """SELECT s.cpse_code, COUNT(*) AS materials,
                  SUM(CASE WHEN c.status='active' THEN 1 ELSE 0 END) AS active,
                  SUM(CASE WHEN c.status='quarantined' THEN 1 ELSE 0 END) AS quarantined
           FROM source_material s LEFT JOIN canonical_record c ON c.source_id = s.id
           GROUP BY s.cpse_code ORDER BY materials DESC"""
    ).fetchall()]
    for row in by_cpse:
        row["sector"] = sectors.sector_of(row["cpse_code"])

    uom_anomalies = _uom_anomalies(r)
    curve = _value_curve(r)
    learning = _learning_curve(r)

    return {
        "counts": counts,
        "funnel": run.get("funnel") or [],
        "run_stats": stats,
        "by_class": by_class,
        "duplicates_by_class": dupes,
        "by_cpse": by_cpse,
        "cross_sector": cross_sector(r.nmc_groups(multi_cpse_only=True), by_cpse),
        "uom_anomalies": uom_anomalies,
        "value_curve": curve,
        "learning_curve": learning,
        "knowledge_graph": kg.stats(),
        "backends": {"embedding": stats.get("embedding_backend"),
                     "rerank": stats.get("rerank_backend")},
    }


def cross_sector(groups: Sequence[dict], by_cpse: Sequence[dict]) -> dict:
    """National codes whose members span more than one sector - the refinery bearing
    that a power plant already holds. Pure function over nmc_groups output."""
    per_sector: Dict[str, dict] = {}
    for row in by_cpse:
        s = per_sector.setdefault(row["sector"], {"sector": row["sector"], "cpses": [],
                                                  "materials": 0, "shared_codes": 0})
        s["cpses"].append(row["cpse_code"])
        s["materials"] += row.get("materials") or 0

    pairs: Dict[Tuple[str, str], int] = {}
    codes, spend = 0, 0.0
    for g in groups:
        spanned = sectors.sectors_of(m["cpse_code"] for m in g["members"])
        if len(spanned) < 2:
            continue
        codes += 1
        spend += sum(float(m.get("annual_demand") or 0) * float(m.get("unit_price") or 0)
                     for m in g["members"])
        for i, a in enumerate(spanned):
            if a in per_sector:
                per_sector[a]["shared_codes"] += 1
            for b in spanned[i + 1:]:
                pairs[(a, b)] = pairs.get((a, b), 0) + 1

    return {
        "cross_sector_codes": codes,
        "cross_sector_spend": round(spend, 2),
        "by_sector": sorted(per_sector.values(), key=lambda s: -s["materials"]),
        "pairs": [{"sectors": list(k), "codes": v}
                  for k, v in sorted(pairs.items(), key=lambda kv: -kv[1])],
    }


def _uom_anomalies(r: Repo, limit: int = 25) -> List[dict]:
    """Where two CPSEs appear to pay wildly different prices but do not - because one
    buys in metres and the other in drums. Every spend chart built on raw prices is
    wrong until this is reconciled."""
    out: List[dict] = []
    for group in r.nmc_groups(multi_cpse_only=True):
        members = [m for m in group["members"] if m.get("unit_price")]
        if len(members) < 2:
            continue
        members.sort(key=lambda m: m.get("unit_price") or 0)
        lo, hi = members[0], members[-1]
        report = uom.anomaly_ratio(
            lo.get("unit_price") or 0, lo.get("uom") or "", lo.get("description") or "",
            hi.get("unit_price") or 0, hi.get("uom") or "", hi.get("description") or "",
        )
        if report["raw_ratio"] and report["raw_ratio"] > 2.5:
            out.append({
                "nmc": group["nmc"],
                "golden_description": group["golden_description"],
                "low": {"cpse": lo["cpse_code"], "uom": lo.get("uom"), "price": lo.get("unit_price")},
                "high": {"cpse": hi["cpse_code"], "uom": hi.get("uom"), "price": hi.get("unit_price")},
                **report,
            })
    out.sort(key=lambda x: -(x["raw_ratio"] or 0))
    return out[:limit]


def _value_curve(r: Repo, buckets: int = 20) -> List[dict]:
    rows = r.conn.execute(
        """SELECT priority, annual_spend FROM proposal
           WHERE status='open' AND tier='review' ORDER BY priority DESC"""
    ).fetchall()
    total = sum((x["annual_spend"] or 0) for x in rows) or 1.0
    out, running = [], 0.0
    if not rows:
        return out
    step = max(1, len(rows) // buckets)
    for i, row in enumerate(rows, 1):
        running += row["annual_spend"] or 0
        if i % step == 0 or i == len(rows):
            out.append({"decisions": i, "share_of_queue": round(i / len(rows), 4),
                        "spend_resolved": round(running, 2),
                        "share_of_spend": round(running / total, 4)})
    return out


def _learning_curve(r: Repo) -> dict:
    model = scorer_mod.LogisticScorer.load()
    return {
        "trained_on": model.trained_on,
        "epochs": model.epochs,
        "history": model.history[-12:],
        "weights": model.weight_table(),
        "note": (
            "every steward decision is a training label; the queue is ranked by "
            "uncertainty x cluster size x spend, so the human hours go where the money is"
        ),
    }


def value_model() -> dict:
    r = repo()
    groups = r.nmc_groups(multi_cpse_only=True)
    counts = r.counts()
    duplicate_codes = max(counts["nmc_members"] - counts["nmc_active"], 0)
    return economics.model(groups, duplicate_codes)


# ============================================================== buy or borrow

def buy_or_borrow(query: str = "", nmc_code: str = "", qty: float = 0,
                  requesting_cpse: str = "", min_idle_days: Optional[int] = None) -> dict:
    """The closer.

    A buyer enters a requirement. The system resolves it to an NMC, follows the
    DIRECTED substitutability edges, and answers: raise a transfer, not a tender.

    Symmetric deduplication saves code count. Directed substitutability satisfies a
    live requisition from another CPSE's non-moving stock. One is a data-quality
    metric; the other is money moving.
    """
    r = repo()
    min_idle_days = config.NON_MOVING_DAYS if min_idle_days is None else min_idle_days

    resolved = None
    considered: List[dict] = []
    if nmc_code:
        resolved = r.nmc_by_code(nmc_code)
    elif query:
        check = duplicate_check({"description": query, "cpse_code": requesting_cpse,
                                 "matnr": "REQUISITION", "uom": "NO"}, top_k=8)
        # Rank the candidate codes by whether they can actually answer the question.
        # Resolving to the closest-looking singleton code and reporting "no stock" is
        # technically correct and operationally useless.
        for match in check.get("matches", []):
            if not match.get("nmc"):
                continue
            cand = r.nmc_by_code(match["nmc"])
            if not cand:
                continue
            holdable = sum(
                1 for m in cand["members"]
                if (m.get("stock_qty") or 0) > 0
                and m["cpse_code"] != requesting_cpse
                and (m.get("last_movement_days") or 0) >= min_idle_days
            )
            considered.append({"nmc": cand["code"], "score": match["score"],
                               "cpses": len({m["cpse_code"] for m in cand["members"]}),
                               "holders": holdable})
            cand["_rank"] = (holdable > 0, len({m["cpse_code"] for m in cand["members"]}),
                             match["score"])
            if resolved is None or cand["_rank"] > resolved["_rank"]:
                resolved = cand
        if not resolved:
            return {"resolved": None, "requirement": query,
                    "message": "could not resolve the requirement to a national material code",
                    "candidates": check.get("matches", [])[:5]}
    if not resolved:
        return {"error": "provide either a description query or an nmc code"}

    direct = [
        m for m in resolved["members"]
        if (m.get("stock_qty") or 0) > 0
        and m["cpse_code"] != requesting_cpse
        and (m.get("last_movement_days") or 0) >= min_idle_days
    ]
    offers: List[dict] = []
    for m in direct:
        offers.append({
            "relation": "IDENTICAL/EQUIVALENT", "direction": "",
            "cpse_code": m["cpse_code"], "plant": m.get("plant"), "matnr": m["matnr"],
            "description": m["description"], "stock_qty": m.get("stock_qty"),
            "idle_days": m.get("last_movement_days"),
            "unit_price_base": m.get("unit_price_base"),
            "value": round((m.get("stock_qty") or 0) * (m.get("unit_price_base") or 0), 2),
            "caveats": [],
            "citations": [],
        })

    # Directed edges: something that is not the same item but SATISFIES the duty.
    member_ids = [m["canonical_id"] for m in resolved["members"]]
    if member_ids:
        marks = ",".join("?" for _ in member_ids)
        rows = r.conn.execute(
            f"""SELECT sub.*, s.cpse_code, s.plant, s.matnr, s.description, s.stock_qty,
                       s.last_movement_days, c.unit_price_base, c.golden_description
                FROM substitution sub
                JOIN canonical_record c ON c.id = sub.from_id
                JOIN source_material s ON s.id = c.source_id
                WHERE sub.to_id IN ({marks}) AND s.stock_qty > 0""",
            member_ids,
        ).fetchall()
        for row in rows:
            if row["cpse_code"] == requesting_cpse:
                continue
            if (row["last_movement_days"] or 0) < min_idle_days:
                continue
            offers.append({
                "relation": "SUBSTITUTABLE", "direction": "offered -> required",
                "cpse_code": row["cpse_code"], "plant": row["plant"], "matnr": row["matnr"],
                "description": row["description"], "stock_qty": row["stock_qty"],
                "idle_days": row["last_movement_days"],
                "unit_price_base": row["unit_price_base"],
                "value": round((row["stock_qty"] or 0) * (row["unit_price_base"] or 0), 2),
                "caveats": json.loads(row["caveats"] or "[]"),
                "citations": json.loads(row["citations"] or "[]"),
                "substitution_id": row["id"],
                "status": row["status"],
            })

    offers.sort(key=lambda o: (-(o["stock_qty"] or 0), o["relation"] != "IDENTICAL/EQUIVALENT"))
    available = sum(o["stock_qty"] or 0 for o in offers)
    covered = min(available, qty) if qty else available
    prices = [m.get("unit_price_base") for m in resolved["members"] if m.get("unit_price_base")]
    avoided = round(covered * (sum(prices) / len(prices)), 2) if prices and covered else 0.0

    if offers:
        top = offers[0]
        verdict = "borrow"
        message = (
            f"{top['stock_qty']:g} units of "
            f"{'an accepted substitute' if top['relation']=='SUBSTITUTABLE' else 'this item'} "
            f"are sitting as non-moving stock at {top['cpse_code']} {top.get('plant') or ''} "
            f"({top['idle_days']} days idle). Raise a transfer instead of a tender."
        )
    else:
        verdict = "buy"
        message = "No non-moving stock found at another CPSE against this code. Proceed to tender."

    return {
        "verdict": verdict,
        "message": " ".join(message.split()),
        "nmc": resolved["code"],
        "golden_description": resolved["golden_description"],
        "qty_required": qty,
        "qty_available": available,
        "qty_covered": covered,
        "procurement_avoided": avoided,
        "offers": offers,
        "candidates_considered": considered[:8],
        "disclosure_note": (
            "in a federated deployment the holders are notified and choose to respond - "
            "a stock position is disclosed by an act, never by default"
        ),
    }


# ============================================================== misc reads

def federation_report() -> dict:
    r = repo()
    sample = r.canonical_all("active")[:1]
    report = federation.boundary_report()
    if sample:
        report["example_published_payload"] = federation.publishable(sample[0], salt="demo-salt")
    return report


def system_info() -> dict:
    r = repo()
    return {
        "version": "1.0.0",
        "config": config.as_dict(),
        "counts": r.counts(),
        "cpses": r.cpses(),
        "knowledge_graph": kg.stats(),
        "nmc_scheme": nmc_mod.describe(),
        "relations": constraints.relation_summary(),
        "classes": [
            {"class_code": c,
             "item_name": __import__("samanvay.core.classify", fromlist=["x"]).class_spec(c)["item_name"]}
            for c in __import__("samanvay.core.classify", fromlist=["x"]).all_classes()
        ],
        "ledger": {"events": r.ledger.count(), "head": r.ledger.head()},
    }
