"""Request handlers. Each takes (params, body) and returns a dict."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .. import __version__, config
from ..core import classify, extract, kg, nmc as nmc_mod, uom
from . import service


def _int(params: dict, key: str, default: int = 0) -> int:
    try:
        return int(params.get(key, default))
    except (TypeError, ValueError):
        return default


def _float(params: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(params.get(key, default))
    except (TypeError, ValueError):
        return default


def _bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).lower() in ("1", "true", "yes", "on")


def _actor(body: dict, params: dict) -> tuple:
    actor = body.get("actor") or params.get("actor") or "steward"
    cpse = body.get("actor_cpse") or params.get("actor_cpse") or ""
    role = body.get("actor_role") or params.get("actor_role") or "steward"
    return actor, cpse, role


# ------------------------------------------------------------------ system

def health(params: dict, body: dict) -> dict:
    return {"status": "ok", "version": __version__, "node_role": config.NODE_ROLE}


def info(params: dict, body: dict) -> dict:
    return service.system_info()


def cpses(params: dict, body: dict) -> dict:
    return {"cpses": service.repo().cpses()}


def routes(params: dict, body: dict) -> dict:
    from .router import ROUTES
    return {"routes": [{"method": m, "path": p, "summary": s} for m, p, _h, s in ROUTES]}


# ------------------------------------------------------------------ ingestion

def ingest(params: dict, body: dict) -> dict:
    records = body.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("body must contain a non-empty 'records' array")
    actor = body.get("actor") or "connector"
    return service.repo().ingest(records, actor=actor,
                                 source_system=body.get("source_system") or "api")


def ingest_demo(params: dict, body: dict) -> dict:
    """Load the synthetic corpus. Generates it if var/corpus.json is absent."""
    path = Path(body.get("path") or params.get("path") or (config.VAR_DIR / "corpus.json"))
    if not path.exists():
        import sys
        sys.path.insert(0, str(config.ROOT / "scripts"))
        from seed import generate  # type: ignore
        payload = generate(int(body.get("groups") or 200))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    corpus = json.loads(path.read_text(encoding="utf-8"))
    out = service.repo().ingest(corpus["records"], actor="demo-loader", source_system="synthetic")
    out["truth_pairs"] = len(corpus.get("truth_pairs") or [])
    out["note"] = corpus.get("_note")
    return out


def pipeline_run(params: dict, body: dict) -> dict:
    return service.run_pipeline(
        train=_bool(body.get("train"), True),
        mine_lexicon=_bool(body.get("mine_lexicon"), True),
    )


def pipeline_latest(params: dict, body: dict) -> dict:
    run = service.repo().latest_run()
    return run or {"error": "no pipeline run recorded yet"}


# ------------------------------------------------------------------ proposals

def proposals(params: dict, body: dict) -> dict:
    r = service.repo()
    items, total = r.proposals(
        tier=params.get("tier") or "review",
        status=params.get("status") or "open",
        relation=params.get("relation") or None,
        class_code=params.get("class_code") or None,
        cpse=params.get("cpse") or None,
        limit=min(_int(params, "limit", 25), 200),
        offset=_int(params, "offset", 0),
        order=params.get("order") or "priority",
    )
    for item in items:
        item.pop("feature_vector", None)
    return {"total": total, "items": items,
            "note": "ranked by expected value of information: uncertainty x cluster size x spend"}


def proposal_detail(params: dict, body: dict) -> dict:
    p = service.repo().proposal(_int(params, "id"))
    return p or {"error": "proposal not found"}


def proposal_endorse(params: dict, body: dict) -> dict:
    actor, cpse, role = _actor(body, params)
    if not cpse:
        raise ValueError("actor_cpse is required - a steward endorses for their own organisation")
    return service.endorse(_int(params, "id"), actor, cpse, body.get("reason", ""), role)


def proposal_reject(params: dict, body: dict) -> dict:
    actor, cpse, _role = _actor(body, params)
    return service.reject(_int(params, "id"), actor, cpse, body.get("reason", ""),
                          distinct=_bool(body.get("distinct"), False))


# ------------------------------------------------------------------ registry

def catalogue(params: dict, body: dict) -> dict:
    filters = {}
    for key, value in params.items():
        if key.startswith("attr."):
            filters[key[5:]] = value
    items, total = service.repo().search_catalogue(
        query=params.get("q") or "",
        class_code=params.get("class_code") or "",
        cpse=params.get("cpse") or "",
        attribute_filters=filters or None,
        limit=min(_int(params, "limit", 25), 200),
        offset=_int(params, "offset", 0),
    )
    return {"total": total, "items": items, "filters": filters}


def nmc_detail(params: dict, body: dict) -> dict:
    rec = service.repo().nmc_by_code(params.get("code", ""))
    return rec or {"error": "national material code not found"}


def nmc_validate(params: dict, body: dict) -> dict:
    return nmc_mod.validate(params.get("code", ""))


def nmc_unmerge(params: dict, body: dict) -> dict:
    ids = body.get("canonical_ids") or []
    if not ids:
        raise ValueError("canonical_ids is required")
    actor, _cpse, _role = _actor(body, params)
    reason = body.get("reason") or "steward-initiated un-merge"
    return service.unmerge(params.get("code", ""), [int(i) for i in ids], actor, reason)


# ------------------------------------------------------------------ directed

def substitutions(params: dict, body: dict) -> dict:
    items = service.repo().substitutions(
        status=params.get("status") or None,
        limit=min(_int(params, "limit", 100), 500),
    )
    return {"total": len(items), "items": items,
            "note": ("directed edges are deliberately excluded from clustering - "
                     "'can serve the duty of' is not 'is the same as'")}


def substitution_approve(params: dict, body: dict) -> dict:
    actor, _cpse, _role = _actor(body, params)
    return service.approve_substitution(_int(params, "id"), actor, body.get("reason", ""))


# ------------------------------------------------------------------ operations

def redeploy(params: dict, body: dict) -> dict:
    return service.buy_or_borrow(
        query=body.get("query") or params.get("query") or "",
        nmc_code=body.get("nmc") or params.get("nmc") or "",
        qty=float(body.get("qty") or params.get("qty") or 0),
        requesting_cpse=body.get("cpse") or params.get("cpse") or "",
        min_idle_days=int(body["min_idle_days"]) if body.get("min_idle_days") else None,
    )


def dup_check(params: dict, body: dict) -> dict:
    record = body.get("record") or body
    if not record.get("description"):
        raise ValueError("record.description is required")
    return service.duplicate_check(record, top_k=min(int(body.get("top_k") or 5), 20))


# ------------------------------------------------------------------ analytics

def analytics(params: dict, body: dict) -> dict:
    return service.analytics()


def value(params: dict, body: dict) -> dict:
    return service.value_model()


def federation_report(params: dict, body: dict) -> dict:
    return service.federation_report()


# ------------------------------------------------------------------ ledger

def ledger_events(params: dict, body: dict) -> dict:
    r = service.repo()
    return {
        "total": r.ledger.count(),
        "head": r.ledger.head(),
        "events": r.ledger.events(
            limit=min(_int(params, "limit", 50), 500),
            offset=_int(params, "offset", 0),
            subject=params.get("subject") or None,
            event_type=params.get("event_type") or None,
        ),
    }


def ledger_verify(params: dict, body: dict) -> dict:
    r = service.repo()
    out = r.ledger.verify()
    out["projection_matches_tables"] = r.ledger.projection_matches_tables()
    return out


def ledger_replay(params: dict, body: dict) -> dict:
    upto = params.get("upto_seq")
    return service.repo().ledger.replay(int(upto) if upto else None)


def ledger_merkle(params: dict, body: dict) -> dict:
    r = service.repo()
    out = r.ledger.merkle_root(params.get("day"), publish=_bool(params.get("publish"), False))
    out["published_roots"] = r.ledger.published_roots()
    return out


# ------------------------------------------------------------------ ERP

def change_requests(params: dict, body: dict) -> dict:
    items = service.repo().change_requests(status=params.get("status") or None,
                                           limit=min(_int(params, "limit", 100), 500))
    return {"total": len(items), "items": items,
            "note": ("the AI never writes production master data; each of these enters "
                     "the CPSE's own approval workflow")}


def erp_sync(params: dict, body: dict) -> dict:
    from ..connectors.sap import SapConnector
    r = service.repo()
    connector = SapConnector.from_config(body.get("connection") or {})
    queued = r.change_requests(status="queued")
    sent = []
    for cr in queued:
        outcome = connector.submit_change_request(cr["payload"])
        r.set_change_request_status(cr["id"], outcome["status"], outcome.get("external_ref", ""))
        sent.append({"id": cr["id"], **outcome})
    return {"dispatched": len(sent), "mode": connector.mode, "results": sent,
            "outbox_dir": str(config.OUTBOX_DIR)}


# ------------------------------------------------------------------ inspection

def kg_compare(params: dict, body: dict) -> dict:
    a = params.get("a") or ""
    b = params.get("b") or ""
    if not a or not b:
        raise ValueError("both 'a' and 'b' query parameters are required")
    order = params.get("order")
    if order:
        return {"kind": "ordered", "order": order, **kg.compare_ordered(order, a, b).to_dict()}
    return {
        "kind": "material",
        "a_family": kg.family_of(a), "b_family": kg.family_of(b),
        **kg.compare_material(a, b).to_dict(),
    }


def uom_resolve(params: dict, body: dict) -> dict:
    res = uom.resolve(params.get("uom") or "", params.get("description") or "")
    price = params.get("price")
    out = res.to_dict()
    if price:
        base, _ = uom.base_unit_price(float(price), params.get("uom") or "",
                                      params.get("description") or "")
        out["unit_price_base"] = base
    return out


def extract_preview(params: dict, body: dict) -> dict:
    description = params.get("description") or body.get("description") or ""
    if not description:
        raise ValueError("description is required")
    cls = classify.classify(description)
    ex = extract.extract(description, cls.class_code)
    from ..core import render, text
    return {
        "classification": cls.to_dict(),
        "extraction": ex.to_dict(),
        "coverage": extract.coverage(ex),
        "completeness": text.completeness(description),
        "golden_description": render.golden_description(cls.class_code, ex.values()),
        "trace": render.trace(cls.class_code, ex.to_dict()["attributes"]),
        "normalised": text.normalise(description),
    }
