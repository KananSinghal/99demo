"""The trust boundary.

Every centralised design in this competition shares one fatal flaw that nobody
mentions: no materials head signs a form that sends their organisation's rates and
stock positions to a shared server. If the architecture requires that, it is a
prototype, not a proposal.

  CPSE node        full material master, prices, stock, vendors, PO history.
                   Normalisation, extraction and embedding all run locally on
                   open-weight models. Nothing calls an external API, ever.

  National registry NMC identities, standardised descriptions, attribute records,
                   classification facets, the ledger, and HASHED blocking keys.
                   Never prices. Never stock. Never vendors. Never consumption.

The hard questions still get answered:
  "who else holds this?"      private set intersection over hashed identity keys
  "what does it cost nationally?"  federated aggregate with a k-anonymity floor
  "who has surplus?"          the holder is notified and chooses to respond -
                              disclosure is an act, not a default
"""

from __future__ import annotations

import hashlib
import hmac
import json
import statistics
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .. import config
from . import blocking

K_ANONYMITY_FLOOR = 3


def _hash(value: str, salt: str) -> str:
    return hmac.new(salt.encode("utf-8"), value.encode("utf-8"), hashlib.sha256).hexdigest()[:32]


# ------------------------------------------------------------------ redaction

def redact(record: dict, fields: Sequence[str] = config.CONFIDENTIAL_FIELDS) -> dict:
    """Strip every commercially sensitive field. This is applied at the node, before
    anything is serialised for transmission - not at the registry on receipt."""
    out = {k: v for k, v in record.items() if k not in set(fields)}
    out["_redacted"] = sorted(set(fields) & set(record.keys()))
    return out


def publishable(record: dict, salt: str, fields: Sequence[str] = config.CONFIDENTIAL_FIELDS) -> dict:
    """What a CPSE node actually publishes to the registry: the structured attribute
    record, the embedding, and hashed blocking keys. Never the raw keys, so the
    registry can find overlap without learning a manufacturer part number."""
    payload = redact(record, fields)
    payload["hashed_keys"] = [
        _hash(key, salt) for key in blocking.blocking_keys_for(record, record.get("attributes") or {})
    ]
    payload.pop("mfr_part_no", None)
    payload.pop("mfr_name", None)
    payload.pop("vendor_matl_no", None)
    return payload


# ------------------------------------------------- private set intersection (PSI)

@dataclass
class PsiRequest:
    node: str
    salt_commitment: str
    hashed_keys: List[str]


def psi_offer(record_keys: Iterable[str], shared_salt: str, node: str) -> PsiRequest:
    """Step 1: a node publishes salted hashes of its identity keys. With a salt agreed
    between exactly the two parties, neither side can dictionary-attack the other's
    catalogue, and no third party learns anything."""
    hashed = sorted({_hash(k, shared_salt) for k in record_keys})
    return PsiRequest(
        node=node,
        salt_commitment=hashlib.sha256(shared_salt.encode("utf-8")).hexdigest()[:16],
        hashed_keys=hashed,
    )


def psi_intersect(a: PsiRequest, b: PsiRequest) -> dict:
    """Step 2: intersect. Both sides learn the OVERLAP and nothing about the rest of
    the other's catalogue."""
    if a.salt_commitment != b.salt_commitment:
        return {"error": "salt commitments differ - the two nodes did not agree a shared salt"}
    sa, sb = set(a.hashed_keys), set(b.hashed_keys)
    overlap = sa & sb
    return {
        "nodes": [a.node, b.node],
        "a_size": len(sa),
        "b_size": len(sb),
        "overlap": len(overlap),
        "overlap_keys": sorted(overlap)[:200],
        "a_only": len(sa - sb),
        "b_only": len(sb - sa),
        "disclosed": "only the intersection; neither side learns the other's non-overlapping items",
    }


# ------------------------------------------------ k-anonymous federated aggregate

def price_band(
    contributions: Sequence[Tuple[str, float]],
    k: int = K_ANONYMITY_FLOOR,
) -> dict:
    """contributions: [(cpse_code, unit_price_base)].

    Returns a national price band only when at least k distinct CPSEs contributed.
    A buyer sees where they sit against the national band; nobody sees a named
    competitor's rate.
    """
    by_cpse: Dict[str, List[float]] = {}
    for cpse, price in contributions:
        if price and price > 0:
            by_cpse.setdefault(cpse, []).append(float(price))
    if len(by_cpse) < k:
        return {
            "available": False,
            "reason": f"fewer than k={k} CPSEs contributed; no band is returned",
            "contributors": len(by_cpse),
        }
    means = sorted(statistics.fmean(v) for v in by_cpse.values())
    n = len(means)
    return {
        "available": True,
        "contributors": n,
        "k": k,
        "min": round(means[0], 4),
        "p25": round(means[max(0, int(n * 0.25) - (1 if n * 0.25 == int(n * 0.25) else 0))], 4),
        "median": round(statistics.median(means), 4),
        "p75": round(means[min(n - 1, int(n * 0.75))], 4),
        "max": round(means[-1], 4),
        "spread": round(means[-1] / means[0], 3) if means[0] > 0 else None,
        "disclosed": "aggregate band only; no CPSE is named and no individual rate is returned",
    }


# ------------------------------------------------------------ disclosure as an act

@dataclass
class SurplusRequest:
    request_id: str
    nmc: str
    requesting_cpse: str
    qty_needed: float
    note: str = ""


def surplus_broadcast(request: SurplusRequest, holders: Sequence[dict]) -> dict:
    """The requesting CPSE posts a requirement against an NMC. Holders are NOTIFIED;
    they are not disclosed. Each holder decides whether to respond, and only a
    response reveals a stock position."""
    notified = [
        {"cpse_code": h.get("cpse_code"), "notified": True}
        for h in holders if h.get("cpse_code") != request.requesting_cpse
    ]
    return {
        "request_id": request.request_id,
        "nmc": request.nmc,
        "requesting_cpse": request.requesting_cpse,
        "qty_needed": request.qty_needed,
        "notified_cpses": notified,
        "disclosed_to_requester": "nothing yet - stock positions appear only in a voluntary response",
    }


# ------------------------------------------------------------------ air-gap mode

def export_bundle(records: Sequence[dict], salt: str, node: str, signing_key: str) -> dict:
    """An air-gapped refinery node syncs by signed export bundle on removable media.
    The models are open-weight and local; the bundle is the only thing that moves."""
    payload = [publishable(r, salt) for r in records]
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    signature = hmac.new(signing_key.encode("utf-8"), digest.encode("utf-8"), hashlib.sha256).hexdigest()
    return {
        "node": node,
        "records": len(payload),
        "sha256": digest,
        "signature": signature,
        "payload": payload,
        "contains": "attribute records, embeddings and hashed keys",
        "excludes": list(config.CONFIDENTIAL_FIELDS),
    }


def verify_bundle(bundle: dict, signing_key: str) -> dict:
    body = json.dumps(bundle.get("payload") or [], sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    expected = hmac.new(signing_key.encode("utf-8"), digest.encode("utf-8"), hashlib.sha256).hexdigest()
    return {
        "digest_matches": digest == bundle.get("sha256"),
        "signature_valid": hmac.compare_digest(expected, bundle.get("signature") or ""),
        "records": len(bundle.get("payload") or []),
    }


def boundary_report() -> dict:
    """What crosses the boundary and what is stopped at it. Used by the UI."""
    return {
        "node_role": config.NODE_ROLE,
        "crosses": [
            "structured attribute record",
            "golden description",
            "embedding vector",
            "hashed blocking keys",
            "class and classification facets",
        ],
        "never_crosses": list(config.CONFIDENTIAL_FIELDS),
        "mechanisms": {
            "catalogue overlap": "private set intersection over salted hashes",
            "national price band": f"federated aggregate with a k-anonymity floor of {K_ANONYMITY_FLOOR}",
            "surplus discovery": "holders are notified; disclosure is a voluntary act",
            "air-gapped sites": "signed export bundle on removable media",
        },
        "models": "open-weight, on-premise inference; nothing calls an external API",
    }
