"""The attribute constraint solver, and the typed relation it produces.

The PS asks you to distinguish truly identical items from functionally equivalent
or superficially similar ones. Those are three different relations, and one of them
is directional - a Schedule 80 pipe serves a Schedule 40 duty; a Schedule 40 pipe
does not serve a Schedule 80 duty.

  IDENTICAL       symmetric    same MPN, or full match on every mandatory attribute
  EQUIVALENT      symmetric    same form, fit and function against the class spec
  SUBSTITUTABLE   DIRECTED     a satisfies every requirement of b, not the reverse
  VARIANT_OF      symmetric    same family, different size or option
  DISTINCT        symmetric    a hard attribute conflict exists, named explicitly

Rule of precedence: a hard conflict on a blocking attribute beats everything. A
false merge puts the wrong part on a wellhead; a false split merely wastes money.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple

from . import classify, features, kg

IDENTICAL = "IDENTICAL"
EQUIVALENT = "EQUIVALENT"
SUBSTITUTABLE = "SUBSTITUTABLE"
VARIANT_OF = "VARIANT_OF"
DISTINCT = "DISTINCT"
UNDETERMINED = "UNDETERMINED"

SYMMETRIC = {IDENTICAL, EQUIVALENT, VARIANT_OF, DISTINCT}
MERGEABLE = {IDENTICAL, EQUIVALENT}


@dataclass
class Relation:
    relation: str
    direction: str = ""                 # "a->b" | "b->a" | "" for symmetric
    reasons: List[str] = field(default_factory=list)
    conflicts: List[dict] = field(default_factory=list)
    caveats: List[str] = field(default_factory=list)
    citations: List[str] = field(default_factory=list)
    identity_key: bool = False
    proof_strength: float = 0.0         # 0..1, how much of the class spec was proven
    blocked_by: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _proof_strength(cmp: dict) -> float:
    cls = cmp["class_code"]
    specs = classify.attribute_specs(cls)
    if not specs:
        return 0.0
    weight_total, weight_proven = 0.0, 0.0
    for spec, row in zip(specs, cmp["rows"]):
        w = 2.0 if spec.get("mandatory") else 1.0
        weight_total += w
        if row["status"] == "exact":
            weight_proven += w
        elif row["status"] == "kg_equiv":
            weight_proven += w * 0.75
        elif row["status"] == "directed":
            weight_proven += w * 0.5
    return round(weight_proven / weight_total, 4) if weight_total else 0.0


def _thread_series(value: str) -> str:
    """Metric, imperial-unified and pipe threads are different series, not different
    sizes of one scale. M12 and 1/2"-13UNC are 12.00 mm and 12.70 mm with different
    pitches - near the same size and interchangeable in nothing."""
    v = (value or "").upper()
    if v.startswith("M") and any(c.isdigit() for c in v[:4]):
        return "metric"
    if "UNC" in v or "UNF" in v or "BSW" in v or "BSF" in v or v.endswith("UN"):
        return "unified"
    if "NPT" in v or "BSP" in v:
        return "pipe"
    return "unknown"


def _variant_dimension(cmp: dict) -> Optional[dict]:
    """A single dimensional conflict on an otherwise-identical record means 'same
    family, different size' - roll it up for spend, never merge it.

    A dimensional conflict only counts as a variant when both values lie on the SAME
    scale. Different scales are DISTINCT, not a bigger version of the same thing.
    """
    cls = cmp["class_code"]
    numeric_keys = {s["key"] for s in classify.attribute_specs(cls) if s.get("numeric")}
    dim_keys = numeric_keys | {"nominal_size", "thread", "length_mm", "bore_mm", "csa_sqmm", "designation"}
    conflicts = cmp["conflicts"]
    if len(conflicts) != 1:
        return None
    row = conflicts[0]
    if row["key"] not in dim_keys:
        return None
    if row["key"] == "thread":
        sa, sb = _thread_series(row.get("a") or ""), _thread_series(row.get("b") or "")
        if sa != sb or "unknown" in (sa, sb):
            return None
    return row


def solve(a: dict, b: dict, cmp: Optional[dict] = None) -> Relation:
    """Type the relation between two canonical records."""
    cmp = cmp or features.compare_attributes(a, b)
    rel = Relation(relation=UNDETERMINED)
    rel.proof_strength = _proof_strength(cmp)

    # ------------------------------------------------------------ class mismatch
    if a.get("class_code") != b.get("class_code"):
        rel.relation = DISTINCT
        rel.blocked_by = "class"
        rel.reasons.append(
            f"different material classes: {a.get('class_code')} vs {b.get('class_code')}"
        )
        return rel

    # ------------------------------------------------- non-inferable service gate
    # Checked before anything else: absence of a sour-service qualifier means
    # UNKNOWN, never NOT REQUIRED, and no score can override that.
    for row in cmp["rows"]:
        spec = classify.attribute_spec(cmp["class_code"], row["key"])
        if spec.get("service") and row["status"] == "unknown":
            rel.relation = UNDETERMINED
            rel.blocked_by = "service_qualifier"
            rel.caveats.append(row["caveat"] or "service qualification stated on one record only")
            if row["citation"]:
                rel.citations.append(row["citation"])
            rel.reasons.append(
                "service qualification is non-inferable - routed to a domain engineer"
            )
            return rel

    # ------------------------------------------------------------ hard conflicts
    blocking_conflicts = cmp["blocking_conflicts"]
    if blocking_conflicts:
        variant = _variant_dimension(cmp)
        if variant is not None and variant["blocking"]:
            rel.relation = VARIANT_OF
            rel.reasons.append(
                f"same family, different {variant['label'].lower()}: "
                f"{variant['a']} vs {variant['b']}"
            )
            rel.conflicts = [variant]
            rel.caveats.append("rolled up for spend analysis; never merged into one code")
            return rel
        rel.relation = DISTINCT
        rel.blocked_by = blocking_conflicts[0]["key"]
        rel.conflicts = blocking_conflicts
        for row in blocking_conflicts:
            detail = row.get("caveat") or f"{row['a']} vs {row['b']}"
            rel.reasons.append(f"hard conflict on {row['label']}: {detail}")
            if row.get("citation"):
                rel.citations.append(row["citation"])
        if any(r["key"] == "thread" for r in blocking_conflicts):
            rel.caveats.append(
                "thread series differ - a metric and a unified thread of similar "
                "diameter are interchangeable in no application"
            )
        return rel

    # Non-blocking conflicts still prevent a merge, but they are softer.
    other_conflicts = [r for r in cmp["conflicts"] if not r["blocking"]]

    # ------------------------------------------------------- directed relations
    directed = cmp["directed"]
    if directed:
        row = directed[0]
        # A directed substitution is only valid when the attributes the standard says
        # must be equal actually are equal (a higher pressure class does not bolt to a
        # lower one unless the end connection matches).
        for required in row.get("requires_equal", []):
            match = next((r for r in cmp["rows"] if r["key"] == required), None)
            if match and match["status"] not in ("exact", "kg_equiv", "absent_both"):
                rel.relation = DISTINCT
                rel.blocked_by = required
                rel.reasons.append(
                    f"{row['label']} would substitute, but {match['label']} differs "
                    f"({match['a']} vs {match['b']}) and the standard requires it to match"
                )
                if row.get("citation"):
                    rel.citations.append(row["citation"])
                return rel
        if other_conflicts:
            rel.relation = DISTINCT
            rel.conflicts = other_conflicts
            rel.reasons.append(
                "conflicting attribute alongside a directional difference: "
                + ", ".join(f"{r['label']} {r['a']} vs {r['b']}" for r in other_conflicts)
            )
            return rel
        rel.relation = SUBSTITUTABLE
        rel.direction = row["direction"] or "a->b"
        rel.reasons.append(
            f"{row['label']} {row['a']} satisfies the duty of {row['b']}"
            if rel.direction == "a->b" else
            f"{row['label']} {row['b']} satisfies the duty of {row['a']}"
        )
        if row.get("caveat"):
            rel.caveats.append(row["caveat"])
        if row.get("citation"):
            rel.citations.append(row["citation"])
        rel.caveats.append(
            "directional substitution is an engineering judgment - the system proposes "
            "and evidences it, an engineer approves it"
        )
        return rel

    if other_conflicts:
        rel.relation = DISTINCT
        rel.conflicts = other_conflicts
        for row in other_conflicts:
            rel.reasons.append(f"conflict on {row['label']}: {row['a']} vs {row['b']}")
        return rel

    # ------------------------------------------------------- identical vs equivalent
    id_hit, id_reason = features.identity_match(a, b)
    rel.identity_key = id_hit

    mandatory = classify.mandatory_keys(cmp["class_code"])
    mand_rows = [r for r in cmp["rows"] if r["key"] in mandatory]
    mand_missing = [r for r in mand_rows if r["status"] in ("missing", "absent_both", "unknown")]
    mand_exact = all(r["status"] == "exact" for r in mand_rows) if mand_rows else False
    partial_rows = [r for r in cmp["rows"] if r["status"] == "kg_equiv"]

    for row in cmp["rows"]:
        if row.get("citation") and row["status"] in ("exact", "kg_equiv"):
            rel.citations.append(row["citation"])
    rel.citations = list(dict.fromkeys(c for c in rel.citations if c))

    if id_hit:
        rel.relation = IDENTICAL
        rel.reasons.append(id_reason)
        if partial_rows:
            rel.caveats.append(
                "identity is established by the identity key; the standards mapping on "
                + ", ".join(r["label"].lower() for r in partial_rows)
                + " is one-to-many and does not itself prove identity"
            )
        return rel

    if mand_exact and not mand_missing and not partial_rows:
        rel.relation = IDENTICAL
        rel.reasons.append(
            "every mandatory attribute of the class matches verbatim and none conflict"
        )
        return rel

    if mand_missing:
        rel.relation = EQUIVALENT if not mand_exact else EQUIVALENT
        rel.reasons.append(
            "form-fit-function equivalence holds, but "
            + ", ".join(r["label"].lower() for r in mand_missing)
            + " is not stated on both records, so part-level identity is unproven"
        )
        rel.caveats.append("missing is not the same as matching")
        return rel

    rel.relation = EQUIVALENT
    if partial_rows:
        rel.reasons.append(
            "matched through the standards graph on "
            + ", ".join(f"{r['label'].lower()}" for r in partial_rows)
        )
        for row in partial_rows:
            if row.get("caveat"):
                rel.caveats.append(row["caveat"])
    else:
        rel.reasons.append("all comparable attributes agree")
    return rel


def counterfactual(a: dict, b: dict, cmp: Optional[dict] = None) -> Optional[str]:
    """What single change would flip this verdict? An engineer can argue with a
    counterfactual; they cannot argue with a similarity score."""
    cmp = cmp or features.compare_attributes(a, b)
    rel = solve(a, b, cmp)
    if rel.relation in (DISTINCT, UNDETERMINED):
        if rel.conflicts:
            row = rel.conflicts[0]
            return (
                f"if {row['label']} agreed on both records, this would be reconsidered "
                f"as {EQUIVALENT}"
            )
        if rel.blocked_by == "service_qualifier":
            return "if the service qualification were stated on both records, this could be auto-typed"
        return None
    partial = [r for r in cmp["rows"] if r["status"] == "kg_equiv" and r.get("partial")]
    if partial:
        row = partial[0]
        return (
            f"if {row['label'].lower()} were stated to the same grade on both records "
            f"this would strengthen to {IDENTICAL}; if it were stated to different "
            f"grades within the group it would drop to {DISTINCT}"
        )
    missing = [r for r in cmp["rows"] if r["status"] == "missing" and r["mandatory"]]
    if missing:
        row = missing[0]
        return (
            f"if {row['label'].lower()} were stated on both records and agreed, "
            f"this would strengthen to {IDENTICAL}"
        )
    return None


def relation_summary() -> List[dict]:
    return [
        {"relation": IDENTICAL, "symmetry": "symmetric",
         "meaning": "same manufacturer part number, or a full match on every mandatory attribute",
         "unlocks": "safe auto-merge into one NMC; duplicate code elimination"},
        {"relation": EQUIVALENT, "symmetry": "symmetric",
         "meaning": "different manufacturers, same form fit and function against the class specification",
         "unlocks": "common NMC with variant sub-codes; multi-vendor tendering"},
        {"relation": SUBSTITUTABLE, "symmetry": "directed",
         "meaning": "a satisfies every requirement of b, but not the reverse",
         "unlocks": "cross-CPSE stock redeployment - the largest single saving"},
        {"relation": VARIANT_OF, "symmetry": "symmetric",
         "meaning": "same family, different size or option",
         "unlocks": "family-level spend rollup and catalogue rationalisation"},
        {"relation": DISTINCT, "symmetry": "symmetric",
         "meaning": "a hard attribute conflict exists, recorded explicitly",
         "unlocks": "suppresses re-surfacing of rejected pairs; trains the model"},
    ]
