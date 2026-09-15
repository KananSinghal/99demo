"""The standards knowledge graph.

Embeddings cannot know that SS316 in one CPSE's catalogue and A4-70 in another's
name the same metal. A graph can, and it can cite the clause it read that from.

The graph is populated offline from standards cross-reference tables, reviewed,
and is READ-ONLY at runtime. When no edge exists the answer is "unknown", which
routes the pair to a domain engineer. A language model is never the authority on
whether two grades are equivalent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict, field
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from . import data, text

IDENTICAL = "IDENTICAL"
EQUIVALENT = "EQUIVALENT"
SUBSTITUTABLE = "SUBSTITUTABLE"     # directed: a serves b's duty
VARIANT = "VARIANT"
UNKNOWN = "UNKNOWN"
CONFLICT = "CONFLICT"


@dataclass
class Verdict:
    relation: str
    citation: str = ""
    path: List[str] = field(default_factory=list)
    partial: bool = False
    caveat: str = ""
    direction: str = ""              # "a->b" for directed verdicts

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------- indexing

class _Index:
    def __init__(self, kg: dict):
        self.kg = kg
        self.alias_to_family: Dict[str, str] = {}
        for fam, spec in kg["families"].items():
            for alias in list(spec.get("aliases", [])) + [fam]:
                self.alias_to_family[_key(alias)] = fam
        self.adj: Dict[str, List[dict]] = {}
        for edge in kg.get("edges", []):
            self.adj.setdefault(edge["from"], []).append(edge)
            rev = dict(edge)
            rev["from"], rev["to"] = edge["to"], edge["from"]
            rev["_reversed"] = True
            self.adj.setdefault(edge["to"], []).append(rev)
        # service qualifiers
        self.service_alias: Dict[str, str] = {}
        for sid, spec in kg.get("service_qualifiers", {}).items():
            for alias in list(spec.get("aliases", [])) + [sid]:
                self.service_alias[_key(alias)] = sid
        # superseded standards
        self.superseded: Dict[str, dict] = {}
        for row in kg.get("superseded", []):
            self.superseded[_key(row["old"])] = row
        # orderings
        self.orders: Dict[str, dict] = kg.get("orders", {})
        self.order_pos: Dict[str, Dict[str, int]] = {}
        self.order_equiv: Dict[str, Dict[str, str]] = {}
        for name, spec in self.orders.items():
            canon: Dict[str, str] = {}
            for group in spec.get("equivalences", []):
                head = group[0]
                for member in group:
                    canon[_key(member)] = head
            self.order_equiv[name] = canon
            pos: Dict[str, int] = {}
            rank = 0
            seen: Dict[str, int] = {}
            for value in spec.get("sequence", []):
                head = canon.get(_key(value), value)
                if _key(head) in seen:
                    pos[_key(value)] = seen[_key(head)]
                else:
                    seen[_key(head)] = rank
                    pos[_key(value)] = rank
                    rank += 1
            self.order_pos[name] = pos


def _key(value: str) -> str:
    return re.sub(r"[\s\.\-_]+", "", text.basic(value))


@lru_cache(maxsize=1)
def index() -> _Index:
    return _Index(data.standards_kg())


def reload() -> None:
    index.cache_clear()
    data.invalidate("standards_kg")


# ------------------------------------------------------------------- resolution

#: Aliases at or below this length are only matched as whole tokens, never as
#: substrings. Without this, "BALL BEARING" contains "AL" and a ball bearing acquires
#: an aluminium material. Short designations are real (AL, CU, A2) so they cannot be
#: dropped - they just have to be matched properly.
_SHORT_ALIAS = 3


def _token_keys(value: str) -> set:
    return {_key(t) for t in text.clean(value).split(" ") if t}


def family_of(designation: str) -> Optional[str]:
    """Map any observed designation string onto a family node, or None."""
    if not designation:
        return None
    idx = index()
    k = _key(designation)
    if k in idx.alias_to_family:
        return idx.alias_to_family[k]
    toks = _token_keys(designation)
    # Longest alias first, so "A216 WCB" beats "WCB" and "SS316" beats "SS".
    best: Tuple[int, Optional[str]] = (0, None)
    for alias, fam in idx.alias_to_family.items():
        if len(alias) <= _SHORT_ALIAS:
            if alias in toks and len(alias) > best[0]:
                best = (len(alias), fam)
        elif alias in k and len(alias) > best[0]:
            best = (len(alias), fam)
    return best[1]


def label(family: str) -> str:
    fam = index().kg["families"].get(family)
    return fam["label"] if fam else family


def citation_for(family: str) -> str:
    fam = index().kg["families"].get(family)
    return fam.get("citation", "") if fam else ""


def find_designations(value: str) -> List[str]:
    """Every family named anywhere in a free-text string, longest alias first."""
    idx = index()
    k = _key(value)
    toks = _token_keys(value)
    hits: List[Tuple[int, str]] = []
    for alias, fam in idx.alias_to_family.items():
        if len(alias) <= _SHORT_ALIAS:
            if alias in toks:
                hits.append((len(alias), fam))
        elif alias in k:
            hits.append((len(alias), fam))
    hits.sort(reverse=True)
    out: List[str] = []
    for _n, fam in hits:
        if fam not in out:
            out.append(fam)
    return out


# ----------------------------------------------------------------- graph search

def _bfs(src: str, dst: str, max_depth: int = 3) -> Optional[List[dict]]:
    """Shortest edge path between two family nodes."""
    if src == dst:
        return []
    idx = index()
    frontier: List[Tuple[str, List[dict]]] = [(src, [])]
    seen = {src}
    for _depth in range(max_depth):
        nxt: List[Tuple[str, List[dict]]] = []
        for node, path in frontier:
            for edge in idx.adj.get(node, []):
                target = edge["to"]
                if target in seen:
                    continue
                new_path = path + [edge]
                if target == dst:
                    return new_path
                seen.add(target)
                nxt.append((target, new_path))
        frontier = nxt
        if not frontier:
            break
    return None


_DIRECTED_TYPES = {"gradeSatisfiedBy", "variantOf", "castEquivalentOf", "sourServiceVariantOf", "memberOf"}


def compare_material(a: str, b: str) -> Verdict:
    """Compare two material designation strings through the graph."""
    if not a and not b:
        return Verdict(UNKNOWN, caveat="neither record states a material")
    if not a or not b:
        return Verdict(UNKNOWN, caveat="material stated on only one record")

    ka, kb = _key(a), _key(b)
    if ka == kb:
        return Verdict(IDENTICAL, citation="verbatim match")

    fa, fb = family_of(a), family_of(b)
    if fa is None or fb is None:
        missing = a if fa is None else b
        return Verdict(UNKNOWN, caveat=f"designation '{missing}' is not in the standards graph")
    if fa == fb:
        return Verdict(
            IDENTICAL,
            citation=citation_for(fa),
            path=[fa],
            caveat=f"both designations resolve to {label(fa)}",
        )

    path = _bfs(fa, fb)
    if path is None:
        return Verdict(
            CONFLICT,
            caveat=f"{label(fa)} and {label(fb)} are different materials with no equivalence edge",
            path=[fa, fb],
        )

    partial = any(e.get("partial") for e in path)
    directed = any(e["type"] in _DIRECTED_TYPES for e in path)
    citations = " | ".join(dict.fromkeys(e.get("citation", "") for e in path if e.get("citation")))
    node_path = [fa] + [e["to"] for e in path]

    if partial or directed:
        return Verdict(
            EQUIVALENT,
            citation=citations,
            path=node_path,
            partial=True,
            caveat=(
                "the standards mapping is one-to-many, so form-fit-function equivalence "
                "holds but part-level identity is not proven"
            ),
        )
    return Verdict(EQUIVALENT, citation=citations, path=node_path)


# ------------------------------------------------------------ ordered attributes

def compare_ordered(order_name: str, a: str, b: str) -> Verdict:
    """Compare two values on a standards ordering (pipe schedule, property class,
    pressure class...). This is where directed substitutability comes from."""
    idx = index()
    spec = idx.orders.get(order_name)
    if not spec:
        return Verdict(UNKNOWN, caveat=f"no ordering named {order_name}")
    pos = idx.order_pos.get(order_name, {})
    ka, kb = _key(a), _key(b)
    if ka == kb:
        return Verdict(IDENTICAL, citation=spec.get("citation", ""))
    pa, pb = pos.get(ka), pos.get(kb)
    if pa is None or pb is None:
        unknown = a if pa is None else b
        return Verdict(UNKNOWN, caveat=f"value '{unknown}' is not on the {order_name} scale")
    if pa == pb:
        return Verdict(
            EQUIVALENT,
            citation=spec.get("citation", ""),
            caveat=spec.get("equivalence_caveat", ""),
        )
    if not spec.get("higher_substitutes_lower", False):
        return Verdict(
            CONFLICT,
            citation=spec.get("citation", ""),
            caveat=spec.get("caveat", f"{a} and {b} are not on a substitutable scale"),
        )
    higher, lower = (a, b) if pa > pb else (b, a)
    return Verdict(
        SUBSTITUTABLE,
        citation=spec.get("citation", ""),
        caveat=spec.get("caveat", ""),
        direction="a->b" if pa > pb else "b->a",
        path=[higher, lower],
    )


def order_requires_equal(order_name: str) -> List[str]:
    spec = index().orders.get(order_name, {})
    return list(spec.get("requires_equal", []))


# ------------------------------------------------------------ service qualifiers

def service_qualifier(value: str) -> Optional[str]:
    if not value:
        return None
    idx = index()
    k = _key(value)
    if k in idx.service_alias:
        return idx.service_alias[k]
    for alias, sid in idx.service_alias.items():
        if len(alias) >= 4 and alias in k:
            return sid
    return None


def find_service_qualifiers(value: str) -> List[str]:
    idx = index()
    k = _key(value)
    toks = _token_keys(value)
    out: List[str] = []
    for alias, sid in idx.service_alias.items():
        hit = alias in toks if len(alias) <= _SHORT_ALIAS else alias in k
        if hit and sid not in out:
            out.append(sid)
    return out


def service_rule(sid: str) -> dict:
    return index().kg.get("service_qualifiers", {}).get(sid, {})


def compare_service(a: str, b: str) -> Verdict:
    """Service qualification is non-inferable. Absent means UNKNOWN, never
    NOT REQUIRED - a record without it can never auto-match one with it."""
    sa = set(find_service_qualifiers(a or ""))
    sb = set(find_service_qualifiers(b or ""))
    if sa == sb:
        if not sa:
            return Verdict(IDENTICAL, citation="no service qualifier on either record")
        return Verdict(IDENTICAL, citation=" | ".join(service_rule(s).get("citation", "") for s in sorted(sa)))
    only = (sa ^ sb)
    names = ", ".join(sorted(service_rule(s).get("label", s) for s in only))
    cits = " | ".join(dict.fromkeys(service_rule(s).get("citation", "") for s in only))
    return Verdict(
        UNKNOWN,
        citation=cits,
        caveat=(
            f"{names} is stated on one record and absent from the other. Absence means "
            f"unknown, not 'not required', so this pair cannot be auto-matched."
        ),
    )


# -------------------------------------------------------------------- standards

def canonical_standard(value: str) -> Tuple[str, str]:
    """Map a superseded standard reference onto its successor. Returns (canonical, note)."""
    if not value:
        return "", ""
    idx = index()
    k = _key(value)
    row = idx.superseded.get(k)
    if row:
        return row["new"], f"{value} is superseded by {row['new']} ({row.get('citation','')})"
    for old, r in idx.superseded.items():
        if len(old) >= 5 and old in k:
            return r["new"], f"{value} is superseded by {r['new']} ({r.get('citation','')})"
    return value, ""


def compare_standard(a: str, b: str) -> Verdict:
    ca, na = canonical_standard(a)
    cb, nb = canonical_standard(b)
    if not a or not b:
        return Verdict(UNKNOWN, caveat="standard reference stated on only one record")
    if _key(ca) == _key(cb):
        note = " ".join(x for x in (na, nb) if x)
        return Verdict(IDENTICAL, citation=note or "standard references match")
    return Verdict(
        CONFLICT,
        caveat=f"different governing standards: {a} vs {b}",
        citation=" ".join(x for x in (na, nb) if x),
    )


def stats() -> dict:
    idx = index()
    kg = idx.kg
    return {
        "version": kg.get("version"),
        "families": len(kg.get("families", {})),
        "aliases": len(idx.alias_to_family),
        "edges": len(kg.get("edges", [])),
        "orders": list(idx.orders.keys()),
        "service_qualifiers": len(kg.get("service_qualifiers", {})),
        "superseded": len(kg.get("superseded", [])),
    }
