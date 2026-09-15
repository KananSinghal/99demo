"""Pairs are not clusters.

Pairwise decisions violate transitivity constantly: A is equivalent to B, B to C,
and A conflicts with C. Naive union-find on positive pairs produces catastrophic
mega-clusters - one bad link merges four hundred distinct bearings, and pairwise F1
hides it completely.

Correlation clustering minimises total disagreement instead. This is the pivot
algorithm (KwikCluster), seeded by confidence rather than at random, and extended
with hard constraints so a steward's decision is never silently overridden by the
optimiser:

    must_link     a steward said these are the same -> they end up together
    cannot_link   a steward said these are different -> they never do

Directed SUBSTITUTABLE edges are deliberately NOT used for clustering. They live in
a separate relation graph, because "can serve the duty of" is not "is the same as".
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .. import config


@dataclass
class Edge:
    a: int
    b: int
    weight: float          # positive = pull together, negative = push apart
    relation: str = ""

    def key(self) -> Tuple[int, int]:
        return (self.a, self.b) if self.a < self.b else (self.b, self.a)


@dataclass
class ClusterResult:
    clusters: List[List[int]] = field(default_factory=list)
    singletons: int = 0
    disagreements: float = 0.0
    max_cluster_size: int = 0
    constraint_violations: List[Tuple[int, int]] = field(default_factory=list)
    split_for_size: int = 0

    def cluster_of(self) -> Dict[int, int]:
        out: Dict[int, int] = {}
        for idx, members in enumerate(self.clusters):
            for m in members:
                out[m] = idx
        return out

    def to_dict(self) -> dict:
        return {
            "clusters": len(self.clusters),
            "singletons": self.singletons,
            "largest_cluster": self.max_cluster_size,
            "disagreements": round(self.disagreements, 3),
            "constraint_violations": len(self.constraint_violations),
            "split_for_size": self.split_for_size,
        }


def correlation_cluster(
    nodes: Iterable[int],
    edges: Sequence[Edge],
    must_link: Optional[Iterable[Tuple[int, int]]] = None,
    cannot_link: Optional[Iterable[Tuple[int, int]]] = None,
    max_size: Optional[int] = None,
) -> ClusterResult:
    """Pivot / KwikCluster with must-link and cannot-link constraints."""
    max_size = max_size or config.MAX_CLUSTER_SIZE
    node_list = sorted(set(int(n) for n in nodes))
    pos: Dict[int, Dict[int, float]] = defaultdict(dict)
    for e in edges:
        if e.weight > 0:
            pos[e.a][e.b] = max(pos[e.a].get(e.b, 0.0), e.weight)
            pos[e.b][e.a] = max(pos[e.b].get(e.a, 0.0), e.weight)

    cannot: Dict[int, Set[int]] = defaultdict(set)
    for a, b in (cannot_link or []):
        cannot[int(a)].add(int(b))
        cannot[int(b)].add(int(a))

    # Must-links are applied first as a hard pre-merge: a human already decided.
    parent: Dict[int, int] = {n: n for n in node_list}

    def find(x: int) -> int:
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    for a, b in (must_link or []):
        union(int(a), int(b))

    groups: Dict[int, List[int]] = defaultdict(list)
    for n in node_list:
        groups[find(n)].append(n)

    # Collapse the graph onto must-link super-nodes.
    rep_of = {n: find(n) for n in node_list}
    super_pos: Dict[int, Dict[int, float]] = defaultdict(dict)
    for a, nbrs in pos.items():
        ra = rep_of.get(a, a)
        for b, w in nbrs.items():
            rb = rep_of.get(b, b)
            if ra == rb:
                continue
            super_pos[ra][rb] = max(super_pos[ra].get(rb, 0.0), w)
    super_cannot: Dict[int, Set[int]] = defaultdict(set)
    for a, others in cannot.items():
        ra = rep_of.get(a, a)
        for b in others:
            rb = rep_of.get(b, b)
            if ra != rb:
                super_cannot[ra].add(rb)
                super_cannot[rb].add(ra)

    # Pivot order: strongest, best-connected first, so the most confident evidence
    # anchors a cluster instead of an arbitrary node.
    reps = sorted(groups.keys(), key=lambda r: (-sum(super_pos.get(r, {}).values()), -len(super_pos.get(r, {})), r))

    unassigned = set(reps)
    clusters: List[List[int]] = []
    split_for_size = 0

    while unassigned:
        pivot = next(r for r in reps if r in unassigned)
        unassigned.discard(pivot)
        members = [pivot]
        member_set = {pivot}
        # Attach positive neighbours, strongest first, skipping any that a steward
        # (or an accumulated cannot-link) forbids against ANY current member.
        neighbours = sorted(super_pos.get(pivot, {}).items(), key=lambda kv: -kv[1])
        for other, _w in neighbours:
            if other not in unassigned:
                continue
            if any(other in super_cannot.get(m, set()) for m in member_set):
                continue
            size_now = sum(len(groups[m]) for m in members) + len(groups[other])
            if size_now > max_size:
                split_for_size += 1
                continue
            members.append(other)
            member_set.add(other)
            unassigned.discard(other)
        expanded: List[int] = []
        for m in members:
            expanded.extend(groups[m])
        clusters.append(sorted(set(expanded)))

    assignment: Dict[int, int] = {}
    for idx, members in enumerate(clusters):
        for m in members:
            assignment[m] = idx

    disagreements = 0.0
    for e in edges:
        same = assignment.get(e.a) == assignment.get(e.b)
        if e.weight > 0 and not same:
            disagreements += e.weight
        elif e.weight < 0 and same:
            disagreements += -e.weight

    violations = [
        (a, b) for a, b in (cannot_link or [])
        if assignment.get(int(a)) is not None and assignment.get(int(a)) == assignment.get(int(b))
    ]

    result = ClusterResult(
        clusters=clusters,
        singletons=len([c for c in clusters if len(c) == 1]),
        disagreements=disagreements,
        max_cluster_size=max((len(c) for c in clusters), default=0),
        constraint_violations=violations,
        split_for_size=split_for_size,
    )
    return result


def union_find_baseline(nodes: Iterable[int], edges: Sequence[Edge]) -> ClusterResult:
    """The naive transitive-closure clustering, kept so the demo can show side by side
    what it does to the largest cluster. This is what most implementations ship."""
    node_list = sorted(set(int(n) for n in nodes))
    parent = {n: n for n in node_list}

    def find(x: int) -> int:
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for e in edges:
        if e.weight > 0:
            rx, ry = find(e.a), find(e.b)
            if rx != ry:
                parent[ry] = rx

    groups: Dict[int, List[int]] = defaultdict(list)
    for n in node_list:
        groups[find(n)].append(n)
    clusters = [sorted(v) for v in groups.values()]
    return ClusterResult(
        clusters=clusters,
        singletons=len([c for c in clusters if len(c) == 1]),
        max_cluster_size=max((len(c) for c in clusters), default=0),
    )


def purity(result: ClusterResult, truth: Dict[int, str]) -> dict:
    """Cluster purity against a ground-truth group label. Catches the mega-cluster
    failure that pairwise F1 completely hides."""
    if not truth:
        return {"purity": None, "clusters_scored": 0}
    total, correct = 0, 0
    impure = 0
    for members in result.clusters:
        labels = [truth.get(m) for m in members if truth.get(m) is not None]
        if not labels:
            continue
        counts: Dict[str, int] = defaultdict(int)
        for label in labels:
            counts[label] += 1
        best = max(counts.values())
        correct += best
        total += len(labels)
        if len(counts) > 1:
            impure += 1
    return {
        "purity": round(correct / total, 4) if total else None,
        "clusters_scored": len(result.clusters),
        "impure_clusters": impure,
        "largest_cluster": result.max_cluster_size,
    }
