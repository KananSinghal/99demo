"""Abbreviation mining.

Every team hardcodes {"brg": "bearing"}. A hand-written dictionary can never hold
the shorthand one storekeeper at one location invented in 1997. So we mine it.

Method: find record pairs whose token multisets differ by exactly one token on each
side. Those two tokens are candidate synonyms. Score candidates by co-occurrence
count and by a pointwise-mutual-information style lift against their independent
frequencies, and keep pairs where one side is plausibly an abbreviation of the
other (subsequence, prefix, or consonant skeleton).
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

from . import data, text


def _is_plausible_abbrev(short: str, long: str) -> bool:
    """Is `short` a plausible abbreviation of `long`?"""
    if len(short) >= len(long):
        return False
    if len(short) < 3:          # two-character "abbreviations" are almost always noise
        return False
    if not short[0].isalpha() or not long[0].isalpha():
        return False
    if short[0] != long[0]:
        return False
    if long.startswith(short):                      # prefix: BRG / BRGSOMETHING
        return True
    # subsequence: BRG is a subsequence of BEARING
    it = iter(long)
    if all(ch in it for ch in short):
        return True
    # consonant skeleton: GALV / GALVANISED
    cons = "".join(c for c in long if c not in "AEIOU")
    return cons.startswith(short)


def _anchors(toks: Sequence[str], k: int = 3) -> List[str]:
    """The rarest, most distinctive tokens in a record - the ones that identify which
    other records describe the same thing. Used to find near-duplicates cheaply."""
    scored = sorted(
        {t for t in toks if len(t) >= 3},
        key=lambda t: (-len(t), t),
    )
    return scored[:k]


def mine_abbreviations(
    descriptions: Iterable[str],
    min_count: int = 2,
    min_lift: float = 1.2,
    max_pairs: int = 400,
    max_bucket: int = 60,
) -> Dict[str, str]:
    """Return {abbreviation: expansion} mined from the corpus.

    Two records that describe the same item in different house styles share most of
    their distinctive tokens. Index on those anchors, then compare the token sets of
    co-located records: where the symmetric difference is one token on each side,
    those two tokens are a synonym candidate.
    """
    rows: List[List[str]] = []
    for desc in descriptions:
        toks = text.content_tokens(desc)
        if 2 <= len(toks) <= 24:
            rows.append(toks)
    if len(rows) < 4:
        return {}

    freq = Counter(t for toks in rows for t in set(toks))
    total_rows = len(rows)

    # Inverted index on anchor tokens, so near-duplicates collide without an O(n^2) scan.
    buckets: Dict[str, List[int]] = defaultdict(list)
    sets: List[set] = [set(toks) for toks in rows]
    for i, toks in enumerate(rows):
        for anchor in _anchors(toks):
            buckets[anchor].append(i)

    candidates: Counter = Counter()
    compared: set = set()
    for _anchor, members in buckets.items():
        if len(members) < 2 or len(members) > max_bucket:
            continue
        for x in range(len(members)):
            for y in range(x + 1, len(members)):
                i, j = members[x], members[y]
                pair_key = (i, j) if i < j else (j, i)
                if pair_key in compared:
                    continue
                compared.add(pair_key)
                sa, sb = sets[i], sets[j]
                only_a, only_b = sa - sb, sb - sa
                shared = len(sa & sb)
                # A small symmetric difference against a solid shared backbone. Real
                # house-style variants rarely differ by exactly one token - they differ
                # by two or three, and the abbreviation is one of them.
                if not only_a or not only_b or shared < 3:
                    continue
                if len(only_a) > 2 or len(only_b) > 2:
                    continue
                for a in only_a:
                    for b in only_b:
                        short, long = (a, b) if len(a) < len(b) else (b, a)
                        if _is_plausible_abbrev(short, long):
                            candidates[(short, long)] += 1

    seed = data.load("abbreviations").get("seed", {})
    mined: Dict[str, str] = {}
    scored: List[Tuple[float, str, str, int]] = []
    for (short, long), count in candidates.items():
        if count < min_count or short in seed:
            continue
        p_joint = count / total_rows
        p_short = freq[short] / total_rows
        p_long = freq[long] / total_rows
        if p_short <= 0 or p_long <= 0:
            continue
        lift = p_joint / (p_short * p_long)
        if lift < min_lift:
            continue
        scored.append((math.log(lift) * count, short, long, count))

    scored.sort(reverse=True)
    for _score, short, long, _count in scored[:max_pairs]:
        if short not in mined:
            mined[short] = long
    return mined


def mine_and_persist(descriptions: Iterable[str], **kwargs) -> Dict[str, str]:
    """Mine, write to data/mined_abbreviations.json and invalidate the cache so the
    next normalisation call uses the new lexicon."""
    descriptions = list(descriptions)
    mined = mine_abbreviations(descriptions, **kwargs)
    data.save(
        "mined_abbreviations",
        {
            "_comment": "Mined from the ingested corpus by core.lexicon. Regenerated on every full ingest.",
            "corpus_size": len(descriptions),
            "mined": mined,
        },
    )
    data.invalidate("abbreviations")
    return mined
