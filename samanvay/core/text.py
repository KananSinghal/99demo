"""Text normalisation primitives.

Material descriptions are not English prose. They are compressed, punctuation-free
engineering shorthand written by different people over thirty years. Everything
here is deterministic and reversible enough to explain in an evidence card.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable, List

from . import data

# Separators that carry no meaning in a material description.
_SEP_RE = re.compile(r"[,\;\|\t\n\r\\]+")
_MULTISPACE_RE = re.compile(r"\s+")
# Keep . - # " ' + % / and alphanumerics; they all carry meaning in this domain
# (A106 GR.B, 150#, 1/2", M12x1.75, S/S, W/O). Dropping "/" silently destroys every
# imperial fraction in the corpus, which is how 1/2" becomes 2".
_KEEP_RE = re.compile(r"[^A-Z0-9\.\-#\"'\+%×x/\s]")
_X_RE = re.compile(r"(?<=[0-9])\s*[xX×]\s*(?=[0-9])")
# Cable core notation: 3CX95 / 3C X 95 -> "3C X 95" so the blocker sees the same tokens.
_CORE_X_RE = re.compile(r"(?<=\d)(C)\s*[xX×]\s*(?=\d)")
_UNIT_GLUE_RE = re.compile(r"(?<=[0-9])(MM|CM|KG|MTR|SQMM|MM2|KV|V|NB|IN|INCH)\b")


def strip_accents(value: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", value) if not unicodedata.combining(c))


def basic(value: str) -> str:
    """Upper-case, de-accent, collapse separators. The first thing done to any string."""
    if value is None:
        return ""
    out = strip_accents(str(value)).upper()
    out = out.replace("×", "X")  # multiplication sign
    out = _SEP_RE.sub(" ", out)
    out = _MULTISPACE_RE.sub(" ", out).strip()
    return out


def transliterate(value: str) -> str:
    table = data.transliteration()
    if not table:
        return value
    out = value
    for src, dst in table.items():
        if src in out:
            out = out.replace(src, f" {dst} ")
    return out


def clean(value: str) -> str:
    """Normalise to the canonical comparison surface: upper case, ASCII, spaced
    dimensions, glued units separated, noise punctuation removed."""
    out = basic(transliterate(str(value or "")))
    out = _KEEP_RE.sub(" ", out)
    out = _CORE_X_RE.sub(r"\1 X ", out)
    out = _X_RE.sub(" X ", out)
    out = _UNIT_GLUE_RE.sub(r" \1", out)
    out = out.replace(".", ". ").replace("- ", "-")
    out = _MULTISPACE_RE.sub(" ", out).strip()
    # Re-join decimal numbers broken by the dot spacing above (1. 75 -> 1.75)
    out = re.sub(r"(\d)\.\s+(\d)", r"\1.\2", out)
    out = re.sub(r"\s+\.\s*", " ", out)
    return _MULTISPACE_RE.sub(" ", out).strip()


def tokens(value: str) -> List[str]:
    return [t for t in clean(value).split(" ") if t]


def content_tokens(value: str) -> List[str]:
    noise = data.noise_tokens()
    return [t for t in tokens(value) if t not in noise and len(t) > 1]


def expand_abbreviations(value: str) -> str:
    """Expand the merged (seed + mined) lexicon, token by token."""
    lex = data.abbreviations()
    out = []
    for tok in tokens(value):
        out.append(lex.get(tok, tok))
    return _MULTISPACE_RE.sub(" ", " ".join(out)).strip()


def normalise(value: str) -> str:
    """Full normalisation: clean, then expand abbreviations. This is the string the
    embedder and the blocker see; the original is always retained on the record."""
    return expand_abbreviations(value)


# ------------------------------------------------------------------- similarity

def char_ngrams(value: str, lo: int = 3, hi: int = 5) -> List[str]:
    padded = f" {value} "
    grams: List[str] = []
    for n in range(lo, hi + 1):
        if len(padded) < n:
            continue
        grams.extend(padded[i:i + n] for i in range(len(padded) - n + 1))
    return grams


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def token_jaccard(a: str, b: str) -> float:
    return jaccard(content_tokens(a), content_tokens(b))


def trigram_similarity(a: str, b: str) -> float:
    return jaccard(char_ngrams(clean(a), 3, 3), char_ngrams(clean(b), 3, 3))


def levenshtein(a: str, b: str, cap: int = 64) -> int:
    """Iterative Levenshtein with a small cap; used only on short designations."""
    a, b = a[:cap], b[:cap]
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def completeness(description: str) -> float:
    """0..1 information score. Records near zero are quarantined rather than guessed at
    - no model recovers information that was never written down."""
    toks = content_tokens(description)
    if not toks:
        return 0.0
    n_tokens = len(toks)
    n_chars = len(clean(description))
    has_digit = any(any(c.isdigit() for c in t) for t in toks)
    distinct = len(set(toks))
    score = 0.0
    score += min(n_tokens / 8.0, 1.0) * 0.40
    score += min(n_chars / 45.0, 1.0) * 0.25
    score += 0.20 if has_digit else 0.0
    score += min(distinct / 6.0, 1.0) * 0.15
    return round(min(score, 1.0), 4)
