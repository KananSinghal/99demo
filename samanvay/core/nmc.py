"""The Common National Material Code.

India already operates a sovereign codification framework: the Directorate of
Standardisation runs National Codification Bureau India, a Tier-1 member of NATO's
AC/135, issuing item identification under the NATO Codification System against
India's two-digit NCB code. Anchoring there instead of minting a fresh sequence is
the strongest single design decision in this system.

    5306-72-014-7723 / 5
    ^^^^ ^^ ^^^^^^^^   ^
    |    |  |          check digit (Verhoeff - the scheme Aadhaar uses). Our
    |    |  |          addition, for keyboard entry; the 13-digit core is NSN-shaped.
    |    |  non-significant serial, PERMANENT, never reissued
    |    India's NCB code
    supply classification (NSC/FSC) - REVISABLE

The principle: identity is permanent, classification is a view. The commonest
failure in national coding schemes is the "intelligent code" - attributes buried in
the digits, rotting the moment a classification changes. Here every descriptive
thing hangs off the identity as a versioned facet, so the code never rots.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Dict, Optional

from .. import config
from . import classify

# ---------------------------------------------------------------- Verhoeff tables
_D = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 2, 3, 4, 0, 6, 7, 8, 9, 5),
    (2, 3, 4, 0, 1, 7, 8, 9, 5, 6),
    (3, 4, 0, 1, 2, 8, 9, 5, 6, 7),
    (4, 0, 1, 2, 3, 9, 5, 6, 7, 8),
    (5, 9, 8, 7, 6, 0, 4, 3, 2, 1),
    (6, 5, 9, 8, 7, 1, 0, 4, 3, 2),
    (7, 6, 5, 9, 8, 2, 1, 0, 4, 3),
    (8, 7, 6, 5, 9, 3, 2, 1, 0, 4),
    (9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
)
_P = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 5, 7, 6, 2, 8, 3, 0, 9, 4),
    (5, 8, 0, 3, 7, 9, 6, 1, 4, 2),
    (8, 9, 1, 6, 0, 4, 3, 5, 2, 7),
    (9, 4, 5, 3, 1, 2, 6, 8, 7, 0),
    (4, 2, 8, 6, 5, 7, 3, 9, 0, 1),
    (2, 7, 9, 3, 8, 0, 6, 4, 1, 5),
    (7, 0, 4, 6, 9, 1, 3, 2, 5, 8),
)
_INV = (0, 4, 3, 2, 1, 5, 6, 7, 8, 9)


def verhoeff_check_digit(number: str) -> int:
    """Compute the Verhoeff check digit for a digit string."""
    digits = [int(c) for c in re.sub(r"\D", "", number)][::-1]
    c = 0
    for i, d in enumerate(digits):
        c = _D[c][_P[(i + 1) % 8][d]]
    return _INV[c]


def verhoeff_validate(number_with_check: str) -> bool:
    digits = [int(c) for c in re.sub(r"\D", "", number_with_check)][::-1]
    c = 0
    for i, d in enumerate(digits):
        c = _D[c][_P[i % 8][d]]
    return c == 0


# ------------------------------------------------------------------------- code

@dataclass
class Nmc:
    code: str            # canonical rendering, e.g. "5306-72-014-7723/5"
    nsn: str             # the 13-digit NSN-shaped core, e.g. "5306721470723"
    nsc: str
    ncb: str
    serial: str
    check_digit: int

    def to_dict(self) -> dict:
        return asdict(self)


_NMC_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{3})-(\d{4})(?:\s*/\s*(\d))?$")


def mint(serial_number: int, nsc: str, ncb: Optional[str] = None) -> Nmc:
    """Mint an NMC from a monotonically increasing registry serial.

    The serial is non-significant on purpose: it encodes nothing, so nothing about
    it can become wrong. Identity never changes; the NSC facet may be revised.
    """
    ncb = (ncb or config.NCB_CODE).zfill(2)[:2]
    nsc = re.sub(r"\D", "", str(nsc) or "9999").zfill(4)[:4]
    serial = str(int(serial_number) % 10_000_000).zfill(7)
    nsn = f"{nsc}{ncb}{serial}"
    check = verhoeff_check_digit(nsn)
    code = f"{nsc}-{ncb}-{serial[:3]}-{serial[3:]}/{check}"
    return Nmc(code=code, nsn=nsn, nsc=nsc, ncb=ncb, serial=serial, check_digit=check)


def mint_for_class(serial_number: int, class_code: str, ncb: Optional[str] = None) -> Nmc:
    return mint(serial_number, classify.class_spec(class_code).get("nsc", "9999"), ncb)


def parse(code: str) -> Optional[Nmc]:
    raw = (code or "").strip().upper().replace(" ", "")
    m = _NMC_RE.match(raw)
    if not m:
        compact = re.sub(r"\D", "", raw)
        if len(compact) in (13, 14):
            nsc, ncb, serial = compact[:4], compact[4:6], compact[6:13]
            nsn = f"{nsc}{ncb}{serial}"
            check = verhoeff_check_digit(nsn)
            return Nmc(code=f"{nsc}-{ncb}-{serial[:3]}-{serial[3:]}/{check}",
                       nsn=nsn, nsc=nsc, ncb=ncb, serial=serial, check_digit=check)
        return None
    nsc, ncb, s1, s2, check = m.groups()
    serial = s1 + s2
    nsn = f"{nsc}{ncb}{serial}"
    expected = verhoeff_check_digit(nsn)
    return Nmc(
        code=f"{nsc}-{ncb}-{s1}-{s2}/{expected}",
        nsn=nsn, nsc=nsc, ncb=ncb, serial=serial,
        check_digit=int(check) if check is not None else expected,
    )


def validate(code: str) -> dict:
    parsed = parse(code)
    if parsed is None:
        return {"valid": False, "reason": "not a well-formed National Material Code"}
    raw = (code or "").strip()
    m = _NMC_RE.match(raw.upper().replace(" ", ""))
    stated = int(m.group(5)) if (m and m.group(5)) else None
    expected = verhoeff_check_digit(parsed.nsn)
    if stated is not None and stated != expected:
        return {
            "valid": False,
            "reason": f"check digit {stated} does not match; expected {expected} - likely a transcription error",
            "expected_check_digit": expected,
        }
    return {"valid": True, "nmc": parsed.to_dict()}


def facets(class_code: str) -> Dict[str, str]:
    """The versioned classification facets. Identity is the code; these are views.

      NSC / FSC      stores, provisioning, NCB interoperability
      UNSPSC         spend analytics, GeM alignment, category management
      HSN            GST, customs, e-invoicing
      eCl@ss         engineering property set (CFIHOS / ISO 15926 aligned)
    """
    return classify.facets(class_code)


def describe() -> dict:
    """Explain the scheme - used by the UI and by the pitch."""
    sample = mint(config.NMC_SERIAL_START, "5306")
    return {
        "example": sample.to_dict(),
        "segments": [
            {"segment": "NSC (4)", "example": sample.nsc, "property": "revisable",
             "source": "National / NATO Supply Classification group and class"},
            {"segment": "NCB (2)", "example": sample.ncb, "property": "fixed",
             "source": "India's national codification bureau code"},
            {"segment": "Serial (7)", "example": sample.serial, "property": "permanent, never reissued",
             "source": "non-significant sequence from the national registry"},
            {"segment": "Check (1)", "example": str(sample.check_digit), "property": "derived",
             "source": "Verhoeff check digit - the scheme Aadhaar uses. Our addition for keyboard entry."},
        ],
        "principle": "identity is permanent, classification is a versioned view",
        "note": (
            "The 13-digit core is NSN-shaped so it interoperates with the codification "
            "infrastructure the Republic already runs. Confirm India's NCB code against "
            "ddpdos.gov.in / ACodP-1 before publishing."
        ),
    }
