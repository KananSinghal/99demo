"""Attribute extraction.

Grammar first: thread designations, dimensions, pressure classes, schedules and
standard references are regular languages. Parse them - do not ask a model.

Model second: only the residue, and only through a decoder constrained to the class
property dictionary, so a value that is not in the dictionary cannot be emitted.
(core.llm holds that optional hook; the default build never needs it.)

Abstain third: unparseable residue is recorded as residue, not guessed. Missing is
not the same as wrong, and a missing MANDATORY attribute makes a record ineligible
for auto-match by construction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple

from . import classify, data, kg, text

# ------------------------------------------------------------------- ISO 261 coarse pitch
COARSE_PITCH = {
    1.6: 0.35, 2: 0.4, 2.5: 0.45, 3: 0.5, 4: 0.7, 5: 0.8, 6: 1.0, 8: 1.25, 10: 1.5,
    12: 1.75, 14: 2.0, 16: 2.0, 18: 2.5, 20: 2.5, 22: 2.5, 24: 3.0, 27: 3.0, 30: 3.5,
    33: 3.5, 36: 4.0, 39: 4.0, 42: 4.5, 45: 4.5, 48: 5.0, 52: 5.0, 56: 5.5, 60: 5.5, 64: 6.0,
}

# ------------------------------------------------------------------- grammars
RE_METRIC_THREAD = re.compile(r"\bM\s?(\d{1,3}(?:\.\d+)?)\s*(?:[X×]\s*(\d(?:\.\d+)?))?(?![0-9A-Z])")
RE_IMPERIAL_THREAD = re.compile(r"\b(\d+(?:/\d+)?|\d+\.\d+)\s*\"?\s*-\s*(\d{1,2})\s*(UNC|UNF|UN|BSW|BSF)\b")
RE_PIPE_THREAD = re.compile(r"\b(\d+(?:/\d+)?)\s*\"?\s*(NPT|BSP|BSPT|BSPP)\b")
RE_LENGTH = re.compile(r"[X×]\s*(\d{1,5}(?:\.\d+)?)\s*(MM|CM|M|IN|INCH|\")?(?![A-Z0-9])")
RE_LENGTH_LABEL = re.compile(r"\b(?:LENGTH|LG|LGTH)\s*[:= ]\s*(\d{1,5}(?:\.\d+)?)\s*(MM|CM|M|IN)?\b")
RE_NOMINAL_SIZE = re.compile(r"\b(\d{1,3}(?:\.\d+)?|\d+\s?\d/\d|\d/\d)\s*(?:\"|IN\b|INCH\b|NB\b|NPS\b|DIA\b)")
RE_NOMINAL_MM = re.compile(r"\b(?:DN|NB)\s?(\d{2,4})\b")
RE_SCHEDULE = re.compile(r"\bSCH(?:EDULE)?\.?\s*(\d{1,3}S?|XXS|XS|STD)\b|\b(XXS|XS|STD)\b")
RE_PRESSURE_CLASS = re.compile(r"\b(\d{3,4})\s*#|\bCLASS\s*(\d{3,4})\b|\bCL\.?\s*(\d{3,4})\b|\b(PN\s?\d{1,3})\b|\b(\d{3,4})\s*LB\b")
RE_STANDARD = re.compile(r"\b(ISO|IS|ASTM|ASME|DIN|EN|BS|API|JIS|IEC|ANSI|SA)\s*[- ]?\s*([A-Z]?\d{1,5}(?:[.\-/]\d{1,4})*[A-Z]?)\b")
RE_BEARING = re.compile(r"\b([1-7]\d{2,4})\s?-?\s?(2RS1|2RSR|2RS|RS1|RS|2Z|ZZ|2RZ|Z|N|NR|K|M)?\s*(C[0-9])?\b")
RE_CABLE = re.compile(r"\b(\d{1,2})\s*C(?:ORE)?\s*[X×]\s*(\d{1,3}(?:\.\d+)?)\s*(?:SQ\.?\s?MM|MM2|MM\^2|SQMM)?\b")
RE_CABLE_ALT = re.compile(r"\b(\d{1,3}(?:\.\d+)?)\s*(?:SQ\.?\s?MM|MM2|SQMM)\b")
RE_VOLTAGE = re.compile(r"\b(\d{1,5}(?:\.\d+)?)\s*(KV|V)\b")
INSULATIONS = [("XLPE", "XLPE"), ("PVC", "PVC"), ("EPR", "EPR"), ("PILC", "PILC"), ("RUBBER", "RUBBER")]
RE_CORES_LABEL = re.compile(r"\b(\d{1,2})\s*CORE\b")

VALVE_TYPES = ["GATE", "GLOBE", "BALL", "CHECK", "NON RETURN", "BUTTERFLY", "NEEDLE", "PLUG", "SAFETY", "RELIEF", "DIAPHRAGM"]
FITTING_TYPES = ["ELBOW", "TEE", "REDUCER", "FLANGE", "COUPLING", "UNION", "NIPPLE", "CAP", "BEND", "CROSS", "PLUG", "BUSH"]
END_CONNECTIONS = [
    ("RING TYPE JOINT", "RTJ"), ("RAISED FACE", "RF"), ("FLAT FACE", "FF"), ("WELD NECK", "WN"),
    ("SLIP ON", "SO"), ("SOCKET WELD", "SW"), ("BUTT WELD", "BW"), ("BUTTWELD", "BW"),
    ("SCREWED", "SCRD"), ("THREADED", "SCRD"), ("FLANGED", "FLGD"), ("WAFER", "WAFER"), ("LUGGED", "LUG"),
]
OPERATIONS = [("HAND WHEEL", "HW"), ("HANDWHEEL", "HW"), ("GEAR OPERATED", "GO"), ("MOTOR OPERATED", "MO"),
              ("PNEUMATIC", "PNEU"), ("LEVER", "LEVER"), ("ACTUATED", "ACT")]
FINISHES = [("HOT DIP GALVANISED", "HDG"), ("GALVANISED", "GALV"), ("ZINC PLATED", "ZP"),
            ("ELECTRO GALVANISED", "EG"), ("PLAIN", "PLAIN"), ("BLACK", "BLACK"), ("PTFE COATED", "PTFE")]
HEADS = [("HEXAGON SOCKET", "SOCKET"), ("SOCKET", "SOCKET"), ("HEXAGON", "HEX"), ("HEX", "HEX"),
         ("SQUARE", "SQUARE"), ("COUNTERSUNK", "CSK"), ("CHEESE", "CHEESE"), ("PAN", "PAN"), ("STUD", "STUD")]
CONSTRUCTIONS = [("SEAMLESS", "SEAMLESS"), ("ERW", "ERW"), ("SAW", "SAW"), ("LSAW", "LSAW"), ("SPIRAL", "SPIRAL"), ("WELDED", "WELDED")]
ARMOURS = [("UNARMOURED", "UNARMOURED"), ("ARMOURED", "ARMOURED"), ("STEEL WIRE ARMOUR", "SWA"),
           ("STEEL STRIP ARMOUR", "STA"), ("SWA", "SWA"), ("STA", "STA")]
GASKET_TYPES = [("SPIRAL WOUND", "SPIRAL WOUND"), ("RING TYPE JOINT", "RTJ"), ("RING JOINT", "RTJ"),
                ("FULL FACE", "FULL FACE"), ("INNER BOLT CIRCLE", "IBC"), ("CAF", "CAF"), ("JOINTING SHEET", "SHEET")]
NUT_STYLES = [("NYLOC", "NYLOC"), ("NYLON INSERT", "NYLOC"), ("LOCK", "LOCK"), ("DOME", "DOME"),
              ("WING", "WING"), ("HEAVY HEXAGON", "HEAVY HEX"), ("HEXAGON", "HEX")]


@dataclass
class Attribute:
    key: str
    value: str
    source: str            # "grammar" | "kg" | "inferred" | "dictionary" | "model"
    evidence: str = ""
    confidence: float = 1.0
    numeric: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Extraction:
    class_code: str
    attributes: Dict[str, Attribute] = field(default_factory=dict)
    residue: List[str] = field(default_factory=list)
    missing_mandatory: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def values(self) -> Dict[str, str]:
        return {k: a.value for k, a in self.attributes.items()}

    def numerics(self) -> Dict[str, Optional[float]]:
        return {k: a.numeric for k, a in self.attributes.items()}

    def to_dict(self) -> dict:
        return {
            "class_code": self.class_code,
            "attributes": {k: a.to_dict() for k, a in self.attributes.items()},
            "residue": self.residue,
            "missing_mandatory": self.missing_mandatory,
            "notes": self.notes,
        }

    @staticmethod
    def from_dict(payload: dict) -> "Extraction":
        ex = Extraction(class_code=payload.get("class_code", "GENERIC"))
        for key, raw in (payload.get("attributes") or {}).items():
            ex.attributes[key] = Attribute(**raw)
        ex.residue = list(payload.get("residue") or [])
        ex.missing_mandatory = list(payload.get("missing_mandatory") or [])
        ex.notes = list(payload.get("notes") or [])
        return ex


# ------------------------------------------------------------------ helpers

def _frac(value: str) -> Optional[float]:
    value = value.strip()
    try:
        if " " in value:                       # 1 1/2
            whole, rest = value.split(" ", 1)
            return float(whole) + _frac(rest or "0")
        if "/" in value:
            num, den = value.split("/", 1)
            return float(num) / float(den) if float(den) else None
        return float(value)
    except (ValueError, ZeroDivisionError, TypeError):
        return None


def _to_mm(value: float, unit: Optional[str]) -> float:
    u = (unit or "MM").upper()
    if u in ("M",):
        return value * 1000.0
    if u in ("CM",):
        return value * 10.0
    if u in ("IN", "INCH", '"'):
        return value * 25.4
    return value


def _first(hay: str, table) -> Optional[Tuple[str, str]]:
    """Longest-match lookup over a (phrase, canonical) table."""
    best: Tuple[int, Optional[Tuple[str, str]]] = (0, None)
    for phrase, canon in table:
        needle = f" {phrase} "
        if needle in hay and len(phrase) > best[0]:
            best = (len(phrase), (canon, phrase))
    return best[1]


def _set(ex: Extraction, key: str, value, source: str, evidence: str = "",
         confidence: float = 1.0, numeric: Optional[float] = None) -> None:
    if value in (None, "", []):
        return
    if key in ex.attributes and ex.attributes[key].confidence >= confidence:
        return
    ex.attributes[key] = Attribute(
        key=key, value=str(value), source=source, evidence=evidence,
        confidence=confidence, numeric=numeric,
    )


# ------------------------------------------------------------------ shared grammar

def _extract_thread(ex: Extraction, hay: str) -> None:
    m = RE_METRIC_THREAD.search(hay)
    if m:
        dia = float(m.group(1))
        pitch = float(m.group(2)) if m.group(2) else COARSE_PITCH.get(dia)
        if pitch:
            src = "grammar" if m.group(2) else "inferred"
            ev = m.group(0).strip() if m.group(2) else f"{m.group(0).strip()} + ISO 261 coarse pitch series"
            _set(ex, "thread", f"M{dia:g}x{pitch:g}", src, ev, 1.0 if m.group(2) else 0.92, dia)
        else:
            _set(ex, "thread", f"M{dia:g}", "grammar", m.group(0).strip(), 0.85, dia)
        return
    m = RE_IMPERIAL_THREAD.search(hay)
    if m:
        dia = _frac(m.group(1))
        _set(ex, "thread", f'{m.group(1)}"-{m.group(2)}{m.group(3)}', "grammar", m.group(0).strip(),
             1.0, (dia * 25.4) if dia else None)
        ex.notes.append("imperial thread series - not interchangeable with a metric thread of similar diameter")
        return
    m = RE_PIPE_THREAD.search(hay)
    if m:
        _set(ex, "thread", f'{m.group(1)}"{m.group(2)}', "grammar", m.group(0).strip())


def _extract_length(ex: Extraction, hay: str) -> None:
    m = RE_LENGTH_LABEL.search(hay)
    if m:
        val = _frac(m.group(1))
        if val is not None:
            mm = _to_mm(val, m.group(2))
            _set(ex, "length_mm", f"{mm:g}", "grammar", m.group(0).strip(), 1.0, mm)
            return
    for m in RE_LENGTH.finditer(hay):
        val = _frac(m.group(1))
        if val is None:
            continue
        mm = _to_mm(val, m.group(2))
        if 1 <= mm <= 5000:
            _set(ex, "length_mm", f"{mm:g}", "grammar", m.group(0).strip(), 0.95, mm)
            return


def _extract_nominal_size(ex: Extraction, hay: str) -> None:
    m = RE_NOMINAL_MM.search(hay)
    if m:
        mm = float(m.group(1))
        _set(ex, "nominal_size", f"DN{mm:g}", "grammar", m.group(0).strip(), 1.0, mm)
        return
    m = RE_NOMINAL_SIZE.search(hay)
    if m:
        val = _frac(m.group(1))
        if val is not None and 0 < val <= 120:
            _set(ex, "nominal_size", f'{m.group(1).strip()}"', "grammar", m.group(0).strip(), 1.0, val * 25.4)


def _extract_schedule(ex: Extraction, hay: str) -> None:
    m = RE_SCHEDULE.search(hay)
    if m:
        raw = (m.group(1) or m.group(2) or "").strip()
        if raw:
            value = raw if raw in ("XS", "XXS", "STD") else f"SCH{raw}"
            _set(ex, "schedule", value, "grammar", m.group(0).strip())


def _extract_pressure_class(ex: Extraction, hay: str) -> None:
    m = RE_PRESSURE_CLASS.search(hay)
    if m:
        raw = next((g for g in m.groups() if g), None)
        if raw:
            _set(ex, "pressure_class", raw.replace(" ", ""), "grammar", m.group(0).strip())


def _extract_standard(ex: Extraction, hay: str) -> None:
    best: Optional[str] = None
    for m in RE_STANDARD.finditer(hay):
        body, num = m.group(1), m.group(2)
        cand = f"{body} {num}"
        # Skip matches that are really a material grade (A105, A106) - kg owns those.
        if kg.family_of(cand) and body in ("ASTM", "SA", "IS", "EN"):
            continue
        if best is None or len(cand) > len(best):
            best = cand
    if best:
        canon, note = kg.canonical_standard(best)
        if note:
            ex.notes.append(note)
            _set(ex, "standard", canon, "kg", note, 0.95)
        else:
            _set(ex, "standard", best, "grammar", best)


def _extract_material(ex: Extraction, raw: str, key: str = "material") -> None:
    families = kg.find_designations(raw)
    if not families:
        return
    # Prefer a specific grade over a generic family (A105 over CARBON STEEL).
    specific = [f for f in families if f != "CS_GENERIC"]
    fam = specific[0] if specific else families[0]
    _set(ex, key, fam, "kg", f"{kg.label(fam)} ({kg.citation_for(fam)})", 0.98)
    if len(specific) > 1 and key == "material":
        ex.notes.append(
            "more than one material designation present: " + ", ".join(kg.label(f) for f in specific[:3])
        )


def _extract_service(ex: Extraction, raw: str) -> None:
    quals = kg.find_service_qualifiers(raw)
    if quals:
        _set(ex, "service", "+".join(sorted(quals)), "kg",
             " | ".join(kg.service_rule(q).get("citation", "") for q in sorted(quals)), 1.0)


def _extract_table(ex: Extraction, hay: str, key: str, table, source: str = "grammar") -> None:
    hit = _first(hay, table)
    if hit:
        _set(ex, key, hit[0], source, hit[1])


def _extract_word(ex: Extraction, hay: str, key: str, words: List[str], glue_suffix: str = "") -> None:
    for w in sorted(words, key=len, reverse=True):
        if f" {w} " in hay:
            _set(ex, key, w, "grammar", w)
            return
    if not glue_suffix:
        return
    # Fall back to a glued match: "GATEVALVE" is "GATE" + "VALVE" run together with
    # no space, ordinary ERP data-entry noise (this exact case is in the synthetic
    # corpus). Without this, the type word is silently dropped - not "unknown", just
    # gone - and a check valve and a globe valve of the same size can compare as
    # EQUIVALENT instead of DISTINCT. Scoped to this one call site (only VALVE_TYPES
    # passes a glue_suffix), so the risk of a stray false match is negligible.
    hay_glued = hay.replace(" ", "")
    for w in sorted(words, key=lambda x: len(x.replace(" ", "")), reverse=True):
        glued_word = w.replace(" ", "")
        if f"{glued_word}{glue_suffix}" in hay_glued:
            _set(ex, key, w, "grammar", f"{glued_word}{glue_suffix} (glued)")
            return


# ------------------------------------------------------------------ per class

def _bearing_bore(designation: str) -> Optional[float]:
    """ISO 15 bore code: 00=10mm, 01=12, 02=15, 03=17, 04+ = code x 5."""
    d = designation.strip()
    if len(d) < 3 or not d.isdigit():
        return None
    code = d[-2:]
    try:
        n = int(code)
    except ValueError:
        return None
    mapping = {0: 10.0, 1: 12.0, 2: 15.0, 3: 17.0}
    if n in mapping:
        return mapping[n]
    if 4 <= n <= 96:
        return float(n * 5)
    return None


def _extract_bearing(ex: Extraction, hay: str) -> None:
    best = None
    for m in RE_BEARING.finditer(hay):
        desig = m.group(1)
        if len(desig) < 3:
            continue
        if _bearing_bore(desig) is None:
            continue
        if best is None or len(desig) > len(best.group(1)):
            best = m
    if not best:
        return
    desig, seal, clearance = best.group(1), best.group(2), best.group(3)
    full = desig + (f"-{seal}" if seal else "")
    _set(ex, "designation", full, "grammar", best.group(0).strip())
    bore = _bearing_bore(desig)
    if bore is not None:
        _set(ex, "bore_mm", f"{bore:g}", "grammar",
             f"ISO 15 bore code '{desig[-2:]}' in designation {desig}", 1.0, bore)
    _set(ex, "series", desig[:-2], "grammar", f"series prefix of {desig}")
    if seal:
        _set(ex, "seal", seal, "grammar", seal)
    if clearance:
        _set(ex, "clearance", clearance, "grammar", clearance)


def _extract_cable(ex: Extraction, hay: str) -> None:
    m = RE_CABLE.search(hay)
    if m:
        cores = float(m.group(1))
        csa = float(m.group(2))
        _set(ex, "cores", f"{cores:g}", "grammar", m.group(0).strip(), 1.0, cores)
        _set(ex, "csa_sqmm", f"{csa:g}", "grammar", m.group(0).strip(), 1.0, csa)
    else:
        mc = RE_CORES_LABEL.search(hay)
        if mc:
            cores = float(mc.group(1))
            _set(ex, "cores", f"{cores:g}", "grammar", mc.group(0).strip(), 1.0, cores)
        ma = RE_CABLE_ALT.search(hay)
        if ma:
            csa = float(ma.group(1))
            _set(ex, "csa_sqmm", f"{csa:g}", "grammar", ma.group(0).strip(), 1.0, csa)
    mv = RE_VOLTAGE.search(hay)
    if mv:
        val, unit = float(mv.group(1)), mv.group(2)
        volts = val * 1000 if unit == "KV" else val
        canon = f"{val:g}KV" if unit == "KV" else f"{val:g}V"
        if abs(volts - 1100) < 1 or abs(volts - 1000) < 1:
            canon = "1100V"
        _set(ex, "voltage", canon, "grammar", mv.group(0).strip(), 1.0, volts)
    # Insulation is a closed vocabulary - never let a conductor designation win it.
    _extract_table(ex, hay, "insulation", INSULATIONS, source="kg")
    for cond in ("COPPER", "ALUMINIUM", "ALUMINUM", "CU", "AL"):
        if f" {cond} " in hay:
            _set(ex, "conductor", "CU" if cond in ("COPPER", "CU") else "AL", "kg", cond)
            break
    _extract_table(ex, hay, "armour", ARMOURS)


# ------------------------------------------------------------------ entry point

def extract(description: str, class_code: Optional[str] = None, extra: str = "") -> Extraction:
    """Extract the attribute vector for one material record."""
    raw = f"{description} {extra}".strip()
    if class_code is None:
        class_code = classify.classify(description, extra).class_code

    hay = " " + text.normalise(raw) + " "
    hay_clean = " " + text.clean(raw) + " "
    both = hay + hay_clean
    ex = Extraction(class_code=class_code)

    # Shared grammar, applied to every class - cheap and never wrong when it fires.
    _extract_standard(ex, both)
    _extract_service(ex, raw)

    if class_code in ("FASTENER_BOLT", "FASTENER_NUT"):
        _extract_thread(ex, both)
        _extract_material(ex, raw)
        _extract_table(ex, hay, "finish", FINISHES)
        if class_code == "FASTENER_BOLT":
            _extract_length(ex, both)
            _extract_table(ex, hay, "head", HEADS)
            if " FULL THREAD " in hay or " FULL THD " in hay_clean or " FULLY THREADED " in hay:
                _set(ex, "thread_length", "FULL", "grammar", "FULL THREAD")
            elif " PART THREAD " in hay or " PARTIALLY THREADED " in hay:
                _set(ex, "thread_length", "PARTIAL", "grammar", "PART THREAD")
        else:
            _extract_table(ex, hay, "style", NUT_STYLES)
        for fam in kg.find_designations(raw):
            if fam.startswith("FC_A"):
                _set(ex, "ss_class", kg.index().kg["families"][fam]["aliases"][0], "kg", kg.citation_for(fam))
            if fam.startswith("PC_") or fam in ("B7", "B7M"):
                _set(ex, "grade", kg.index().kg["families"][fam]["aliases"][0], "kg", kg.citation_for(fam))

    elif class_code == "PIPE":
        _extract_nominal_size(ex, both)
        _extract_schedule(ex, both)
        _extract_material(ex, raw)
        _extract_table(ex, hay, "construction", CONSTRUCTIONS)
        _extract_table(ex, hay, "end_connection", END_CONNECTIONS)
        _extract_table(ex, hay, "finish", FINISHES)

    elif class_code == "PIPE_FITTING":
        _extract_word(ex, hay, "fitting_type", FITTING_TYPES)
        _extract_nominal_size(ex, both)
        _extract_pressure_class(ex, both)
        _extract_schedule(ex, both)
        _extract_material(ex, raw)
        _extract_table(ex, hay, "end_connection", END_CONNECTIONS)
        m = re.search(r"\b(45|90|180)\s*(?:DEG|DEGREE|°)?\b", hay)
        if m and ex.attributes.get("fitting_type", Attribute("", "", "")).value in ("ELBOW", "BEND"):
            _set(ex, "angle", f"{m.group(1)}DEG", "grammar", m.group(0).strip())

    elif class_code == "VALVE":
        _extract_word(ex, hay, "valve_type", VALVE_TYPES, glue_suffix="VALVE")
        _extract_nominal_size(ex, both)
        _extract_pressure_class(ex, both)
        _extract_material(ex, raw)
        _extract_table(ex, hay, "end_connection", END_CONNECTIONS)
        _extract_table(ex, hay, "operation", OPERATIONS)
        m = re.search(r"\bTRIM\s*[:= ]?\s*([A-Z0-9 ]{2,12})", hay)
        if m:
            _extract_material(ex, m.group(1), "trim")

    elif class_code == "BEARING":
        _extract_bearing(ex, both)
        _extract_material(ex, raw)

    elif class_code == "CABLE":
        _extract_cable(ex, both)

    elif class_code == "GASKET":
        _extract_table(ex, hay, "gasket_type", GASKET_TYPES)
        _extract_nominal_size(ex, both)
        _extract_pressure_class(ex, both)
        # A spiral wound gasket names two materials: the metallic winding and the
        # non-metallic filler. Split them rather than letting one overwrite the other.
        fams = kg.find_designations(raw)
        fillers = [f for f in fams if f in ("GRAPHITE", "PTFE", "NBR", "EPDM")]
        windings = [f for f in fams if f.startswith("SS") or f in ("CS_GENERIC", "CF8M")]
        if fillers:
            _set(ex, "material", fillers[0], "kg", kg.citation_for(fillers[0]), 0.98)
        if windings:
            _set(ex, "winding", windings[0], "kg", kg.citation_for(windings[0]), 0.98)
        if not fillers and not windings:
            _extract_material(ex, raw)

    else:  # GENERIC
        toks = text.content_tokens(raw)
        if toks:
            _set(ex, "noun", toks[0], "grammar", "first content token")
        _extract_material(ex, raw)

    # Residue: normalised content tokens not accounted for by any extracted attribute
    # or by the class's own keywords. Residue is recorded, never guessed at.
    consumed = set()
    for attr in ex.attributes.values():
        consumed.update(text.content_tokens(attr.evidence or ""))
        consumed.update(text.content_tokens(attr.value))
        consumed.update(re.sub(r"[^A-Z0-9]", "", text.basic(attr.value)) for _ in (0,))
    for kw in classify.class_spec(class_code).get("keywords", []):
        consumed.update(text.content_tokens(kw))
    squashed = {re.sub(r"[^A-Z0-9]", "", c) for c in consumed if c}
    ex.residue = [
        t for t in text.content_tokens(text.normalise(raw))
        if t not in consumed and re.sub(r"[^A-Z0-9]", "", t) not in squashed
    ][:24]

    ex.missing_mandatory = [
        key for key in classify.mandatory_keys(class_code) if key not in ex.attributes
    ]
    if ex.missing_mandatory:
        ex.notes.append(
            "missing mandatory attribute(s): " + ", ".join(ex.missing_mandatory)
            + " - this record cannot be auto-matched as IDENTICAL"
        )
    return ex


def coverage(ex: Extraction) -> float:
    """Share of the class's declared attributes that were actually extracted."""
    specs = classify.attribute_specs(ex.class_code)
    if not specs:
        return 0.0
    return round(len(ex.attributes) / len(specs), 4)
