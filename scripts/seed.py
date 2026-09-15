#!/usr/bin/env python3
"""Synthetic multi-CPSE corpus generator.

SYNTHETIC DATA. The CPSE names are real organisations across Oil & Gas, Power, Steel,
Mining and Heavy Engineering, used as labels so the demo reads realistically. Every material
code, price, stock figure and vendor reference in here is invented. Nothing in this
file is, or claims to be, real CPSE data.

The corpus is built the way the real problem looks:
  - the same physical item written differently by different organisations
  - different units of measure for the same item (the DRUM / M trap)
  - vendor material numbers that silently identify the same item across CPSEs
  - near-miss pairs planted deliberately (6205 vs 6206, M12 vs 1/2"UNC, +/- NACE)
  - low-information records that must be quarantined, not guessed
  - engineered-to-order items that must be excluded, not harmonised
and it ships GROUND TRUTH, so precision and recall are measurable rather than claimed.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CPSES = [
    ("ONGC", "Oil and Natural Gas Corporation", ["Mumbai High", "Hazira", "Ankleshwar"]),
    ("IOCL", "Indian Oil Corporation", ["Panipat", "Paradip", "Mathura"]),
    ("BPCL", "Bharat Petroleum Corporation", ["Kochi", "Mumbai Refinery"]),
    ("HPCL", "Hindustan Petroleum Corporation", ["Visakhapatnam", "Mumbai"]),
    ("GAIL", "GAIL India", ["Vijaipur", "Pata"]),
    ("OIL", "Oil India", ["Duliajan", "Jodhpur"]),
    ("CPCL", "Chennai Petroleum Corporation", ["Manali", "Nagapattinam"]),
    ("NTPC", "NTPC Limited", ["Vindhyachal", "Korba", "Ramagundam"]),
    ("NHPC", "NHPC Limited", ["Chamera", "Uri"]),
    ("SAIL", "Steel Authority of India", ["Bhilai", "Rourkela", "Bokaro"]),
    ("RINL", "Rashtriya Ispat Nigam", ["Visakhapatnam Steel Plant"]),
    ("CIL", "Coal India", ["Dhanbad", "Singrauli"]),
    ("NMDC", "NMDC Limited", ["Kirandul", "Bacheli"]),
    ("BHEL", "Bharat Heavy Electricals", ["Haridwar", "Tiruchirappalli"]),
]

# Local store shorthand that is NOT in the seed lexicon. A hand-written dictionary
# can never contain one storekeeper's private abbreviations; the miner has to find
# them in the corpus. These exist so that capability is actually exercised.
LOCAL_SHORTHAND = {
    "SEAMLESS": "SMLESS", "SPIRAL": "SPRL", "WOUND": "WND", "BEARING": "BRNG",
    "HEXAGONAL": "HEXGNL", "THREADED": "THRDED", "STAINLESS": "STNLSS",
    "FLANGED": "FLNGD", "SCHEDULE": "SCHDL", "GRAPHITE": "GRPHT",
    "ARMOURED": "ARMRD", "GROOVE": "GRVE", "WELD": "WLD",
}


def _localise(value: str) -> str:
    out = value
    for long, short in LOCAL_SHORTHAND.items():
        out = out.replace(long, short)
    return out


# Rendering styles: the same item, written by seven different materials departments.
STYLES = [
    lambda p: p["a"],
    lambda p: p["b"],
    lambda p: p["c"],
    lambda p: p["a"].replace(" ", ",", 2),
    lambda p: p["b"].upper(),
    lambda p: p["c"].replace("X", " X "),
    lambda p: _localise(p["c"]),
]

VENDORS = [
    ("0000512345", "SKF INDIA"), ("0000518822", "L&T VALVES"), ("0000521190", "POLYCAB"),
    ("0000530471", "SUNDRAM FASTENERS"), ("0000540903", "JINDAL SAW"), ("0000551238", "KIRLOSKAR"),
]


def _bolt(dia, length, material, grade, std):
    return {
        "a": f"HEX BOLT M{dia}X{length} {material} FULL THD",
        "b": f"BOLT,HEXAGONAL,M{dia} X {length}MM,{grade},{std}",
        "c": f"BOLT HEX HD M{dia}X{length} {material} {std} FULLY THREADED",
        "class": "FASTENER_BOLT", "uoms": [("NO", 1), ("EA", 1), ("BOX", 100)],
        "price": 8 + dia * 0.9 + length * 0.05,
    }


def _pipe(size, sch, grade):
    return {
        "a": f'PIPE {size}" SCH {sch} {grade} SMLS BW',
        "b": f"PIPE {size} IN SCH{sch} ASTM {grade} SEAMLESS BUTT WELD",
        "c": f'SEAMLESS PIPE {size}" SCHEDULE {sch} {grade} BW ENDS',
        "class": "PIPE", "uoms": [("M", 1), ("MTR", 1), ("COIL", 100)],
        "price": 900 + size * 210 + int(str(sch).strip("S") or 40) * 3.2,
    }


def _valve(vtype, size, cls, body, nace=False):
    n = " NACE MR0175" if nace else ""
    return {
        "a": f'{vtype} VALVE {size}" {cls}# {body} FLGD RF HW{n}',
        "b": f"{vtype} VALVE {size} IN CLASS {cls} {body} FLANGED RAISED FACE HANDWHEEL{n}",
        "c": f'{vtype}VALVE {size}" CL{cls} BODY {body} FLGD RF HAND WHEEL{n}',
        "class": "VALVE", "uoms": [("NO", 1), ("EA", 1)],
        "price": 14000 + size * 2600 + cls * 42,
    }


def _bearing(desig, seal):
    return {
        "a": f"BEARING {desig}-{seal}",
        "b": f"BRG {desig} {seal}",
        "c": f"BALL BEARING {desig}{seal} DEEP GROOVE",
        "class": "BEARING", "uoms": [("NO", 1), ("EA", 1), ("PKT", 10)],
        "price": 180 + int(desig[-2:]) * 12,
    }


def _cable(cores, csa, volt, cond, armour):
    return {
        "a": f"CABLE XLPE {volt} {cores}C X {csa} SQMM {cond} {armour}",
        "b": f"CABLE,XLPE,{volt},{cores}CX{csa}MM2,{cond},{armour}",
        "c": f"XLPE CABLE {volt} {cores} CORE X {csa} SQMM {cond} {armour}",
        "class": "CABLE", "uoms": [("M", 1), ("MTR", 1), ("DRUM", 500)],
        "price": 42 + csa * 5.1 + cores * 9,
    }


def _gasket(size, cls, filler):
    return {
        "a": f'SPIRAL WOUND GASKET {size}" {cls}# SS316 {filler}',
        "b": f"GASKET SPIRAL WOUND {size} IN CLASS {cls} SS316 {filler} ASME B16.20",
        "c": f'SWG {size}" CL{cls} SS316 WINDING {filler} FILLER',
        "class": "GASKET", "uoms": [("NO", 1), ("EA", 1), ("SET", 8)],
        "price": 210 + size * 64 + cls * 0.7,
    }


def _fitting(ftype, size, cls, material):
    return {
        "a": f'{ftype} {size}" {cls}# {material} BW',
        "b": f"{ftype} {size} IN CLASS {cls} ASTM {material} BUTT WELD",
        "c": f'{ftype} {size}" CL{cls} {material} BUTTWELD ENDS',
        "class": "PIPE_FITTING", "uoms": [("NO", 1), ("EA", 1)],
        "price": 450 + size * 180,
    }


def build_prototypes():
    protos = []
    for dia, length in [(8, 40), (10, 40), (12, 50), (12, 60), (16, 60), (16, 80), (20, 80), (24, 100)]:
        protos.append(_bolt(dia, length, "SS316", "A4-70", "ISO4017"))
        protos.append(_bolt(dia, length, "SS304", "A2-70", "ISO4017"))
    for size in [2, 3, 4, 6, 8, 10, 12]:
        for sch in [40, 80]:
            protos.append(_pipe(size, sch, "A106 GR.B"))
    for vtype in ["GATE", "GLOBE", "BALL", "CHECK"]:
        for size in [2, 4, 6, 8]:
            for cls in [150, 300]:
                protos.append(_valve(vtype, size, cls, "A216 WCB"))
    protos.append(_valve("GATE", 6, 150, "A216 WCB", nace=True))
    protos.append(_valve("BALL", 4, 300, "A216 WCB", nace=True))
    for desig in ["6203", "6204", "6205", "6206", "6207", "6208", "6305", "6306"]:
        for seal in ["2RS", "ZZ"]:
            protos.append(_bearing(desig, seal))
    for cores, csa in [(1, 300), (3, 95), (3, 185), (4, 16), (4, 25)]:
        protos.append(_cable(cores, csa, "1.1KV", "CU", "ARMOURED"))
        protos.append(_cable(cores, csa, "1.1KV", "CU", "UNARMOURED"))
    for size in [2, 4, 6, 8]:
        for cls in [150, 300]:
            protos.append(_gasket(size, cls, "GRAPHITE"))
    for ftype in ["ELBOW", "TEE", "REDUCER"]:
        for size in [4, 6, 8]:
            protos.append(_fitting(ftype, size, 150, "A234 WPB"))
    return protos


LOW_INFO = ["BOLT", "SPARE", "ITEM", "PART", "MISC", "N/A", "ASSY", "SET", "-", "RUBBER"]
ENGINEERED = [
    "PUMP CASING AS PER DRAWING NO 4471-B",
    "SKID MOUNTED METERING PACKAGE AS PER SPEC NO MP-2201",
    "CUSTOM BUILT HEAT EXCHANGER TUBE BUNDLE AS PER DRG 9912",
    "FABRICATED TO ORDER STRUCTURAL SUPPORT ASSEMBLY",
    "TURNKEY COMPRESSOR PACKAGE ENGINEERED TO ORDER",
]


def generate(n_groups=180, seed=20260914):
    rng = random.Random(seed)
    protos = build_prototypes()
    rng.shuffle(protos)
    protos = protos[:n_groups]

    records = []
    truth_groups = {}
    rid = 1000

    for gi, proto in enumerate(protos):
        group_id = f"G{gi:04d}"
        n_cpses = rng.choice([2, 2, 3, 3, 4, 5])
        chosen = rng.sample(CPSES, n_cpses)
        vendor = rng.choice(VENDORS) if rng.random() < 0.55 else None
        vendor_matl = f"{proto['class'][:3]}-{gi:04d}" if vendor else None
        base_price = proto["price"]

        # Plant a deliberate redeployment opportunity in one group in three: one CPSE
        # holds idle stock, another has live demand. This is the real pattern the
        # directed-substitutability screen exists to find, so the corpus has to
        # contain it rather than leave it to chance.
        plant_redeploy = (gi % 3 == 0) and n_cpses >= 2
        holder_idx = 0
        needer_idx = 1

        for ci, (code, name, plants) in enumerate(chosen):
            rid += 1
            style = STYLES[(gi + ci) % len(STYLES)]
            desc = style(proto)
            uom_name, pack = proto["uoms"][(gi + ci) % len(proto["uoms"])]
            if pack > 1:
                desc = f"{desc} {uom_name} OF {pack} " + ("M" if uom_name in ("DRUM", "COIL") else "NOS")
            jitter = 1.0 + rng.uniform(-0.06, 0.06)
            unit_price = round(base_price * pack * jitter, 2)

            has_vendor = vendor is not None and rng.random() < 0.7

            stock = rng.choice([0, 0, 12, 47, 120, 860, 2400])
            demand = rng.choice([0, 0, 40, 180, 600, 5200])
            idle = rng.choice([15, 60, 120, 400, 700, 1100])
            if plant_redeploy and ci == holder_idx:
                stock = rng.choice([180, 420, 960, 2400])
                demand = 0
                idle = rng.choice([420, 640, 900, 1300])
            elif plant_redeploy and ci == needer_idx:
                stock = rng.choice([0, 4])
                demand = rng.choice([220, 540, 1400])
                idle = rng.choice([20, 45])

            records.append({
                "id": rid,
                "cpse_code": code,
                "cpse_name": name,
                "plant": rng.choice(plants),
                "matnr": f"{rng.randint(10,99)}{gi:04d}{ci}{rng.randint(100,999)}",
                "description": desc,
                "uom": uom_name,
                "unit_price": unit_price,
                "currency": "INR",
                "stock_qty": stock,
                "annual_demand": demand,
                "last_movement_days": idle,
                "mfr_name": vendor[1] if (vendor and rng.random() < 0.4) else None,
                "mfr_part_no": f"{vendor_matl}-M" if (vendor and rng.random() < 0.3) else None,
                "vendor_id": vendor[0] if has_vendor else None,
                "vendor_name": vendor[1] if has_vendor else None,
                "vendor_matl_no": vendor_matl if has_vendor else None,
                "_group": group_id,
            })
            truth_groups.setdefault(group_id, []).append(rid)

    # ---- low-information records: must be quarantined, never guessed at
    for i in range(14):
        rid += 1
        cpse = rng.choice(CPSES)
        records.append({
            "id": rid, "cpse_code": cpse[0], "cpse_name": cpse[1], "plant": rng.choice(cpse[2]),
            "matnr": f"77{i:04d}00{rng.randint(100,999)}",
            "description": rng.choice(LOW_INFO), "uom": "NO",
            "unit_price": round(rng.uniform(10, 900), 2),
            "stock_qty": rng.choice([0, 5, 40]), "annual_demand": 0,
            "last_movement_days": rng.choice([300, 900]),
            "mfr_name": None, "mfr_part_no": None,
            "vendor_id": None, "vendor_name": None, "vendor_matl_no": None,
            "_group": None, "_expect": "quarantine",
        })

    # ---- engineered to order: must be excluded by design, not harmonised
    for i, desc in enumerate(ENGINEERED):
        rid += 1
        cpse = rng.choice(CPSES)
        records.append({
            "id": rid, "cpse_code": cpse[0], "cpse_name": cpse[1], "plant": rng.choice(cpse[2]),
            "matnr": f"88{i:04d}00{rng.randint(100,999)}",
            "description": desc, "uom": "NO",
            "unit_price": round(rng.uniform(120000, 2400000), 2),
            "stock_qty": 0, "annual_demand": 1, "last_movement_days": 500,
            "mfr_name": None, "mfr_part_no": None,
            "vendor_id": None, "vendor_name": None, "vendor_matl_no": None,
            "_group": None, "_expect": "excluded",
        })

    truth_pairs = []
    for members in truth_groups.values():
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                truth_pairs.append([members[i], members[j]])

    return {
        "_note": "SYNTHETIC. Every code, price, stock figure and vendor reference is invented.",
        "records": records,
        "truth_groups": truth_groups,
        "truth_pairs": truth_pairs,
        "cpses": [{"code": c, "name": n, "plants": p} for c, n, p in CPSES],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate the synthetic multi-CPSE corpus")
    ap.add_argument("--groups", type=int, default=180, help="number of distinct physical items")
    ap.add_argument("--seed", type=int, default=20260914)
    ap.add_argument("--out", default="var/corpus.json")
    args = ap.parse_args()

    payload = generate(args.groups, args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")

    by_cpse = {}
    for r in payload["records"]:
        by_cpse[r["cpse_code"]] = by_cpse.get(r["cpse_code"], 0) + 1
    print(f"wrote {out}")
    print(f"  records      {len(payload['records'])}")
    print(f"  true groups  {len(payload['truth_groups'])}")
    print(f"  true pairs   {len(payload['truth_pairs'])}")
    print(f"  by CPSE      {by_cpse}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
