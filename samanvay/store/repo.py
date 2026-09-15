"""Repository. All SQL lives here; nothing above this layer writes a query.

The one invariant worth stating in code: the harmonisation process NEVER updates
source_material. A merge inserts into nmc_member - a link. The CPSE's MATNR is
untouched, so no transaction, report, interface or historical document is affected.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .. import config
from ..core import classify, constraints, embed, nmc as nmc_mod, render
from .ledger import Ledger

SOURCE_FIELDS = (
    "cpse_code", "cpse_name", "plant", "matnr", "description", "long_text", "uom",
    "unit_price", "currency", "stock_qty", "annual_demand", "last_movement_days",
    "mfr_name", "mfr_part_no", "vendor_id", "vendor_name", "vendor_matl_no", "source_system",
)


def _j(value) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _unj(value, default=None):
    if not value:
        return default
    try:
        return json.loads(value)
    except Exception:
        return default


class Repo:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self.ledger = Ledger(conn)

    # ================================================================ ingestion
    def upsert_source(self, record: dict) -> int:
        payload = {k: record.get(k) for k in SOURCE_FIELDS}
        payload["raw_json"] = _j(record)
        cols = ", ".join(payload.keys())
        marks = ", ".join("?" for _ in payload)
        updates = ", ".join(f"{k}=excluded.{k}" for k in payload if k not in ("cpse_code", "matnr"))
        cur = self.conn.execute(
            f"""INSERT INTO source_material ({cols}) VALUES ({marks})
                ON CONFLICT(cpse_code, matnr) DO UPDATE SET {updates}
                RETURNING id""",
            list(payload.values()),
        )
        return int(cur.fetchone()["id"])

    def ingest(self, records: Sequence[dict], actor: str = "connector",
               source_system: str = "manual") -> dict:
        ids: List[int] = []
        for rec in records:
            rec = dict(rec)
            rec.setdefault("source_system", source_system)
            if not rec.get("matnr"):
                rec["matnr"] = f"AUTO{len(ids):08d}"
            ids.append(self.upsert_source(rec))
        by_cpse: Dict[str, int] = {}
        for rec in records:
            by_cpse[rec.get("cpse_code", "?")] = by_cpse.get(rec.get("cpse_code", "?"), 0) + 1
        self.ledger.append(
            "INGESTED",
            {"records": len(ids), "by_cpse": by_cpse, "source_system": source_system},
            actor=actor, reason=f"ingested {len(ids)} source material records",
        )
        return {"ingested": len(ids), "ids": ids, "by_cpse": by_cpse}

    def source_records(self, cpse_code: Optional[str] = None, limit: Optional[int] = None) -> List[dict]:
        sql = "SELECT * FROM source_material"
        params: List = []
        if cpse_code:
            sql += " WHERE cpse_code = ?"
            params.append(cpse_code)
        sql += " ORDER BY id ASC"
        if limit:
            sql += " LIMIT ?"
            params.append(limit)
        return [dict(r) for r in self.conn.execute(sql, params).fetchall()]

    # ================================================================ canonical
    def save_canonical(self, canon: dict, status: str = "active", reason: str = "") -> int:
        payload = {
            "source_id": canon["id"],
            "class_code": canon["class_code"],
            "class_confidence": canon.get("class_confidence"),
            "item_name": canon.get("item_name"),
            "standardisable": 1 if canon.get("standardisable", True) else 0,
            "normalised_text": canon.get("normalised_text"),
            "attributes_json": _j(canon.get("attributes") or {}),
            "numerics_json": _j(canon.get("numerics") or {}),
            "extraction_json": _j(canon.get("extraction") or {}),
            "golden_description": canon.get("golden_description"),
            "short_description": canon.get("short_description"),
            "facets_json": _j(canon.get("facets") or {}),
            "coverage": canon.get("coverage"),
            "completeness": canon.get("completeness"),
            "missing_mandatory": _j(canon.get("missing_mandatory") or []),
            "residue": _j(canon.get("residue") or []),
            "uom_resolution_json": _j(canon.get("uom_resolution") or {}),
            "unit_price_base": canon.get("unit_price_base"),
            "blocking_keys": _j(canon.get("blocking_keys") or []),
            "identity_keys": _j(canon.get("identity_keys") or []),
            "embedding": embed.pack(canon.get("embedding") or []),
            "status": status,
            "status_reason": reason,
        }
        cols = ", ".join(payload)
        marks = ", ".join("?" for _ in payload)
        updates = ", ".join(f"{k}=excluded.{k}" for k in payload if k != "source_id")
        cur = self.conn.execute(
            f"""INSERT INTO canonical_record ({cols}) VALUES ({marks})
                ON CONFLICT(source_id) DO UPDATE SET {updates}, updated_at=datetime('now')
                RETURNING id""",
            list(payload.values()),
        )
        return int(cur.fetchone()["id"])

    def canonical(self, canonical_id: int) -> Optional[dict]:
        row = self.conn.execute(
            """SELECT c.*, s.cpse_code, s.cpse_name, s.plant, s.matnr, s.description,
                      s.uom, s.unit_price, s.currency, s.stock_qty, s.annual_demand,
                      s.last_movement_days, s.mfr_name, s.mfr_part_no, s.vendor_id,
                      s.vendor_name, s.vendor_matl_no
               FROM canonical_record c JOIN source_material s ON s.id = c.source_id
               WHERE c.id = ?""",
            (canonical_id,),
        ).fetchone()
        return self._hydrate(row) if row else None

    def canonical_many(self, ids: Sequence[int]) -> List[dict]:
        if not ids:
            return []
        marks = ",".join("?" for _ in ids)
        rows = self.conn.execute(
            f"""SELECT c.*, s.cpse_code, s.cpse_name, s.plant, s.matnr, s.description,
                       s.uom, s.unit_price, s.currency, s.stock_qty, s.annual_demand,
                       s.last_movement_days, s.mfr_name, s.mfr_part_no, s.vendor_id,
                       s.vendor_name, s.vendor_matl_no
                FROM canonical_record c JOIN source_material s ON s.id = c.source_id
                WHERE c.id IN ({marks})""",
            list(ids),
        ).fetchall()
        return [self._hydrate(r) for r in rows]

    def canonical_all(self, status: str = "active") -> List[dict]:
        rows = self.conn.execute(
            """SELECT c.*, s.cpse_code, s.cpse_name, s.plant, s.matnr, s.description,
                      s.uom, s.unit_price, s.currency, s.stock_qty, s.annual_demand,
                      s.last_movement_days, s.mfr_name, s.mfr_part_no, s.vendor_id,
                      s.vendor_name, s.vendor_matl_no
               FROM canonical_record c JOIN source_material s ON s.id = c.source_id
               WHERE c.status = ? ORDER BY c.id""",
            (status,),
        ).fetchall()
        return [self._hydrate(r) for r in rows]

    @staticmethod
    def _hydrate(row: sqlite3.Row) -> dict:
        d = dict(row)
        d["attributes"] = _unj(d.pop("attributes_json", None), {}) or {}
        d["numerics"] = _unj(d.pop("numerics_json", None), {}) or {}
        d["extraction"] = _unj(d.pop("extraction_json", None), {}) or {}
        d["facets"] = _unj(d.pop("facets_json", None), {}) or {}
        d["uom_resolution"] = _unj(d.pop("uom_resolution_json", None), {}) or {}
        d["missing_mandatory"] = _unj(d.get("missing_mandatory"), []) or []
        d["residue"] = _unj(d.get("residue"), []) or []
        d["blocking_keys"] = _unj(d.get("blocking_keys"), []) or []
        d["identity_keys"] = _unj(d.get("identity_keys"), []) or []
        blob = d.pop("embedding", None)
        d["embedding"] = embed.unpack(blob) if blob else []
        return d

    # ================================================================ proposals
    def save_proposal(self, proposal: dict, canonical_by_source: Dict[int, int]) -> int:
        a = canonical_by_source.get(int(proposal["a_id"]), int(proposal["a_id"]))
        b = canonical_by_source.get(int(proposal["b_id"]), int(proposal["b_id"]))
        if a > b:
            a, b = b, a
        payload = {
            "a_id": a, "b_id": b,
            "class_code": proposal.get("class_code"),
            "relation": proposal["relation"],
            "direction": proposal.get("direction", ""),
            "score": proposal.get("score"),
            "bi_score": proposal.get("bi_score"),
            "tier": proposal.get("tier"),
            "priority": proposal.get("priority", 0),
            "annual_spend": proposal.get("annual_spend", 0),
            "identity_key": 1 if proposal.get("identity_key") else 0,
            "proof_strength": proposal.get("proof_strength"),
            "blocked_by": proposal.get("blocked_by", ""),
            "strategies": _j(proposal.get("strategies") or []),
            "evidence_json": _j(proposal.get("evidence") or {}),
            "decision_json": _j(proposal.get("decision") or {}),
            "relation_json": _j(proposal.get("relation_detail") or {}),
            "feature_json": _j(proposal.get("feature_vector") or []),
            "narrative": proposal.get("narrative", ""),
        }
        cols = ", ".join(payload)
        marks = ", ".join("?" for _ in payload)
        updates = ", ".join(f"{k}=excluded.{k}" for k in payload if k not in ("a_id", "b_id"))
        cur = self.conn.execute(
            f"""INSERT INTO proposal ({cols}) VALUES ({marks})
                ON CONFLICT(a_id, b_id) DO UPDATE SET {updates}
                RETURNING id""",
            list(payload.values()),
        )
        return int(cur.fetchone()["id"])

    def proposal(self, proposal_id: int) -> Optional[dict]:
        row = self.conn.execute("SELECT * FROM proposal WHERE id = ?", (proposal_id,)).fetchone()
        return self._hydrate_proposal(row) if row else None

    def proposals(
        self,
        tier: Optional[str] = None,
        status: Optional[str] = "open",
        relation: Optional[str] = None,
        class_code: Optional[str] = None,
        cpse: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
        order: str = "priority",
    ) -> Tuple[List[dict], int]:
        where, params = ["1=1"], []
        if tier:
            where.append("p.tier = ?")
            params.append(tier)
        if status:
            where.append("p.status = ?")
            params.append(status)
        if relation:
            where.append("p.relation = ?")
            params.append(relation)
        if class_code:
            where.append("p.class_code = ?")
            params.append(class_code)
        if cpse:
            where.append("(sa.cpse_code = ? OR sb.cpse_code = ?)")
            params.extend([cpse, cpse])
        join = """FROM proposal p
                  JOIN canonical_record ca ON ca.id = p.a_id
                  JOIN source_material sa ON sa.id = ca.source_id
                  JOIN canonical_record cb ON cb.id = p.b_id
                  JOIN source_material sb ON sb.id = cb.source_id"""
        clause = " AND ".join(where)
        total = int(self.conn.execute(
            f"SELECT COUNT(*) AS n {join} WHERE {clause}", params
        ).fetchone()["n"])
        order_sql = {
            "priority": "p.priority DESC, p.annual_spend DESC",
            "score": "p.score DESC",
            "spend": "p.annual_spend DESC",
            "recent": "p.id DESC",
        }.get(order, "p.priority DESC")
        rows = self.conn.execute(
            f"SELECT p.* {join} WHERE {clause} ORDER BY {order_sql} LIMIT ? OFFSET ?",
            params + [limit, offset],
        ).fetchall()
        return [self._hydrate_proposal(r) for r in rows], total

    @staticmethod
    def _hydrate_proposal(row: sqlite3.Row) -> dict:
        d = dict(row)
        d["evidence"] = _unj(d.pop("evidence_json", None), {}) or {}
        d["decision"] = _unj(d.pop("decision_json", None), {}) or {}
        d["relation_detail"] = _unj(d.pop("relation_json", None), {}) or {}
        d["feature_vector"] = _unj(d.pop("feature_json", None), []) or []
        d["strategies"] = _unj(d.get("strategies"), []) or []
        d["endorsements"] = _unj(d.pop("endorsements_json", None), []) or []
        d["identity_key"] = bool(d.get("identity_key"))
        return d

    # ============================================================ registry / NMC
    def next_serial(self) -> int:
        self.conn.execute(
            "UPDATE registry_counter SET value = value + 1 WHERE name = 'nmc_serial'"
        )
        row = self.conn.execute(
            "SELECT value FROM registry_counter WHERE name = 'nmc_serial'"
        ).fetchone()
        return int(row["value"])

    def mint_nmc(self, class_code: str, golden_description: str, attributes: dict,
                 actor: str = "registrar") -> dict:
        serial = self.next_serial()
        code = nmc_mod.mint_for_class(serial, class_code)
        facets = nmc_mod.facets(class_code)
        cur = self.conn.execute(
            """INSERT INTO nmc (code, nsn, nsc, ncb, serial, check_digit, class_code,
                                golden_description, attributes_json, facets_json, facet_version)
               VALUES (?,?,?,?,?,?,?,?,?,?,?) RETURNING id""",
            (code.code, code.nsn, code.nsc, code.ncb, code.serial, code.check_digit,
             class_code, golden_description, _j(attributes), _j(facets),
             facets.get("facet_version")),
        )
        nmc_id = int(cur.fetchone()["id"])
        self.ledger.append(
            "MINTED",
            {"nmc": code.code, "nsn": code.nsn, "class_code": class_code,
             "golden_description": golden_description, "facets": facets},
            actor=actor, actor_role="national_registrar", subject=code.code,
            reason="national material code issued",
        )
        return {"id": nmc_id, **code.to_dict(), "class_code": class_code,
                "golden_description": golden_description, "facets": facets}

    def link_members(self, nmc_id: int, canonical_ids: Sequence[int], actor: str,
                     role: str = "member", reason: str = "") -> dict:
        code = self.conn.execute("SELECT code FROM nmc WHERE id = ?", (nmc_id,)).fetchone()["code"]
        linked = []
        for cid in canonical_ids:
            self.conn.execute(
                """INSERT INTO nmc_member (nmc_id, canonical_id, role, linked_by, active)
                   VALUES (?,?,?,?,1)
                   ON CONFLICT(nmc_id, canonical_id) DO UPDATE SET active = 1, role = excluded.role""",
                (nmc_id, int(cid), role, actor),
            )
            linked.append(int(cid))
        self.ledger.append(
            "MERGED", {"nmc": code, "members": linked, "role": role},
            actor=actor, subject=code,
            reason=reason or "members linked to the national material code",
        )
        return {"nmc": code, "linked": linked}

    def unlink_members(self, nmc_code: str, canonical_ids: Sequence[int], actor: str,
                       reason: str) -> dict:
        """Un-merge. A merge added a LINK; removing it restores the prior state exactly,
        because the source records were never rewritten."""
        row = self._resolve_nmc_row(nmc_code)
        if not row:
            return {"error": f"unknown national material code {nmc_code}"}
        nmc_id = int(row["id"])
        nmc_code = row["code"]
        for cid in canonical_ids:
            self.conn.execute(
                "UPDATE nmc_member SET active = 0 WHERE nmc_id = ? AND canonical_id = ?",
                (nmc_id, int(cid)),
            )
        self.ledger.append(
            "SPLIT", {"nmc": nmc_code, "members": [int(c) for c in canonical_ids]},
            actor=actor, subject=nmc_code, reason=reason,
        )
        return {"nmc": nmc_code, "unlinked": [int(c) for c in canonical_ids],
                "note": "the source records were never rewritten, so nothing had to be recovered"}

    def _resolve_nmc_row(self, code: str) -> Optional[sqlite3.Row]:
        """Accept every reasonable spelling of a national material code:
        '5306-72-014-7723/9', '5306-72-014-7723', '5306720147723', '5306 72 014 7723'.
        A code people type on a shop floor has to survive being typed."""
        raw = (code or "").strip()
        row = self.conn.execute("SELECT * FROM nmc WHERE code = ?", (raw,)).fetchone()
        if row:
            return row
        parsed = nmc_mod.parse(raw)
        if parsed:
            row = self.conn.execute("SELECT * FROM nmc WHERE nsn = ?", (parsed.nsn,)).fetchone()
            if row:
                return row
            return self.conn.execute("SELECT * FROM nmc WHERE code = ?", (parsed.code,)).fetchone()
        return None

    def nmc_by_code(self, code: str) -> Optional[dict]:
        row = self._resolve_nmc_row(code)
        if not row:
            return None
        d = dict(row)
        d["attributes"] = _unj(d.pop("attributes_json", None), {}) or {}
        d["facets"] = _unj(d.pop("facets_json", None), {}) or {}
        d["members"] = self.nmc_members(int(row["id"]))
        return d

    def nmc_members(self, nmc_id: int) -> List[dict]:
        rows = self.conn.execute(
            """SELECT m.role, c.id AS canonical_id, c.class_code, c.golden_description,
                      c.unit_price_base, c.attributes_json,
                      s.cpse_code, s.cpse_name, s.plant, s.matnr, s.description, s.uom,
                      s.unit_price, s.stock_qty, s.annual_demand, s.last_movement_days
               FROM nmc_member m
               JOIN canonical_record c ON c.id = m.canonical_id
               JOIN source_material s ON s.id = c.source_id
               WHERE m.nmc_id = ? AND m.active = 1
               ORDER BY s.cpse_code""",
            (nmc_id,),
        ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["attributes"] = _unj(d.pop("attributes_json", None), {}) or {}
            out.append(d)
        return out

    def nmc_for_canonical(self, canonical_id: int) -> Optional[dict]:
        row = self.conn.execute(
            """SELECT n.* FROM nmc n JOIN nmc_member m ON m.nmc_id = n.id
               WHERE m.canonical_id = ? AND m.active = 1 LIMIT 1""",
            (int(canonical_id),),
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        d["attributes"] = _unj(d.pop("attributes_json", None), {}) or {}
        d["facets"] = _unj(d.pop("facets_json", None), {}) or {}
        return d

    def nmc_groups(self, limit: Optional[int] = None, multi_cpse_only: bool = False) -> List[dict]:
        rows = self.conn.execute("SELECT * FROM nmc WHERE status = 'active' ORDER BY id").fetchall()
        out = []
        for row in rows:
            members = self.nmc_members(int(row["id"]))
            if multi_cpse_only and len({m["cpse_code"] for m in members}) < 2:
                continue
            out.append({
                "nmc": row["code"], "class_code": row["class_code"],
                "golden_description": row["golden_description"],
                "members": members, "cpse_count": len({m["cpse_code"] for m in members}),
            })
            if limit and len(out) >= limit:
                break
        return out

    def search_catalogue(self, query: str = "", class_code: str = "", cpse: str = "",
                         attribute_filters: Optional[Dict[str, str]] = None,
                         limit: int = 50, offset: int = 0) -> Tuple[List[dict], int]:
        where, params = ["n.status = 'active'"], []
        if query:
            where.append("(n.golden_description LIKE ? OR n.code LIKE ?)")
            params.extend([f"%{query.upper()}%", f"%{query.upper()}%"])
        if class_code:
            where.append("n.class_code = ?")
            params.append(class_code)
        clause = " AND ".join(where)
        total = int(self.conn.execute(
            f"SELECT COUNT(*) AS n FROM nmc n WHERE {clause}", params
        ).fetchone()["n"])
        rows = self.conn.execute(
            f"SELECT * FROM nmc n WHERE {clause} ORDER BY n.id LIMIT ? OFFSET ?",
            params + [limit, offset],
        ).fetchall()
        out = []
        for row in rows:
            members = self.nmc_members(int(row["id"]))
            if cpse and cpse not in {m["cpse_code"] for m in members}:
                continue
            attrs = _unj(row["attributes_json"], {}) or {}
            if attribute_filters and not all(
                str(attrs.get(k, "")).upper() == str(v).upper() for k, v in attribute_filters.items()
            ):
                continue
            out.append({
                "nmc": row["code"], "nsn": row["nsn"], "class_code": row["class_code"],
                "golden_description": row["golden_description"],
                "attributes": attrs, "facets": _unj(row["facets_json"], {}) or {},
                "members": members, "cpse_count": len({m["cpse_code"] for m in members}),
            })
        return out, total

    # ========================================================== steward actions
    def record_decision(self, proposal_id: Optional[int], actor: str, decision: str,
                        reason: str = "", a_id: Optional[int] = None,
                        b_id: Optional[int] = None, actor_cpse: str = "",
                        actor_role: str = "steward") -> int:
        cur = self.conn.execute(
            """INSERT INTO steward_decision
               (proposal_id, a_id, b_id, actor, actor_cpse, actor_role, decision, reason)
               VALUES (?,?,?,?,?,?,?,?) RETURNING id""",
            (proposal_id, a_id, b_id, actor, actor_cpse, actor_role, decision, reason),
        )
        return int(cur.fetchone()["id"])

    def add_constraint(self, a_id: int, b_id: int, kind: str, actor: str, reason: str = "") -> None:
        a, b = (int(a_id), int(b_id)) if a_id < b_id else (int(b_id), int(a_id))
        self.conn.execute(
            """INSERT INTO constraint_link (a_id, b_id, kind, actor, reason)
               VALUES (?,?,?,?,?) ON CONFLICT(a_id, b_id, kind) DO NOTHING""",
            (a, b, kind, actor, reason),
        )

    def constraints(self) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        rows = self.conn.execute("SELECT a_id, b_id, kind FROM constraint_link").fetchall()
        must = [(int(r["a_id"]), int(r["b_id"])) for r in rows if r["kind"] == "must"]
        cannot = [(int(r["a_id"]), int(r["b_id"])) for r in rows if r["kind"] == "cannot"]
        return must, cannot

    def steward_labels(self) -> List[dict]:
        """Every steward decision, as a training label. The model gets better as it
        is used, and the curve is visible on the analytics screen."""
        rows = self.conn.execute(
            """SELECT d.a_id, d.b_id, d.decision, p.a_id AS pa, p.b_id AS pb
               FROM steward_decision d LEFT JOIN proposal p ON p.id = d.proposal_id
               WHERE d.decision IN ('endorse','reject','distinct')"""
        ).fetchall()
        out = []
        for r in rows:
            a = r["a_id"] or r["pa"]
            b = r["b_id"] or r["pb"]
            if a is None or b is None:
                continue
            out.append({"a_id": int(a), "b_id": int(b),
                        "label": 1 if r["decision"] == "endorse" else 0})
        return out

    # ================================================================ pipeline
    def record_run(self, records_in: int, stats: dict, training: dict, funnel: list) -> int:
        cur = self.conn.execute(
            """INSERT INTO pipeline_run (records_in, stats_json, training_json, funnel_json,
                                         finished_at)
               VALUES (?,?,?,?, datetime('now')) RETURNING id""",
            (records_in, _j(stats), _j(training), _j(funnel)),
        )
        run_id = int(cur.fetchone()["id"])
        self.ledger.append(
            "PIPELINE_RUN",
            {"run_id": run_id, "records_in": records_in,
             "duplicate_rate": stats.get("duplicate_rate"),
             "tiers": stats.get("tiers"), "clusters": stats.get("clusters")},
            actor="engine", actor_role="proposal_engine",
            reason="cascade run completed",
        )
        return run_id

    def latest_run(self) -> Optional[dict]:
        row = self.conn.execute(
            "SELECT * FROM pipeline_run ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        d["stats"] = _unj(d.pop("stats_json", None), {}) or {}
        d["training"] = _unj(d.pop("training_json", None), {}) or {}
        d["funnel"] = _unj(d.pop("funnel_json", None), []) or []
        return d

    def save_substitution(self, sub: dict, canonical_by_source: Dict[int, int]) -> int:
        f = canonical_by_source.get(int(sub["from_id"]), int(sub["from_id"]))
        t = canonical_by_source.get(int(sub["to_id"]), int(sub["to_id"]))
        cur = self.conn.execute(
            """INSERT INTO substitution (from_id, to_id, class_code, score, reasons, caveats, citations)
               VALUES (?,?,?,?,?,?,?)
               ON CONFLICT(from_id, to_id) DO UPDATE SET
                   score=excluded.score, reasons=excluded.reasons,
                   caveats=excluded.caveats, citations=excluded.citations
               RETURNING id""",
            (f, t, sub.get("class_code"), sub.get("score"),
             _j(sub.get("reasons") or []), _j(sub.get("caveats") or []),
             _j(sub.get("citations") or [])),
        )
        return int(cur.fetchone()["id"])

    def substitutions(self, status: Optional[str] = None, limit: int = 200) -> List[dict]:
        sql = """SELECT sub.*, ca.golden_description AS from_desc, cb.golden_description AS to_desc,
                        sa.cpse_code AS from_cpse, sb.cpse_code AS to_cpse,
                        sa.matnr AS from_matnr, sb.matnr AS to_matnr,
                        sa.stock_qty AS from_stock, sb.annual_demand AS to_demand,
                        sa.last_movement_days AS from_idle_days, sa.plant AS from_plant,
                        sb.plant AS to_plant, ca.unit_price_base AS from_price,
                        cb.unit_price_base AS to_price
                 FROM substitution sub
                 JOIN canonical_record ca ON ca.id = sub.from_id
                 JOIN source_material sa ON sa.id = ca.source_id
                 JOIN canonical_record cb ON cb.id = sub.to_id
                 JOIN source_material sb ON sb.id = cb.source_id"""
        params: List = []
        if status:
            sql += " WHERE sub.status = ?"
            params.append(status)
        sql += " ORDER BY sub.score DESC LIMIT ?"
        params.append(limit)
        out = []
        for r in self.conn.execute(sql, params).fetchall():
            d = dict(r)
            d["reasons"] = _unj(d.get("reasons"), []) or []
            d["caveats"] = _unj(d.get("caveats"), []) or []
            d["citations"] = _unj(d.get("citations"), []) or []
            out.append(d)
        return out

    # ================================================================ ERP outbox
    def queue_change_request(self, nmc_code: str, cpse_code: str, matnr: str,
                             mechanism: str, payload: dict, target_system: str = "SAP",
                             proposal_id: Optional[int] = None, actor: str = "registrar") -> int:
        cur = self.conn.execute(
            """INSERT INTO erp_change_request
               (nmc_code, cpse_code, matnr, target_system, mechanism, payload_json, proposal_id)
               VALUES (?,?,?,?,?,?,?) RETURNING id""",
            (nmc_code, cpse_code, matnr, target_system, mechanism, _j(payload), proposal_id),
        )
        cr_id = int(cur.fetchone()["id"])
        self.ledger.append(
            "ERP_WRITEBACK",
            {"change_request": cr_id, "nmc": nmc_code, "cpse": cpse_code,
             "matnr": matnr, "mechanism": mechanism},
            actor=actor, subject=nmc_code,
            reason=(
                "change request raised - the AI never writes production master data; "
                "this enters the CPSE's own approval workflow"
            ),
        )
        return cr_id

    def change_requests(self, status: Optional[str] = None, limit: int = 100) -> List[dict]:
        sql = "SELECT * FROM erp_change_request"
        params: List = []
        if status:
            sql += " WHERE status = ?"
            params.append(status)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        out = []
        for r in self.conn.execute(sql, params).fetchall():
            d = dict(r)
            d["payload"] = _unj(d.pop("payload_json", None), {}) or {}
            out.append(d)
        return out

    def set_change_request_status(self, cr_id: int, status: str, external_ref: str = "") -> None:
        self.conn.execute(
            "UPDATE erp_change_request SET status=?, external_ref=?, updated_at=datetime('now') WHERE id=?",
            (status, external_ref, cr_id),
        )

    # =================================================================== counts
    def counts(self) -> dict:
        def n(sql: str, *params) -> int:
            return int(self.conn.execute(sql, params).fetchone()["n"])
        return {
            "source_materials": n("SELECT COUNT(*) AS n FROM source_material"),
            "cpses": n("SELECT COUNT(DISTINCT cpse_code) AS n FROM source_material"),
            "canonical_active": n("SELECT COUNT(*) AS n FROM canonical_record WHERE status='active'"),
            "quarantined": n("SELECT COUNT(*) AS n FROM canonical_record WHERE status='quarantined'"),
            "excluded": n("SELECT COUNT(*) AS n FROM canonical_record WHERE status='excluded'"),
            "proposals_open": n("SELECT COUNT(*) AS n FROM proposal WHERE status='open'"),
            "proposals_review": n("SELECT COUNT(*) AS n FROM proposal WHERE status='open' AND tier='review'"),
            "proposals_auto": n("SELECT COUNT(*) AS n FROM proposal WHERE tier='auto_accept'"),
            "nmc_active": n("SELECT COUNT(*) AS n FROM nmc WHERE status='active'"),
            "nmc_members": n("SELECT COUNT(*) AS n FROM nmc_member WHERE active=1"),
            "substitutions": n("SELECT COUNT(*) AS n FROM substitution"),
            "steward_decisions": n("SELECT COUNT(*) AS n FROM steward_decision"),
            "ledger_events": n("SELECT COUNT(*) AS n FROM ledger_event"),
            "change_requests": n("SELECT COUNT(*) AS n FROM erp_change_request"),
        }

    def cpses(self) -> List[dict]:
        rows = self.conn.execute(
            """SELECT cpse_code, MAX(cpse_name) AS cpse_name, COUNT(*) AS materials,
                      COUNT(DISTINCT plant) AS plants
               FROM source_material GROUP BY cpse_code ORDER BY cpse_code"""
        ).fetchall()
        from ..core import sectors
        return [dict(r, sector=sectors.sector_of(r["cpse_code"])) for r in rows]
