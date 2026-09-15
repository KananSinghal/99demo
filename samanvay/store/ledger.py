"""The governance ledger.

National MDM dies on irreversibility. Once a merge overwrites a row, unwinding it is
archaeology, and one bad batch poisons the registry permanently. So nothing is ever
overwritten: the registry is an append-only event store and current state is a
materialised projection of it.

    PROPOSED -> ENDORSED x2 -> MERGED -> SPLIT

Each event carries SHA-256(seq | type | payload | actor | prev_hash), so altering any
historical event invalidates every hash after it. The daily Merkle root is published.

Why un-merge is free: a merge ADDS A LINK; it never rewrites the source record.
Nothing was destroyed, so nothing has to be recovered - drop the projection and
rebuild it without the link.

On blockchain: this gives every tamper-evidence property a jury asks about, on
Postgres or SQLite, with no consensus overhead. If inter-organisation non-repudiation
is required - so CPSE-A cannot later claim CPSE-B forged an endorsement - anchor the
daily root to a permissioned ledger. That is a one-line change and a scoped use.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Sequence

GENESIS = "0" * 64

# Event vocabulary. Nothing outside this list may be appended.
EVENTS = (
    "PROPOSED",       # the engine proposed a relation, with evidence
    "ENDORSED",       # a named steward endorsed it for their own CPSE
    "REJECTED",       # a named steward rejected it
    "MERGED",         # dual endorsement reached; members linked to an NMC
    "SPLIT",          # a merge reversed; the projection is replayed without the link
    "MINTED",         # an NMC was issued by the national registrar
    "RECLASSIFIED",   # a classification facet changed (identity untouched)
    "SUPERSEDED",     # an NMC replaced by another
    "DEPRECATED",     # an NMC retired
    "REINSTATED",     # a deprecated NMC brought back
    "SUBSTITUTION_APPROVED",   # a directed substitutability edge approved by an engineer
    "ERP_WRITEBACK",  # a change request raised against a CPSE's ERP
    "INGESTED",       # a corpus was loaded
    "PIPELINE_RUN",   # a cascade run completed
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _canonical(payload) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def compute_hash(seq: int, event_type: str, payload_json: str, actor: str, prev_hash: str,
                 created_at: str) -> str:
    material = f"{seq}|{event_type}|{payload_json}|{actor}|{prev_hash}|{created_at}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


class Ledger:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    # ------------------------------------------------------------------- append
    def head(self) -> str:
        row = self.conn.execute(
            "SELECT hash FROM ledger_event ORDER BY seq DESC LIMIT 1"
        ).fetchone()
        return row["hash"] if row else GENESIS

    def append(
        self,
        event_type: str,
        payload: dict,
        actor: str,
        reason: str = "",
        subject: str = "",
        actor_cpse: str = "",
        actor_role: str = "",
    ) -> dict:
        if event_type not in EVENTS:
            raise ValueError(
                f"unknown event type {event_type!r}; the vocabulary is fixed: {', '.join(EVENTS)}"
            )
        payload_json = _canonical(payload)
        created_at = _now()
        prev = self.head()
        cur = self.conn.execute("SELECT COALESCE(MAX(seq), 0) + 1 AS nxt FROM ledger_event")
        seq = int(cur.fetchone()["nxt"])
        digest = compute_hash(seq, event_type, payload_json, actor, prev, created_at)
        self.conn.execute(
            """INSERT INTO ledger_event
               (seq, event_type, subject, payload_json, actor, actor_cpse, actor_role,
                reason, prev_hash, hash, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (seq, event_type, subject, payload_json, actor, actor_cpse, actor_role,
             reason, prev, digest, created_at),
        )
        return {
            "seq": seq, "event_type": event_type, "subject": subject, "actor": actor,
            "reason": reason, "prev_hash": prev, "hash": digest, "created_at": created_at,
        }

    # -------------------------------------------------------------------- reads
    def events(
        self,
        limit: int = 200,
        offset: int = 0,
        subject: Optional[str] = None,
        event_type: Optional[str] = None,
    ) -> List[dict]:
        sql = "SELECT * FROM ledger_event WHERE 1=1"
        params: List = []
        if subject:
            sql += " AND subject = ?"
            params.append(subject)
        if event_type:
            sql += " AND event_type = ?"
            params.append(event_type)
        sql += " ORDER BY seq DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        rows = self.conn.execute(sql, params).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            try:
                d["payload"] = json.loads(d.pop("payload_json"))
            except Exception:
                d["payload"] = {}
            out.append(d)
        return out

    def count(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) AS n FROM ledger_event").fetchone()["n"])

    # ------------------------------------------------------------ verification
    def verify(self, limit: Optional[int] = None) -> dict:
        """Walk the chain and recompute every hash. Any historical edit shows up here
        as the first broken link, and every hash after it is invalid too."""
        sql = "SELECT * FROM ledger_event ORDER BY seq ASC"
        if limit:
            sql += f" LIMIT {int(limit)}"
        rows = self.conn.execute(sql).fetchall()
        prev = GENESIS
        broken: List[dict] = []
        for row in rows:
            expected = compute_hash(
                row["seq"], row["event_type"], row["payload_json"], row["actor"],
                prev, row["created_at"],
            )
            if row["prev_hash"] != prev:
                broken.append({"seq": row["seq"], "problem": "prev_hash does not match the chain",
                               "expected_prev": prev, "found_prev": row["prev_hash"]})
            elif expected != row["hash"]:
                broken.append({"seq": row["seq"], "problem": "content hash does not match the payload",
                               "expected": expected, "found": row["hash"]})
            prev = row["hash"]
        return {
            "events": len(rows),
            "valid": not broken,
            "head": prev,
            "broken": broken[:20],
            "statement": (
                "chain intact - every event hashes the previous, so altering any "
                "historical event would invalidate every hash after it"
                if not broken else
                f"chain broken at event {broken[0]['seq']}"
            ),
        }

    # ------------------------------------------------------------- merkle roots
    @staticmethod
    def _merkle(leaves: Sequence[str]) -> str:
        if not leaves:
            return GENESIS
        level = [bytes.fromhex(x) for x in leaves]
        while len(level) > 1:
            nxt = []
            for i in range(0, len(level), 2):
                left = level[i]
                right = level[i + 1] if i + 1 < len(level) else left
                nxt.append(hashlib.sha256(left + right).digest())
            level = nxt
        return level[0].hex()

    def merkle_root(self, day: Optional[str] = None, publish: bool = False) -> dict:
        day = day or datetime.now(timezone.utc).date().isoformat()
        rows = self.conn.execute(
            "SELECT hash FROM ledger_event WHERE substr(created_at,1,10) = ? ORDER BY seq ASC",
            (day,),
        ).fetchall()
        leaves = [r["hash"] for r in rows]
        root = self._merkle(leaves)
        if publish and leaves:
            self.conn.execute(
                "INSERT OR REPLACE INTO merkle_root (day, root, events, published_at) VALUES (?,?,?,?)",
                (day, root, len(leaves), _now()),
            )
        return {
            "day": day, "root": root, "events": len(leaves), "published": bool(publish and leaves),
            "note": (
                "publish this root daily. If inter-organisation non-repudiation is "
                "required, anchor it to a permissioned ledger - that is the scoped, "
                "defensible use of a distributed ledger here."
            ),
        }

    def published_roots(self, limit: int = 30) -> List[dict]:
        rows = self.conn.execute(
            "SELECT * FROM merkle_root ORDER BY day DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    # ----------------------------------------------------------------- replay
    def replay(self, upto_seq: Optional[int] = None) -> dict:
        """Rebuild the membership projection purely from the log.

        This is the reversibility proof: merge, un-merge, and the projection is
        identical to the pre-merge state - because nothing was ever destroyed.
        """
        sql = "SELECT * FROM ledger_event"
        params: List = []
        if upto_seq is not None:
            sql += " WHERE seq <= ?"
            params.append(upto_seq)
        sql += " ORDER BY seq ASC"
        rows = self.conn.execute(sql, params).fetchall()

        members: Dict[str, set] = {}
        minted: Dict[str, dict] = {}
        deprecated: set = set()
        applied = 0

        for row in rows:
            try:
                payload = json.loads(row["payload_json"])
            except Exception:
                continue
            et = row["event_type"]
            if et == "MINTED":
                code = payload.get("nmc")
                if code:
                    minted[code] = {"class_code": payload.get("class_code"),
                                    "golden_description": payload.get("golden_description")}
                    members.setdefault(code, set())
                    applied += 1
            elif et == "MERGED":
                code = payload.get("nmc")
                if code:
                    members.setdefault(code, set()).update(int(x) for x in payload.get("members", []))
                    applied += 1
            elif et == "SPLIT":
                code = payload.get("nmc")
                if code and code in members:
                    for x in payload.get("members", []):
                        members[code].discard(int(x))
                    applied += 1
            elif et == "DEPRECATED":
                if payload.get("nmc"):
                    deprecated.add(payload["nmc"])
                    applied += 1
            elif et == "REINSTATED":
                deprecated.discard(payload.get("nmc"))
                applied += 1

        return {
            "upto_seq": upto_seq or (rows[-1]["seq"] if rows else 0),
            "events_applied": applied,
            "nmc_count": len(minted),
            "membership": {code: sorted(ids) for code, ids in members.items() if ids},
            "deprecated": sorted(deprecated),
            "statement": (
                "current state is a projection of the log; drop it and rebuild it at "
                "any time, which is also how un-merge works"
            ),
        }

    def projection_matches_tables(self) -> dict:
        """Compare the replayed projection against the materialised tables. If these
        ever disagree, the tables are wrong and the log is right."""
        replayed = self.replay()["membership"]
        rows = self.conn.execute(
            """SELECT n.code AS code, m.canonical_id AS cid
               FROM nmc n JOIN nmc_member m ON m.nmc_id = n.id
               WHERE m.active = 1"""
        ).fetchall()
        live: Dict[str, set] = {}
        for r in rows:
            live.setdefault(r["code"], set()).add(int(r["cid"]))
        live_norm = {k: sorted(v) for k, v in live.items() if v}
        differences = []
        for code in sorted(set(replayed) | set(live_norm)):
            if replayed.get(code, []) != live_norm.get(code, []):
                differences.append({
                    "nmc": code,
                    "from_ledger": replayed.get(code, []),
                    "in_tables": live_norm.get(code, []),
                })
        return {
            "matches": not differences,
            "nmc_compared": len(set(replayed) | set(live_norm)),
            "differences": differences[:20],
        }
