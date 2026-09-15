"""Schema and connection handling.

Two rules the schema enforces structurally rather than by convention:

  1. ledger_event has NO update or delete path. Current state is a projection of the
     log, which is why un-merge is a replay rather than a recovery.
  2. source_material is never rewritten by the harmonisation process. A merge adds a
     row to nmc_member - a LINK. The CPSE's material number is untouched, always.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

from .. import config

SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- ------------------------------------------------------------------ source data
-- A faithful copy of what the CPSE's ERP holds. NEVER mutated by harmonisation.
CREATE TABLE IF NOT EXISTS source_material (
    id                  INTEGER PRIMARY KEY,
    cpse_code           TEXT NOT NULL,
    cpse_name           TEXT,
    plant               TEXT,
    matnr               TEXT NOT NULL,
    description         TEXT NOT NULL,
    long_text           TEXT,
    uom                 TEXT,
    unit_price          REAL,
    currency            TEXT DEFAULT 'INR',
    stock_qty           REAL DEFAULT 0,
    annual_demand       REAL DEFAULT 0,
    last_movement_days  INTEGER,
    mfr_name            TEXT,
    mfr_part_no         TEXT,
    vendor_id           TEXT,
    vendor_name         TEXT,
    vendor_matl_no      TEXT,
    source_system       TEXT DEFAULT 'manual',
    ingested_at         TEXT DEFAULT (datetime('now')),
    raw_json            TEXT,
    UNIQUE (cpse_code, matnr)
);
CREATE INDEX IF NOT EXISTS ix_source_cpse ON source_material (cpse_code);
CREATE INDEX IF NOT EXISTS ix_source_vendor ON source_material (vendor_id, vendor_matl_no);
CREATE INDEX IF NOT EXISTS ix_source_mpn ON source_material (mfr_part_no);

-- --------------------------------------------------------------- canonical form
CREATE TABLE IF NOT EXISTS canonical_record (
    id                  INTEGER PRIMARY KEY,
    source_id           INTEGER NOT NULL UNIQUE REFERENCES source_material(id) ON DELETE CASCADE,
    class_code          TEXT NOT NULL,
    class_confidence    REAL,
    item_name           TEXT,
    standardisable      INTEGER DEFAULT 1,
    normalised_text     TEXT,
    attributes_json     TEXT,
    numerics_json       TEXT,
    extraction_json     TEXT,
    golden_description  TEXT,
    short_description   TEXT,
    facets_json         TEXT,
    coverage            REAL,
    completeness        REAL,
    missing_mandatory   TEXT,
    residue             TEXT,
    uom_resolution_json TEXT,
    unit_price_base     REAL,
    blocking_keys       TEXT,
    identity_keys       TEXT,
    embedding           BLOB,
    status              TEXT DEFAULT 'active',   -- active | quarantined | excluded
    status_reason       TEXT,
    updated_at          TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS ix_canon_class ON canonical_record (class_code);
CREATE INDEX IF NOT EXISTS ix_canon_status ON canonical_record (status);

-- ------------------------------------------------------------------- proposals
CREATE TABLE IF NOT EXISTS proposal (
    id                  INTEGER PRIMARY KEY,
    a_id                INTEGER NOT NULL REFERENCES canonical_record(id) ON DELETE CASCADE,
    b_id                INTEGER NOT NULL REFERENCES canonical_record(id) ON DELETE CASCADE,
    class_code          TEXT,
    relation            TEXT NOT NULL,
    direction           TEXT DEFAULT '',
    score               REAL,
    bi_score            REAL,
    tier                TEXT,
    priority            REAL DEFAULT 0,
    annual_spend        REAL DEFAULT 0,
    identity_key        INTEGER DEFAULT 0,
    proof_strength      REAL,
    blocked_by          TEXT,
    strategies          TEXT,
    evidence_json       TEXT,
    decision_json       TEXT,
    relation_json       TEXT,
    feature_json        TEXT,
    narrative           TEXT,
    status              TEXT DEFAULT 'open',   -- open | endorsed | rejected | merged | superseded
    endorsements_json   TEXT DEFAULT '[]',
    created_at          TEXT DEFAULT (datetime('now')),
    UNIQUE (a_id, b_id)
);
CREATE INDEX IF NOT EXISTS ix_prop_status ON proposal (status, tier);
CREATE INDEX IF NOT EXISTS ix_prop_priority ON proposal (priority DESC);
CREATE INDEX IF NOT EXISTS ix_prop_relation ON proposal (relation);

-- --------------------------------------------------------- directed relations
-- SUBSTITUTABLE lives here, NOT in the cluster graph. "can serve the duty of" is
-- not "is the same as", and conflating them is how a national registry goes wrong.
CREATE TABLE IF NOT EXISTS substitution (
    id                  INTEGER PRIMARY KEY,
    from_id             INTEGER NOT NULL REFERENCES canonical_record(id) ON DELETE CASCADE,
    to_id               INTEGER NOT NULL REFERENCES canonical_record(id) ON DELETE CASCADE,
    class_code          TEXT,
    score               REAL,
    reasons             TEXT,
    caveats             TEXT,
    citations           TEXT,
    approved_by         TEXT,
    approved_at         TEXT,
    status              TEXT DEFAULT 'proposed',
    UNIQUE (from_id, to_id)
);

-- ------------------------------------------------------------------- registry
CREATE TABLE IF NOT EXISTS nmc (
    id                  INTEGER PRIMARY KEY,
    code                TEXT NOT NULL UNIQUE,
    nsn                 TEXT NOT NULL UNIQUE,
    nsc                 TEXT,
    ncb                 TEXT,
    serial              TEXT,
    check_digit         INTEGER,
    class_code          TEXT,
    golden_description  TEXT,
    attributes_json     TEXT,
    facets_json         TEXT,
    facet_version       TEXT,
    status              TEXT DEFAULT 'active',   -- active | deprecated | superseded
    superseded_by       TEXT,
    created_at          TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS ix_nmc_class ON nmc (class_code);
CREATE INDEX IF NOT EXISTS ix_nmc_status ON nmc (status);

CREATE TABLE IF NOT EXISTS nmc_member (
    id                  INTEGER PRIMARY KEY,
    nmc_id              INTEGER NOT NULL REFERENCES nmc(id) ON DELETE CASCADE,
    canonical_id        INTEGER NOT NULL REFERENCES canonical_record(id) ON DELETE CASCADE,
    role                TEXT DEFAULT 'member',   -- member | variant
    linked_by           TEXT,
    linked_at           TEXT DEFAULT (datetime('now')),
    active              INTEGER DEFAULT 1,
    UNIQUE (nmc_id, canonical_id)
);
CREATE INDEX IF NOT EXISTS ix_member_canon ON nmc_member (canonical_id, active);

CREATE TABLE IF NOT EXISTS registry_counter (
    name    TEXT PRIMARY KEY,
    value   INTEGER NOT NULL
);

-- ------------------------------------------------------------- steward decisions
CREATE TABLE IF NOT EXISTS steward_decision (
    id                  INTEGER PRIMARY KEY,
    proposal_id         INTEGER REFERENCES proposal(id) ON DELETE SET NULL,
    a_id                INTEGER,
    b_id                INTEGER,
    actor               TEXT NOT NULL,
    actor_cpse          TEXT,
    actor_role          TEXT,
    decision            TEXT NOT NULL,      -- endorse | reject | distinct | escalate | unmerge
    reason              TEXT,
    created_at          TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS ix_decision_proposal ON steward_decision (proposal_id);

-- must_link / cannot_link constraints fed back into clustering, so a human decision
-- is never silently overridden by the optimiser on the next run.
CREATE TABLE IF NOT EXISTS constraint_link (
    id                  INTEGER PRIMARY KEY,
    a_id                INTEGER NOT NULL,
    b_id                INTEGER NOT NULL,
    kind                TEXT NOT NULL,      -- must | cannot
    actor               TEXT,
    reason              TEXT,
    created_at          TEXT DEFAULT (datetime('now')),
    UNIQUE (a_id, b_id, kind)
);

-- ------------------------------------------------------------------ the ledger
-- APPEND ONLY. No UPDATE. No DELETE. Enforced by triggers below, not by convention.
CREATE TABLE IF NOT EXISTS ledger_event (
    seq                 INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type          TEXT NOT NULL,
    subject             TEXT,
    payload_json        TEXT NOT NULL,
    actor               TEXT NOT NULL,
    actor_cpse          TEXT,
    actor_role          TEXT,
    reason              TEXT,
    prev_hash           TEXT NOT NULL,
    hash                TEXT NOT NULL UNIQUE,
    created_at          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_ledger_subject ON ledger_event (subject);
CREATE INDEX IF NOT EXISTS ix_ledger_type ON ledger_event (event_type);

CREATE TRIGGER IF NOT EXISTS ledger_no_update
BEFORE UPDATE ON ledger_event
BEGIN
    SELECT RAISE(ABORT, 'ledger_event is append-only: events are never updated');
END;

CREATE TRIGGER IF NOT EXISTS ledger_no_delete
BEFORE DELETE ON ledger_event
BEGIN
    SELECT RAISE(ABORT, 'ledger_event is append-only: events are never deleted');
END;

CREATE TABLE IF NOT EXISTS merkle_root (
    day         TEXT PRIMARY KEY,
    root        TEXT NOT NULL,
    events      INTEGER NOT NULL,
    published_at TEXT DEFAULT (datetime('now'))
);

-- ------------------------------------------------------------------- ERP outbox
CREATE TABLE IF NOT EXISTS erp_change_request (
    id                  INTEGER PRIMARY KEY,
    nmc_code            TEXT,
    cpse_code           TEXT,
    matnr               TEXT,
    target_system       TEXT,
    mechanism           TEXT,       -- mdg_change_request | classification | z_xref
    payload_json        TEXT,
    proposal_id         INTEGER,
    status              TEXT DEFAULT 'queued',   -- queued | sent | applied | failed
    external_ref        TEXT,
    created_at          TEXT DEFAULT (datetime('now')),
    updated_at          TEXT
);
CREATE INDEX IF NOT EXISTS ix_cr_status ON erp_change_request (status);

-- ------------------------------------------------------------------- run history
CREATE TABLE IF NOT EXISTS pipeline_run (
    id                  INTEGER PRIMARY KEY,
    started_at          TEXT DEFAULT (datetime('now')),
    finished_at         TEXT,
    records_in          INTEGER,
    stats_json          TEXT,
    training_json       TEXT,
    funnel_json         TEXT
);
"""


def connect(path: Optional[str] = None) -> sqlite3.Connection:
    db_path = Path(path or config.DB_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=30.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 30000")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    cur = conn.execute("SELECT value FROM registry_counter WHERE name = 'nmc_serial'")
    if cur.fetchone() is None:
        conn.execute(
            "INSERT INTO registry_counter (name, value) VALUES ('nmc_serial', ?)",
            (config.NMC_SERIAL_START,),
        )


def reset_db(path: Optional[str] = None) -> sqlite3.Connection:
    """Drop and recreate. Used by `make demo` to restore known-good state in seconds.

    Note this deletes the LEDGER too, which is only ever acceptable for a demo
    database. In a deployment the ledger is the system of record and is never reset.
    """
    db_path = Path(path or config.DB_PATH)
    for suffix in ("", "-wal", "-shm"):
        candidate = Path(str(db_path) + suffix)
        if candidate.exists():
            candidate.unlink()
    conn = connect(str(db_path))
    init_db(conn)
    return conn
