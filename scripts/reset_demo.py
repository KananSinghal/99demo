#!/usr/bin/env python3
"""Reset the database to a clean, known-good demo state in a few seconds:
drop and recreate the schema, load the synthetic multi-CPSE corpus, and run
the nine-stage cascade once. This is what `make demo` calls before starting
the server, and it is safe to re-run at any point during a demo if state gets
messy -- that is the whole design point of an append-only ledger over a
disposable projection.

    python3 scripts/reset_demo.py
    python3 scripts/reset_demo.py --groups 300   # a bigger corpus

This ships with zero steward decisions made, so the review queue, the
Buy-or-Borrow screen and the ledger are all exactly as a judge would first see
them.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--groups", type=int, default=200)
    args = ap.parse_args()

    from samanvay.store import db
    from samanvay.api import handlers, service

    corpus_path = ROOT / "var" / "corpus.json"
    if corpus_path.exists():
        corpus_path.unlink()

    print("[1/3] resetting database...")
    db.reset_db()

    print(f"[2/3] loading synthetic corpus ({args.groups} groups)...")
    ingested = handlers.ingest_demo({}, {"groups": args.groups})
    print(f"      {ingested['ingested']} records across {len(ingested['by_cpse'])} CPSEs, "
          f"{ingested['truth_pairs']} planted true-duplicate pairs")

    print("[3/3] running the nine-stage cascade...")
    run = handlers.pipeline_run({}, {})
    stats = run.get("stats") or run
    tiers = stats["tiers"]
    print(f"      canonical={stats['canonical']}  quarantined={stats['quarantined']}  "
          f"excluded={stats['excluded']}")
    print(f"      auto-accept={tiers['auto_accept']}  review={tiers['review']}  "
          f"auto-reject={tiers['auto_reject']}  duplicate-rate={stats['duplicate_rate']}")

    counts = service.repo().counts()
    verify = service.repo().ledger.verify()
    ok = counts["source_materials"] == ingested["ingested"] and counts["steward_decisions"] == 0 \
        and verify["valid"]
    print()
    print(json.dumps(counts, indent=2))
    print(f"ledger: {'valid' if verify['valid'] else 'BROKEN'}, {verify['events']} events")
    print("READY." if ok else "WARNING: state did not come out clean - see counts above.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
