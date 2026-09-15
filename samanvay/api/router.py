"""A tiny router over pure handler functions.

Handlers take (params: dict, body: dict) and return a dict or (status, dict). No web
framework is imported here, which is why the stdlib server and the FastAPI adapter
serve exactly the same behaviour with no duplicated logic.
"""

from __future__ import annotations

import re
from typing import Callable, Dict, List, Optional, Tuple

from . import handlers

Handler = Callable[[dict, dict], object]

# (method, path pattern, handler, summary)
ROUTES: List[Tuple[str, str, Handler, str]] = [
    ("GET",  "/api/health",              handlers.health,             "liveness and version"),
    ("GET",  "/api/info",                handlers.info,               "config, counts, code scheme, relation types"),
    ("GET",  "/api/cpses",               handlers.cpses,              "participating organisations"),

    ("POST", "/api/ingest",              handlers.ingest,             "ingest source material records"),
    ("POST", "/api/ingest/demo",         handlers.ingest_demo,        "load the synthetic multi-CPSE corpus"),
    ("POST", "/api/pipeline/run",        handlers.pipeline_run,       "run the nine-stage cascade"),
    ("GET",  "/api/pipeline/latest",     handlers.pipeline_latest,    "the last run's funnel and stats"),

    ("GET",  "/api/proposals",           handlers.proposals,          "the steward review queue"),
    ("GET",  "/api/proposals/{id}",      handlers.proposal_detail,    "one proposal with its evidence card"),
    ("POST", "/api/proposals/{id}/endorse", handlers.proposal_endorse, "endorse for your own CPSE (dual key)"),
    ("POST", "/api/proposals/{id}/reject",  handlers.proposal_reject,  "reject, optionally marking DISTINCT"),

    ("GET",  "/api/catalogue",           handlers.catalogue,          "attribute-faceted national catalogue search"),
    # A national material code carries a "/" before its check digit, so these routes
    # use the :path form. Both "5306-72-014-7723/9" and the percent-encoded spelling
    # resolve, as does the bare 13-digit NSN.
    ("GET",  "/api/nmc/{code:path}/validate", handlers.nmc_validate,  "check-digit validation"),
    ("GET",  "/api/nmc/{code:path}",     handlers.nmc_detail,         "one national material code and its members"),

    ("GET",  "/api/substitutions",       handlers.substitutions,      "directed substitutability edges"),
    ("POST", "/api/substitutions/{id}/approve", handlers.substitution_approve, "engineer approval"),

    ("POST", "/api/redeploy",            handlers.redeploy,           "buy or borrow"),
    ("POST", "/api/duplicate-check",     handlers.dup_check,          "create-time duplicate check"),

    ("GET",  "/api/analytics",           handlers.analytics,          "harmonisation analytics"),
    ("GET",  "/api/value",               handlers.value,              "the transparent value model"),
    ("GET",  "/api/federation",          handlers.federation_report,  "what crosses the trust boundary"),

    ("GET",  "/api/ledger",              handlers.ledger_events,      "the append-only event log"),
    ("GET",  "/api/ledger/verify",       handlers.ledger_verify,      "recompute and verify the hash chain"),
    ("GET",  "/api/ledger/replay",       handlers.ledger_replay,      "rebuild the projection from the log"),
    ("GET",  "/api/ledger/merkle",       handlers.ledger_merkle,      "daily Merkle root"),
    ("POST", "/api/nmc/{code:path}/unmerge", handlers.nmc_unmerge,    "reverse a merge and prove the chain still holds"),

    ("GET",  "/api/erp/change-requests", handlers.change_requests,    "the ERP outbox"),
    ("POST", "/api/erp/sync",            handlers.erp_sync,           "dispatch queued change requests"),

    ("GET",  "/api/kg/compare",          handlers.kg_compare,         "ask the standards graph about two designations"),
    ("GET",  "/api/uom/resolve",         handlers.uom_resolve,        "resolve a unit of measure"),
    ("GET",  "/api/extract",             handlers.extract_preview,    "parse one description into attributes"),
    ("GET",  "/api/routes",              handlers.routes,             "this list"),
]

_PARAM = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)(:path)?\}")


def _compile(pattern: str) -> re.Pattern:
    def sub(m: re.Match) -> str:
        name, is_path = m.group(1), m.group(2)
        # ":path" params match slashes too, non-greedily, so a trailing literal
        # segment (".../validate") still wins over the parameter.
        return f"(?P<{name}>.+?)" if is_path else f"(?P<{name}>[^/]+)"
    return re.compile(f"^{_PARAM.sub(sub, pattern)}$")


_COMPILED = [(method, _compile(path), handler, path, summary)
             for method, path, handler, summary in ROUTES]


class Router:
    @staticmethod
    def match(method: str, path: str) -> Tuple[Optional[Handler], dict, Optional[str]]:
        allowed: List[str] = []
        for m, rx, handler, raw, _summary in _COMPILED:
            hit = rx.match(path)
            if hit:
                if m == method.upper():
                    return handler, hit.groupdict(), raw
                allowed.append(m)
        return None, {"_allowed": allowed}, None


def dispatch(method: str, path: str, query: dict, body: dict) -> Tuple[int, dict]:
    handler, path_params, _raw = Router.match(method, path)
    if handler is None:
        allowed = path_params.get("_allowed") or []
        if allowed:
            return 405, {"error": f"{method} not allowed on {path}", "allowed": allowed}
        return 404, {"error": f"no route for {method} {path}",
                     "hint": "GET /api/routes lists every endpoint"}
    params = dict(query or {})
    params.update(path_params)
    try:
        result = handler(params, body or {})
    except ValueError as exc:
        return 400, {"error": str(exc)}
    except KeyError as exc:
        return 400, {"error": f"missing field: {exc}"}
    except Exception as exc:  # pragma: no cover - surfaced to the caller, not swallowed
        import traceback
        return 500, {"error": str(exc), "type": type(exc).__name__,
                     "traceback": traceback.format_exc().splitlines()[-6:]}
    if isinstance(result, tuple) and len(result) == 2:
        return int(result[0]), result[1]
    if isinstance(result, dict) and "error" in result and len(result) <= 4:
        return 400, result
    return 200, result
