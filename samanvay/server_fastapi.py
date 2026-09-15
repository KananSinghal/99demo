"""FastAPI adapter.

Optional. Exactly the same routes, served through uvicorn with OpenAPI docs at /docs,
for teams who want them. It imports the same handlers as samanvay.server, so the two
can never drift apart.

    pip install fastapi uvicorn
    uvicorn samanvay.server_fastapi:app --reload
"""

from __future__ import annotations

from typing import Any, Dict

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from . import __version__, config
from .api.router import ROUTES, dispatch
from .store import connect, init_db

app = FastAPI(
    title="Samanvay",
    version=__version__,
    description=(
        "AI-driven standardisation and harmonisation of material codes across CPSEs "
        "(SIH26099, Ministry of Petroleum & Natural Gas).\n\n"
        "**Similarity proposes. Standards decide. Stewards approve. The ledger remembers.**"
    ),
)


@app.on_event("startup")
def _startup() -> None:
    conn = connect()
    init_db(conn)
    conn.close()


def _make_endpoint(method: str, path: str, summary: str):
    async def endpoint(request: Request) -> JSONResponse:
        query: Dict[str, Any] = dict(request.query_params)
        body: Dict[str, Any] = {}
        if method == "POST":
            try:
                parsed = await request.json()
                body = parsed if isinstance(parsed, dict) else {"records": parsed}
            except Exception:
                body = {}
        status, payload = dispatch(method, request.url.path, query, body)
        return JSONResponse(status_code=status, content=payload)

    endpoint.__name__ = f"{method.lower()}_{path.strip('/').replace('/', '_').replace('{', '').replace('}', '')}"
    endpoint.__doc__ = summary
    return endpoint


for _method, _path, _handler, _summary in ROUTES:
    app.add_api_route(
        _path, _make_endpoint(_method, _path, _summary),
        methods=[_method], summary=_summary, tags=[_path.split("/")[2] if len(_path.split("/")) > 2 else "api"],
    )

# The console is served last so it does not shadow the API routes.
app.mount("/", StaticFiles(directory=str(config.WEB_DIR), html=True), name="console")
