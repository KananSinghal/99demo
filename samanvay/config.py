"""Runtime configuration. Everything is overridable by environment variable so the
same build runs on a laptop, in a CPSE data centre, and inside an air-gapped node."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("SAMANVAY_DATA_DIR", ROOT / "data"))
VAR_DIR = Path(os.environ.get("SAMANVAY_VAR_DIR", ROOT / "var"))
WEB_DIR = Path(__file__).resolve().parent / "web"

VAR_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = Path(os.environ.get("SAMANVAY_DB", VAR_DIR / "samanvay.db"))
OUTBOX_DIR = Path(os.environ.get("SAMANVAY_OUTBOX", VAR_DIR / "outbox"))

# ---------------------------------------------------------------- node identity
# A deployment is either the national registry or one CPSE edge node. The edge
# node never forwards price, stock, vendor or consumption data across the boundary.
NODE_ROLE = os.environ.get("SAMANVAY_NODE_ROLE", "registry")  # registry | cpse_node
NODE_CPSE = os.environ.get("SAMANVAY_NODE_CPSE", "")
REGISTRY_URL = os.environ.get("SAMANVAY_REGISTRY_URL", "")

# Fields that are stripped from any payload leaving a CPSE node. See core.federation.
CONFIDENTIAL_FIELDS = (
    "unit_price", "unit_price_base", "currency", "stock_qty", "stock_value",
    "vendor_id", "vendor_name", "annual_demand", "annual_spend", "last_po_date",
)

# ---------------------------------------------------------------- engine tuning
# Embedding backend: "hash" needs nothing at all and is deterministic; "st" uses
# sentence-transformers when the team has installed it and has the weights cached.
EMBEDDING_BACKEND = os.environ.get("SAMANVAY_EMBEDDING", "hash")
EMBEDDING_MODEL = os.environ.get("SAMANVAY_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
EMBEDDING_DIM = int(os.environ.get("SAMANVAY_EMBEDDING_DIM", "256"))

# Reranker backend: "features" is a numpy logistic model trained on distantly
# supervised labels; "cross" uses a sentence-transformers CrossEncoder if present.
RERANK_BACKEND = os.environ.get("SAMANVAY_RERANK", "features")
RERANK_MODEL = os.environ.get("SAMANVAY_RERANK_MODEL", "BAAI/bge-reranker-base")

ANN_TOP_K = int(os.environ.get("SAMANVAY_ANN_TOP_K", "50"))
BI_ENCODER_CUTOFF = float(os.environ.get("SAMANVAY_BI_CUTOFF", "0.42"))
MAX_CLUSTER_SIZE = int(os.environ.get("SAMANVAY_MAX_CLUSTER", "60"))

# ---------------------------------------------------------------- risk appetite
# Conformal alpha by consequence, not one number for everything. A gasket on a
# pressure boundary and a whiteboard marker do not deserve the same error budget.
DEFAULT_ALPHA = float(os.environ.get("SAMANVAY_ALPHA", "0.01"))
CLASS_ALPHA = {
    "VALVE": 0.001,
    "PIPE": 0.001,
    "PIPE_FITTING": 0.001,
    "GASKET": 0.001,
    "FASTENER_BOLT": 0.001,
    "FASTENER_NUT": 0.001,
    "BEARING": 0.01,
    "CABLE": 0.01,
    "GENERIC": 0.05,
}
# Classes where auto-accept additionally requires a hard identity key
# (manufacturer part number or vendor material number), never text alone.
IDENTITY_KEY_REQUIRED = {"VALVE", "PIPE", "PIPE_FITTING", "GASKET", "FASTENER_BOLT", "FASTENER_NUT"}

# ---------------------------------------------------------------- registry / NMC
NCB_CODE = os.environ.get("SAMANVAY_NCB_CODE", "72")  # India. Verify against ACodP-1.
NMC_SERIAL_START = int(os.environ.get("SAMANVAY_NMC_SERIAL_START", "147723"))

# ---------------------------------------------------------------- economics
INVENTORY_CARRYING_RATE = float(os.environ.get("SAMANVAY_CARRYING_RATE", "0.22"))
POOLING_REALISABILITY = float(os.environ.get("SAMANVAY_POOLING_REALISABILITY", "0.45"))
CODE_UPKEEP_COST_PER_YEAR = float(os.environ.get("SAMANVAY_CODE_UPKEEP", "850"))
NON_MOVING_DAYS = int(os.environ.get("SAMANVAY_NON_MOVING_DAYS", "365"))

# ---------------------------------------------------------------- server
HOST = os.environ.get("SAMANVAY_HOST", "127.0.0.1")
PORT = int(os.environ.get("SAMANVAY_PORT", "8000"))


def alpha_for(class_code: str) -> float:
    return CLASS_ALPHA.get(class_code, DEFAULT_ALPHA)


def as_dict() -> dict:
    return {
        "version": "1.0.0",
        "node_role": NODE_ROLE,
        "node_cpse": NODE_CPSE,
        "embedding_backend": EMBEDDING_BACKEND,
        "embedding_dim": EMBEDDING_DIM,
        "rerank_backend": RERANK_BACKEND,
        "ann_top_k": ANN_TOP_K,
        "bi_encoder_cutoff": BI_ENCODER_CUTOFF,
        "default_alpha": DEFAULT_ALPHA,
        "class_alpha": CLASS_ALPHA,
        "ncb_code": NCB_CODE,
        "db": str(DB_PATH),
    }
