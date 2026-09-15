"""Samanvay - AI-driven standardisation and harmonisation of material codes across CPSEs.

SIH26099 | Ministry of Petroleum & Natural Gas.

Design maxim:  Similarity proposes. Standards decide. Stewards approve. The ledger remembers.

The package is layered so that each layer can be tested without the one above it:

    samanvay.core        pure resolution engine - stdlib only (numpy used if present)
    samanvay.store       sqlite3 persistence + hash-chained event ledger
    samanvay.api         pure handler functions (dict in, dict out) + a tiny router
    samanvay.server_*    two interchangeable HTTP adapters (stdlib / FastAPI)
    samanvay.connectors  SAP and generic ERP connectors
    samanvay.web         build-free single-page console
"""

__version__ = "1.0.0"
__all__ = ["__version__"]
