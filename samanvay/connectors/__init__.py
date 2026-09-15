"""ERP connectors. SAP is first-class; everything else lands on the same canonical
schema so a non-SAP CPSE joins without a special case."""

from .sap import SapConnector, build_change_request     # noqa: F401
from .generic import CsvConnector, JsonConnector        # noqa: F401
