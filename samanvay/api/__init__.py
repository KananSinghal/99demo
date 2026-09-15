"""Pure handler functions (dict in, dict out) plus a tiny router.

Nothing here imports a web framework, which is why the same handlers are served by
both the stdlib server and the FastAPI adapter without a line of duplication.
"""

from .router import Router, ROUTES, dispatch          # noqa: F401
