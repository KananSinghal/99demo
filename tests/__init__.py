"""Samanvay test suite.

Zero test dependencies on purpose, matching the rest of the build: everything
here runs with the stdlib `unittest` runner, nothing else. From the repo root:

    python3 -m unittest discover -s tests -t . -v

or simply `make test`.
"""
