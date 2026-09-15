SHELL := /bin/bash
PY := python3
PORT ?= 8000

.PHONY: help demo reset serve test bench seed clean

help:
	@echo "Samanvay -- make targets:"
	@echo "  make demo     reset the DB to a clean state, load the synthetic corpus,"
	@echo "                run the cascade, then start the server on :$(PORT)"
	@echo "  make reset    same as demo, minus starting the server"
	@echo "  make serve    start the server against whatever is already in the DB"
	@echo "  make test     run the full test suite (stdlib unittest, no deps)"
	@echo "  make bench    run the benchmark against synthetic ground truth"
	@echo "  make seed     regenerate var/corpus.json only, no DB changes"
	@echo "  make clean    remove generated var/ artifacts (DB, corpus, logs)"
	@echo ""
	@echo "PORT=8080 make demo   overrides the port (default $(PORT))"

demo: reset
	$(PY) -m samanvay.server --port $(PORT)

reset:
	$(PY) scripts/reset_demo.py

serve:
	$(PY) -m samanvay.server --port $(PORT)

test:
	$(PY) -m unittest discover -s tests -t . -v

bench:
	$(PY) scripts/benchmark.py --strict

seed:
	$(PY) scripts/seed.py --out var/corpus.json

clean:
	rm -f var/*.db var/*.db-wal var/*.db-shm var/corpus.json var/bench_corpus.json \
	      var/benchmark.json var/conformal.json var/scorer.json var/server.log \
	      var/ui-*.png
	rm -rf var/outbox
	@mkdir -p var/outbox
	@echo "var/ cleaned. Run 'make demo' to rebuild."
