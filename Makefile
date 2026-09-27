.PHONY: help dev test lint markdownlint pycompile clean all

help:
	@echo "llm-audit-bench - AuditBench v2 and Capability Check v1"
	@echo ""
	@echo "  make dev   - Create the uv-managed venv and install the CLIs"
	@echo "  make test  - Run the unit tests"
	@echo "  make lint  - Run all linters"
	@echo "  make all   - Run lint and test"
	@echo "  make clean - Remove the venv and build artifacts"

dev:
	uv sync

test:
	@uv run python -m unittest discover -s tests -p 'test_*.py'

lint: markdownlint pycompile

markdownlint:
	markdownlint-cli2

pycompile:
	@uv run python -m compileall -q src tests

all: lint test

clean:
	rm -rf .venv build dist src/llm_audit_bench.egg-info \
		__pycache__ src/__pycache__ tests/__pycache__
