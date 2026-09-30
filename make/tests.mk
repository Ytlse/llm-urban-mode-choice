# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────

.PHONY: tests test-gateway test-mobility test-all lint-imports lint typecheck

# Interpreter for the packages (installed as editable in the llm-agents venv).
PKG_PYTHON ?= $(VENV_PYTHON)

## Tests of the generic LLM gateway (unit + contract + integration; e2e if LLM_GATEWAY_E2E_URL)
test-gateway:
	cd packages/llm_gateway && $(PKG_PYTHON) -m pytest

## Tests of the mobility domain (mobility_core) and of the LLM categories (mobility_llm)
test-mobility:
	cd packages/mobility_core && $(PKG_PYTHON) -m pytest
	cd packages/mobility_llm && $(PKG_PYTHON) -m pytest

## The three packages, then the architecture contracts
test-all: test-gateway test-mobility lint-imports

## Historical alias
tests: test-all

## import-linter contracts of the three packages (.importlinter at the root)
lint-imports:
	$(PKG_PYTHON) -c "from importlinter.cli import lint_imports_command; lint_imports_command()"

## ruff on the three packages
lint:
	$(PKG_PYTHON) -m ruff check packages/llm_gateway packages/mobility_core packages/mobility_llm

## mypy: public contract of the gateway (core, ports, sdk) in strict mode
typecheck:
	cd packages/llm_gateway && $(PKG_PYTHON) -m mypy src/llm_gateway/core src/llm_gateway/ports src/llm_gateway/sdk
