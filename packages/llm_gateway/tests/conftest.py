"""Markers per tier: the folder of a test says what it is (unit, contract, integration, e2e).

A test can also carry its marker explicitly; `--strict-markers` rejects unknown ones.
"""
from __future__ import annotations

import pytest

_BY_DIR = ("unit", "contract", "integration", "e2e")


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        parts = set(item.path.parts)
        for marker in _BY_DIR:
            if marker in parts:
                item.add_marker(getattr(pytest.mark, marker))
                break
