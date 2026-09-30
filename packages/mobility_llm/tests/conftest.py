"""Markers per folder: unit/ and e2e/."""
from __future__ import annotations

import pytest


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        parts = set(item.path.parts)
        if "e2e" in parts:
            item.add_marker(pytest.mark.e2e)
        elif "unit" in parts:
            item.add_marker(pytest.mark.unit)
