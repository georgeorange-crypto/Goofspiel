from __future__ import annotations

from scripts.validate_requirements_trace import validate_trace


def test_requirements_trace_has_no_unfinished_rows():
    assert validate_trace("REQUIREMENTS_TRACE.md") == []
