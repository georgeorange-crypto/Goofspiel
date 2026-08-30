"""Validate that REQUIREMENTS_TRACE.md contains no unfinished acceptance rows."""

from __future__ import annotations

from pathlib import Path

FORBIDDEN_STATUSES = {"PARTIAL", "MISSING", "BLOCKED"}


def validate_trace(path: str | Path = "REQUIREMENTS_TRACE.md") -> list[str]:
    trace = Path(path).read_text(encoding="utf-8")
    errors: list[str] = []
    rows = [line for line in trace.splitlines() if line.startswith("|") and not line.startswith("|---")]
    data_rows = [line for line in rows if line.count("|") >= 6 and not line.startswith("| ID ")]
    root = Path(path).resolve().parent
    for line in data_rows:
        cols = [part.strip() for part in line.strip("|").split("|")]
        if len(cols) < 6:
            continue
        req_id, _source, _requirement, status, implementation, tests = cols[:6]
        if status in FORBIDDEN_STATUSES:
            errors.append(f"{req_id} has unfinished status {status}")
        if status != "DONE":
            errors.append(f"{req_id} is not DONE: {status}")
        for field_name, field in (("implementation", implementation), ("tests", tests)):
            for raw in field.split(","):
                candidate = raw.strip().strip("`")
                if candidate and not (root / candidate).exists():
                    errors.append(f"{req_id} references missing {field_name} path: {candidate}")
    rows = [line for line in data_rows if "| DONE |" in line]
    if len(rows) < 10:
        errors.append("trace has too few DONE rows to cover the order specs")
    order_dir = root / "order"
    if order_dir.exists():
        for doc in sorted(order_dir.glob("*.md")):
            if doc.name not in trace:
                errors.append(f"order document is not represented in trace: {doc.name}")
    return errors


def main() -> None:
    errors = validate_trace()
    if errors:
        raise SystemExit("\n".join(errors))
    print("REQUIREMENTS_TRACE.md OK")


if __name__ == "__main__":
    main()
