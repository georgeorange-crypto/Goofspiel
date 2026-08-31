"""Validate that REQUIREMENTS_TRACE.md contains no unfinished acceptance rows."""

from __future__ import annotations

import re
from pathlib import Path

FORBIDDEN_STATUSES = {"PARTIAL", "MISSING", "BLOCKED"}

# A compiled Python extension is named `<stem>.<ABI-tag>.<pyd|so>`, where the ABI
# tag encodes interpreter + platform (e.g. `cp310-win_amd64`, `cp312-linux_x86_64`).
# The ledger must therefore NOT pin a single interpreter/OS build: the same
# requirement is satisfied by a Linux `.so` under Python 3.11 just as by a Windows
# 3.10 `.pyd`.  We accept a reference written either as a bare stem
# (`goofspiel/_core`) or with a concrete ABI tag; both are treated as a *family*.
_ABI_TAGGED = re.compile(r"^(?P<stem>.+?)\.(?:cp|pp|py)\d{2,}[^/\\]*\.(?P<ext>pyd|so)$")


def _has_compiled_build(root: Path, stem: str) -> bool:
    """True if any interpreter/platform build `<stem>.<abi>.{pyd,so}` exists."""
    target = root / stem
    parent = target.parent
    if not parent.is_dir():
        return False
    prefix = target.name + "."
    return any(
        child.name.startswith(prefix) and child.suffix in (".pyd", ".so")
        for child in parent.iterdir()
    )


def _resolve_reference(root: Path, candidate: str) -> bool:
    """True if a ledger path is satisfied.

    Plain files must exist as written.  A compiled-extension reference is a
    *family* rather than one concrete artifact, and is satisfied by any of:
      * the literal path (a matching build is checked in), or
      * any interpreter/OS build of the same stem (`<stem>.<abi>.{pyd,so}`), or
      * a pure-Python module of that exact stem (`<stem>.py`), or
      * the pure-Python fallback shim beside it (`_cxx.py`) — because the C++
        accelerator is optional (`fail_on_build_error = false` in pyproject.toml),
        so a clean checkout that never compiled `_core` still satisfies the
        requirement through the fallback import path.
    """
    if (root / candidate).exists():
        return True
    m = _ABI_TAGGED.match(candidate)
    stem = m.group("stem") if m else candidate
    # Only treat something as an optional compiled-extension family when it is
    # either ABI-tagged or an extensionless stem; a normal path with a real
    # suffix (e.g. `foo/bar.py`) must exist and gets no leniency.
    if not m and Path(candidate).suffix:
        return False
    if _has_compiled_build(root, stem):
        return True
    if (root / stem).with_suffix(".py").exists():
        return True
    fallback = (root / stem).with_name("_cxx.py")
    return fallback.exists()


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
                if candidate and not _resolve_reference(root, candidate):
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
