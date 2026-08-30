"""Checkpoint kind registry for promotion, resume, and artifact discovery."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from goofspiel.training.checkpoint import sha256_file

CHECKPOINT_KINDS = (
    "latest",
    "best_robust",
    "best_raw",
    "best_search",
    "best_adaptive",
    "best_generalization",
    "best_opponent_model",
    "teacher_ema",
)


@dataclass
class CheckpointRegistryEntry:
    kind: str
    source_path: str
    registry_path: str
    sha256: str
    global_step: int
    metrics: dict[str, float] = field(default_factory=dict)


class CheckpointRegistry:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "checkpoint_registry.json"

    def register(
        self,
        kind: str,
        checkpoint_path: str | Path,
        *,
        global_step: int,
        metrics: dict[str, float] | None = None,
    ) -> CheckpointRegistryEntry:
        if kind not in CHECKPOINT_KINDS:
            raise ValueError(f"unsupported checkpoint kind {kind!r}; expected one of {CHECKPOINT_KINDS}")
        src = Path(checkpoint_path)
        if not src.exists():
            raise FileNotFoundError(src)
        dst = self.root / f"{kind}{src.suffix}"
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
            manifest_src = src.with_suffix(src.suffix + ".sha256.json")
            if manifest_src.exists():
                shutil.copy2(manifest_src, dst.with_suffix(dst.suffix + ".sha256.json"))
        entry = CheckpointRegistryEntry(
            kind=kind,
            source_path=str(src),
            registry_path=str(dst),
            sha256=sha256_file(dst),
            global_step=int(global_step),
            metrics=metrics or {},
        )
        index = self.read()
        index[kind] = asdict(entry)
        self.index_path.write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")
        return entry

    def read(self) -> dict[str, Any]:
        if not self.index_path.exists():
            return {}
        return json.loads(self.index_path.read_text(encoding="utf-8"))

    def missing_kinds(self) -> list[str]:
        index = self.read()
        return [kind for kind in CHECKPOINT_KINDS if kind not in index]
