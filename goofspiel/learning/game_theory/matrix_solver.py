"""Matrix-game solution dataclass."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass
class MatrixGameSolution:
    row_policy: Tensor
    column_policy: Tensor
    value: Tensor
    duality_gap: Tensor
    iterations: int
    valid: Tensor | None = None
