"""Canonical data-schema constants and encoding-boundary helpers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from typing import Any

MAX_N = 13
DOMAIN_MIN_RANK = 1
DOMAIN_MAX_RANK = 13
TENSOR_MIN_INDEX = 0
TENSOR_MAX_INDEX = 12
PAD_INDEX = 13
IGNORE_INDEX = -100
SCHEMA_VERSION = "goofspiel.training.v1"


def rank_to_index(rank: int) -> int:
    if not (DOMAIN_MIN_RANK <= int(rank) <= DOMAIN_MAX_RANK):
        raise ValueError(f"rank must be in [1,13], got {rank}")
    return int(rank) - 1


def index_to_rank(index: int) -> int:
    if not (TENSOR_MIN_INDEX <= int(index) <= TENSOR_MAX_INDEX):
        raise ValueError(f"index must be in [0,12], got {index}")
    return int(index) + 1


def canonical_json(obj: Any) -> str:
    if is_dataclass(obj):
        obj = asdict(obj)
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_state_hash(state: Any) -> str:
    return hashlib.sha256(canonical_json(state).encode("utf-8")).hexdigest()
