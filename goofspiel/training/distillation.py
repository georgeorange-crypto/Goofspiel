"""Strong and fast student distillation interfaces."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DistillationPlan:
    teacher_checkpoint: str
    student_kind: str
    temperature: float = 1.0
    alpha_policy: float = 1.0
    alpha_value: float = 0.5


def strong_student_plan(teacher_checkpoint: str) -> DistillationPlan:
    return DistillationPlan(teacher_checkpoint=teacher_checkpoint, student_kind="strong")


def fast_student_plan(teacher_checkpoint: str) -> DistillationPlan:
    return DistillationPlan(teacher_checkpoint=teacher_checkpoint, student_kind="fast", temperature=1.5, alpha_value=0.25)
