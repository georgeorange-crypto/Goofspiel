"""Small helpers for gradient-routing tests."""

from __future__ import annotations

import torch.nn as nn


def zero_existing_gradients(module: nn.Module) -> None:
    for param in module.parameters():
        param.grad = None


def assert_no_forbidden_gradients(module: nn.Module) -> None:
    offenders = []
    for name, param in module.named_parameters():
        if param.grad is not None and param.grad.detach().abs().max().item() > 0:
            offenders.append(name)
    if offenders:
        raise AssertionError("forbidden gradients: " + ", ".join(offenders[:10]))
