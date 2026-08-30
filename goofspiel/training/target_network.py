"""Target-network lifecycle helpers."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TargetNetworkConfig:
    ema_tau: float = 0.005
    hard_update_interval: int = 1000


class TargetNetwork:
    def __init__(self, model: Any, config: TargetNetworkConfig | None = None) -> None:
        self.config = config or TargetNetworkConfig()
        self.module = copy.deepcopy(model)
        for param in self.module.parameters():
            param.requires_grad_(False)

    def hard_update(self, model: Any) -> None:
        self.module.load_state_dict(model.state_dict())

    def ema_update(self, model: Any) -> None:
        src = dict(model.named_parameters())
        for name, tgt in self.module.named_parameters():
            tgt.data.mul_(1.0 - self.config.ema_tau).add_(src[name].data, alpha=self.config.ema_tau)
        src_buffers = dict(model.named_buffers())
        for name, tgt in self.module.named_buffers():
            if name in src_buffers:
                tgt.data.copy_(src_buffers[name].data)

    def maybe_update(self, model: Any, global_step: int) -> None:
        if global_step > 0 and global_step % self.config.hard_update_interval == 0:
            self.hard_update(model)
        else:
            self.ema_update(model)
