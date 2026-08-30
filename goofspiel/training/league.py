"""League registry, role integrity, and opponent sampling."""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


ROLE_ROBUST = "ROBUST"
ROLE_AGGRESSIVE = "AGGRESSIVE"
ROLE_EXPLOITER = "EXPLOITER"
LEAGUE_ROLES = (ROLE_ROBUST, ROLE_AGGRESSIVE, ROLE_EXPLOITER)


@dataclass
class LeagueAgent:
    agent_id: str
    role: str
    checkpoint_path: str | None
    policy_version: int
    metrics: dict[str, float] = field(default_factory=dict)
    frozen: bool = True
    created_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if self.role not in LEAGUE_ROLES:
            raise ValueError(f"unknown league role={self.role!r}")


class LeagueRegistry:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.agents: dict[str, LeagueAgent] = {}
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            for row in data.get("agents", []):
                agent = LeagueAgent(**row)
                self.agents[agent.agent_id] = agent

    def add(self, agent: LeagueAgent) -> None:
        if not agent.frozen:
            raise ValueError("historical league agents must be frozen")
        self.agents[agent.agent_id] = agent
        self.save()

    def save(self) -> None:
        payload = {"agents": [asdict(agent) for agent in sorted(self.agents.values(), key=lambda a: a.agent_id)]}
        self.path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def counts_by_role(self) -> dict[str, int]:
        return {role: sum(1 for agent in self.agents.values() if agent.role == role) for role in LEAGUE_ROLES}

    def sample(self, rng: random.Random | None = None) -> LeagueAgent | None:
        if not self.agents:
            return None
        rng = rng or random.Random()
        agents = list(self.agents.values())
        weights = [max(0.01, agent.metrics.get("priority", 1.0)) for agent in agents]
        return rng.choices(agents, weights=weights, k=1)[0]
