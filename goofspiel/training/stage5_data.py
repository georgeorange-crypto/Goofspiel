"""Stage5 dataset generation, canonical hashing, caching, and budget resolution.

Phase 3B extracts the Stage5 opponent-session dataset out of the monolithic
``run_stage5_adaptive`` loop so that the three budgets it used to conflate can be
controlled independently and *proven* to be independent:

    D  ``opponent_sessions``   how many opponent sessions the dataset contains
    H  ``games_per_session``   how many games share one opponent/session identity
    U  ``adaptation_steps``    how many optimizer updates run over the fixed data

Under the legacy single ``steps`` contract these were all the same number and the
data RNG (``stage_seed``) was itself derived from ``steps`` -- so changing the
optimizer budget silently changed the dataset.  This module makes the coupling
explicit via :func:`resolve_stage5_budget` and a data-contract *version*:

    version 1  legacy: ``stage_seed = seed + 503 + steps + n_cards`` -- byte
               identical to the pre-Phase3B behaviour, reproduced here exactly.
    version 2  decoupled: ``data_seed`` never depends on ``adaptation_steps``, so
               a fixed training set can be reused verbatim across a U-sweep.

Nothing here trains a model or touches the firewall; it only produces and
fingerprints the data that the Stage5 trainer and the held-out validation
evaluator consume.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from goofspiel.game import GameState, legal_cards, transition
from goofspiel.training.adaptive import default_opponent_curriculum, opponent_action_for_regime
from goofspiel.training.data import JsonlStore, OpponentSession, RoundRecord, _to_jsonable

# The historical, hard-coded games-per-session used by every pre-Phase3B Stage5
# run.  Kept as the version-1 / unspecified default so legacy callers are byte
# identical.
LEGACY_GAMES_PER_SESSION = 3

# Deterministic, U-independent seed offsets used only when the caller does not
# pass explicit seeds under the decoupled (version-2) contract.  The validation
# offset is a large prime-ish constant so the held-out stream never collides with
# the training stream for any realistic (seed, D) pair.
_V2_VALIDATION_SEED_OFFSET = 90_007


@dataclass(frozen=True)
class ResolvedStage5Budget:
    """The fully-resolved Stage5 budget after legacy/decoupled reconciliation.

    ``contract_version`` records which seed contract produced ``data_seed``:
    1 = legacy (``seed+503+steps+n_cards``), 2 = decoupled (``data_seed`` is
    independent of ``adaptation_steps``).  ``optimization_seed`` and
    ``validation_seed`` are ``None`` under the legacy contract (legacy Stage5
    seeded neither and had no held-out set), which is how the trainer knows to
    preserve byte-identical legacy behaviour.
    """

    contract_version: int
    opponent_sessions: int  # D
    games_per_session: int  # H
    adaptation_steps: int  # U
    n_cards: int
    data_seed: int
    optimization_seed: int | None
    validation_seed: int | None
    validation_sessions: int
    legacy_steps: int | None


def resolve_stage5_budget(
    *,
    steps: int | None = None,
    opponent_sessions: int | None = None,
    games_per_session: int | None = None,
    adaptation_steps: int | None = None,
    n_cards: int,
    seed: int,
    data_seed: int | None = None,
    optimization_seed: int | None = None,
    validation_seed: int | None = None,
    validation_sessions: int | None = None,
    contract_version: int | None = None,
) -> ResolvedStage5Budget:
    """Reconcile the legacy ``steps`` knob with the decoupled D/H/U API.

    The contract version is auto-selected: any explicit decoupled field (or an
    explicit ``contract_version=2``) selects the decoupled contract; otherwise
    the legacy contract is used so a bare ``steps=`` call is byte identical to
    the pre-Phase3B implementation.

    Under the legacy contract the historical derivations are preserved exactly:
    ``D = U = steps``, ``H = 3``, ``data_seed = seed + 503 + steps + n_cards``.

    Under the decoupled contract ``data_seed`` depends only on ``(seed, D,
    n_cards)`` -- crucially **never** on ``adaptation_steps`` -- so a fixed
    dataset is reused verbatim while U is swept.  ``optimization_seed`` defaults
    to ``seed`` and ``validation_seed`` to ``seed + 90007``; both are independent
    of U.
    """
    decoupled_signal = any(
        v is not None
        for v in (
            opponent_sessions,
            games_per_session,
            adaptation_steps,
            data_seed,
            optimization_seed,
            validation_seed,
            validation_sessions,
        )
    )
    if contract_version is None:
        contract_version = 2 if decoupled_signal else 1
    if contract_version not in (1, 2):
        raise ValueError(f"stage5_data_contract_version must be 1 or 2, got {contract_version!r}")

    if contract_version == 1:
        if steps is None:
            raise ValueError("legacy Stage5 data contract (version 1) requires steps=")
        legacy_stage_seed = int(seed) + 503 + int(steps) + int(n_cards)
        d = int(opponent_sessions) if opponent_sessions is not None else int(steps)
        u = int(adaptation_steps) if adaptation_steps is not None else int(steps)
        h = int(games_per_session) if games_per_session is not None else LEGACY_GAMES_PER_SESSION
        ds = int(data_seed) if data_seed is not None else legacy_stage_seed
        vsess = int(validation_sessions) if validation_sessions is not None else 0
        return ResolvedStage5Budget(
            contract_version=1,
            opponent_sessions=max(1, d),
            games_per_session=max(1, h),
            adaptation_steps=max(1, u),
            n_cards=int(n_cards),
            data_seed=ds,
            optimization_seed=int(optimization_seed) if optimization_seed is not None else None,
            validation_seed=int(validation_seed) if validation_seed is not None else None,
            validation_sessions=max(0, vsess),
            legacy_steps=int(steps),
        )

    # Decoupled (version-2) contract.
    if opponent_sessions is None and steps is None:
        raise ValueError(
            "decoupled Stage5 data contract (version 2) requires opponent_sessions= "
            "(steps= is accepted only as a fallback for D)"
        )
    d = int(opponent_sessions) if opponent_sessions is not None else int(steps)
    u = (
        int(adaptation_steps)
        if adaptation_steps is not None
        else (int(steps) if steps is not None else d)
    )
    h = int(games_per_session) if games_per_session is not None else LEGACY_GAMES_PER_SESSION
    d = max(1, d)
    # data_seed depends only on (seed, D, n_cards) -- NEVER on U -- so the exact
    # same training set is regenerated for every adaptation_steps in a sweep.
    ds = int(data_seed) if data_seed is not None else (int(seed) + 503 + d + int(n_cards))
    os_ = int(optimization_seed) if optimization_seed is not None else int(seed)
    vs = int(validation_seed) if validation_seed is not None else (int(seed) + _V2_VALIDATION_SEED_OFFSET)
    vsess = int(validation_sessions) if validation_sessions is not None else max(1, d // 5)
    return ResolvedStage5Budget(
        contract_version=2,
        opponent_sessions=d,
        games_per_session=max(1, h),
        adaptation_steps=max(1, u),
        n_cards=int(n_cards),
        data_seed=ds,
        optimization_seed=os_,
        validation_seed=vs,
        validation_sessions=max(1, vsess),
        legacy_steps=int(steps) if steps is not None else None,
    )


def generate_opponent_sessions(
    *,
    opponent_sessions: int,
    games_per_session: int,
    n_cards: int,
    data_seed: int,
) -> list[OpponentSession]:
    """Generate opponent sessions with the **exact** legacy RNG order.

    This is a verbatim extraction of the pre-Phase3B generation loop in
    ``_run_stage5_adaptive_rank0``.  For legacy parameters
    (``opponent_sessions = max(1, steps)``, ``games_per_session = 3``,
    ``data_seed = seed + 503 + steps + n_cards``) it produces byte-identical
    ``opponent_sessions.jsonl`` content -- the RNG draw sequence per round is
    ``first_prize`` (once per game) then, each round, ``self_action`` ->
    ``opponent_action_for_regime`` -> ``next_prize``.
    """
    rng = random.Random(int(data_seed))
    regimes = default_opponent_curriculum()
    sessions: list[OpponentSession] = []
    for session_idx in range(max(1, int(opponent_sessions))):
        regime = regimes[session_idx % len(regimes)]
        games: list[list[RoundRecord]] = []
        for _game_idx in range(max(1, int(games_per_session))):
            first_prize = rng.choice(list(range(1, n_cards + 1)))
            state = GameState.initial(n_cards, current_prize=first_prize)
            rounds: list[RoundRecord] = []
            while not state.done:
                self_action = rng.choice(state.self_actions)
                opp_action = opponent_action_for_regime(
                    regime.regime_id,
                    state.opponent_actions,
                    stake=state.stake,
                    n_cards=n_cards,
                    rng=rng,
                )
                next_prize = rng.choice(legal_cards(state.prize_mask, state.n)) if state.prize_mask else None
                result = transition(state, self_action, opp_action, next_prize=next_prize)
                rounds.append(
                    RoundRecord(
                        round_index=state.round_index,
                        prize=state.current_prize,
                        self_action=self_action,
                        opponent_action=opp_action,
                        reward_self=result.reward_self,
                        reward_opponent=result.reward_opp,
                        carry_in=state.carry_pool,
                        carry_out=result.state.carry_pool,
                        done=result.state.done,
                    )
                )
                state = result.state
            games.append(rounds)
        sessions.append(
            OpponentSession(
                session_id=f"adaptive_session_{session_idx}:seed{int(data_seed)}:n{n_cards}",
                opponent_id=f"curriculum_{regime.regime_id}",
                strategy_regime_id=regime.regime_id,
                games=games,
            )
        )
    return sessions


def _canonical_json(obj) -> str:
    """Stable, whitespace-free JSON with sorted keys -- the canonical byte form."""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def dataset_sha256(sessions: Sequence[OpponentSession]) -> str:
    """Content hash over the *ordered* sessions.

    Hashes the exact jsonable content each session is persisted as (session id,
    opponent id, regime, and every round's boundaries/prize/actions/rewards/carry
    -- i.e. the observations, labels and legal-action-determining fields), so two
    datasets share a hash iff they would serialise identically.  Order is part of
    the identity (generation order is deterministic and meaningful).
    """
    payload = [_to_jsonable(s) for s in sessions]
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def canonical_session_content(session: OpponentSession) -> str:
    """A per-session content key with the seed-bearing ``session_id`` removed.

    Used for train/validation overlap detection: two sessions collide iff their
    opponent identity, regime and full round content are identical, regardless of
    which seed/index produced them.
    """
    payload = _to_jsonable(session)
    payload.pop("session_id", None)
    return _canonical_json(payload)


def train_val_overlap_count(
    train_sessions: Sequence[OpponentSession],
    val_sessions: Sequence[OpponentSession],
) -> int:
    """Number of validation sessions whose content also appears in training.

    Overlap is measured on :func:`canonical_session_content` (seed-independent),
    so a non-zero count is a genuine content duplication, not a seed coincidence.
    """
    train_keys = {canonical_session_content(s) for s in train_sessions}
    return sum(1 for s in val_sessions if canonical_session_content(s) in train_keys)


def row_count(sessions: Sequence[OpponentSession]) -> int:
    """Total number of decision rounds (training rows) across all sessions/games."""
    return sum(len(game) for s in sessions for game in s.games)


def write_sessions_jsonl(sessions: Sequence[OpponentSession], path: str | Path, *, reset: bool = True) -> Path:
    """Persist sessions as JSONL using the same serialisation as :class:`JsonlStore`.

    With ``reset=True`` (the default) any existing file is removed first, matching
    the legacy Stage5 behaviour of rewriting ``opponent_sessions.jsonl`` from
    scratch on a non-resume run.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if reset:
        path.unlink(missing_ok=True)
    store: JsonlStore = JsonlStore(path)
    for session in sessions:
        store.append(session)
    return path


def sessions_from_jsonl(path: str | Path) -> list[OpponentSession]:
    """Reconstruct :class:`OpponentSession` objects from a persisted JSONL file."""
    store: JsonlStore = JsonlStore(Path(path))
    round_fields = {
        "round_index",
        "prize",
        "self_action",
        "opponent_action",
        "reward_self",
        "reward_opponent",
        "carry_in",
        "carry_out",
        "done",
    }
    sessions: list[OpponentSession] = []
    for d in store.iter_dicts():
        games = [
            [RoundRecord(**{k: r[k] for k in round_fields if k in r}) for r in game]
            for game in d["games"]
        ]
        sessions.append(
            OpponentSession(
                session_id=d["session_id"],
                opponent_id=d["opponent_id"],
                strategy_regime_id=d["strategy_regime_id"],
                games=games,
            )
        )
    return sessions


@dataclass
class Stage5Dataset:
    """An immutable, fingerprinted Stage5 opponent-session dataset.

    ``role`` is ``"train"`` or ``"validation"``.  ``content_hash`` is the
    :func:`dataset_sha256` of the ordered sessions and is the single fact that
    proves two runs consumed the same data.
    """

    role: str
    contract_version: int
    opponent_sessions: int
    games_per_session: int
    n_cards: int
    data_seed: int
    sessions: list[OpponentSession]
    content_hash: str
    session_count: int
    row_count: int

    @classmethod
    def generate(
        cls,
        *,
        role: str,
        contract_version: int,
        opponent_sessions: int,
        games_per_session: int,
        n_cards: int,
        data_seed: int,
    ) -> "Stage5Dataset":
        sessions = generate_opponent_sessions(
            opponent_sessions=opponent_sessions,
            games_per_session=games_per_session,
            n_cards=n_cards,
            data_seed=data_seed,
        )
        return cls(
            role=role,
            contract_version=int(contract_version),
            opponent_sessions=int(opponent_sessions),
            games_per_session=int(games_per_session),
            n_cards=int(n_cards),
            data_seed=int(data_seed),
            sessions=sessions,
            content_hash=dataset_sha256(sessions),
            session_count=len(sessions),
            row_count=row_count(sessions),
        )

    def manifest(self) -> dict:
        """The persisted, metadata-only manifest (no session payload)."""
        return {
            "role": self.role,
            "stage5_data_contract_version": self.contract_version,
            "opponent_sessions": self.opponent_sessions,
            "games_per_session": self.games_per_session,
            "n_cards": self.n_cards,
            "data_seed": self.data_seed,
            "session_count": self.session_count,
            "row_count": self.row_count,
            "content_hash": self.content_hash,
        }

    def _params_match(self, cached: dict) -> bool:
        return (
            int(cached.get("stage5_data_contract_version", -1)) == self.contract_version
            and int(cached.get("opponent_sessions", -1)) == self.opponent_sessions
            and int(cached.get("games_per_session", -1)) == self.games_per_session
            and int(cached.get("n_cards", -1)) == self.n_cards
            and int(cached.get("data_seed", -1)) == self.data_seed
        )


def load_or_generate_dataset(
    cache_dir: str | Path,
    *,
    role: str,
    contract_version: int,
    opponent_sessions: int,
    games_per_session: int,
    n_cards: int,
    data_seed: int,
) -> tuple[Stage5Dataset, str]:
    """Load a cached dataset if present and parameter-matching, else generate+cache.

    This is the mechanism that guarantees a *fixed* training (or validation) set
    is reused verbatim across an entire U-sweep: the first call generates and
    writes ``<role>_opponent_sessions.jsonl`` + ``<role>_dataset_manifest.json``
    into ``cache_dir``; every later call with the same (contract, D, H, n_cards,
    data_seed) loads that exact jsonl and verifies its content hash.

    Returns ``(dataset, "loaded" | "generated")``.  A cache whose manifest params
    disagree with the request is an error (never silently regenerated), and a
    cache whose reloaded content hash disagrees with its manifest is treated as
    corrupt.
    """
    cache_dir = Path(cache_dir)
    manifest_path = cache_dir / f"{role}_dataset_manifest.json"
    jsonl_path = cache_dir / f"{role}_opponent_sessions.jsonl"

    probe = Stage5Dataset(
        role=role,
        contract_version=int(contract_version),
        opponent_sessions=int(opponent_sessions),
        games_per_session=int(games_per_session),
        n_cards=int(n_cards),
        data_seed=int(data_seed),
        sessions=[],
        content_hash="",
        session_count=0,
        row_count=0,
    )

    if manifest_path.exists() and jsonl_path.exists():
        cached = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not probe._params_match(cached):
            raise ValueError(
                f"Stage5 dataset cache parameter mismatch at {cache_dir} (role={role}): "
                f"cached={ {k: cached.get(k) for k in ('stage5_data_contract_version','opponent_sessions','games_per_session','n_cards','data_seed')} } "
                f"!= requested contract={contract_version} D={opponent_sessions} H={games_per_session} "
                f"n_cards={n_cards} data_seed={data_seed}"
            )
        sessions = sessions_from_jsonl(jsonl_path)
        recomputed = dataset_sha256(sessions)
        if str(cached.get("content_hash")) != recomputed:
            raise ValueError(
                f"Stage5 dataset cache is corrupt at {jsonl_path}: reloaded content_hash "
                f"{recomputed} != manifest {cached.get('content_hash')}"
            )
        dataset = Stage5Dataset(
            role=role,
            contract_version=int(contract_version),
            opponent_sessions=int(opponent_sessions),
            games_per_session=int(games_per_session),
            n_cards=int(n_cards),
            data_seed=int(data_seed),
            sessions=sessions,
            content_hash=recomputed,
            session_count=len(sessions),
            row_count=row_count(sessions),
        )
        return dataset, "loaded"

    dataset = Stage5Dataset.generate(
        role=role,
        contract_version=contract_version,
        opponent_sessions=opponent_sessions,
        games_per_session=games_per_session,
        n_cards=n_cards,
        data_seed=data_seed,
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    write_sessions_jsonl(dataset.sessions, jsonl_path)
    manifest_path.write_text(json.dumps(dataset.manifest(), indent=2, ensure_ascii=False), encoding="utf-8")
    return dataset, "generated"
