"""Phase 3B — decoupling Stage5's data / history / adaptation budgets.

The pre-Phase3B Stage5 conflated three distinct quantities into one ``steps``
knob, and derived the *data* RNG seed from it, so changing the optimizer budget
silently changed the training set.  Phase 3B splits them into three scientific
axes and makes their independence *provable*:

    D  opponent_sessions   how many opponent sessions the dataset contains
    H  games_per_session   how many games share one opponent/session identity
    U  adaptation_steps     how many optimizer updates run over the fixed data

The causal question this capability answers is: *given exactly the same training
data, validation data, initialization, RNG and optimizer configuration, how does
opponent-model quality change as the number of optimizer updates grows?*  That
question is only meaningful if U genuinely does not perturb the data — which is
the central fact these tests re-execute.

Per the project testing principle every test RE-EXECUTES the fact rather than
reading a stored field: dataset identity is proven by recomputing the sha256 over
the *persisted bytes*; the firewall by comparing actual saved weights to the
parent; validation neutrality by re-running training with and without an
interleaved validation call and comparing the resulting parameters; legacy
byte-identity by regenerating the sessions independently and diffing the file the
real stage wrote.  No test asserts on a value merely because the code also wrote
that value into a manifest.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

try:
    import torch
except OSError as exc:  # pragma: no cover - machine environment guard
    pytest.skip(f"torch cannot be imported in this environment: {exc}", allow_module_level=True)

import torch.nn.functional as F

from goofspiel.models import GoofspielModel
from goofspiel.training.checkpoint import load_checkpoint
from goofspiel.training.stage5_data import (
    LEGACY_GAMES_PER_SESSION,
    _V2_VALIDATION_SEED_OFFSET,
    Stage5Dataset,
    canonical_session_content,
    dataset_sha256,
    generate_opponent_sessions,
    load_or_generate_dataset,
    resolve_stage5_budget,
    row_count,
    sessions_from_jsonl,
    train_val_overlap_count,
    write_sessions_jsonl,
)
from goofspiel.training.stage5_validation import evaluate_opponent_model, model_param_signature
from goofspiel.training.stages import (
    _build_adaptive_training_tensors,
    run_stage1_pretrain,
    run_stage5_adaptive,
)


# ===========================================================================
# Shared fixtures / helpers
# ===========================================================================
@pytest.fixture(scope="module")
def parent_ckpt(tmp_path_factory) -> str:
    """A real (tiny) Stage4-equivalent parent so init_from_checkpoint is a strict
    full load — the model init is then identical across every Stage5 run for free,
    which is exactly the fixed-initialization condition Phase 3B needs."""
    d = tmp_path_factory.mktemp("p1_parent")
    p1 = run_stage1_pretrain(steps=1, batch_size=4, out_dir=d, n_cards=3, seed=1)
    assert p1.checkpoint is not None
    return p1.checkpoint


def _persisted_train_sessions(out_dir: Path):
    """Reconstruct the training sessions from the file the stage actually wrote."""
    return sessions_from_jsonl(Path(out_dir) / "adaptive" / "opponent_sessions.jsonl")


def _persisted_train_hash(out_dir: Path) -> str:
    """Recompute the dataset sha256 from the persisted bytes (never read a field)."""
    return dataset_sha256(_persisted_train_sessions(out_dir))


def _legacy_data_seed(*, seed: int, steps: int, n_cards: int) -> int:
    """The historical (pre-Phase3B) stage-seed derivation, spelled out here so the
    test pins the exact formula rather than trusting the module to define it."""
    return seed + 503 + steps + n_cards


def _small_sessions(*, D=4, H=2, n_cards=3, data_seed=101):
    return generate_opponent_sessions(
        opponent_sessions=D, games_per_session=H, n_cards=n_cards, data_seed=data_seed
    )


# ===========================================================================
# A. Budget resolution — the coupling is made explicit and correct.
# ===========================================================================
def test_legacy_steps_couples_D_H_U_and_uses_legacy_seed():
    r = resolve_stage5_budget(steps=100, n_cards=5, seed=1)
    # A bare steps= call is the legacy contract: D == U == steps, H == 3, and the
    # data seed is the exact historical derivation.
    assert r.contract_version == 1
    assert (r.opponent_sessions, r.games_per_session, r.adaptation_steps) == (100, 3, 100)
    assert r.games_per_session == LEGACY_GAMES_PER_SESSION
    assert r.data_seed == _legacy_data_seed(seed=1, steps=100, n_cards=5) == 609
    assert r.legacy_steps == 100
    # Legacy seeded neither the optimizer nor a held-out set -> the trainer keeps
    # byte-identical legacy behaviour precisely because these stay None/0.
    assert r.optimization_seed is None
    assert r.validation_seed is None
    assert r.validation_sessions == 0


def test_decoupled_field_selects_v2_and_derives_independent_seeds():
    r = resolve_stage5_budget(opponent_sessions=100, n_cards=5, seed=1)
    assert r.contract_version == 2
    assert r.opponent_sessions == 100
    assert r.games_per_session == LEGACY_GAMES_PER_SESSION  # H defaults to 3
    assert r.adaptation_steps == 100  # U falls back to D when unspecified
    assert r.data_seed == 1 + 503 + 100 + 5 == 609
    assert r.optimization_seed == 1  # defaults to seed
    assert r.validation_seed == 1 + _V2_VALIDATION_SEED_OFFSET == 90008
    assert r.validation_sessions == max(1, 100 // 5) == 20


def test_data_seed_is_independent_of_adaptation_steps():
    """THE Phase 3B invariant at the resolution layer: sweeping U leaves the data
    seed, validation seed, D and H untouched — only the update count moves."""
    a = resolve_stage5_budget(opponent_sessions=200, adaptation_steps=50, n_cards=5, seed=7)
    b = resolve_stage5_budget(opponent_sessions=200, adaptation_steps=500, n_cards=5, seed=7)
    assert a.data_seed == b.data_seed
    assert a.validation_seed == b.validation_seed
    assert a.validation_sessions == b.validation_sessions
    assert a.opponent_sessions == b.opponent_sessions
    assert a.games_per_session == b.games_per_session
    # ...and U is the ONLY thing that differs.
    assert a.adaptation_steps == 50 and b.adaptation_steps == 500


def test_data_seed_depends_on_D_and_n_cards():
    base = resolve_stage5_budget(opponent_sessions=100, n_cards=5, seed=1).data_seed
    more_d = resolve_stage5_budget(opponent_sessions=200, n_cards=5, seed=1).data_seed
    more_cards = resolve_stage5_budget(opponent_sessions=100, n_cards=7, seed=1).data_seed
    assert more_d != base and more_cards != base
    assert more_d == 1 + 503 + 200 + 5
    assert more_cards == 1 + 503 + 100 + 7


def test_contract_version_guards_required_knobs():
    with pytest.raises(ValueError):
        resolve_stage5_budget(n_cards=5, seed=1)  # neither steps nor D
    with pytest.raises(ValueError):
        resolve_stage5_budget(contract_version=1, opponent_sessions=10, n_cards=5, seed=1)  # v1 needs steps
    with pytest.raises(ValueError):
        resolve_stage5_budget(contract_version=3, steps=10, n_cards=5, seed=1)  # unknown version


# ===========================================================================
# B. Generation — deterministic, and identical across a U-sweep.
# ===========================================================================
def test_generation_is_deterministic_given_data_seed():
    a = generate_opponent_sessions(opponent_sessions=5, games_per_session=2, n_cards=3, data_seed=42)
    b = generate_opponent_sessions(opponent_sessions=5, games_per_session=2, n_cards=3, data_seed=42)
    assert dataset_sha256(a) == dataset_sha256(b)


def test_same_D_regenerates_identical_training_set_across_U():
    """RE-EXECUTED end-to-end at the generation layer: resolve two very different
    U budgets for the same D/seed, generate each resulting training set, and prove
    they are byte-for-byte the same data."""
    lo = resolve_stage5_budget(opponent_sessions=8, adaptation_steps=50, n_cards=3, seed=3)
    hi = resolve_stage5_budget(opponent_sessions=8, adaptation_steps=500, n_cards=3, seed=3)
    ds_lo = Stage5Dataset.generate(
        role="train", contract_version=2, opponent_sessions=lo.opponent_sessions,
        games_per_session=lo.games_per_session, n_cards=lo.n_cards, data_seed=lo.data_seed,
    )
    ds_hi = Stage5Dataset.generate(
        role="train", contract_version=2, opponent_sessions=hi.opponent_sessions,
        games_per_session=hi.games_per_session, n_cards=hi.n_cards, data_seed=hi.data_seed,
    )
    assert ds_lo.content_hash == ds_hi.content_hash


def test_D_changes_data_quantity():
    d4 = _small_sessions(D=4, H=3, n_cards=3, data_seed=513)
    d8 = _small_sessions(D=8, H=3, n_cards=3, data_seed=513)
    assert len(d4) == 4 and len(d8) == 8
    # Goofspiel plays exactly n_cards rounds per game, so rows == D*H*n_cards.
    assert row_count(d4) == 4 * 3 * 3
    assert row_count(d8) == 8 * 3 * 3
    assert dataset_sha256(d4) != dataset_sha256(d8)


def test_H_changes_history_not_session_count():
    h1 = _small_sessions(D=4, H=1, n_cards=3, data_seed=513)
    h3 = _small_sessions(D=4, H=3, n_cards=3, data_seed=513)
    assert len(h1) == len(h3) == 4  # H does not change the number of sessions
    assert row_count(h1) == 4 * 1 * 3
    assert row_count(h3) == 4 * 3 * 3  # three times the games -> three times the rows
    assert dataset_sha256(h1) != dataset_sha256(h3)


# ===========================================================================
# C. Canonical hashing — the dataset-identity primitive itself.
# ===========================================================================
def test_identical_content_has_identical_sha256():
    a = _small_sessions(data_seed=77)
    b = _small_sessions(data_seed=77)
    assert dataset_sha256(a) == dataset_sha256(b)


def test_mutating_a_single_action_changes_the_sha256():
    sessions = _small_sessions(data_seed=77)
    before = dataset_sha256(sessions)
    mutated = copy.deepcopy(sessions)
    # Flip one opponent action in one round; the hash must move (it covers labels).
    r = mutated[0].games[0][0]
    r.opponent_action = (r.opponent_action % 3) + 1
    after = dataset_sha256(mutated)
    assert after != before


def test_canonical_content_ignores_the_seed_bearing_session_id():
    """Overlap detection must be seed-independent: two sessions that differ ONLY in
    their (seed-bearing) session_id have identical canonical content."""
    s = _small_sessions(D=1, data_seed=77)[0]
    twin = copy.deepcopy(s)
    twin.session_id = "totally_different_id:seed999:n3"
    assert twin.session_id != s.session_id
    assert canonical_session_content(twin) == canonical_session_content(s)


# ===========================================================================
# D. Train / validation overlap — genuinely held out.
# ===========================================================================
def test_train_and_validation_are_disjoint_for_independent_seeds():
    r = resolve_stage5_budget(opponent_sessions=6, n_cards=3, seed=1)
    train = generate_opponent_sessions(
        opponent_sessions=r.opponent_sessions, games_per_session=r.games_per_session,
        n_cards=r.n_cards, data_seed=r.data_seed,
    )
    val = generate_opponent_sessions(
        opponent_sessions=r.validation_sessions, games_per_session=r.games_per_session,
        n_cards=r.n_cards, data_seed=r.validation_seed,
    )
    # Held out: no validation session's content appears in training.
    assert train_val_overlap_count(train, val) == 0
    # Non-vacuous: the detector DOES flag overlap when it exists (train vs itself).
    assert train_val_overlap_count(train, train) == len(train)


# ===========================================================================
# E. Held-out validation has NO effect on the training run (§14).
# ===========================================================================
def _eval_fixture():
    torch.manual_seed(0)
    model = GoofspielModel(max_cards=13)
    model.set_robust_requires_grad(False)
    val_sessions = _small_sessions(D=3, H=2, n_cards=3, data_seed=555)
    return model, val_sessions


def test_validation_does_not_mutate_model_parameters():
    model, val_sessions = _eval_fixture()
    before = model_param_signature(model)
    evaluate_opponent_model(model, val_sessions, max_cards=13, device="cpu")
    after = model_param_signature(model)
    assert after == before, "held-out evaluation mutated a model parameter/buffer"


def test_validation_is_rng_neutral():
    """A validation call between optimizer updates must not perturb the training
    RNG stream; the evaluator snapshots and restores it, so the state is byte-equal
    across the call."""
    model, val_sessions = _eval_fixture()
    torch.manual_seed(12345)
    rng_before = torch.get_rng_state()
    evaluate_opponent_model(model, val_sessions, max_cards=13, device="cpu")
    rng_after = torch.get_rng_state()
    assert torch.equal(rng_before, rng_after)


def test_interleaved_validation_leaves_training_trajectory_identical():
    """The definitive §14 fact, re-executed: two identically-initialised models take
    the same seeded optimizer steps; the only difference is that one runs a
    validation pass *between* its steps.  If validation had any observable effect
    (RNG, params, mode) the second model's weights would diverge.  They must not."""
    train_sessions = _small_sessions(D=4, H=2, n_cards=3, data_seed=11)
    val_sessions = _small_sessions(D=2, H=2, n_cards=3, data_seed=999)
    built = _build_adaptive_training_tensors(train_sessions, max_cards=13, device="cpu")
    assert built is not None
    batch, history, memory, target_t, _ = built

    def make_model():
        m = GoofspielModel(max_cards=13)
        m.set_robust_requires_grad(False)
        return m

    torch.manual_seed(0)
    m1 = make_model()
    m2 = make_model()
    m2.load_state_dict(m1.state_dict())  # identical initialisation
    assert model_param_signature(m1) == model_param_signature(m2)

    def step(m, opt):
        m.train()
        out = m(batch, current_game_history=history, long_term_memory=memory)
        loss = F.cross_entropy(out.opponent_fused_logits, target_t)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.adaptive_parameters(), 1.0)
        opt.step()

    opt1 = torch.optim.AdamW(m1.adaptive_parameters(), lr=1e-3)
    opt2 = torch.optim.AdamW(m2.adaptive_parameters(), lr=1e-3)

    torch.manual_seed(123)
    step(m1, opt1)
    step(m1, opt1)

    torch.manual_seed(123)
    step(m2, opt2)
    evaluate_opponent_model(m2, val_sessions, max_cards=13, device="cpu")  # the only difference
    step(m2, opt2)

    assert model_param_signature(m1) == model_param_signature(m2), (
        "an interleaved validation call changed the training trajectory"
    )


def test_validation_metrics_are_wellformed():
    model, _ = _eval_fixture()
    val_sessions = _small_sessions(D=4, H=2, n_cards=3, data_seed=222)
    m = evaluate_opponent_model(model, val_sessions, max_cards=13, device="cpu")
    for key in ("val_nll", "val_uniform_reference_nll", "val_nll_gain",
                "val_accuracy", "val_ece", "val_brier", "val_rows"):
        assert key in m, f"missing validation metric {key}"
    assert m["val_rows"] == 4 * 2 * 3  # rows == D*H*n_cards, re-derived
    assert all(v == v for v in m.values())  # no NaN
    # gain is exactly the uniform reference minus the model NLL.
    assert abs(m["val_nll_gain"] - (m["val_uniform_reference_nll"] - m["val_nll"])) < 1e-9


# ===========================================================================
# F. End-to-end through the REAL run_stage5_adaptive (needs a parent checkpoint).
# ===========================================================================
def test_legacy_run_matches_independent_generation_byte_for_byte(parent_ckpt, tmp_path):
    """A bare steps= (v1) run must persist opponent_sessions.jsonl byte-identical to
    an independent regeneration under the documented legacy seed formula — the
    concrete meaning of 'byte-identical to the pre-Phase3B behaviour'."""
    steps, n_cards, seed = 4, 3, 1
    out = tmp_path / "v1"
    run_stage5_adaptive(steps=steps, out_dir=out, n_cards=n_cards, seed=seed,
                        init_from_checkpoint=parent_ckpt)

    ref_sessions = generate_opponent_sessions(
        opponent_sessions=steps, games_per_session=LEGACY_GAMES_PER_SESSION,
        n_cards=n_cards, data_seed=_legacy_data_seed(seed=seed, steps=steps, n_cards=n_cards),
    )
    ref_path = tmp_path / "ref_opponent_sessions.jsonl"
    write_sessions_jsonl(ref_sessions, ref_path)

    got_bytes = (out / "adaptive" / "opponent_sessions.jsonl").read_bytes()
    ref_bytes = ref_path.read_bytes()
    assert got_bytes == ref_bytes, "legacy session bytes diverged from independent regeneration"


def test_legacy_run_writes_no_phase3b_artifacts(parent_ckpt, tmp_path):
    out = tmp_path / "v1_artifacts"
    run_stage5_adaptive(steps=4, out_dir=out, n_cards=3, seed=1, init_from_checkpoint=parent_ckpt)
    adaptive = out / "adaptive"
    # None of the decoupled-contract artifacts exist on the legacy path.
    for name in ("stage5_training_curve.jsonl", "stage5_manifest.json",
                 "train_dataset_manifest.json", "validation_dataset_manifest.json"):
        assert not (adaptive / name).exists(), f"legacy run wrote {name}"
    # ...and the gate report carries none of the v2-only keys.
    report = json.loads((adaptive / "adaptive_gate_report.json").read_text(encoding="utf-8"))
    assert "stage5_data_contract_version" not in report
    assert "decoupled_budget" not in report
    assert "validation" not in report


def test_legacy_checkpoint_config_is_unchanged(parent_ckpt, tmp_path):
    steps, n_cards, seed = 4, 3, 1
    out = tmp_path / "v1_ckpt"
    result = run_stage5_adaptive(steps=steps, out_dir=out, n_cards=n_cards, seed=seed,
                                 init_from_checkpoint=parent_ckpt)
    cfg = load_checkpoint(result.checkpoint)["metadata"]["config"]
    # Exactly the historical keys, and stage_seed is the legacy derivation.
    assert set(cfg.keys()) == {"steps", "n_cards", "lr", "games_per_session", "seed", "stage_seed"}
    assert cfg["steps"] == steps
    assert cfg["games_per_session"] == LEGACY_GAMES_PER_SESSION
    assert cfg["stage_seed"] == _legacy_data_seed(seed=seed, steps=steps, n_cards=n_cards)


def test_adaptation_budget_does_not_change_training_data_end_to_end(parent_ckpt, tmp_path):
    """§11 core gate, re-executed through the real stage: generate once into a shared
    cache, train a short U and a longer U against it, and prove BOTH runs consumed
    the identical training set (sha over persisted bytes) even though the number of
    optimizer updates differed."""
    cache = tmp_path / "cache"
    out_lo = tmp_path / "run_U2"
    out_hi = tmp_path / "run_U5"

    r_lo = run_stage5_adaptive(
        opponent_sessions=6, games_per_session=3, adaptation_steps=2, n_cards=3, seed=1,
        init_from_checkpoint=parent_ckpt, out_dir=out_lo, dataset_cache_dir=cache,
    )
    r_hi = run_stage5_adaptive(
        opponent_sessions=6, games_per_session=3, adaptation_steps=5, n_cards=3, seed=1,
        init_from_checkpoint=parent_ckpt, out_dir=out_hi, dataset_cache_dir=cache,
    )

    # The optimizer budget genuinely differed between the two runs.
    assert r_lo.steps == 2 and r_hi.steps == 5

    # Recompute the dataset identity from the bytes each run persisted, and from the
    # shared cache file, and from an independent regeneration — all four must agree.
    hash_lo = _persisted_train_hash(out_lo)
    hash_hi = _persisted_train_hash(out_hi)
    cache_file = cache / "train_opponent_sessions.jsonl"
    assert cache_file.exists(), "shared cache was never written"
    hash_cache = dataset_sha256(sessions_from_jsonl(cache_file))
    independent = resolve_stage5_budget(opponent_sessions=6, adaptation_steps=999, n_cards=3, seed=1)
    hash_ref = Stage5Dataset.generate(
        role="train", contract_version=2, opponent_sessions=independent.opponent_sessions,
        games_per_session=independent.games_per_session, n_cards=independent.n_cards,
        data_seed=independent.data_seed,
    ).content_hash

    assert hash_lo == hash_hi == hash_cache == hash_ref


def test_validation_set_is_fixed_across_U_end_to_end(parent_ckpt, tmp_path):
    """§5, re-executed: the held-out set is generated from a U-independent seed, so
    two runs at different U over a shared cache use the identical validation data —
    proven by recomputing the validation sha from the cached bytes and matching an
    independent regeneration."""
    cache = tmp_path / "cache_val"
    run_stage5_adaptive(
        opponent_sessions=6, games_per_session=3, adaptation_steps=2, n_cards=3, seed=1,
        init_from_checkpoint=parent_ckpt, out_dir=tmp_path / "vU2", dataset_cache_dir=cache,
    )
    run_stage5_adaptive(
        opponent_sessions=6, games_per_session=3, adaptation_steps=5, n_cards=3, seed=1,
        init_from_checkpoint=parent_ckpt, out_dir=tmp_path / "vU5", dataset_cache_dir=cache,
    )
    val_file = cache / "validation_opponent_sessions.jsonl"
    assert val_file.exists(), "no held-out validation set was cached"
    hash_cached_val = dataset_sha256(sessions_from_jsonl(val_file))

    r = resolve_stage5_budget(opponent_sessions=6, n_cards=3, seed=1)
    ref_val = generate_opponent_sessions(
        opponent_sessions=r.validation_sessions, games_per_session=r.games_per_session,
        n_cards=r.n_cards, data_seed=r.validation_seed,
    )
    assert hash_cached_val == dataset_sha256(ref_val)


def test_v2_firewall_holds_end_to_end(parent_ckpt, tmp_path):
    """The gradient firewall must survive the decoupled path exactly as on the legacy
    path: re-execute a real v2 run, then compare the SAVED weights to the parent —
    robust weights byte-equal (frozen backbone carried through), adaptive moved."""
    out = tmp_path / "v2_firewall"
    result = run_stage5_adaptive(
        opponent_sessions=6, games_per_session=3, adaptation_steps=4, n_cards=3, seed=1,
        init_from_checkpoint=parent_ckpt, out_dir=out,
    )
    assert result.checkpoint is not None

    parent_state = load_checkpoint(parent_ckpt)["model_state"]
    trained_state = load_checkpoint(result.checkpoint)["model_state"]

    m = GoofspielModel(max_cards=13)
    robust_names = set(m._robust_modules().keys())
    adaptive_names = set(m._adaptive_modules().keys())

    robust_checked = adaptive_moved = 0
    for name, tensor in trained_state.items():
        top = name.split(".")[0]
        if top in robust_names:
            assert torch.equal(tensor, parent_state[name]), f"robust param {name} changed under v2"
            robust_checked += 1
        elif top in adaptive_names:
            if not torch.equal(tensor, parent_state[name]):
                adaptive_moved += 1
    assert robust_checked > 0, "no robust params checked — partition/owner mismatch"
    assert adaptive_moved > 0, "adaptive params did not move — v2 did not train"


def test_trajectory_records_train_and_val_rows(parent_ckpt, tmp_path):
    """§9/§10/§17, re-executed: a v2 run with a trajectory interval must persist a
    curve whose rows carry the train + held-out-val + gradient + walltime fields,
    with the first and last optimizer update always present and the post-clip norm
    exactly min(pre-clip, clip_threshold)."""
    out = tmp_path / "v2_traj"
    U = 3
    run_stage5_adaptive(
        opponent_sessions=6, games_per_session=3, adaptation_steps=U, n_cards=3, seed=1,
        init_from_checkpoint=parent_ckpt, out_dir=out, trajectory_log_interval=1,
    )
    curve = out / "adaptive" / "stage5_training_curve.jsonl"
    assert curve.exists(), "trajectory file was not written"
    rows = [json.loads(line) for line in curve.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows, "trajectory file is empty"

    # First and last optimizer updates are captured.
    assert rows[0]["update"] == 1
    assert rows[-1]["update"] == U

    required = {
        "update", "train_nll", "train_accuracy", "train_ece",
        "adaptive_grad_norm_preclip", "adaptive_grad_norm_postclip", "grad_clip_norm",
        "grad_finite", "adaptive_param_norm", "lr", "robust_param_delta_l1",
        "elapsed_s", "ms_per_update",
        # held-out validation fields (D//5 >= 1 so validation is on)
        "val_nll", "val_nll_gain", "val_accuracy", "val_ece", "val_brier", "val_rows",
    }
    for row in rows:
        missing = required - row.keys()
        assert not missing, f"trajectory row missing fields {missing}"
        # Post-clip 2-norm is exactly min(pre-clip, threshold) — re-derived, not read.
        assert row["adaptive_grad_norm_postclip"] == min(
            row["adaptive_grad_norm_preclip"], row["grad_clip_norm"]
        )
        assert row["grad_finite"] is True
        # The firewall held at every logged update (robust delta stays zero).
        assert row["robust_param_delta_l1"] == 0.0
