"""NeuRD-style robust actor losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


NEURD_LOGIT_THRESHOLD = 2.0
NEURD_THRESHOLD_LR = 1.0


def legal_logits(logits: Tensor, legal_mask: Tensor) -> Tensor:
    """Center legal logits and mask illegal ones.

    NeuRD relies on the pre-softmax logit differences.  Centering the legal
    logits is policy-preserving because softmax is shift-invariant, and it makes
    the logit-gap threshold operate around zero.
    """
    mask = legal_mask.bool()
    legal = logits.masked_fill(~mask, 0.0)
    counts = mask.float().sum(dim=-1, keepdim=True).clamp_min(1.0)
    centered = legal - (legal.sum(dim=-1, keepdim=True) / counts)
    return centered.masked_fill(~mask, -1e9)


def apply_force_with_threshold(
    logits: Tensor,
    force: Tensor,
    threshold: float,
    *,
    step_size: float = NEURD_THRESHOLD_LR,
    legal_mask: Tensor | None = None,
) -> Tensor:
    """Project a raw-logit NeuRD force through the paper's thresholding rule.

    Hennes et al. (NeuRD, arXiv:1906.00190v5) note that raw logits may diverge
    and propose a clipping gradient with indicator
    ``z(theta) + eta * grad z(theta) in [-beta, beta]``.  OpenSpiel's PyTorch
    NeuRD implementation uses beta=2.0 and splits positive/negative regrets so
    logits stop moving farther beyond the threshold.  This helper applies the
    same raw-logit sign split to centered legal logits and uses ``step_size`` as
    eta; it is not a log-prob policy-gradient surrogate.
    """
    if threshold < 0:
        raise ValueError("NeuRD logit threshold must be non-negative")
    if step_size <= 0:
        raise ValueError("NeuRD threshold step_size must be positive")

    if legal_mask is None:
        mask = torch.ones_like(logits, dtype=torch.bool)
    else:
        mask = legal_mask.bool()
    force = force.to(dtype=logits.dtype, device=logits.device) * mask.to(logits.dtype)
    positive = F.relu(force)
    negative = -F.relu(-force)
    eta = float(step_size)
    beta = float(threshold)

    can_increase = (logits < beta) & ((logits + eta * positive) <= beta)
    can_decrease = (logits > -beta) & ((logits + eta * negative) >= -beta)
    # If a restored or inherited model is already outside the band, keep the
    # force component that moves it back toward the legal centered-logit range.
    can_increase = can_increase | (logits < -beta)
    can_decrease = can_decrease | (logits > beta)
    thresholded = can_increase.to(logits.dtype) * positive + can_decrease.to(logits.dtype) * negative
    return thresholded * mask.to(logits.dtype)


def neurd_logit_force(
    logits: Tensor,
    regret: Tensor,
    legal_mask: Tensor,
    *,
    threshold: float | None = NEURD_LOGIT_THRESHOLD,
    threshold_step_size: float = NEURD_THRESHOLD_LR,
) -> tuple[Tensor, Tensor]:
    """Return centered legal logits and the raw-logit NeuRD force.

    The NeuRD update is ``logit += advantage/regret`` at the policy-output
    level.  If ``threshold`` is set, positive and negative force components are
    clipped against the centered legal-logit band ``[-threshold, threshold]``.
    """
    mask = legal_mask.bool()
    centered_logits = legal_logits(logits, mask)
    centered_for_threshold = centered_logits.masked_fill(~mask, 0.0)
    force = regret.detach().to(dtype=logits.dtype, device=logits.device) * mask.to(logits.dtype)
    if threshold is not None:
        force = apply_force_with_threshold(
            centered_for_threshold,
            force,
            threshold=float(threshold),
            step_size=float(threshold_step_size),
            legal_mask=mask,
        ) * mask.to(logits.dtype)
    return centered_logits, force


def neurd_actor_loss_from_regret(
    logits: Tensor,
    regret: Tensor,
    legal_mask: Tensor,
    *,
    threshold: float | None = NEURD_LOGIT_THRESHOLD,
    threshold_step_size: float = NEURD_THRESHOLD_LR,
) -> tuple[Tensor, Tensor, Tensor]:
    """Raw-logit NeuRD actor loss plus the centered logits and effective force."""
    mask_f = legal_mask.float()
    centered_logits, force = neurd_logit_force(
        logits,
        regret,
        legal_mask,
        threshold=threshold,
        threshold_step_size=threshold_step_size,
    )
    denom = mask_f.sum(dim=-1).clamp_min(1.0)
    raw_legal_logits = logits.masked_fill(~legal_mask.bool(), 0.0)
    loss = -((force.detach() * raw_legal_logits * mask_f).sum(dim=-1) / denom).mean()
    return loss, centered_logits, force


def row_action_regret(q_matrix: Tensor, row_policy: Tensor, col_policy: Tensor, row_mask: Tensor) -> Tensor:
    q = q_matrix.detach().float()
    row = row_policy.float() * row_mask.float()
    row = row / row.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    col = col_policy.float()
    col = col / col.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    action_value = torch.bmm(q, col[:, :, None]).squeeze(-1)
    value = (row * action_value).sum(dim=-1, keepdim=True)
    return (action_value - value) * row_mask.float()


def neurd_loss(
    logits_self: Tensor,
    logits_opp: Tensor,
    q_matrix: Tensor,
    self_mask: Tensor,
    opp_mask: Tensor,
    *,
    threshold: float | None = NEURD_LOGIT_THRESHOLD,
    threshold_step_size: float = NEURD_THRESHOLD_LR,
) -> Tensor:
    """NeuRD raw-logit loss with thresholding.

    The loss remains a raw-logit force on the actor; it is not converted into a
    log-prob policy-gradient objective.
    """
    q = q_matrix.detach().float()
    row_mask = self_mask.bool()
    col_mask = opp_mask.bool()
    row_logits = legal_logits(logits_self, row_mask)
    col_logits = legal_logits(logits_opp, col_mask)
    row_policy = F.softmax(row_logits, dim=-1)
    col_policy = F.softmax(col_logits, dim=-1)
    regret = row_action_regret(q, row_policy, col_policy, row_mask)
    loss, _centered, _force = neurd_actor_loss_from_regret(
        logits_self,
        regret,
        row_mask,
        threshold=threshold,
        threshold_step_size=threshold_step_size,
    )
    return loss


def nash_anchor_kl(logits: Tensor, target_policy: Tensor, legal_mask: Tensor) -> Tensor:
    logp = F.log_softmax(logits.masked_fill(~legal_mask.bool(), -1e9), dim=-1)
    target = target_policy.float() * legal_mask.float()
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return F.kl_div(logp, target, reduction="none").sum(dim=-1).mean()
