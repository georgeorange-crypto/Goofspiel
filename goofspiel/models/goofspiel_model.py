"""Full Goofspiel neural architecture implementation.

The implementation is deliberately compact but keeps the hard contracts from
the design documents: variable N, joint-action Q matrices, public robust path
isolated from opponent history, LSTM short-term memory, Mamba-style long-term
memory, adaptive residual branch, masks, and ensemble heads.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .types import HistoryBatch, OpponentMemoryBatch, PublicStateBatch


def _masked_softmax(logits: Tensor, mask: Tensor, dim: int = -1) -> Tensor:
    masked = logits.masked_fill(~mask, -1e9)
    return F.softmax(masked, dim=dim)


def _masked_mean(x: Tensor, mask: Tensor, dim: int) -> Tensor:
    w = mask.to(dtype=x.dtype).unsqueeze(-1)
    denom = w.sum(dim=dim).clamp_min(1.0)
    return (x * w).sum(dim=dim) / denom


class RankEncoder(nn.Module):
    def __init__(self, dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(6, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, n_cards: Tensor, max_cards: int) -> Tensor:
        batch = int(n_cards.shape[0])
        ranks = torch.arange(1, max_cards + 1, device=n_cards.device, dtype=torch.float32)
        r = ranks[None, :] / n_cards.float().clamp_min(1.0)[:, None]
        basis = torch.stack(
            [r, r.square(), torch.sin(torch.pi * r), torch.cos(torch.pi * r),
             torch.sin(2 * torch.pi * r), torch.cos(2 * torch.pi * r)],
            dim=-1,
        )
        return self.net(basis.reshape(batch * max_cards, 6)).reshape(batch, max_cards, -1)


class ResidualMatrixBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(1, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        y = self.conv2(F.silu(self.conv1(self.norm1(x))))
        out = (x + y) * mask[:, None].to(dtype=x.dtype)
        return out


class SimpleMambaMemory(nn.Module):
    """Mamba-compatible long-memory fallback without an external dependency.

    It uses depthwise temporal convolution plus a GRU gate.  The class keeps
    the lifecycle and interface distinct from LSTM so the rest of the project
    can use a state-space memory implementation through the same caller API.
    """

    def __init__(self, input_dim: int = 192, hidden_dim: int = 192):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.depthwise = nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim)
        self.gate = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, memory: OpponentMemoryBatch | None, batch: int, device: torch.device) -> Tensor:
        if memory is None:
            return torch.zeros(batch, self.norm.normalized_shape[0], device=device)
        x = self.in_proj(memory.game_summary_sequence.float())
        x = self.depthwise(x.transpose(1, 2)).transpose(1, 2)
        x = F.silu(x)
        out, _ = self.gate(x)
        mask = memory.valid_mask.to(dtype=out.dtype)
        idx = mask.sum(dim=1).long().clamp_min(1) - 1
        last = out[torch.arange(out.shape[0], device=out.device), idx]
        return self.norm(last)


@dataclass
class GoofspielModelOutput:
    q_robust: Tensor
    q_robust_heads: Tensor
    robust_policy_logits: Tensor
    robust_score_logits: Tensor
    opponent_short_logits: Tensor
    opponent_long_logits: Tensor
    opponent_fused_logits: Tensor
    opponent_fused_heads: Tensor
    lstm_state: Tensor
    mamba_state: Tensor
    opponent_embedding: Tensor
    q_adaptive: Tensor
    q_adaptive_heads: Tensor
    adaptive_policy_logits: Tensor
    adaptive_score_logits: Tensor
    self_action_mask: Tensor
    opponent_action_mask: Tensor
    joint_action_mask: Tensor
    public_embedding: Tensor
    self_action_embeddings: Tensor
    opponent_action_embeddings: Tensor


class GoofspielModel(nn.Module):
    def __init__(
        self,
        *,
        max_cards: int = 13,
        d_model: int = 192,
        gnn_dim: int = 128,
        pair_dim: int = 96,
        matrix_dim: int = 128,
        q_heads: int = 4,
        outcome_bins: int = 201,
    ) -> None:
        super().__init__()
        self.max_cards = max_cards
        self.q_heads = q_heads
        self.outcome_bins = outcome_bins

        self.rank_encoder = RankEncoder(64)
        self.global_encoder = nn.Sequential(
            nn.Linear(8, 128), nn.SiLU(), nn.Linear(128, d_model), nn.LayerNorm(d_model)
        )
        self.card_projector = nn.Sequential(
            nn.Linear(68, d_model), nn.LayerNorm(d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=6, dim_feedforward=768, dropout=0.05,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.card_transformer = nn.TransformerEncoder(enc_layer, num_layers=4)

        self.role_embed = nn.Embedding(3, 32)
        self.node_projector = nn.Sequential(nn.Linear(64 + 32 + 2 + d_model, gnn_dim), nn.SiLU())
        self.gnn_layers = nn.ModuleList(
            [nn.Sequential(nn.LayerNorm(gnn_dim), nn.Linear(gnn_dim, gnn_dim), nn.SiLU())
             for _ in range(3)]
        )
        self.gnn_global = nn.Linear(gnn_dim, 1)

        self.t_to_action = nn.Linear(d_model, d_model)
        self.g_to_action = nn.Linear(gnn_dim, d_model)
        self.card_gate = nn.Sequential(nn.Linear(d_model * 3, d_model), nn.Sigmoid())
        self.global_fusion = nn.Sequential(
            nn.Linear(d_model + gnn_dim + d_model, 256),
            nn.SiLU(),
            nn.Linear(256, d_model),
            nn.LayerNorm(d_model),
        )

        self.pair_projector = nn.Sequential(
            nn.Linear(d_model * 3 + 7, 256), nn.SiLU(), nn.Linear(256, pair_dim), nn.LayerNorm(pair_dim)
        )
        self.matrix_in = nn.Sequential(nn.Conv2d(pair_dim, 96, 3, padding=1), nn.SiLU())
        self.matrix_blocks = nn.ModuleList([ResidualMatrixBlock(96) for _ in range(4)])
        self.matrix_out = nn.Conv2d(96, matrix_dim, 1)

        self.q_head = nn.ModuleList([
            nn.Sequential(nn.Linear(matrix_dim, 64), nn.SiLU(), nn.Linear(64, 1))
            for _ in range(q_heads)
        ])
        self.policy_head = nn.Sequential(nn.Linear(d_model + matrix_dim + d_model, d_model), nn.SiLU(), nn.Linear(d_model, 1))
        self.value_head = nn.Sequential(nn.Linear(d_model + matrix_dim, 256), nn.SiLU(), nn.Linear(256, outcome_bins))

        self.history_projector = nn.Sequential(nn.Linear(64 * 3 + 3, 128), nn.LayerNorm(128), nn.SiLU())
        self.intra_game_lstm = nn.LSTM(128, d_model, num_layers=2, dropout=0.05, batch_first=True)
        self.game_summary_projector = nn.Linear(d_model, d_model)
        self.inter_game_mamba = SimpleMambaMemory(d_model, d_model)
        self.memory_fusion = nn.Sequential(nn.Linear(d_model * 3, d_model), nn.SiLU(), nn.Linear(d_model, d_model), nn.LayerNorm(d_model))
        self.opp_short_head = nn.Sequential(nn.Linear(d_model * 3, d_model), nn.SiLU(), nn.Linear(d_model, 1))
        self.opp_long_head = nn.Sequential(nn.Linear(d_model * 3, d_model), nn.SiLU(), nn.Linear(d_model, 1))
        self.opp_fused_head = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model * 3, d_model), nn.SiLU(), nn.Linear(d_model, 1))
            for _ in range(q_heads)
        ])

        self.adaptive_film = nn.Linear(d_model, matrix_dim * 2)
        self.adaptive_cnn = nn.ModuleList([ResidualMatrixBlock(matrix_dim) for _ in range(2)])
        self.adaptive_delta_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(matrix_dim, 64), nn.SiLU(), nn.Linear(64, 1))
            for _ in range(q_heads)
        ])
        self.adaptive_policy_head = nn.Sequential(nn.Linear(d_model + matrix_dim + d_model, d_model), nn.SiLU(), nn.Linear(d_model, 1))
        self.adaptive_value_head = nn.Sequential(nn.Linear(d_model + matrix_dim + d_model, 256), nn.SiLU(), nn.Linear(256, outcome_bins))

    def _global_features(self, public_state: PublicStateBatch) -> Tensor:
        n = public_state.n_cards.float().clamp_min(1.0)
        nmax = float(self.max_cards)
        total = n * (n + 1.0) / 2.0
        remaining_mass = (public_state.remaining_prizes * torch.arange(
            1, self.max_cards + 1, device=public_state.device, dtype=torch.float32
        )[None]).sum(dim=1)
        return torch.stack([
            n / nmax,
            public_state.round_idx.float() / n,
            public_state.self_action_mask.sum(dim=1).float() / n,
            public_state.current_prize.float() / n,
            public_state.self_score.float() / total,
            public_state.opponent_score.float() / total,
            (public_state.self_score.float() - public_state.opponent_score.float()) / total,
            remaining_mass / total,
        ], dim=-1)

    def _encode_public(self, s: PublicStateBatch) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        batch, max_cards = s.batch_size, s.max_cards
        rank_emb = self.rank_encoder(s.n_cards, max_cards)
        ranks = torch.arange(1, max_cards + 1, device=s.device)
        current = (ranks[None] == s.current_prize[:, None]).float()
        card_in = torch.cat([rank_emb, s.self_cards[..., None], s.opponent_cards[..., None],
                             s.remaining_prizes[..., None], current[..., None]], dim=-1)
        card_tokens = self.card_projector(card_in)
        global_emb = self.global_encoder(self._global_features(s))
        tokens = torch.cat([global_emb[:, None], card_tokens], dim=1)
        key_padding = torch.cat([torch.zeros(batch, 1, device=s.device, dtype=torch.bool), ~s.rank_mask], dim=1)
        t_out = self.card_transformer(tokens, src_key_padding_mask=key_padding)
        t_cards = t_out[:, 1:]
        t_global = t_out[:, 0]

        role_ids = torch.tensor([0, 1, 2], device=s.device).repeat_interleave(max_cards)
        role = self.role_embed(role_ids)[None].expand(batch, -1, -1)
        rank3 = rank_emb.repeat(1, 3, 1)
        available = torch.cat([s.self_cards, s.opponent_cards, s.remaining_prizes], dim=1)[..., None]
        current3 = torch.cat([torch.zeros_like(current), torch.zeros_like(current), current], dim=1)[..., None]
        global3 = global_emb[:, None].expand(batch, 3 * max_cards, -1)
        nodes = self.node_projector(torch.cat([rank3, role, available, current3, global3], dim=-1))
        node_mask = torch.cat([s.rank_mask, s.rank_mask, s.rank_mask], dim=1)
        for layer in self.gnn_layers:
            pooled = _masked_mean(nodes, node_mask, dim=1)[:, None]
            nodes = (nodes + layer(nodes + pooled)) * node_mask[..., None].to(nodes.dtype)
        attn = self.gnn_global(nodes).squeeze(-1).masked_fill(~node_mask, -1e9)
        g_global = torch.sum(F.softmax(attn, dim=-1)[..., None] * nodes, dim=1)
        g_self = nodes[:, :max_cards]
        g_opp = nodes[:, max_cards:2 * max_cards]

        public = self.global_fusion(torch.cat([t_global, g_global, global_emb], dim=-1))
        self_action = self._fuse_action(t_cards, g_self, public)
        opp_action = self._fuse_action(t_cards, g_opp, public)
        return self_action, opp_action, public, global_emb

    def _fuse_action(self, t: Tensor, g: Tensor, public: Tensor) -> Tensor:
        t2 = self.t_to_action(t)
        g2 = self.g_to_action(g)
        p = public[:, None].expand_as(t2)
        gate = self.card_gate(torch.cat([t2, g2, p], dim=-1))
        return gate * t2 + (1.0 - gate) * g2

    def _pair_features(self, s: PublicStateBatch, self_action: Tensor, opp_action: Tensor, public: Tensor) -> Tensor:
        batch, n = s.batch_size, s.max_cards
        hs = self_action[:, :, None].expand(batch, n, n, -1)
        ho = opp_action[:, None, :].expand(batch, n, n, -1)
        hp = public[:, None, None].expand(batch, n, n, -1)
        ranks = torch.arange(1, n + 1, device=s.device, dtype=torch.float32)
        ri = ranks[None, :, None] / s.n_cards.float().clamp_min(1)[:, None, None]
        rj = ranks[None, None, :] / s.n_cards.float().clamp_min(1)[:, None, None]
        ri = ri.expand(batch, n, n)
        rj = rj.expand(batch, n, n)
        sign = torch.sign(ri - rj)
        prize = s.current_prize.float()[:, None, None] / s.n_cards.float().clamp_min(1)[:, None, None]
        total = s.n_cards.float() * (s.n_cards.float() + 1.0) / 2.0
        immediate = s.current_prize.float()[:, None, None] * sign / total[:, None, None]
        small = torch.stack([ri, rj, ri - rj, (ri - rj).abs(), sign, prize.expand_as(ri), immediate], dim=-1)
        return self.pair_projector(torch.cat([hs, ho, hp, small], dim=-1))

    def _encode_history(self, hist: HistoryBatch | None, rank_emb: Tensor, batch: int, device: torch.device) -> Tensor:
        if hist is None:
            return torch.zeros(batch, 192, device=device)
        def gather_rank(cards: Tensor) -> Tensor:
            idx = cards.long().clamp_min(1).clamp_max(rank_emb.shape[1]) - 1
            idx = idx[..., None].expand(*idx.shape, rank_emb.shape[-1])
            return torch.gather(rank_emb, 1, idx)

        p = gather_rank(hist.prize)
        a = gather_rank(hist.self_action)
        b = gather_rank(hist.opponent_action)
        extras = torch.stack([
            torch.sign((hist.self_action - hist.opponent_action).float()),
            hist.score_diff.float(),
            hist.round_idx.float() / float(self.max_cards),
        ], dim=-1)
        tokens = self.history_projector(torch.cat([p, a, b, extras], dim=-1))
        tokens = tokens * hist.valid_mask[..., None].to(tokens.dtype)
        out, (h, _) = self.intra_game_lstm(tokens)
        lengths = hist.valid_mask.sum(dim=1).long().clamp_min(1) - 1
        last = out[torch.arange(batch, device=device), lengths]
        empty = hist.valid_mask.sum(dim=1) == 0
        last = last.masked_fill(empty[:, None], 0.0)
        return last

    def forward(
        self,
        public_state: PublicStateBatch,
        current_game_history: HistoryBatch | None = None,
        long_term_memory: OpponentMemoryBatch | None = None,
        return_intermediates: bool = False,
    ) -> GoofspielModelOutput:
        del return_intermediates
        self_action, opp_action, public, _global = self._encode_public(public_state)
        batch, n = public_state.batch_size, public_state.max_cards
        joint_mask = public_state.self_action_mask[:, :, None] & public_state.opponent_action_mask[:, None, :]

        pair = self._pair_features(public_state, self_action, opp_action, public)
        matrix = self.matrix_in(pair.permute(0, 3, 1, 2)) * joint_mask[:, None].to(pair.dtype)
        for block in self.matrix_blocks:
            matrix = block(matrix, joint_mask)
        matrix = self.matrix_out(matrix) * joint_mask[:, None].to(matrix.dtype)
        cells = matrix.permute(0, 2, 3, 1)

        q_heads = torch.stack([head(cells).squeeze(-1) for head in self.q_head], dim=1)
        q_robust = q_heads.mean(dim=1)

        row_pool = _masked_mean(cells, public_state.opponent_action_mask[:, None].expand(batch, n, n), dim=2)
        pol_in = torch.cat([self_action, row_pool, public[:, None].expand(batch, n, -1)], dim=-1)
        robust_logits = self.policy_head(pol_in).squeeze(-1).masked_fill(~public_state.self_action_mask, -1e9)
        m_pool = _masked_mean(cells.reshape(batch, n * n, -1), joint_mask.reshape(batch, n * n), dim=1)
        robust_score = self.value_head(torch.cat([public, m_pool], dim=-1))

        rank_emb = self.rank_encoder(public_state.n_cards, n)
        lstm_state = self._encode_history(current_game_history, rank_emb, batch, public_state.device)
        mamba_state = self.inter_game_mamba(long_term_memory, batch, public_state.device)
        opponent_embedding = self.memory_fusion(torch.cat([lstm_state, mamba_state, public.detach()], dim=-1))

        def opp_logits(head: nn.Module, memory: Tensor) -> Tensor:
            x = torch.cat([opp_action.detach(), memory[:, None].expand(batch, n, -1),
                           public.detach()[:, None].expand(batch, n, -1)], dim=-1)
            return head(x).squeeze(-1).masked_fill(~public_state.opponent_action_mask, -1e9)

        opp_short = opp_logits(self.opp_short_head, lstm_state)
        opp_long = opp_logits(self.opp_long_head, mamba_state)
        opp_fused_heads = torch.stack([opp_logits(head, opponent_embedding) for head in self.opp_fused_head], dim=1)
        opp_fused = opp_fused_heads.mean(dim=1)

        gamma_beta = self.adaptive_film(opponent_embedding)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        adaptive = matrix * (1.0 + torch.tanh(gamma)[:, :, None, None]) + beta[:, :, None, None]
        adaptive = adaptive * joint_mask[:, None].to(adaptive.dtype)
        for block in self.adaptive_cnn:
            adaptive = block(adaptive, joint_mask)
        adaptive_cells = adaptive.permute(0, 2, 3, 1)
        delta_heads = torch.stack([head(adaptive_cells).squeeze(-1) for head in self.adaptive_delta_heads], dim=1)
        q_adaptive_heads = q_heads.detach() + delta_heads
        q_adaptive = q_robust.detach() + delta_heads.mean(dim=1)
        adaptive_row = _masked_mean(adaptive_cells, public_state.opponent_action_mask[:, None].expand(batch, n, n), dim=2)
        apol_in = torch.cat([self_action.detach(), adaptive_row, opponent_embedding[:, None].expand(batch, n, -1)], dim=-1)
        adaptive_logits = self.adaptive_policy_head(apol_in).squeeze(-1).masked_fill(~public_state.self_action_mask, -1e9)
        adaptive_pool = _masked_mean(adaptive_cells.reshape(batch, n * n, -1), joint_mask.reshape(batch, n * n), dim=1)
        adaptive_score = self.adaptive_value_head(torch.cat([public.detach(), adaptive_pool, opponent_embedding], dim=-1))

        return GoofspielModelOutput(
            q_robust=q_robust,
            q_robust_heads=q_heads,
            robust_policy_logits=robust_logits,
            robust_score_logits=robust_score,
            opponent_short_logits=opp_short,
            opponent_long_logits=opp_long,
            opponent_fused_logits=opp_fused,
            opponent_fused_heads=opp_fused_heads,
            lstm_state=lstm_state,
            mamba_state=mamba_state,
            opponent_embedding=opponent_embedding,
            q_adaptive=q_adaptive,
            q_adaptive_heads=q_adaptive_heads,
            adaptive_policy_logits=adaptive_logits,
            adaptive_score_logits=adaptive_score,
            self_action_mask=public_state.self_action_mask,
            opponent_action_mask=public_state.opponent_action_mask,
            joint_action_mask=joint_mask,
            public_embedding=public,
            self_action_embeddings=self_action,
            opponent_action_embeddings=opp_action,
        )

    def parameter_count_by_module(self) -> dict[str, int]:
        groups = {
            "rank_encoder": self.rank_encoder,
            "card_transformer": self.card_transformer,
            "relational_gnn": nn.ModuleList([self.role_embed, self.node_projector, self.gnn_layers]),
            "matrix_cnn": nn.ModuleList([self.pair_projector, self.matrix_in, self.matrix_blocks, self.matrix_out]),
            "lstm": self.intra_game_lstm,
            "mamba_memory": self.inter_game_mamba,
            "adaptive_branch": nn.ModuleList([self.adaptive_film, self.adaptive_cnn, self.adaptive_delta_heads]),
            "heads": nn.ModuleList([self.q_head, self.policy_head, self.value_head,
                                    self.opp_short_head, self.opp_long_head, self.opp_fused_head,
                                    self.adaptive_policy_head, self.adaptive_value_head]),
        }
        counts = {name: sum(p.numel() for p in module.parameters()) for name, module in groups.items()}
        counts["total"] = sum(p.numel() for p in self.parameters())
        counts["trainable"] = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return counts
