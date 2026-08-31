"""
Author: 陈子聪 (Chen Zicong)
Date: 2026-08-30
Purpose: FastAPI backend for Goofspiel — player-configurable N (1..13) cards
         AND bot choice (Random / Heuristic / Exact Nash).

The browser NEVER holds authoritative state; it only renders what the server
returns after each action. The server keeps a single in-memory session.

Endpoints:
  GET  /api/game/config -> 前端下拉框的合法选项 (num_cards 范围 + bot 列表)
  POST /api/game/new    -> 接受 {num_cards, bot_type}; bot_type = nash 时若 N>7
                           自动回落到 Heuristic 并在响应里标明 reason
  GET  /api/game/state  -> 当前 UI 状态
  POST /api/game/play   -> 出一张牌;同时出 AI 牌, last_round 带 AI 的决策分布条

Card-Value to Display (牌面显示): 1→A, 11→J, 12→Q, 13→K, 其余→数字。
"""

from __future__ import annotations

import secrets
import time
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from starlette.requests import Request

from goofspiel import (
    GoofspielEnv,
    PLAYER_0,
    PLAYER_1,
    # Bot 工厂 & 枚举
    create_bot,
    BOT_TYPES,
    BOT_DESCRIPTIONS,
    BOT_RANDOM,
    BOT_HEURISTIC,
    BOT_NASH,
    BOT_NASH_CARRY,          # NEW — carry-over 精确 Nash
    EXACT_MODE_CLASSIC,
    EXACT_MODE_CARRY,
    NASH_MAX_N,
    NASH_CARRY_MAX_N,
    # 规则枚举 (§29 平局变体 + VEIL §15 情报粒度)
    TIE_RULE_ROLLOVER,
    TIE_RULE_DISCARD,
    TIE_RULE_SPLIT,
    TIE_RULES,
    DEFAULT_TIE_RULE,
    INFO_MODES,
)
from goofspiel.bots import BaseBot, NashBot

# ------------------------------------------------------------------------
# App setup
# ------------------------------------------------------------------------
app = FastAPI(title="Goofspiel")

import os as _os
_PROJECT_DIR = _os.path.dirname(_os.path.abspath(__file__))
templates = Jinja2Templates(directory=_os.path.join(_PROJECT_DIR, "templates"))
app.mount(
    "/static",
    StaticFiles(directory=_os.path.join(_PROJECT_DIR, "static")),
    name="static",
)

MIN_NUM_CARDS = 1
MAX_NUM_CARDS = 13
DEFAULT_NUM_CARDS = 13
DEFAULT_BOT = BOT_RANDOM  # 保持老 UI 的默认行为

# VEIL 机制默认全部关闭 (与 GoofspielEnv 构造器保持一致)
DEFAULT_VEIL_HIDDEN_PRIZE = False
DEFAULT_VEIL_SUIT_TIEBREAK = False
DEFAULT_VEIL_INFO_REWARD = False
# VEIL §15 情报粒度 (前端下拉) — 默认值 = "auto":
#   info_reward=False → none, info_reward=True → half (避免出现 2 个独立 bool+enum 开不同步)
DEFAULT_INFO_BITS_MODE = "auto"
# §29 平局处理变体三选一 (复用 goofspiel.DEFAULT_TIE_RULE = rollover 作为页面默认)

# ------------------------------------------------------------------------
# In-memory single-user session (符合 demo 场景,不需要数据库)
# ------------------------------------------------------------------------
_env: Optional[GoofspielEnv] = None
_bot: Optional[BaseBot] = None
_bot_type: str = DEFAULT_BOT
_num_cards_used: int = DEFAULT_NUM_CARDS
_bot_fallback_reason: Optional[str] = None  # 如果选了 Nash 且 N 超了 → 记录（ONLY true fallback）
_bot_nash_precalc_note: Optional[str] = None  # Nash 精确策略首次预计算耗时提示（NOT a fallback）
_requested_bot: str = DEFAULT_BOT  # 用户选中的 (用于 /new meta.requested_bot & 副标题显示)
# ---- VEIL 会话级开关 (新游戏时由 POST /api/game/new 写入) ----
_veil_hidden_prize: bool = DEFAULT_VEIL_HIDDEN_PRIZE
_veil_suit_tiebreak: bool = DEFAULT_VEIL_SUIT_TIEBREAK
_veil_info_reward: bool = DEFAULT_VEIL_INFO_REWARD
_info_bits_mode: str = DEFAULT_INFO_BITS_MODE    # "auto" | "none" | "half"
_tie_rule: str = DEFAULT_TIE_RULE                # "rollover" | "discard" | "split" (§29)


def _nash_exact_limit(bot_type: str) -> int:
    """Each exact-Nash variant has its own default size limit (state-space blow-up)."""
    if bot_type == BOT_NASH:
        return NASH_MAX_N
    if bot_type == BOT_NASH_CARRY:
        return NASH_CARRY_MAX_N
    return MAX_NUM_CARDS


def _get_env_or_400() -> GoofspielEnv:
    if _env is None:
        raise HTTPException(
            status_code=400,
            detail="No active game. Call POST /api/game/new first.",
        )
    return _env


# ------------------------------------------------------------------------
# Pydantic schemas
# ------------------------------------------------------------------------
class NewGameRequest(BaseModel):
    num_cards: int = Field(
        DEFAULT_NUM_CARDS,
        ge=MIN_NUM_CARDS,
        le=MAX_NUM_CARDS,
        description="Deck size. 1=only A, 13=full deck A..K.",
    )
    bot_type: str = Field(
        DEFAULT_BOT,
        description="Bot type id. Must be one of goofspiel.BOT_TYPES",
    )
    # ---- VEIL 可选机制 (全部默认 False = 经典行为, 零破坏) ----
    veil_hidden_prize: bool = Field(
        DEFAULT_VEIL_HIDDEN_PRIZE,
        description="VEIL §9 Hidden Prize 模式: 出牌前隐藏当前奖励, 双方联合出价后揭晓。",
    )
    veil_suit_tiebreak: bool = Field(
        DEFAULT_VEIL_SUIT_TIEBREAK,
        description="VEIL §6 Suit Tiebreak: 每轮随机分配 ♣♦♥♠ 中 2 个 suit, rank 平局时 suit (♣<♦<♥<♠) 高者胜, 消除平局。",
    )
    veil_info_reward: bool = Field(
        DEFAULT_VEIL_INFO_REWARD,
        description="VEIL §11-12 Information Reward: 最低出牌者获得下一轮奖励 HIGH/LOW 的 1-bit 私人信号 (最后一轮不出情报)。",
    )
    info_bits_mode: str = Field(
        DEFAULT_INFO_BITS_MODE,
        description=(
            "VEIL §15 情报粒度枚举："
            f"'auto' = 跟随 veil_info_reward (关→none / 开→half, 默认); "
            f"'none' = 显式关闭; "
            f"'half' = HIGH/LOW 二分 (当前仅实现的精确模式); "
            f"其余 (quartile/parity/exact) 仅作为 schema 占位, 实际传入将 422。"
        ),
    )
    tie_rule: str = Field(
        DEFAULT_TIE_RULE,
        description=(
            "§29 平局处理变体 (三选一)："
            f"'{TIE_RULE_ROLLOVER}' = 平局滚入下一轮 carry (项目默认, 末轮平局丢弃); "
            f"'{TIE_RULE_DISCARD}'  = 平局即丢弃奖池 (Nash 经典奖牌型); "
            f"'{TIE_RULE_SPLIT}'    = 平局双方各分一半 (向下取整, 无 carry)。"
        ),
    )


class PlayRequest(BaseModel):
    action: int = Field(..., ge=1, le=13, description="Human's card value to play.")


# ------------------------------------------------------------------------
# Root page
# ------------------------------------------------------------------------
@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# ------------------------------------------------------------------------
# Configuration endpoint (前端下拉框选项来源)
# ------------------------------------------------------------------------
@app.get("/api/game/config")
async def game_config() -> Dict[str, Any]:
    """Frontend uses this to populate the "New Game" form options."""
    return {
        "num_cards": {
            "min": MIN_NUM_CARDS,
            "max": MAX_NUM_CARDS,
            "default": DEFAULT_NUM_CARDS,
        },
        "bots": [
            {
                "id": bt,
                "label": BOT_DESCRIPTIONS.get(bt, bt),
                "max_n_for_exact_nash": _nash_exact_limit(bt),
                "nash_rule_model": (
                    EXACT_MODE_CLASSIC if bt == BOT_NASH else
                    EXACT_MODE_CARRY   if bt == BOT_NASH_CARRY else
                    None
                ),
                # Exact Nash 只有在 0 个 VEIL 开关开启时才可能真跑 exact,
                # 否则 Nash 会诚实回落 Heuristic (见 NashBot._choose 回落 #0)。
                "nash_exact_incompatible_with_veil": bt in (BOT_NASH, BOT_NASH_CARRY),
            }
            for bt in BOT_TYPES
        ],
        "card_display": {
            "1": "A", "11": "J", "12": "Q", "13": "K",
        },
        # ---- VEIL 可选机制 (前端新游戏面板的唯一数据源, schema-driven) ----
        #      type ∈ {"checkbox", "select", "radio"}
        #      前端 populateVeilOptions 按 type 切换渲染组件; collectVeilOptions 按
        #      id 抓对应字段作为 POST /api/game/new 的顶层键 (与 NewGameRequest 字段 1:1 对应)。
        "veil_options": [
            {
                "id": "veil_hidden_prize",
                "type": "checkbox",
                "default": DEFAULT_VEIL_HIDDEN_PRIZE,
                "label": "隐藏奖励 (VEIL §9 Hidden Prize)",
                "description": "出牌前不显示当前奖金牌；双方同时出牌后才揭晓。你需要根据剩余奖金分布概率做决策。",
                "category": "信息不完全",
            },
            {
                "id": "veil_info_reward",
                "type": "checkbox",
                "default": DEFAULT_VEIL_INFO_REWARD,
                "label": "信息奖励 (VEIL §11-12 最低出牌获情报)",
                "description": "每轮出价最低者获得『下一轮奖金区属』的私人信号。高牌争分，低牌买信息。玩家的私人信号会在 GUI 手牌区实时显示。",
                "category": "信息不对称",
            },
            {
                "id": "info_bits_mode",
                "type": "select",
                "default": DEFAULT_INFO_BITS_MODE,
                "label": "情报信号粒度 (VEIL §15 info_bits_mode)",
                "description": "信息奖励开启时，拿到的信号有多具体：默认 HIGH/LOW 二分。精确奖金数/相对排名等粒度尚未实现 (422)。",
                "category": "信息不对称",
                "options": [
                    {"value": "auto",  "label": "跟随「信息奖励」开关 (默认)", "disabled": False,
                     "hint": "关=none, 开=HIGH/LOW 二分 (最省心)"},
                    {"value": "none",  "label": "关闭 (none)", "disabled": False,
                     "hint": "显式关闭情报通道 (即使信息奖励开关打开)"},
                    {"value": "half",  "label": "HIGH / LOW 二分 (half · 已实现)", "disabled": False,
                     "hint": "VEIL §12: ≥(N+1)/2 的奖金 = HIGH, 其余 LOW"},
                    {"value": "quartile", "label": "四分位 (quartile · Coming Soon)", "disabled": True,
                     "hint": "未实现, POST 会 422 拒绝"},
                    {"value": "parity",   "label": "奇偶 (parity · Coming Soon)",   "disabled": True,
                     "hint": "未实现, POST 会 422 拒绝"},
                    {"value": "exact",    "label": "精确奖金数 (exact · Coming Soon)", "disabled": True,
                     "hint": "未实现, POST 会 422 拒绝"},
                ],
            },
            {
                "id": "veil_suit_tiebreak",
                "type": "checkbox",
                "default": DEFAULT_VEIL_SUIT_TIEBREAK,
                "label": "花色 Tie-breaker (VEIL §6 ♣<♦<♥<♠)",
                "description": "每轮随机给双方分配♣♦♥♠中的两个临时花色；相同 rank 时由花色决胜负（♠最强，♣最弱），消除平局。",
                "category": "消除平局",
            },
            {
                "id": "tie_rule",
                "type": "radio",
                "default": DEFAULT_TIE_RULE,
                "label": "平局处理规则 (§29 三类变体)",
                "description": (
                    "Goofspiel 并没有唯一官方平局规则，不同论文/代码仓库会用不同模型。"
                    "精确 Nash 只有『Nash(nash) + 平局弃奖』或『Nash-carry + 平局滚入』两条合法组合，其余组合会诚实回落 Heuristic。"
                ),
                "category": "核心规则变体",
                "options": [
                    {"value": TIE_RULE_ROLLOVER, "label": "[默认] Variant A 平局滚入 Carry-Over",
                     "hint": "非末轮平局 → 奖金整包滚入下一轮 carry_pool；末轮平局 → 丢弃。(= 本项目默认规则 / nash_carry 奖牌型)"},
                    {"value": TIE_RULE_DISCARD,  "label": "Variant C 平局即弃奖 (经典 Discard)",
                     "hint": "任何平局双方 0 分，奖金直接丢弃，无 carry。(= Nash 经典教科书奖牌型 / OpenSpiel 默认)"},
                    {"value": TIE_RULE_SPLIT,    "label": "Variant B 平局 Split 各分一半",
                     "hint": "平局时双方各拿 prize_at_stake // 2 分，向下取整，余数静默丢弃。无精确 Nash Solver 支持 → 必回落 Heuristic。"},
                ],
            },
        ],
    }


# ------------------------------------------------------------------------
# Game lifecycle endpoints
# ------------------------------------------------------------------------
@app.post("/api/game/new")
async def new_game(req: Optional[NewGameRequest] = None) -> Dict[str, Any]:
    """
    Start a brand new game.  Human = player_0, Bot = player_1.

    Behavior:
      * If bot_type ∈ {nash, nash_carry} 且 req.num_cards > 各自 NASH 上限:
        自动回落到 Heuristic 并在响应 meta 里返回 `bot_fallback_reason`
        (前端在副标题里 show 给用户)。
      * If any VEIL flag is on AND user selected Nash exact bot: the bot will
        itself HONESTLY fall back to Heuristic on every _choose call (see
        NashBot._choose honest fallback #0).  We additionally annotate
        `meta.fallback_reason` preemptively so the new-game banner shows it.
      * If no body / empty body: fall back to old defaults (N=13, Random, 0 VEIL)
        以便 curl 手工调试 & 兼容老的 /static/app.js (如果有人暂未升级)。
    """
    global _env, _bot, _bot_type, _num_cards_used, _bot_fallback_reason, _requested_bot
    global _veil_hidden_prize, _veil_suit_tiebreak, _veil_info_reward
    global _info_bits_mode, _tie_rule

    n = DEFAULT_NUM_CARDS
    requested_bot = DEFAULT_BOT
    v_hidden = DEFAULT_VEIL_HIDDEN_PRIZE
    v_suit   = DEFAULT_VEIL_SUIT_TIEBREAK
    v_info   = DEFAULT_VEIL_INFO_REWARD
    # 新参数: info_bits_mode / tie_rule — 默认跟随 session 默认值
    v_bits   = DEFAULT_INFO_BITS_MODE
    v_tie    = DEFAULT_TIE_RULE
    if req is not None:
        n = int(req.num_cards)
        requested_bot = str(req.bot_type).strip() or DEFAULT_BOT
        v_hidden = bool(req.veil_hidden_prize)
        v_suit   = bool(req.veil_suit_tiebreak)
        v_info   = bool(req.veil_info_reward)
        v_bits   = str(req.info_bits_mode) if req.info_bits_mode else DEFAULT_INFO_BITS_MODE
        v_tie    = str(req.tie_rule)      if req.tie_rule      else DEFAULT_TIE_RULE

    # --- Validate ---
    if not (MIN_NUM_CARDS <= n <= MAX_NUM_CARDS):
        raise HTTPException(
            status_code=422,
            detail=f"num_cards must be in [{MIN_NUM_CARDS}, {MAX_NUM_CARDS}], got {n}",
        )
    if requested_bot not in BOT_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown bot_type '{requested_bot}'. Must be one of {list(BOT_TYPES)}",
        )
    # tie_rule: 必须 ∈ TIE_RULES (三者都已实现, 无 Coming Soon)
    if v_tie not in TIE_RULES:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unknown tie_rule '{v_tie}'. Must be one of {list(TIE_RULES)} "
                f"(rollover=平局滚入 / discard=平局弃奖 / split=各分一半)。"
            ),
        )
    # info_bits_mode: 允许 "auto" (由后端解释) + INFO_MODES 已实现值;
    # 对 quartile / parity / exact 等 Coming Soon 值 422 明确拒绝 (不静默回落)
    COMING_SOON_BITS = {"quartile", "parity", "exact", "noisy_half", "rank"}
    valid_bits_with_auto = {"auto", *INFO_MODES}
    if v_bits in COMING_SOON_BITS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"info_bits_mode='{v_bits}' 是 VEIL §15 预留粒度, 当前尚未实现。"
                f"可接受值: {sorted(valid_bits_with_auto)} (其中 'half' = HIGH/LOW 二分, 'auto' = 跟随信息奖励开关)。"
            ),
        )
    if v_bits not in valid_bits_with_auto:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown info_bits_mode '{v_bits}'. Must be one of {sorted(valid_bits_with_auto)}。",
        )

    # --- 解释 info_bits_mode "auto" → 实际 env 形参值 ---
    #   info_reward=False + auto → none (关闭)
    #   info_reward=True  + auto → half (HIGH/LOW 二分)
    if v_bits == "auto":
        actual_bits_mode = "half" if v_info else "none"
    else:
        actual_bits_mode = v_bits
        # 语义对齐: 如果用户显式指定 "half" 但 info_reward=False, env 会 implicitly 开
        # info_reward; 如果用户显式 "none" 但 info_reward=True, env 会 silently 修回 half.
        # 我们完全让 env 自己 normalize, 这里不重复做。

    # Commit flags to session globals (before env construction)
    _veil_hidden_prize = v_hidden
    _veil_suit_tiebreak = v_suit
    _veil_info_reward   = v_info
    _info_bits_mode     = actual_bits_mode
    _tie_rule           = v_tie
    # "any_veil_on" 扩展判定: 不仅看 3 个 bool, 还包括:
    #   - info_bits_mode 非 none (无论 bool 开关) → 情报通道确实开启
    #   - tie_rule 不是 DEFAULT_TIE_RULE (rollover) → 属于规则变体, Nash solver 不可兼容
    any_veil_on = (
        v_hidden or v_suit or v_info
        or (actual_bits_mode != "none")
    )
    tie_rule_is_nondefault = (v_tie != DEFAULT_TIE_RULE)

    # Instantiate env (所有可选参数显式传入; 缺省行为 = GoofspielEnv() 默认)
    env = GoofspielEnv(
        num_cards=n,
        tie_rule=v_tie,
        hidden_prize=v_hidden,
        suit_tiebreak=v_suit,
        info_reward_enabled=v_info,
        info_bits_mode=actual_bits_mode,
    )
    env.reset()

    # --- Nash rule-model helpers ---
    is_any_exact_nash = requested_bot in (BOT_NASH, BOT_NASH_CARRY)
    exact_n_limit = _nash_exact_limit(requested_bot)

    # Decide bot (带 Nash → heuristic 自动降级处理)
    fallback_reason: Optional[str] = None
    actual_bot_type = requested_bot
    # --- 仅用于诊断：精确 Nash 预计算耗时 (秒)，若 <0 表示未触发 ---
    # NOTE: 绝对不能把"慢但精确"的 timing 日志塞进 fallback_reason，
    # 否则前端会把一个 *仍在跑精确 Nash* 的状态误判成"已回落 Heuristic"。
    # 语义红线：fallback_reason 非空 ⇔ actual_bot != requested_bot。
    nash_warmup_sec: Optional[float] = None
    try:
        # ===== 抢先回落判定 (Nash solver 假设不成立, 见 NashBot 诚实回落) =====
        #   触发两类独立不兼容, 可以同时发生 (理由会合并):
        #     A) 用户开启了 VEIL 机制 (信息不完全/消除平局/私人信号任一)
        #     B) 精确 Nash Solver 的"奖牌型"和用户选的 tie_rule 对不上:
        #          BOT_NASH        ↔ TIE_RULE_DISCARD  (经典平局弃奖)
        #          BOT_NASH_CARRY  ↔ TIE_RULE_ROLLOVER (平局滚入)
        #        TIE_RULE_SPLIT  与任一精确 Nash 奖牌型均不兼容 (无 Solver 支持)
        #        兼容契约: BOT_NASH + rollover 允许开局使用 classic table,
        #        但一旦 carry_pool>0, NashBot._choose 会逐回合诚实回落并解释原因。
        incompat_reasons: List[str] = []
        if is_any_exact_nash:
            if any_veil_on:
                if v_hidden: incompat_reasons.append("VEIL: HiddenPrize(隐藏奖励)")
                if v_info:   incompat_reasons.append("VEIL: InfoReward(私人情报)")
                if v_suit:   incompat_reasons.append("VEIL: SuitTiebreak(花色Tie-break)")
                if actual_bits_mode != "none" and not v_info:
                    incompat_reasons.append("VEIL: InfoBitsMode(私人情报通道开)")
            # tie-rule ↔ solver 奖牌型一致性检查
            solver_model = EXACT_MODE_CLASSIC if requested_bot == BOT_NASH else EXACT_MODE_CARRY
            required_tie = TIE_RULE_DISCARD if requested_bot == BOT_NASH else TIE_RULE_ROLLOVER
            runtime_carry_fallback_supported = (
                requested_bot == BOT_NASH and v_tie == TIE_RULE_ROLLOVER
            )
            if v_tie != required_tie and not runtime_carry_fallback_supported:
                incompat_reasons.append(
                    f"TieRule 不匹配: 当前规则={v_tie!r} (§29), "
                    f"但所选精确 Nash={BOT_DESCRIPTIONS.get(requested_bot)!r} 内部奖牌型={solver_model!r} "
                    f"要求 tie_rule={required_tie!r}"
                )
        if incompat_reasons:
            fallback_reason = (
                f"所选精确 Nash 为「{BOT_DESCRIPTIONS.get(requested_bot)}」；"
                f"但存在不兼容项：{' / '.join(incompat_reasons)}。"
                f"已诚实预先回落为 Heuristic (NashBot 每次 _choose 也会再次检查并诚实回落)。"
            )
            actual_bot_type = BOT_HEURISTIC
            bot: BaseBot = create_bot(BOT_HEURISTIC)
        elif is_any_exact_nash and n > exact_n_limit:
            limit_name = (
                f"NASH_MAX_N={NASH_MAX_N} (经典平局弃奖牌型)"
                if requested_bot == BOT_NASH else
                f"NASH_CARRY_MAX_N={NASH_CARRY_MAX_N} (平局滚入奖牌型)"
            )
            fallback_reason = (
                f"所选精确 Nash 为「{BOT_DESCRIPTIONS.get(requested_bot)}」；"
                f"N={n} 超过 {limit_name}，自动回落为 Heuristic 启发式。"
            )
            actual_bot_type = BOT_HEURISTIC
            bot = create_bot(BOT_HEURISTIC)
        else:
            t0 = time.perf_counter()
            bot = create_bot(requested_bot)
            # 如果是 NashBot: ensure policy is precomputed NOW (before returning new-game)
            # 这样用户点 "开始" 后是一次性等待，而不是首轮出牌才卡住 30s。
            if isinstance(bot, NashBot):
                # 类级 cache; 首次调用会跑 solve, 之后瞬时
                _result = bot._ensure_policy(n)  # type: ignore[attr-defined]
                del _result
            nash_warmup_sec = time.perf_counter() - t0
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # Commit
    _env = env
    _bot = bot
    _bot_type = actual_bot_type
    _num_cards_used = n
    _bot_fallback_reason = fallback_reason
    _requested_bot = requested_bot

    nash_rule_model_meta: Optional[str] = None
    if requested_bot == BOT_NASH:
        nash_rule_model_meta = (
            EXACT_MODE_CLASSIC if actual_bot_type == BOT_NASH else None
        )
    elif requested_bot == BOT_NASH_CARRY:
        nash_rule_model_meta = (
            EXACT_MODE_CARRY if actual_bot_type == BOT_NASH_CARRY else None
        )

    # ---- VEIL + 规则变体 元数据 (供前端显示"当前规则模式"标签 & banner) ----
    veil_flags_active: List[str] = []
    if v_hidden: veil_flags_active.append("HiddenPrize")
    if v_suit:   veil_flags_active.append("SuitTiebreak")
    if v_info:   veil_flags_active.append(f"InfoReward({actual_bits_mode})")
    elif actual_bits_mode != "none":
        veil_flags_active.append(f"InfoBitsMode({actual_bits_mode})")
    # tie_rule 独立展示 (不属于 VEIL 标签, 但用户 GUI 上会和 VEIL 一起在机制面板)
    tie_label = {
        TIE_RULE_ROLLOVER: "Rollover(平局滚入·默认)",
        TIE_RULE_DISCARD:  "Discard(平局即弃奖·经典)",
        TIE_RULE_SPLIT:    "Split(平局各分一半)",
    }.get(v_tie, v_tie)
    veil_meta = {
        "hidden_prize": v_hidden,
        "suit_tiebreak": v_suit,
        "info_reward_enabled": v_info,
        "info_bits_mode": actual_bits_mode,
        "any_enabled": any_veil_on,
        "active_tags": veil_flags_active,
        "tie_rule": v_tie,
        "tie_rule_label": tie_label,
        "tie_rule_nondefault": tie_rule_is_nondefault,
    }

    return {
        "state": _build_state(env),
        "last_round": None,
        "meta": {
            "num_cards": n,
            "requested_bot": requested_bot,
            "actual_bot": actual_bot_type,
            "actual_bot_label": BOT_DESCRIPTIONS.get(actual_bot_type, actual_bot_type),
            # 严格语义：非空 ⇔ actual != requested（真回落）
            "fallback_reason": fallback_reason,
            # 精确 Nash 预计算耗时（秒）。未触发预计算时 = None。
            # 慢 ≠ 回落；这个字段用于前端显示"加载提示条"而非回落 banner。
            "nash_warmup_sec": nash_warmup_sec,
            # 对两类精确 Nash: 明确告诉前端当前实际模型
            "nash_rule_model": nash_rule_model_meta,
            # ---- VEIL 新增 (旧前端忽略此字段 = 安全无影响) ----
            "veil": veil_meta,
        },
    }


@app.get("/api/game/state")
async def get_state() -> Dict[str, Any]:
    env = _get_env_or_400()
    any_on = _veil_hidden_prize or _veil_suit_tiebreak or _veil_info_reward
    tags: List[str] = []
    if _veil_hidden_prize: tags.append("HiddenPrize")
    if _veil_suit_tiebreak: tags.append("SuitTiebreak")
    if _veil_info_reward:   tags.append("InfoReward(half)")
    return {
        "state": _build_state(env),
        "last_round": env.history[-1] if env.history else None,
        "meta": {
            "num_cards": _num_cards_used,
            "actual_bot": _bot_type,
            "actual_bot_label": BOT_DESCRIPTIONS.get(_bot_type, _bot_type),
            "fallback_reason": _bot_fallback_reason,
            # ---- VEIL ----
            "veil": {
                "hidden_prize": _veil_hidden_prize,
                "suit_tiebreak": _veil_suit_tiebreak,
                "info_reward_enabled": _veil_info_reward,
                "info_bits_mode": "half" if _veil_info_reward else "none",
                "any_enabled": any_on,
                "active_tags": tags,
            },
        },
    }


@app.post("/api/game/play")
async def play(req: PlayRequest) -> Dict[str, Any]:
    """
    Submit HUMAN action; BOT picks SIMULTANEOUSLY from the PRE-step state;
    single env.step resolves both.

    Response `last_round.ai_policy` 携带 AI 决策的分布条, 前端画 "AI thinking bars".
    """
    global _bot
    env = _get_env_or_400()
    if _bot is None:
        raise HTTPException(status_code=400, detail="Bot not initialized. New game?")
    if env.done:
        raise HTTPException(
            status_code=400,
            detail="Game is already over. Start a new one.",
        )

    # 1) Validate human action
    human_action: int = int(req.action)
    if human_action not in env.legal_actions(PLAYER_0):
        raise HTTPException(
            status_code=400,
            detail=f"Illegal action {human_action}. Legal: {env.legal_actions(PLAYER_0)}",
        )

    # 2) Bot picks *simultaneously* on pre-step state
    bot_action, bot_policy_info = _bot.choose_action_with_policy(env, PLAYER_1)
    bot_action = int(bot_action)
    if bot_action not in env.legal_actions(PLAYER_1):
        # 双重防线: bot 出 bug 的时候 server 给用户可读错误, 避免 silent 脏数据
        raise HTTPException(
            status_code=500,
            detail=(
                f"Bot '{_bot_type}' produced illegal action {bot_action}. "
                f"Legal: {env.legal_actions(PLAYER_1)}"
            ),
        )

    # 2b) Human "what if" counterfactual analysis (carry-aware).
    #
    # Given the *same bot action* bot_action, for EVERY card the human had in
    # hand pre-step we compute win / tie / lose + delta-score.  Crucially,
    # the reward at stake is NOT just prize — it includes the carry pool so
    # the user can see "If I win right now I take the whole 25 (10+15 rollover)".
    # Tie semantics also respect rule-variant:
    #   tie + NOT last round  → roll over (player still 0 score this round but
    #                                       carry_pool grows)
    #   tie + LAST round      → discard permanently (score 0)
    round_prize: int = int(env.current_prize or 0)
    carry: int = int(getattr(env, "carry_pool", 0))
    prize_at_stake = round_prize + carry
    last_round_idx = env.round                        # 本轮 reveal 后的 round 号 (step 前)
    is_last_round_before_step = (len(env.remaining_prizes) == 0)  # remaining_prizes 在 reveal 后就已不含 current_prize；空=这是最后一张
    human_legal_prestep = sorted(env.legal_actions(PLAYER_0))
    bot_cv: int = int(bot_action)                    # 假设 bot 固定出这张牌
    bot_cd: str = card_display_name(bot_cv)
    human_counterfactual: List[Dict[str, Any]] = []
    for hc in human_legal_prestep:
        if hc > bot_action:
            outcome = "win"          # 你会稳稳拿下 prize_at_stake (含 carry)
            delta = prize_at_stake
            bot_delta = 0            # bot 拿 0 分 (你赢)
        elif hc < bot_action:
            outcome = "lose"         # AI 拿下 prize_at_stake (含 carry)
            delta = 0
            bot_delta = prize_at_stake
        else:
            outcome = "tie"
            delta = 0                # 平局本轮不得分; 非末轮会滚入下一轮奖池
            bot_delta = 0            # bot 本轮同样 0 分
        # Outcome 描述文字 (carry-aware, UI 侧直接显示在条上 tooltip 里)
        if outcome == "tie":
            if is_last_round_before_step:
                oc_desc = (f"平局 · 末轮，{prize_at_stake}={round_prize}+carry{carry} 丢弃")
            else:
                if carry > 0:
                    oc_desc = (f"平局 · {round_prize}+carry{carry}={prize_at_stake} 滚入下一轮")
                else:
                    oc_desc = f"平局 · {round_prize} 滚入下一轮"
        else:  # win/lose
            if carry > 0:
                oc_desc = (
                    f"{'你赢' if outcome == 'win' else 'Bot赢'} "
                    f"{round_prize}+carry{carry}={prize_at_stake}"
                )
            else:
                oc_desc = f"{'你赢' if outcome == 'win' else 'Bot赢'} {round_prize}"
        # ===== 新增维度：对手亏牌 + 你自己的子力性价比 =====
        #   opponent_net_gain = bot 的奖分所得 − bot 实际出掉的牌面值
        #   (相当于 bot 花了 bot_cv 这么多「子力价值」去换 bot_delta 这么多奖分)
        #   · 输奖 + 平局 (bot_delta=0) 时 bot 纯亏 bot_cv
        #   · 赢奖时净收益 = prize_at_stake − bot_cv（正=赚牌，负=亏牌）
        #   my_net_gain = 你的奖分 − 你出掉的牌面值（对称维度，用户视角自评）
        bot_net_gain = int(bot_delta - bot_cv)
        my_net_gain  = int(delta - hc)
        if bot_net_gain > 0:
            bot_eff = "profitable"   # 赚牌：K 花得值，对手用小牌抢了大奖
            bot_eff_label = "Bot 赚牌"
        elif bot_net_gain < 0:
            bot_eff = "wasted"       # 亏牌：对手大牌打蚊子 / 输了 / 打平
            bot_eff_label = "Bot 亏牌"
        else:
            bot_eff = "even"         # 平本：bot_cv 刚好等于收益
            bot_eff_label = "Bot 平本"
        # bot 效率的解释性文字（tooltip 用，直接给人话）
        if outcome == "win":
            bot_eff_desc = (
                f"你改出 {card_display_name(hc)} 会赢 {prize_at_stake}。"
                f"对手出 {bot_cd}（值 {bot_cv}）啥都拿不到，纯亏子力 {bot_cv} "
                f"（净收益 {bot_net_gain:+d}）"
            )
        elif outcome == "lose":
            if bot_net_gain > 0:
                bot_eff_desc = (
                    f"对手出 {bot_cd} 赢了 {prize_at_stake}，"
                    f"子力花掉 {bot_cv}，净 +{bot_net_gain} = 赚牌（很划算这张牌）"
                )
            elif bot_net_gain < 0:
                bot_eff_desc = (
                    f"对手出 {bot_cd} 虽然赢了 {prize_at_stake}，"
                    f"但子力花掉 {bot_cv}，净 {bot_net_gain:+d} = 亏牌。"
                    f"后面大奖 K/Q/J 对手机会变少，你改出 {card_display_name(hc)} 这招虽丢分但骗了大牌。"
                )
            else:
                bot_eff_desc = (
                    f"对手出 {bot_cd} 赢 {prize_at_stake}，"
                    f"收益 {prize_at_stake} = 花费 {bot_cv}，打平。"
                )
        else:  # tie
            tie_note = "末轮弃奖" if is_last_round_before_step else "奖滚入下一轮"
            bot_eff_desc = (
                f"平局（{tie_note}）。对手出 {bot_cd} 本轮 0 分，"
                f"子力纯亏 {bot_cv}（净收益 {bot_net_gain:+d} = 亏牌）"
            )
        human_counterfactual.append({
            "card_value": int(hc),
            "card_display": card_display_name(hc),
            "outcome": outcome,
            "delta": int(delta),
            "outcome_desc": oc_desc,
            "played": (hc == human_action),
            # --- 对手视角：对手亏牌 / 赚牌 分析 ---
            "bot_card_value": int(bot_cv),
            "bot_card_display": bot_cd,
            "bot_prize_delta": int(bot_delta),          # 对手从奖池得到的分
            "bot_net_gain": bot_net_gain,                # 净收益 = 奖分 - 牌面值（可能负数=亏牌）
            "bot_efficiency": bot_eff,                   # "wasted" | "even" | "profitable"
            "bot_efficiency_label": bot_eff_label,       # 中文标签 "Bot 亏牌" 等
            "bot_eff_desc": bot_eff_desc,                # 完整解释 (tooltip)
            # --- 你自己视角对称维度（顺便给）---
            "my_net_gain": my_net_gain,                  # 你净收益 = 你奖分 - 你出牌面值
        })
    # ---------------------------------------------------------- end 2b

    # 3) Single step -> both actions applied atomically
    obs, rewards, done, info = env.step({
        PLAYER_0: human_action,
        PLAYER_1: bot_action,
    })
    last_raw = env.history[-1]
    h_round_p = int(last_raw["round_prize"])
    h_carry_in  = int(last_raw["carry_in"])
    h_stake     = int(last_raw["prize_at_stake"])
    h_carry_out = int(last_raw["carry_out"])
    h_discard   = bool(last_raw["discarded"])
    human_r = int(rewards[PLAYER_0])
    bot_r   = int(rewards[PLAYER_1])

    # ---- VEIL §6 suit 信息 (本轮) ----
    suits_r = last_raw.get("suits") or {}
    h_suit_idx = suits_r.get(PLAYER_0) if suits_r else None
    b_suit_idx = suits_r.get(PLAYER_1) if suits_r else None
    h_suit_sym = _suit_symbol(h_suit_idx)
    b_suit_sym = _suit_symbol(b_suit_idx)
    tie_broken_by_suit = bool(last_raw.get("tie_broken_by_suit"))

    # ---- VEIL §11-12 info reward (本轮 lowest bid 归属 + signal) ----
    low_player = last_raw.get("lowest_bid_player")
    info_sig_r = last_raw.get("info_signal")
    prize_was_hidden = bool(last_raw.get("prize_was_hidden_before_bid"))

    # ---- Human-readable banner text for THIS round (tie-carry + VEIL aware) ----
    winner = last_raw["winner"]
    stake_disp = _stake_display_text(h_round_p, h_carry_in, h_stake)
    if winner == PLAYER_0:
        banner = (
            f"You played {h_suit_sym}{card_display_name(human_action)} · "
            f"Bot played {b_suit_sym}{card_display_name(bot_action)} → "
            f"You win {stake_disp}"
        )
    elif winner == PLAYER_1:
        banner = (
            f"You played {h_suit_sym}{card_display_name(human_action)} · "
            f"Bot played {b_suit_sym}{card_display_name(bot_action)} → "
            f"Bot wins {stake_disp}"
        )
    else:  # tie
        if h_discard:
            banner = (
                f"You played {h_suit_sym}{card_display_name(human_action)} · "
                f"Bot played {b_suit_sym}{card_display_name(bot_action)} → "
                f"Tie (final round). {stake_disp} discarded · no rollover"
            )
        else:
            banner = (
                f"You played {h_suit_sym}{card_display_name(human_action)} · "
                f"Bot played {b_suit_sym}{card_display_name(bot_action)} → "
                f"Tie · {stake_disp} rolls over to next round "
                f"(carry={h_carry_out})"
            )
    if tie_broken_by_suit:
        winner_suit = h_suit_sym if winner == PLAYER_0 else (b_suit_sym if winner == PLAYER_1 else "")
        banner += f" · 同点花色解平 (♣<♦<♥<♠ 中 {winner_suit} 高者胜)"
    if low_player and info_sig_r:
        who = "You" if low_player == PLAYER_0 else "Bot"
        banner += f" · 【情报】最低出牌者{who}获下一轮奖励信号: {info_sig_r}"
    if prize_was_hidden:
        banner += f" · 本轮奖励出牌前为Hidden (揭晓值: {card_display_name(h_round_p)})"

    last_round = {
        "round": last_raw["round"],
        # prize-level fields (含 carry 扩展)
        "prize": h_round_p,                        # backward compat
        "round_prize": h_round_p,
        "round_prize_display": card_display_name(h_round_p),
        "prize_display": card_display_name(h_round_p),   # backward compat = round_prize_display
        "carry_in": h_carry_in,
        "carry_in_display": card_display_name(h_carry_in) if h_carry_in > 0 else None,
        "prize_at_stake": h_stake,
        "prize_at_stake_display": _stake_short(h_round_p, h_carry_in, h_stake),
        "carry_out": h_carry_out,
        "discarded": h_discard,
        # actions
        "human_action": human_action,
        "human_action_display": card_display_name(human_action),
        "human_suit": h_suit_idx,
        "human_suit_symbol": h_suit_sym,
        "bot_action": bot_action,
        "bot_action_display": card_display_name(bot_action),
        "bot_suit": b_suit_idx,
        "bot_suit_symbol": b_suit_sym,
        "winner": winner,
        "human_reward": human_r,
        "bot_reward": bot_r,
        "result_text": banner,
        # VEIL 扩展字段
        "tie_broken_by_suit": tie_broken_by_suit,
        "info_award_to": ("human" if low_player == PLAYER_0 else
                          "bot" if low_player == PLAYER_1 else None),
        "info_award_signal": info_sig_r,
        "prize_was_hidden_before_bid": prize_was_hidden,
        # AI 决策"透明化"数据 (AI 每张牌的出手概率%)
        "ai_policy": {
            "bot_type": bot_policy_info.get("bot_type", _bot_type),
            "value": bot_policy_info.get("value", float("nan")),
            "note": bot_policy_info.get("note", ""),
            # distribution: [[card_value, pct_0_100], ...]
            "distribution": bot_policy_info.get("distribution", []),
        },
        # 反事实 "如果你刚才改出 X …"
        "human_policy": {
            "note": (
                f"Bot played {b_suit_sym}{card_display_name(bot_action)}; "
                f"prize_at_stake this round = "
                f"{h_round_p}(round)+{h_carry_in}(carry)={h_stake}. "
                f"Outcome for every card in YOUR pre-step hand: "
                f"win=you take {h_stake}, lose=bot takes {h_stake}, "
                f"tie=0+rollover or final discard."
                + (" · suit 同点时花色决胜负 (♣<♦<♥<♠)" if tie_broken_by_suit or h_suit_idx is not None else "")
            ),
            "bot_action_played": int(bot_action),
            "prize_at_stake": int(h_stake),
            "round_prize": int(h_round_p),
            "carry_in": int(h_carry_in),
            "bars": human_counterfactual,
        },
    }

    any_on = _veil_hidden_prize or _veil_suit_tiebreak or _veil_info_reward
    tags: List[str] = []
    if _veil_hidden_prize: tags.append("HiddenPrize")
    if _veil_suit_tiebreak: tags.append("SuitTiebreak")
    if _veil_info_reward:   tags.append("InfoReward(half)")

    return {
        "state": _build_state(env),
        "last_round": last_round,
        "ai_policy": last_round["ai_policy"],
        "meta": {
            "num_cards": _num_cards_used,
            "actual_bot": _bot_type,
            "actual_bot_label": BOT_DESCRIPTIONS.get(_bot_type, _bot_type),
            "fallback_reason": _bot_fallback_reason,
            # ---- VEIL (新增, 同 /api/game/state & /api/game/new) ----
            "veil": {
                "hidden_prize": _veil_hidden_prize,
                "suit_tiebreak": _veil_suit_tiebreak,
                "info_reward_enabled": _veil_info_reward,
                "info_bits_mode": "half" if _veil_info_reward else "none",
                "any_enabled": any_on,
                "active_tags": tags,
            },
        },
    }


# ------------------------------------------------------------------------
# Helpers: state builder & card-display
# ------------------------------------------------------------------------
def _stake_short(round_prize: int, carry_in: int, stake: int) -> str:
    """Short form used in bar / history cell headings.
    prize=10 carry=0 → 10
    prize=10 carry=5 → 10(+5)=15
    """
    rp = int(round_prize)
    c = int(carry_in)
    s = int(stake)
    if c <= 0:
        return str(s)
    return f"{rp}(+{c})={s}"

def _stake_display_text(round_prize: int, carry_in: int, stake: int) -> str:
    """Longer form used in banners / round result sentences.
    Examples:
      prize 10                  (carry 0)
      prize 10 + carry 5 = 15   (carry>0)
    """
    rp_d = card_display_name(round_prize)
    s_d  = card_display_name(stake)
    if int(carry_in) <= 0:
        return f"prize {rp_d} (={s_d})"
    c_d = card_display_name(carry_in)
    return f"prize {rp_d} + carry {c_d} = {s_d}"

def _round_result_text(h: Dict[str, Any]) -> str:
    """Generate the `result_text` line shown in History (R1 / R2 / …) cell."""
    pv = int(h["round_prize"])
    carry_in = int(h["carry_in"])
    stake = int(h["prize_at_stake"])
    stake_d = _stake_short(pv, carry_in, stake)
    winner = h["winner"]
    tie_suit_note = ""
    if bool(h.get("tie_broken_by_suit")):
        tie_suit_note = " · 同点花色解平"
    winner_line = ""
    if winner == PLAYER_0:
        winner_line = f"You win {stake_d}"
    elif winner == PLAYER_1:
        winner_line = f"Bot wins {stake_d}"
    else:
        if bool(h.get("discarded")):
            winner_line = f"Tie · {stake_d} discarded (final round)"
        else:
            carry_out = int(h["carry_out"])
            winner_line = f"Tie · {stake_d} rolls over (carry={carry_out})"
    base = winner_line + tie_suit_note
    low = h.get("lowest_bid_player")
    sig = h.get("info_signal")
    if low and sig:
        who = "You" if low == PLAYER_0 else "Bot"
        base += f" · Info→{who}:{sig}"
    if bool(h.get("prize_was_hidden_before_bid")):
        base += " · HiddenPrize"
    return base


# ---- suit int -> symbol fallback (env.py already defines SUIT_SYMBOLS but we don't import directly) ----
SUIT_SYMBOLS_FALLBACK: Dict[int, str] = {0: "\u2663", 1: "\u2666", 2: "\u2665", 3: "\u2660"}


def _suit_symbol(suit_idx: Optional[int]) -> str:
    if suit_idx is None:
        return ""
    return SUIT_SYMBOLS_FALLBACK.get(int(suit_idx), "")


def _build_state(env: GoofspielEnv) -> Dict[str, Any]:
    """
    构建 UI 友好的完整状态 (server-owned 所有真值)。
    This is the SINGLE source of truth the frontend will render.

    Compatibility (兼容性):
      - 旧字段 (scores/carry_pool 等) 100% 保留原名和语义。
      - viewer=PLAYER_0 (HUMAN perspective) 用于:
          * 隐藏 hidden_prize 模式下的 current_prize
          * 仅向玩家展示"自己的"私人情报, 对手情报强制隐藏
    """
    obs = env.get_observation(viewer=PLAYER_0)   # HUMAN perspective (VEIL §12 私人信息过滤)
    veil_obs: Dict[str, Any] = obs.get("veil") or {}
    human_numeric = sorted(obs["remaining_cards"][PLAYER_0])
    bot_numeric   = sorted(obs["remaining_cards"][PLAYER_1])
    carry_pool    = int(obs.get("carry_pool", 0))
    total_stake   = int(obs.get("total_prize_at_stake",
                                  (int(obs.get("current_prize") or 0) + carry_pool)))

    human_used: List[Dict[str, Any]] = []
    bot_used:   List[Dict[str, Any]] = []
    round_log:  List[Dict[str, Any]] = []

    for h in env.history:
        hv = int(h["actions"][PLAYER_0])
        bv = int(h["actions"][PLAYER_1])
        pv = int(h["round_prize"])
        # suit per round (VEIL §6 suit_tiebreak)
        suits_r = h.get("suits") or {}
        h_suit_idx = suits_r.get(PLAYER_0) if suits_r else None
        b_suit_idx = suits_r.get(PLAYER_1) if suits_r else None
        h_suit_sym = _suit_symbol(h_suit_idx)
        b_suit_sym = _suit_symbol(b_suit_idx)
        human_used.append({
            "value": hv, "display": card_display_name(hv),
            "suit": h_suit_idx, "suit_symbol": h_suit_sym,
        })
        bot_used.append({
            "value": bv, "display": card_display_name(bv),
            "suit": b_suit_idx, "suit_symbol": b_suit_sym,
        })
        prize_d = card_display_name(pv)
        result_line = _round_result_text(h)
        carry_in = int(h["carry_in"])
        stake    = int(h["prize_at_stake"])
        carry_out= int(h["carry_out"])
        # info reward
        low_p = h.get("lowest_bid_player")
        info_r = h.get("info_signal")
        info_for_r = h.get("info_for_round")
        info_award_to = None
        info_award_sig = None
        if low_p and info_r:
            info_award_to = "human" if low_p == PLAYER_0 else "bot"
            info_award_sig = info_r
        round_log.append({
            "round":        int(h["round"]),
            "prize":        pv,
            "round_prize":  pv,
            "prize_display": prize_d,
            "carry_in":     carry_in,
            "carry_in_display": (card_display_name(carry_in) if carry_in > 0 else None),
            "prize_at_stake":    stake,
            "prize_at_stake_display": _stake_short(pv, carry_in, stake),
            "carry_out":    carry_out,
            "discarded":    bool(h.get("discarded", False)),
            "human_action": hv,
            "human_action_display": card_display_name(hv),
            "human_suit":   h_suit_idx,
            "human_suit_symbol": h_suit_sym,
            "bot_action":   bv,
            "bot_action_display": card_display_name(bv),
            "bot_suit":     b_suit_idx,
            "bot_suit_symbol": b_suit_sym,
            "winner":       h["winner"],
            "result_text":  result_line,
            # VEIL 扩展 (suit_tiebreak/info_reward/hidden_prize)
            "tie_broken_by_suit": bool(h.get("tie_broken_by_suit")),
            "lowest_bid_is_human": (low_p == PLAYER_0) if low_p else None,
            "info_award_to": info_award_to,
            "info_award_signal": info_award_sig,
            "info_award_for_round": info_for_r,
            "prize_was_hidden": bool(h.get("prize_was_hidden_before_bid")),
        })

    # ---- 当前轮 suit (suit_tiebreak 模式下为玩家展示"你本轮拿到的花色") ----
    current_suit_h_idx = None
    current_suit_b_idx = None
    current_suit_h_sym = ""
    current_suit_b_sym = ""
    if veil_obs.get("suit_tiebreak"):
        suits_now = veil_obs.get("suits") or {}
        current_suit_h_idx = suits_now.get(PLAYER_0)
        current_suit_b_idx = suits_now.get(PLAYER_1)
        current_suit_h_sym = _suit_symbol(current_suit_h_idx)
        current_suit_b_sym = _suit_symbol(current_suit_b_idx)

    # ---- 私人情报 (VEIL §12 — 只展示玩家自己的 HIGH/LOW) ----
    my_private_info = None
    private_info_map = veil_obs.get("private_info") or {}
    if private_info_map.get(PLAYER_0):
        my_private_info = str(private_info_map[PLAYER_0])

    # ---- HiddenPrize: 当前奖励是否隐藏 (UI 显示 "?" 替代牌面) ----
    prize_hidden_now = bool(veil_obs.get("prize_is_currently_hidden"))
    hidden_display = None
    if prize_hidden_now:
        hidden_display = "?"

    veil_state_ui = {
        "enabled": any([
            veil_obs.get("hidden_prize"),
            veil_obs.get("suit_tiebreak"),
            veil_obs.get("info_reward_enabled"),
        ]),
        "hidden_prize": bool(veil_obs.get("hidden_prize")),
        "suit_tiebreak": bool(veil_obs.get("suit_tiebreak")),
        "info_reward_enabled": bool(veil_obs.get("info_reward_enabled")),
        "info_bits_mode": str(veil_obs.get("info_bits_mode") or "none"),
        # §29 平局处理规则 (前端顶部 banner / 得分卡角落小标用)
        "tie_rule": str(obs.get("tie_rule") or TIE_RULE_ROLLOVER),
        # 当前轮 suit 显示
        "current_round_suit_human_symbol": current_suit_h_sym,
        "current_round_suit_bot_symbol": current_suit_b_sym,
        # 当前隐藏奖励占位
        "prize_is_hidden_now": prize_hidden_now,
        "prize_hidden_placeholder": hidden_display,
        # 玩家私人情报 (HIGH/LOW 或 None) — GUI 手牌区顶部直接显式显示
        "my_private_info_next_prize_half": my_private_info,
        # 历史: 上一轮谁拿到了情报 (公开信息)
        "last_info_awarded_to": veil_obs.get("last_info_awarded_to"),
    }

    return {
        "round": obs["round"],
        "num_cards": env.num_cards,
        "current_prize": obs["current_prize"],
        "current_prize_display": (
            hidden_display if prize_hidden_now else (
                card_display_name(obs["current_prize"])
                if obs["current_prize"] is not None else None
            )
        ),
        # --- carry-over UI fields ---------------------------------------
        "carry_pool": carry_pool,
        "carry_pool_display": (
            card_display_name(carry_pool) if carry_pool > 0 else "0"
        ),
        "total_prize_at_stake": total_stake,
        "total_prize_at_stake_display": (
            hidden_display if prize_hidden_now else (
                _stake_short(int(obs["current_prize"] or 0), carry_pool, total_stake)
            )
        ),
        # ----------------------------------------------------------------
        "scores": {
            "human": obs["scores"][PLAYER_0],
            "bot":   obs["scores"][PLAYER_1],
        },
        "remaining_cards": {
            "human": [
                {"value": v, "display": card_display_name(v)}
                for v in human_numeric
            ],
            "bot": [
                {"value": v, "display": card_display_name(v)}
                for v in bot_numeric
            ],
        },
        "remaining_prizes_display": [
            card_display_name(v) for v in obs["remaining_prizes"]
        ],
        "used_cards": {
            "human": human_used,
            "bot":   bot_used,
        },
        "history": round_log,
        "done": obs["done"],
        "result": obs["result"],
        # ---- VEIL UI state (新增, 旧前端忽略 = 安全无影响) ----
        "veil": veil_state_ui,
    }


def card_display_name(v: int) -> str:
    mapping = {1: "A", 11: "J", 12: "Q", 13: "K"}
    return mapping.get(int(v), str(int(v)))


# ------------------------------------------------------------------------
# Direct execution launcher (auto port fallback 跟旧版完全一致)
# ------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import os
    import socket
    import sys
    import uvicorn

    def _port_is_free(host: str, port: int) -> bool:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False
        finally:
            s.close()

    def _pick_port(host: str, preferred: int, max_tries: int) -> int:
        for i in range(max_tries):
            p = preferred + i
            if i == 0:
                print(f"Goofspiel: checking http://{host}:{p} ...", flush=True)
            else:
                print(f"  -> port {preferred + i - 1} busy, trying :{p} ...", flush=True)
            if _port_is_free(host, p):
                return p
        return -1

    parser = argparse.ArgumentParser(description="Goofspiel FastAPI server")
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    parser.add_argument("--max-port-tries", type=int,
                        default=int(os.environ.get("MAX_PORT_TRIES", "50")))
    args = parser.parse_args()

    start_port = max(1, int(args.port))
    max_tries  = max(1, int(args.max_port_tries))
    chosen = _pick_port(args.host, start_port, max_tries)
    if chosen < 0:
        print(
            f"\nGoofspiel: FAILED to find a free port. "
            f"Tried ports {start_port}-{start_port + max_tries - 1}.\n"
            "Tip: pass a higher start port, e.g.  python app.py --port 9000",
            file=sys.stderr,
        )
        sys.exit(1)
    if chosen != start_port:
        print(f"Goofspiel: port {start_port} not available; using :{chosen}.")
    print(f"Goofspiel: starting server on http://{args.host}:{chosen}")
    print("           (Press CTRL+C to stop)")
    uvicorn.run(app, host=args.host, port=chosen, log_level="info")
