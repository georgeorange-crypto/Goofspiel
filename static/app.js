/*
 * Author: 陈子聪 (Chen Zicong)
 * Date: 2026-08-30
 * Purpose: Vanilla JS controller for the NEW Goofspiel web UI:
 *   - configurable N (1..13) cards
 *   - AI choice dropdown (Random / Heuristic / Exact Nash)
 *   - AI "thinking bars" showing the per-card policy distribution (%)
 *
 * The authoritative state lives on the server; this file ONLY renders
 * what the server returns and submits user actions.
 */

(function () {
    "use strict";

    const $ = (sel) => document.querySelector(sel);

    // ---------------------------------------------------------------- state
    const ui = {
        // Header / setup form
        subtitle:       $("#subtitle"),
        btnRestart:     $("#btn-restart"),

        form:           $("#setup-form"),
        inputNumCards:  $("#input-num-cards"),
        inputBot:       $("#input-bot"),
        btnStart:       $("#btn-start"),
        loadingHint:    $("#loading-hint"),
        fallbackWarn:   $("#fallback-warning"),
        numCardsHint:   $("#num-cards-hint"),

        // VEIL — setup form checkboxes
        veilCheckboxes: $("#veil-checkboxes"),
        veilNashWarn:   $("#veil-nash-warn"),

        // VEIL — in-game status strip
        veilStatus:     $("#veil-status"),
        veilActiveTags: $("#veil-active-tags"),
        veilStatusBody: $("#veil-status-body"),

        // Status
        roundNum:       $("#round-num"),
        roundTotal:     $("#round-total"),
        currentPrize:   $("#current-prize"),
        carryPool:      $("#carry-pool"),
        stakeTotal:     $("#stake-total"),
        scoreHuman:     $("#score-human"),
        scoreBot:       $("#score-bot"),

        // AI panel
        aiPanel:        $("#ai-thinking"),
        aiBotType:      $("#ai-thinking-bot-type"),
        aiNote:         $("#ai-thinking-note"),
        aiValueEl:      $("#ai-value"),
        aiBars:         $("#ai-bars"),

        // Human counterfactual panel
        hCfPanel:       $("#human-counterfactual"),
        hCfNote:        $("#human-cf-note"),
        hCfBars:        $("#human-cf-bars"),

        // Hands / history
        humanCards:     $("#human-cards"),
        botCards:       $("#bot-cards"),
        humanUsed:      $("#human-used"),
        botUsed:        $("#bot-used"),
        historyList:    $("#history-list"),

        // Banners
        roundBanner:     $("#round-banner"),
        roundBannerText: $("#round-banner-text"),
        finalBanner:     $("#final-banner"),
        finalBannerText: $("#final-banner-text"),
    };

    /** @type {{num_cards: {min,max,default}, bots: Array<{id,label,max_n_for_exact_nash}>}|null} */
    let CONFIG = null;
    let pending = false;       // API-in-flight guard
    let currentState = null;   // last rendered server state
    let selectedBot = null;    // user's current form selection (tracked for Nash hint)

    // -------------------------------------------------------------- helpers
    function el(tag, cls, text) {
        const node = document.createElement(tag);
        if (cls)  node.className = cls;
        if (text !== undefined && text !== null) node.textContent = String(text);
        return node;
    }

    function showBanner(text, isRound) {
        const target = isRound ? ui.roundBanner : ui.finalBanner;
        const textEl = isRound ? ui.roundBannerText : ui.finalBannerText;
        textEl.textContent = text;
        target.classList.remove("hidden");
    }
    function hideBanner(isRound) {
        const target = isRound ? ui.roundBanner : ui.finalBanner;
        target.classList.add("hidden");
    }

    function clampN(n) {
        const c = CONFIG.num_cards;
        return Math.min(c.max, Math.max(c.min, n|0 || c.default));
    }

    function updateNashHint() {
        const bot = ui.inputBot.value;
        const n = clampN(ui.inputNumCards.value);
        const botMeta = (CONFIG.bots || []).find(b => b.id === bot) || {};
        const maxN = botMeta.max_n_for_exact_nash;
        const isExactNashBot = bot === "nash" || bot === "nash_carry";
        let anyWarn = false;
        const warnings = [];

        // (a) 经典 N 超 Nash 上限检测
        if (isExactNashBot && typeof maxN === "number" && n > maxN) {
            warnings.push(
                `⚠️ N=${n} 超过「${botMeta.label || bot}」精确 Nash 支持上限 (${maxN})。`
                + `本局会自动降级为 Heuristic（启发式）。`
            );
        }

        // (b) VEIL + Nash exact 不兼容检测 (任何 VEIL 机制或 tie_rule 变体)
        const opts = collectVeilOptions();
        const anyVeilMechanism = Boolean(
            opts.veil_hidden_prize
            || opts.veil_suit_tiebreak
            || opts.veil_info_reward
            || (opts.info_bits_mode && opts.info_bits_mode !== "auto" && opts.info_bits_mode !== "none")
        );

        // (c) tie_rule × bot 奖牌型一致性检测
        //     nash        ↔ discard  (平局弃奖 经典)
        //     nash_carry  ↔ rollover (平局滚入)
        //     split       与任一精确 Nash 均不兼容
        const tie = opts.tie_rule || "rollover";
        let tieRuleMismatch = false;
        if (bot === "nash" && tie !== "discard") tieRuleMismatch = true;
        if (bot === "nash_carry" && tie !== "rollover") tieRuleMismatch = true;
        if (isExactNashBot && tie === "split") tieRuleMismatch = true;

        if (isExactNashBot && (anyVeilMechanism || tieRuleMismatch)) {
            const parts = [];
            if (opts.veil_hidden_prize) parts.push("隐藏奖励");
            if (opts.veil_suit_tiebreak) parts.push("花色 Tie-break");
            if (opts.veil_info_reward) parts.push("信息奖励/情报通道");
            if (tieRuleMismatch) {
                const need =
                    (bot === "nash")        ? "平局即弃奖 (discard)"  :
                    (bot === "nash_carry")  ? "平局滚入 (rollover)"   :
                    "—";
                parts.push(`平局规则=${tie} (该 Nash 奖牌型要求=${need})`);
            }
            warnings.push(
                `⚠️ 「${botMeta.label || bot}」精确 Nash 与当前机制不兼容：${parts.join(" / ")}。`
                + `本局会诚实回落 Heuristic（启发式）。`
            );
        }

        if (warnings.length) {
            ui.fallbackWarn.classList.remove("hidden");
            ui.fallbackWarn.textContent = warnings.join(" ");
            ui.veilNashWarn.classList.remove("hidden");
            ui.veilNashWarn.textContent = warnings[warnings.length - 1];
            anyWarn = true;
        } else {
            ui.fallbackWarn.classList.add("hidden");
            ui.veilNashWarn.classList.add("hidden");
        }

        return anyWarn;
    }

    /** Returns true if any VEIL checkbox is currently checked OR any non-default enum option. */
    function isAnyVeilChecked() {
        const opts = collectVeilOptions();
        return Boolean(
            opts.veil_hidden_prize
            || opts.veil_suit_tiebreak
            || opts.veil_info_reward
            || (opts.info_bits_mode && opts.info_bits_mode !== "auto" && opts.info_bits_mode !== "none")
            || (opts.tie_rule && opts.tie_rule !== "rollover")
        );
    }

    /**
     * Collect ALL veil options (checkbox bools / select strings / radio-group strings)
     * as a flat {id: value} dict.  Every key id matches NewGameRequest top-level
     * schema field 1:1.  Returns the same shape regardless of user interactions
     * (missing elements → default fallback handled by submitNewGameForm via server).
     */
    function collectVeilOptions() {
        const out = {};
        if (!ui.veilCheckboxes) return out;

        // (1) checkbox
        const cbs = ui.veilCheckboxes.querySelectorAll('input[type="checkbox"][data-veil-id]');
        cbs.forEach(cb => { out[cb.dataset.veilId] = Boolean(cb.checked); });

        // (2) select
        const selects = ui.veilCheckboxes.querySelectorAll('select[data-veil-id]');
        selects.forEach(sel => { out[sel.dataset.veilId] = sel.value; });

        // (3) radio groups: 每个 group 一个 id → 取 checked 值 (未选中 → null,
        //     server schema default 会接住)
        const groups = new Set();
        ui.veilCheckboxes
            .querySelectorAll('input[type="radio"][data-veil-id]')
            .forEach(r => groups.add(r.dataset.veilId));
        groups.forEach(gid => {
            const checked = ui.veilCheckboxes.querySelector(
                `input[type="radio"][data-veil-id="${gid}"]:checked`
            );
            out[gid] = checked ? checked.value : null;
        });
        return out;
    }

    /** @deprecated backward-compat alias (旧引用指向新集合) */
    function collectVeilFlags() { return collectVeilOptions(); }

    // ----------------------------------------------------------- UI wiring
    function populateSetupForm() {
        const c = CONFIG;
        // Num cards
        ui.inputNumCards.min = c.num_cards.min;
        ui.inputNumCards.max = c.num_cards.max;
        ui.inputNumCards.value = c.num_cards.default;
        ui.numCardsHint.textContent =
            `${c.num_cards.min} = 只有 ${cardLabel(c.num_cards.min)}；`
            + `${c.num_cards.max} = 完整一副 ${cardLabel(1)}~${cardLabel(13)}`;

        // Bots dropdown
        ui.inputBot.innerHTML = "";
        c.bots.forEach(b => {
            const opt = el("option", null, b.label);
            opt.value = b.id;
            ui.inputBot.appendChild(opt);
        });
        // Default: "heuristic" is a good middle ground for most users
        // (faster than Nash; stronger than Random)
        const defaultBot = "heuristic";
        if (c.bots.some(b => b.id === defaultBot)) {
            ui.inputBot.value = defaultBot;
        }

        // VEIL 可选机制复选框（根据 /api/game/config → veil_options）
        populateVeilCheckboxes(c.veil_options || []);

        updateNashHint();
    }

    function populateVeilCheckboxes(options) {
        if (!ui.veilCheckboxes) return;
        ui.veilCheckboxes.innerHTML = "";
        if (!options.length) {
            ui.veilCheckboxes.appendChild(
                el("div", "hint", "（后端未提供 VEIL 选项）"));
            return;
        }
        options.forEach(opt => {
            const label = el("label", "veil-option");
            label.setAttribute("for", opt.id);

            const input = document.createElement("input");
            input.type = "checkbox";
            input.id = opt.id;
            input.checked = Boolean(opt.default);
            // Default all-off.  Server provides false default for all 3.
            if (opt.default == null) input.checked = false;

            // ================================================================
            // GUI toggle switch — wraps the native checkbox.
            // Left click on either the slider, the label, or the description
            // toggles the switch.  The <input> is still a real checkbox so
            // collectVeilFlags (querySelectorAll input[type=checkbox]) keeps
            // working unmodified — pure visual upgrade, zero logic contract.
            // ================================================================
            const switchEl     = el("span", "gui-toggle");
            const switchSlider = el("span", "gui-toggle-knob");
            switchEl.appendChild(switchSlider);

            const labelSpan = el("span", "veil-option-label", opt.label || opt.id);
            const catSpan = el("span", "veil-option-cat",
                               opt.category ? `[${opt.category}]` : "");
            const descSmall = el("small", "veil-option-desc",
                                 opt.description || "");

            // Layout: [toggle-switch | checkbox]  |  label-line + description
            const checkboxWrap = el("div", "veil-option-input");
            checkboxWrap.appendChild(input);
            checkboxWrap.appendChild(switchEl);

            const wrap = el("div", "veil-option-content");
            const line1 = el("div", "veil-option-line1");
            line1.appendChild(labelSpan);
            if (opt.category) line1.appendChild(catSpan);
            wrap.appendChild(line1);
            wrap.appendChild(descSmall);

            label.appendChild(checkboxWrap);
            label.appendChild(wrap);
            ui.veilCheckboxes.appendChild(label);

            // Re-trigger Nash-VEIL incompatibility hint on toggle
            input.addEventListener("change", updateNashHint);
        });
    }

    function cardLabel(v) {
        const m = {1:"A", 11:"J", 12:"Q", 13:"K"};
        return m[v] != null ? m[v] : String(v);
    }

    // ----------------------------------------------------------- rendering
    function render(state, lastRound, meta) {
        currentState = state;
        const numCards = state.num_cards || CONFIG?.num_cards?.default || 13;
        const veil = state.veil || null;

        // 1. Top subtitle (反映当前局配置)
        const botLabel = meta?.actual_bot_label || "—";
        const fallback = meta?.fallback_reason ? ` · ${meta.fallback_reason}` : "";
        const veilTag = (veil && veil.active_tags && veil.active_tags.length)
            ? ` · VEIL[${veil.active_tags.join(", ")}]`
            : "";
        ui.subtitle.textContent =
            `${numCards} 张牌 · AI=${botLabel}${fallback}${veilTag}`;
        ui.roundTotal.textContent = numCards;

        // 1b. VEIL status strip — 激活任一机制时展开
        renderVeilStatus(veil);

        // 2. Status
        ui.roundNum.textContent      = state.done ? numCards : (state.round || "–");
        // hidden_prize 下后端在出牌前把 current_prize_display 置为 "?"；
        // 经典模式下为 A..K 字符串，完全保持旧行为。
        ui.currentPrize.textContent  = state.current_prize_display || "–";
        if (veil && veil.prize_is_currently_hidden) {
            ui.currentPrize.classList.add("prize-hidden");
        } else {
            ui.currentPrize.classList.remove("prize-hidden");
        }
        ui.carryPool.textContent     = state.carry_pool_display || "0";
        ui.stakeTotal.textContent    = state.total_prize_at_stake_display
                                        || (state.done ? "0" : "–");
        ui.scoreHuman.textContent    = state.scores.human;
        ui.scoreBot.textContent      = state.scores.bot;

        // 3. Bot remaining cards
        ui.botCards.innerHTML = "";
        (state.remaining_cards.bot || []).forEach(c => {
            ui.botCards.appendChild(el("div", "card bot", c.display));
        });

        // 4. Human remaining cards -> clickable
        ui.humanCards.innerHTML = "";
        const humanDisabled = state.done || pending;
        if (humanDisabled) ui.humanCards.classList.add("disabled");
        else               ui.humanCards.classList.remove("disabled");

        (state.remaining_cards.human || []).forEach(c => {
            const node = el("div", "card", c.display);
            if (humanDisabled) node.classList.add("disabled");
            node.addEventListener("click", () => onHumanCardClicked(c.value, node));
            ui.humanCards.appendChild(node);
        });

        // 5. Used cards — if suit_symbol exists, prepend it (VEIL suit mode)
        ui.humanUsed.innerHTML = "";
        ui.botUsed.innerHTML   = "";
        (state.used_cards.human || []).forEach(c => {
            const s = (c.suit_symbol != null) ? String(c.suit_symbol) : "";
            ui.humanUsed.appendChild(el("div", "card used human-used",
                                        s + c.display));
        });
        (state.used_cards.bot || []).forEach(c => {
            const s = (c.suit_symbol != null) ? String(c.suit_symbol) : "";
            ui.botUsed.appendChild(el("div", "card used bot-used",
                                      s + c.display));
        });

        // 6. History — 经典格式 + (suit 前缀 · info 归属 · hidden标记)
        ui.historyList.innerHTML = "";
        (state.history || []).forEach(h => {
            const hs = h.human_suit_symbol ? String(h.human_suit_symbol) : "";
            const bs = h.bot_suit_symbol   ? String(h.bot_suit_symbol)   : "";
            const hiddenPrizeTag = h.prize_was_hidden ? " 「隐藏奖」" : "";
            const tieSuitTag = h.tie_broken_by_suit ? " 「花色解平」" : "";
            const infoTag = h.info_award_to
                ? ` 「情报→${h.info_award_to === "human" ? "你" : "AI"}`
                    + `${h.info_award_half === "HIGH" ? " HIGH"
                         : h.info_award_half === "LOW" ? " LOW" : ""}」`
                : "";
            const line =
                `R${h.round} · Prize ${h.prize_display}${hiddenPrizeTag}:`
                + ` You ${hs}${h.human_action_display}`
                + ` vs Bot ${bs}${h.bot_action_display}`
                + tieSuitTag
                + infoTag
                + ` — ${h.result_text}`;
            let cls = "";
            if (h.winner === "player_0") cls = "win";
            else if (h.winner === "player_1") cls = "lose";
            else cls = "tie";
            ui.historyList.appendChild(el("li", cls, line));
        });
        ui.historyList.scrollTop = ui.historyList.scrollHeight;

        // 7. AI thinking bars (show ONLY if we have lastRound with ai_policy)
        if (lastRound && lastRound.ai_policy && lastRound.ai_policy.distribution
            && lastRound.ai_policy.distribution.length) {
            renderAiPolicy(lastRound.ai_policy, state.remaining_cards.bot || []);
        } else if (state.history && state.history.length === 0) {
            // New game, no actions yet -> hide AI panel
            ui.aiPanel.classList.add("hidden");
        }

        // 7b. Human counterfactual bars (同样只在有 lastRound 后显示)
        if (lastRound && lastRound.human_policy && lastRound.human_policy.bars
            && lastRound.human_policy.bars.length) {
            renderHumanCounterfactual(lastRound.human_policy);
        } else if (state.history && state.history.length === 0) {
            ui.hCfPanel.classList.add("hidden");
        }

        // 8. Round banner (result of the most recent round)
        // 单真值来源：后端 last_round.result_text 已做 carry-aware 中文措辞，
        // 前端只渲染，不再自拼 "平局作废" 等旧语义文案。
        if (lastRound && typeof lastRound.result_text === "string"
                && lastRound.result_text.length) {
            showBanner(lastRound.result_text, true);
        } else if (lastRound) {
            showBanner(buildRoundBannerText(lastRound), true);
        } else if (state.history && state.history.length === 0) {
            hideBanner(true);
        }

        // 8b. 如果启用了 suit_tiebreak → 给人类手牌加花色前缀 (display 不变)
        //      后端 state.remaining_cards.human[*].suit_symbol 已返回 ♠/♥/♦/♣
        decorateRemainingCardsWithSuit(state.remaining_cards.human || [],
                                       state.remaining_cards.bot || []);

        // 9. Final banner
        if (state.done) {
            hideBanner(true);
            const sH = state.scores.human, sB = state.scores.bot;
            let msg, cls;
            if (state.result === "player_0") {
                msg = `🎉 你赢了! You win! ${sH} – ${sB}`;
                cls = "final win";
            } else if (state.result === "player_1") {
                msg = `🤖 AI 胜 Bot wins. ${sB} – ${sH}`;
                cls = "final lose";
            } else {
                msg = `🤝 平局 Draw. ${sH} – ${sB}`;
                cls = "final draw";
            }
            ui.finalBanner.className = "panel " + cls;
            showBanner(msg, false);
        } else {
            hideBanner(false);
        }
    }

    function buildRoundBannerText(lastRound) {
        const you = lastRound.human_action_display;
        const bot = lastRound.bot_action_display;
        const prize = lastRound.prize_display;
        if (lastRound.winner === "player_0") {
            return `你出 ${you}，AI 出 ${bot}。你赢得奖金 ${prize} (+${lastRound.human_reward})。`;
        }
        if (lastRound.winner === "player_1") {
            return `你出 ${you}，AI 出 ${bot}。AI 赢得奖金 ${prize} (+${lastRound.bot_reward})。`;
        }
        return `你出 ${you}，AI 出 ${bot}。平局 → 奖金 ${prize} 作废 (双方不加分)。`;
    }

    /* ------------------------------------------------------------------
     * VEIL UI helpers
     * ------------------------------------------------------------------ */

    /**
     * Render the in-game VEIL status bar between status-bar and round banner.
     * Shows:
     *   (a) 激活标签（§9 HiddenPrize / §6 Suit Tiebreak / §11 Info Reward）
     *   (b) 你本轮的花色 (Suit Tiebreak 时，每人整局被分配一个固定花色)
     *   (c) 你持有的私人情报 HIGH/LOW（关于"下一轮奖金属于高/低半区"）
     *   (d) 对手是否持有私人情报（只说 Yes/No，不泄露 HIGH/LOW，符合不对称信息语义）
     * Input `veil` == null means "no veil at all" → 收起整个 section.
     */
    function renderVeilStatus(veil) {
        if (!veil || !veil.any_enabled) {
            if (ui.veilStatus) ui.veilStatus.classList.add("hidden");
            return;
        }
        ui.veilStatus.classList.remove("hidden");

        // active_tags bar
        if (ui.veilActiveTags) {
            ui.veilActiveTags.textContent =
                (veil.active_tags || []).map(t => `· ${t}`).join("  ");
        }

        // body chips
        const body = ui.veilStatusBody;
        if (!body) return;
        body.innerHTML = "";

        const addChip = (label, value, cls) => {
            const chip = el("div", "veil-chip" + (cls ? " " + cls : ""));
            chip.appendChild(el("div", "veil-chip-label", label));
            const valEl = el("div", "veil-chip-value");
            if (typeof value === "string") valEl.textContent = value;
            else valEl.innerHTML = value;  // suit symbol HTML safe
            chip.appendChild(valEl);
            body.appendChild(chip);
        };

        // (b) 你的花色 / AI 花色
        if (veil.suit_tiebreak_enabled) {
            addChip("你的花色",
                    veil.player_suit_symbol_human || "—",
                    "chip-suit chip-suit-human");
            addChip("AI 花色",
                    veil.player_suit_symbol_bot   || "—",
                    "chip-suit chip-suit-bot");
        }

        // (c) 你持有的私人情报 (HIGH/LOW 下一轮)
        if (veil.info_reward_enabled) {
            let myInfo = veil.human_private_info_next_prize_half;
            const myCls = myInfo === "HIGH" ? "info-high"
                         : myInfo === "LOW"  ? "info-low"
                         : "info-none";
            const myTxt = (myInfo === "HIGH" || myInfo === "LOW")
                ? `下一轮奖金属 ${myInfo} 半区`
                : "暂未持有";
            addChip("我的私人情报", myTxt, "chip-info " + myCls);

            // (d) 对手情报持有状态（只说有无，保持不对称语义）
            let botHas = veil.bot_holds_private_info;
            let botTxt;
            if (botHas === true)       botTxt = "✓ 已持有（内容未知）";
            else if (botHas === false) botTxt = "✗ 未持有";
            else                       botTxt = "—";
            addChip("AI 情报状态", botTxt,
                    "chip-info " + (botHas === true ? "info-bot-has" : "info-bot-no"));

            // (e) 最近一次情报的归属（谁出的最小牌→获得了 HIGH/LOW）
            if (veil.last_info_awarded_to) {
                const who = veil.last_info_awarded_to === "player_0" ? "你"
                           : veil.last_info_awarded_to === "player_1" ? "AI"
                           : String(veil.last_info_awarded_to);
                const half = (veil.last_info_half === "HIGH"
                              || veil.last_info_half === "LOW")
                    ? " → " + veil.last_info_half
                    : "";
                addChip("上轮情报归属", `${who} 获得${half}`,
                        "chip-info-award");
            } else if (veil.round === 1) {
                addChip("上轮情报归属", "第 1 轮尚无",
                        "chip-info-award info-none");
            }
        }

        // (f) Hidden prize 当前是否隐藏
        if (veil.hidden_prize_enabled) {
            const hid = veil.prize_is_currently_hidden;
            addChip("隐藏奖励状态",
                    hid ? "出牌前隐藏（出价后揭晓）" : "已揭晓",
                    hid ? "chip-hidden on" : "chip-hidden off");
        }
    }

    /**
     * Decorate remaining card nodes with a suit prefix if server provided
     * per-card suit_symbol.  Idempotent: we rely on innerHTML being the
     * raw display already so we simply replace it instead of double-
     * prefixing on re-render (render() already rebuilds the container).
     */
    function decorateRemainingCardsWithSuit(humanCards, botCards) {
        // Human card nodes were just appended; re-run iteration by index
        // to attach suit prefix inside each node's content.
        if (humanCards.length && ui.humanCards) {
            const nodes = ui.humanCards.children;
            humanCards.forEach((c, i) => {
                if (!c || !c.suit_symbol) return;
                const node = nodes[i];
                if (!node) return;
                node.textContent = String(c.suit_symbol)
                                 + (c.display != null ? c.display : "");
            });
        }
        if (botCards.length && ui.botCards) {
            const nodes = ui.botCards.children;
            botCards.forEach((c, i) => {
                if (!c || !c.suit_symbol) return;
                const node = nodes[i];
                if (!node) return;
                node.textContent = String(c.suit_symbol)
                                 + (c.display != null ? c.display : "");
            });
        }
    }

    /**
     * Render the AI policy as bar chart.
     * policy: {bot_type, value (can be NaN), note, distribution: [[cardValue, pct],...]}
     * The played card is *not* highlighted because it's already revealed
     * via the banner + used-cards; but we render the pre-step distribution.
     */
    function renderAiPolicy(policy, botCardsRemainingAfterStep) {
        const botType = policy.bot_type || "?";
        const typeZh = {
            random: "Random · 纯随机",
            heuristic: "Heuristic · 启发式",
            nash: "Nash · 精确纳什混合策略",
        }[botType] || botType;
        ui.aiBotType.textContent = typeZh;
        ui.aiNote.textContent = policy.note || "";

        // Value
        const v = policy.value;
        if (typeof v === "number" && Number.isFinite(v)) {
            ui.aiValueEl.textContent =
                `${v >= 0 ? "+" : ""}${v.toFixed(3)}`;
            ui.aiValueEl.style.color = (v > 0) ? "var(--bot)"
                                       : (v < 0) ? "var(--human)" : "var(--text)";
        } else {
            ui.aiValueEl.textContent = "—";
            ui.aiValueEl.style.color = "var(--muted)";
        }

        // Bars
        ui.aiBars.innerHTML = "";
        const rows = policy.distribution || [];
        rows.forEach((row) => {
            const cardValue = row[0];
            const pct = Math.max(0, Math.min(100, Number(row[1]) || 0));
            const wrap = el("div", "bar-row");

            const lab  = el("div", "bar-label", cardLabel(cardValue));
            const box  = el("div", "bar-box");
            const fill = el("div", "bar-fill");
            fill.style.width = pct.toFixed(1) + "%";
            if (pct > 5) {
                fill.textContent = pct.toFixed(1) + "%";
            }
            box.appendChild(fill);
            const pctLab = el("div", "bar-pct",
                              (pct > 5) ? "" : pct.toFixed(1) + "%");

            wrap.appendChild(lab);
            wrap.appendChild(box);
            wrap.appendChild(pctLab);
            ui.aiBars.appendChild(wrap);
        });

        ui.aiPanel.classList.remove("hidden");
    }

    /**
     * Render the HUMAN counterfactual chart: 3-color bars by outcome.
     * Width of a bar = normalized by prize value, but we always set the
     * "win" bars to width = 100% * prize/max_prize_so_far? Keep it simpler:
     *   win  → 100% green + " +prize"
     *   tie  → 40%  tan   + " tie +0"
     *   lose → 15%  red   + " lose +0"
     * The card you ACTUALLY played gets a dashed black outline.
     *
     * NEW (Opponent-waste dimension, 对手亏牌):
     *   We now ALSO report, for every what-if card, whether the opponent's
     *   actual play was a good deal when matched against your hypothetical
     *   card.  Formula: bot_net_gain = opponent_prize_delta − bot_card_value.
     *     profitable  (green chip) → bot net > 0 — bot used a SMALL card to
     *                                    grab a BIG prize; we lost + bot
     *                                    played efficiently.
     *     even        (gray chip)  → bot net == 0 — break-even.
     *     wasted     (red chip)   → bot net < 0 — typical when you win the
     *                                    prize (bot gets 0 but loses its
     *                                    card) OR when bot used a BIG card
     *                                    on a tiny prize (overkill).  We
     *                                    call this "对手亏牌 / wasted
     *                                    card": you didn't win points but
     *                                    forced the opponent to burn a
     *                                    valuable asset that would've been
     *                                    useful later for a larger prize.
     */
    function renderHumanCounterfactual(policy) {
        // Top note: remind the user which fixed bot-card all of this analysis
        // assumes (counterfactual freezes the bot's real action and varies YOU).
        const bars = policy.bars || [];
        let headerNote = policy.note || "";
        if (bars.length) {
            const first = bars[0];
            const botCardDisp  = (first.bot_card_display  != null) ? String(first.bot_card_display)  : "";
            const botCardValue = (first.bot_card_value != null) ? Number(first.bot_card_value) : null;
            if (botCardDisp) {
                const chipTag = botCardValue != null
                    ? `<span class="bot-card-chip-inline" title="对手出的牌面值 ${botCardValue}（反事实固定这张不变）">Bot出 ${botCardDisp}</span>`
                    : `Bot出 ${botCardDisp}`;
                headerNote = (headerNote ? headerNote + "  ·  " : "") + chipTag;
            }
        }
        ui.hCfNote.innerHTML = headerNote;
        ui.hCfBars.innerHTML = "";

        bars.forEach((bar) => {
            const outcome = bar.outcome;       // "win" | "tie" | "lose"
            const prizeAtStake = Number(policy.prize_at_stake || 0);
            let widthPct, fillCls, deltaTxt;
            if (outcome === "win") {
                widthPct = 100;
                fillCls = "bar-fill-h win-h";
                const delta = Number(bar.delta != null ? bar.delta : prizeAtStake);
                deltaTxt = `+${delta}`;
            } else if (outcome === "tie") {
                widthPct = 40;
                fillCls = "bar-fill-h tie-h";
                deltaTxt = "tie +0";
            } else {
                widthPct = 15;
                fillCls = "bar-fill-h lose-h";
                deltaTxt = "lose +0";
            }
            const wrap = el("div", "bar-row" + (bar.played ? " played-cf-row" : ""));
            const lab  = el("div", "bar-label human-label",
                            bar.card_display || String(bar.card_value));
            const box  = el("div", "bar-box");
            const fill = el("div", fillCls);
            fill.style.width = widthPct.toFixed(1) + "%";
            fill.textContent = deltaTxt;

            // ===== 对手亏牌 badge =====
            //   颜色：亏牌红 / 平本灰 / 赚牌绿
            //   内容：Bot亏牌/赚牌/平本 + net_gain（带正负号）
            let botBadgeEl = null;
            const botEff = String(bar.bot_efficiency || "").toLowerCase();
            const bNet   = Number(bar.bot_net_gain);
            const bLabel = String(bar.bot_efficiency_label || (
                botEff === "wasted" ? "Bot 亏牌" :
                botEff === "profitable" ? "Bot 赚牌" :
                botEff === "even" ? "Bot 平本" : ""));
            if (botEff && bLabel) {
                const netStr = Number.isFinite(bNet)
                    ? ((bNet > 0 ? " +" : (bNet < 0 ? " " : " ±")) + String(bNet))
                    : "";
                botBadgeEl = el("span",
                    "bot-eff-chip bot-eff-" + (botEff || "none"),
                    bLabel + netStr);
            }

            // Combined tooltip = outcome_desc 基础版 + 对手亏牌详细解释（如果有）
            const tipParts = [];
            if (typeof bar.outcome_desc === "string" && bar.outcome_desc) {
                tipParts.push(bar.outcome_desc);
            }
            if (typeof bar.bot_eff_desc === "string" && bar.bot_eff_desc) {
                tipParts.push(bar.bot_eff_desc);
            }
            // 也显示对称的「我净收益」
            const myNet = Number(bar.my_net_gain);
            if (Number.isFinite(myNet)) {
                tipParts.push("你这张子力性价比（你得分 − 你出牌面值）：" +
                              (myNet >= 0 ? "+" : "") + myNet);
            }
            const fullTip = tipParts.join("\n—\n");
            if (fullTip) {
                fill.title = fullTip;
                box.title  = fullTip;
                wrap.title = fullTip;
                if (botBadgeEl) botBadgeEl.title = fullTip;
            }
            box.appendChild(fill);

            // outcome 标签行
            const outcomeSuffix =
                (typeof bar.outcome_desc === "string" && bar.outcome_desc.length)
                    ? ` · ${bar.outcome_desc}`
                    : "";
            const outLab = el("div", "bar-pct");
            const mainTxt = bar.played ? `played${outcomeSuffix}` : `${outcome}${outcomeSuffix}`;
            outLab.appendChild(document.createTextNode(mainTxt));
            // 对手亏牌 badge 直接贴在 outcome 文字后面
            if (botBadgeEl) {
                // 中间补个分隔空格
                outLab.appendChild(document.createTextNode("   "));
                outLab.appendChild(botBadgeEl);
            }
            wrap.appendChild(lab);
            wrap.appendChild(box);
            wrap.appendChild(outLab);
            ui.hCfBars.appendChild(wrap);
        });

        ui.hCfPanel.classList.remove("hidden");
    }

    // ------------------------------------------------------------ actions
    async function onHumanCardClicked(value, _cardEl) {
        if (pending || !currentState || currentState.done) return;
        pending = true;
        render(currentState, null);  // disable cards once

        try {
            const resp = await fetch("/api/game/play", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ action: value }),
            });
            if (!resp.ok) {
                const err = await resp.json().catch(() => ({}));
                alert("Error: " + (err.detail || ("HTTP " + resp.status)));
                return;
            }
            const data = await resp.json();
            render(data.state, data.last_round, data.meta);
        } catch (e) {
            console.error(e);
            alert("请求失败 Request failed: " + (e.message || String(e)));
        } finally {
            pending = false;
            if (currentState && !currentState.done) render(currentState, null);
        }
    }

    async function submitNewGameForm(evt) {
        if (evt) evt.preventDefault();
        if (pending) return;

        const n = clampN(ui.inputNumCards.value);
        const botType = ui.inputBot.value || "heuristic";
        pending = true;
        hideBanner(true);
        hideBanner(false);
        ui.loadingHint.classList.remove("hidden");
        ui.btnStart.disabled = true;
        ui.aiPanel.classList.add("hidden");

        try {
            const veil = collectVeilFlags();
            const payload = {
                num_cards: n,
                bot_type: botType,
                // Expand each veil_* flag as top-level field, matching
                // NewGameRequest schema: veil_hidden_prize / veil_suit_tiebreak /
                // veil_info_reward.  Unknown ids are ignored server-side.
                ...veil,
            };
            const resp = await fetch("/api/game/new", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });
            if (!resp.ok) {
                const err = await resp.json().catch(() => ({}));
                alert("无法开始新游戏: "
                      + (err.detail || ("HTTP " + resp.status)));
                return;
            }
            const data = await resp.json();
            // Hide config panel *after* the user has selected and started.
            // Keep the panel open so they can reconfigure via "New Game".
            render(data.state, data.last_round, data.meta);
            updateNashHint();  // maybe warn if next start would be out-of-range
        } catch (e) {
            console.error(e);
            alert("无法开始新游戏: " + (e.message || String(e)));
        } finally {
            pending = false;
            ui.loadingHint.classList.add("hidden");
            ui.btnStart.disabled = false;
            if (currentState && !currentState.done) render(currentState, null);
        }
    }

    // Restart button just scrolls + focuses the setup form so user can
    // change params before starting — does NOT auto-submit.
    function onRestartClicked() {
        ui.form.scrollIntoView({ behavior: "smooth", block: "start" });
        setTimeout(() => ui.inputNumCards.focus(), 150);
    }

    // -------------------------------------------------------------- bootstrap
    async function bootstrap() {
        // 1. Pull config
        try {
            const resp = await fetch("/api/game/config");
            if (!resp.ok) throw new Error("HTTP " + resp.status);
            CONFIG = await resp.json();
        } catch (e) {
            alert("无法加载游戏配置: " + (e.message || String(e)));
            ui.subtitle.textContent = "加载失败 Load failed.";
            return;
        }

        populateSetupForm();

        // Wire handlers
        ui.form.addEventListener("submit", submitNewGameForm);
        ui.inputNumCards.addEventListener("input", updateNashHint);
        ui.inputBot.addEventListener("change", updateNashHint);
        ui.btnRestart.addEventListener("click", onRestartClicked);

        // Auto start a DEFAULT game so the UI is not empty on first paint.
        // Default: 13 cards + Heuristic (strong & fast).
        await submitNewGameForm(null);
    }

    window.addEventListener("DOMContentLoaded", bootstrap);
})();
