// Author: 陈子聪 (Chen Zicong)
// Date:   2026-08-30
// Purpose: Header-only Goofspiel game-state types and inline helpers.
//          Designed for SIMD / vectorised rollout loops: all per-state data
//          fits in 64 bits (for N <= 13, each bitmask is 13 bits => 39 bits
//          total), i.e. a std::array<uint64_t, M> of M parallel envs can be
//          processed with AVX-2 gather ops via Eigen::Map<Array<uint64_t,Dynamic,1>>.
//
//  State := (human_mask : uint16, bot_mask : uint16, prize_mask : uint16,
//            current_prize_index : uint8, round : uint8,
//            score_h : uint8, score_b : uint8)
//  Packed into uint64_t:
//    bit  0..12  = human_mask  (card v => bit v-1)
//    bit 16..28  = bot_mask
//    bit 32..44  = prize_mask
//    bit 48..54  = remaining prize deck is prize_mask; "current prize" ==
//                  leading (lowest) set bit of prize_mask (== current_prize_value)
//    bit 55..62  = round (uint8) | scores packed into a single uint16
//                  score_h in low-byte, score_b high-byte
//    bit 63      = done flag (1 = terminal)
//
//  The "prize deck" order is implicitly: "lowest set bit first".  To generate
//  a shuffled prize deck, the caller writes an arbitrary 13-bit subset as
//  prize_mask at reset time and rotates it each step.  For deterministic
//  seeds we use xorshift64* and perform a Fisher–Yates shuffle over the
//  index list, then OR the present-bits in that order into a uint16_t array
//  used as a "mask stack".
#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(__SSE2__) || defined(_M_X64) || defined(_M_IX86_FP)
#  include <emmintrin.h>
#endif

namespace cxxgoof {

// ---------- Limits --------------------------------------------------------
static constexpr int   kMaxCards    = 13;
static constexpr int   kMaxPrizeSum = 13 * 14 / 2;  // 91, fits in uint8

// ---------- Bit helpers ---------------------------------------------------
// Number of set bits (for len(remaining_cards)).  Uses builtin if available.
inline int popcnt(uint16_t x) noexcept {
#if defined(_MSC_VER)
    return static_cast<int>(__popcnt16(x));
#elif defined(__GNUC__) || defined(__clang__)
    return static_cast<int>(__builtin_popcount(x));
#else
    int c = 0;
    while (x) { c++; x &= x - 1; }
    return c;
#endif
}

// Lowest set bit => card value (0 => returns 0).
inline int lsb_value(uint16_t x) noexcept {
    if (x == 0) return 0;
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    unsigned long idx;
    _BitScanForward(&idx, x);
    return static_cast<int>(idx) + 1;
#elif defined(__GNUC__) || defined(__clang__)
    return static_cast<int>(__builtin_ctz(x)) + 1;
#else
    int i = 0;
    while ((x & (1u << i)) == 0) i++;
    return i + 1;
#endif
}

// card value v -> 1 << (v-1).
constexpr uint16_t card_mask(int v) noexcept {
    return static_cast<uint16_t>(1u << (v - 1));
}

// ---------- Compact Packed State ------------------------------------------
// For the vectorised path we prefer a struct-of-arrays layout; this packed
// uint64_t is used for *cache keys* and for the exact-Nash recursion state.
struct alignas(8) PackedState {
    uint16_t human_mask;
    uint16_t bot_mask;
    uint16_t prize_mask;
    uint8_t  round;          // round index 1..N (0 = just reset, no step yet)
    uint8_t  score_h;
    uint8_t  score_b;
    uint8_t  done;           // 0/1
    uint8_t  carry_pool;     // tie-rollover pool (prize_at_stake accumulated so far;
                             //   max = N(N+1)/2 = 91 so uint8 is enough)
    // --- Derived quick accessors: ---
    constexpr int current_prize() const noexcept {
        return lsb_value(prize_mask);  // 0 when prize_mask == 0 => done
    }
    constexpr int remaining_human_count() const noexcept { return popcnt(human_mask); }
    constexpr int remaining_bot_count()   const noexcept { return popcnt(bot_mask); }
    constexpr int remaining_prize_count() const noexcept { return popcnt(prize_mask); }
};
static_assert(sizeof(PackedState) <= 16,
              "PackedState must stay tiny so 1M envs fit in ~16 MB.");

// ---------- Step (atomic, no allocations). --------------------------------
// Given (pre-state, human_card_val 1..N, bot_card_val 1..N, N):
//   returns {next_state, reward_h, reward_b, terminal_winner_id}.
// terminal_winner_id in { 0 => human win, 1 => bot win, 2 => draw, 3 => game NOT over }.
struct StepOutcome {
    PackedState next;
    int         reward_h;
    int         reward_b;
    int         winner_id;  // this-round winner (NOT game winner)
                        //   0 = human, 1 = bot, 2 = tie
    int         game_winner_id;  // 0/1/2/3 (3 = not done)
};

inline StepOutcome step(const PackedState& pre, int h, int b, int N) noexcept {
    StepOutcome o{};
    // Clear the two played card bits.  (If the caller passes an already-used
    // card we still &=~mask to be permissive; the Python-layer wrapper is in
    // charge of raising on illegal inputs.)
    const uint16_t hm = pre.human_mask & static_cast<uint16_t>(~card_mask(h));
    const uint16_t bm = pre.bot_mask   & static_cast<uint16_t>(~card_mask(b));
    const int prize     = pre.current_prize();  // round_prize for this round
    const int carry_in  = pre.carry_pool;       // rolled-over stakes from previous ties
    const int stake     = prize + carry_in;     // prize_at_stake the winner takes ALL
    // is_last_round: the current mask has exactly 1 bit set (this round's
    // prize is the last one) — mirrors Python's `len(remaining_prizes) == 0`.
    const bool is_last_round = (pre.remaining_prize_count() == 1);

    int rew_h = 0, rew_b = 0, win = 2;
    int carry_out = 0;
    if (h > b) {
        rew_h = stake;  win = 0;  carry_out = 0;
    } else if (h < b) {
        rew_b = stake;  win = 1;  carry_out = 0;
    } else {
        // tie: 0 score this round.  If not the last round the *entire*
        // prize_at_stake (round_prize + carry_in) rolls over (the rule
        // variant GoofspielEnv implements).  If it IS the last round the
        // total package is PERMANENTLY discarded (only lose-money case).
        win = 2;
        carry_out = is_last_round ? 0 : stake;
    }

    // Consume the *current* prize from the mask (its lsb was the prize).
    const uint16_t pm_new =
        static_cast<uint16_t>(pre.prize_mask & static_cast<uint16_t>(~card_mask(prize)));

    o.next.human_mask = hm;
    o.next.bot_mask   = bm;
    o.next.prize_mask = pm_new;
    o.next.round      = static_cast<uint8_t>(pre.round + 1);
    o.next.score_h    = static_cast<uint8_t>(pre.score_h + rew_h);
    o.next.score_b    = static_cast<uint8_t>(pre.score_b + rew_b);
    o.next.carry_pool = static_cast<uint8_t>(carry_out);
    const bool done = (pm_new == 0) || (hm == 0) || (bm == 0);
    o.next.done = static_cast<uint8_t>(done ? 1 : 0);

    o.reward_h   = rew_h;
    o.reward_b   = rew_b;
    o.winner_id  = win;

    if (done) {
        if (o.next.score_h > o.next.score_b)       o.game_winner_id = 0;
        else if (o.next.score_b > o.next.score_h)  o.game_winner_id = 1;
        else                                       o.game_winner_id = 2;
    } else {
        o.game_winner_id = 3;
    }
    // silence unused-arg warning for N (reserved for future N-bound asserts)
    (void)N;
    return o;
}

// ---------- Canonical-key + sign helper (for Nash solver). -----------------
// Canonical form: if swap(human_mask, bot_mask) is lex-small then swap and
// set sign = -1 else sign = +1.  We store the canonical key in the
// std::unordered_map cache and multiply returned F-value by sign on read.
//
// The 64-bit key is laid out as follows (each field leaves room for future
// growth; current masks are 13-bit so low/high 13-bit only are used):
//   bits  0..15 -> human_mask (or bot_mask if A > B, canonical small first)
//   bits 16..31 -> bot_mask   (or human_mask if swapped)
//   bits 32..47 -> prize_mask R
//   bits 48..55 -> carry_pool (8-bit, max 91 — needed once the solver models
//                              tie-rollover; today Nash recursion is still the
//                              classical no-carry variant, so callers pass 0)
//   bits 56..63 -> reserved (0)
struct CanonicalKey {
    uint64_t key;        // layout above
    int      sign;       // ±1
};
inline CanonicalKey canonicalize(uint16_t A, uint16_t B, uint16_t R,
                                 uint8_t carry = 0) noexcept {
    CanonicalKey k{};
    const uint64_t ucarry = static_cast<uint64_t>(carry);
    if (A > B) {
        k.key = (static_cast<uint64_t>(B))
              | (static_cast<uint64_t>(A) << 16)
              | (static_cast<uint64_t>(R) << 32)
              | (ucarry << 48);
        k.sign = -1;
    } else {
        k.key = (static_cast<uint64_t>(A))
              | (static_cast<uint64_t>(B) << 16)
              | (static_cast<uint64_t>(R) << 32)
              | (ucarry << 48);
        k.sign = 1;
    }
    return k;
}

// ---------- Xorshift64 RNG (tiny, deterministic, header-only). ------------
class Xor64Rng {
public:
    explicit constexpr Xor64Rng(uint64_t seed) noexcept : s_(seed ? seed : 0x9E3779B97F4A7C15ULL) {}
    constexpr uint64_t operator()() noexcept {
        // xorshift64*
        uint64_t x = s_;
        x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
        s_ = x;
        return x * 0x2545F4914F6CDD1DULL;
    }
    // Fisher-Yates over indices [0..n)
    template <class Int, class = std::enable_if_t<std::is_integral_v<Int>>>
    void shuffle(Int* first, Int* last) noexcept {
        const ptrdiff_t n = last - first;
        for (ptrdiff_t i = n - 1; i > 0; --i) {
            const ptrdiff_t j = static_cast<ptrdiff_t>(this->operator()() % (uint64_t)(i + 1));
            std::swap(first[i], first[j]);
        }
    }
private:
    uint64_t s_;
};

// Convert a *shuffled* array of prize_values[0..N-1] to the stack-of-masks
// representation used by the VectorisedEnv below.  mask_stack[i] is the
// prize_mask that should be active at the start of round (i+1).
inline std::array<uint16_t, kMaxCards + 1> build_prize_mask_stack(
    const int* prizes, int N) noexcept
{
    std::array<uint16_t, kMaxCards + 1> stack{};
    uint16_t running = 0;
    for (int i = N - 1; i >= 0; --i) {
        running = static_cast<uint16_t>(running | card_mask(prizes[i]));
        stack[i] = running;
    }
    stack[N] = 0;
    return stack;
}

// ==========================================================================
// Single-environment wrapper (used by exact Nash, tests, Python 1-step).
// ==========================================================================
class SingleEnv {
public:
    explicit SingleEnv(int N) : N_(N) {
        if (N_ <= 0 || N_ > kMaxCards)
            throw std::invalid_argument("N must be in [1, 13]");
    }
    int num_cards() const noexcept { return N_; }
    const PackedState& state() const noexcept { return s_; }

    // Resets with deterministic seed (prize deck = xorshift shuffle of 1..N).
    void reset(uint64_t seed) noexcept {
        Xor64Rng rng(seed);
        int prizes[kMaxCards];
        for (int i = 0; i < N_; ++i) prizes[i] = i + 1;
        rng.shuffle(prizes, prizes + N_);
        const auto stack = build_prize_mask_stack(prizes, N_);
        s_ = PackedState{};
        s_.human_mask = (N_ == 13) ? uint16_t(0x1FFF)
                                   : static_cast<uint16_t>((1u << N_) - 1);
        s_.bot_mask   = s_.human_mask;
        s_.prize_mask = stack[0];
        s_.round      = 1;
        s_.score_h = s_.score_b = 0;
        s_.carry_pool = 0;
        s_.done = 0;
        (void)stack;
    }

    bool is_legal_human(int v) const noexcept {
        return v >= 1 && v <= N_ && ((s_.human_mask & card_mask(v)) != 0);
    }
    bool is_legal_bot(int v) const noexcept {
        return v >= 1 && v <= N_ && ((s_.bot_mask   & card_mask(v)) != 0);
    }

    StepOutcome step(int h, int b) noexcept {
        auto o = cxxgoof::step(s_, h, b, N_);
        s_ = o.next;
        return o;
    }

private:
    int N_;
    PackedState s_;
};

// ==========================================================================
// VectorizedEnv — struct-of-arrays of M envs.  Outputs are raw spans; the
// pybind11 layer casts them directly to numpy views (zero-copy for numpy
// arrays provided by the caller).
// ==========================================================================
class VectorizedEnv {
public:
    VectorizedEnv(int num_cards, int num_envs)
        : N_(num_cards), M_(num_envs)
    {
        if (N_ <= 0 || N_ > kMaxCards)
            throw std::invalid_argument("N must be in [1, 13]");
        if (M_ <= 0)
            throw std::invalid_argument("num_envs must be > 0");
        // Allocate and zero-initialise the SoA.
        hm_.assign(M_, 0); bm_.assign(M_, 0); pm_.assign(M_, 0);
        sh_.assign(M_, 0); sb_.assign(M_, 0); rd_.assign(M_, 0);
        carry_.assign(M_, 0);
        done_.assign(M_, 0);
    }

    int num_cards() const noexcept { return N_; }
    int num_envs()  const noexcept { return M_; }

    // ---- accessors --------------------------------------------------------
    uint16_t* human_masks()   noexcept { return hm_.data(); }
    uint16_t* bot_masks()     noexcept { return bm_.data(); }
    uint16_t* prize_masks()   noexcept { return pm_.data(); }
    uint8_t*  score_human()   noexcept { return sh_.data(); }
    uint8_t*  score_bot()     noexcept { return sb_.data(); }
    uint8_t*  rounds()        noexcept { return rd_.data(); }
    uint8_t*  carry_pool()    noexcept { return carry_.data(); }
    uint8_t*  dones()         noexcept { return done_.data(); }

    // ---- reset ------------------------------------------------------------
    void reset_batch(uint64_t base_seed) noexcept {
        std::vector<uint64_t> v(M_);
        for (int i = 0; i < M_; ++i) v[i] = base_seed + static_cast<uint64_t>(i);
        reset_batch(v.data());
    }

    void reset_batch(const uint64_t seeds[]) noexcept {
        const uint16_t full_mask =
            (N_ == 13) ? uint16_t(0x1FFF)
                       : static_cast<uint16_t>((1u << N_) - 1);
        int prizes[kMaxCards];
        for (int i = 0; i < M_; ++i) {
            Xor64Rng rng(seeds[i] ? seeds[i]
                                  : static_cast<uint64_t>(i) + 1ULL);
            for (int k = 0; k < N_; ++k) prizes[k] = k + 1;
            rng.shuffle(prizes, prizes + N_);
            uint16_t mask = 0;
            for (int k = N_ - 1; k >= 0; --k) mask |= card_mask(prizes[k]);
            hm_[i]   = full_mask;
            bm_[i]   = full_mask;
            pm_[i]   = mask;
            sh_[i]   = 0; sb_[i] = 0;
            rd_[i]   = 1;
            carry_[i] = 0;
            done_[i] = 0;
        }
    }

    // ---- step -------------------------------------------------------------
    void step_batch(const int actions_h[], const int actions_b[],
                    int rew_h[], int rew_b[],
                    int w_id[], int gw_id[]) noexcept {
        const int N = N_;
        const int M = M_;
        uint16_t* __restrict hm = hm_.data();
        uint16_t* __restrict bm = bm_.data();
        uint16_t* __restrict pm = pm_.data();
        uint8_t*  __restrict sh = sh_.data();
        uint8_t*  __restrict sb = sb_.data();
        uint8_t*  __restrict rd = rd_.data();
        uint8_t*  __restrict ca = carry_.data();
        uint8_t*  __restrict dn = done_.data();
        for (int i = 0; i < M; ++i) {
            if (dn[i]) {
                rew_h[i] = 0; rew_b[i] = 0;
                w_id[i]  = 2;
                gw_id[i] = sh[i] > sb[i] ? 0 : (sb[i] > sh[i] ? 1 : 2);
                continue;
            }
            const int h = actions_h[i];
            const int b = actions_b[i];
            const uint16_t new_hm = static_cast<uint16_t>(hm[i] & ~card_mask(h));
            const uint16_t new_bm = static_cast<uint16_t>(bm[i] & ~card_mask(b));
            const int prize     = lsb_value(pm[i]);
            const int carry_in  = ca[i];
            const int stake     = prize + carry_in;
            const bool is_last_round = (popcnt(pm[i]) == 1);

            int rwh = 0, rwb = 0, win = 2;
            int carry_out = 0;
            if (h > b)       { rwh = stake; win = 0; carry_out = 0; }
            else if (h < b)  { rwb = stake; win = 1; carry_out = 0; }
            else             { win = 2; carry_out = is_last_round ? 0 : stake; }

            const uint16_t new_pm = static_cast<uint16_t>(pm[i] & ~card_mask(prize));
            const auto new_sh = uint8_t(sh[i] + rwh);
            const auto new_sb = uint8_t(sb[i] + rwb);
            const bool done = (new_pm == 0) || (new_hm == 0) || (new_bm == 0);
            hm[i] = new_hm; bm[i] = new_bm; pm[i] = new_pm;
            sh[i] = new_sh; sb[i] = new_sb;
            rd[i] = uint8_t(rd[i] + 1);
            ca[i] = static_cast<uint8_t>(carry_out);
            dn[i] = done ? uint8_t(1) : uint8_t(0);
            rew_h[i] = rwh; rew_b[i] = rwb;
            w_id[i]  = win;
            gw_id[i] = done ? (new_sh > new_sb ? 0 : (new_sb > new_sh ? 1 : 2)) : 3;
            (void)N;
        }
    }

    // Reset a single env (auto-respawn on terminal for infinite-horizon).
    void reset_single(int i, uint64_t seed) noexcept {
        Xor64Rng rng(seed ? seed : static_cast<uint64_t>(i) + 0xDEADBEEFull);
        int prizes[kMaxCards];
        for (int k = 0; k < N_; ++k) prizes[k] = k + 1;
        rng.shuffle(prizes, prizes + N_);
        uint16_t mask = 0;
        for (int k = N_ - 1; k >= 0; --k) mask |= card_mask(prizes[k]);
        const uint16_t full_mask =
            (N_ == 13) ? uint16_t(0x1FFF)
                       : static_cast<uint16_t>((1u << N_) - 1);
        hm_[i] = full_mask; bm_[i] = full_mask; pm_[i] = mask;
        sh_[i] = 0; sb_[i] = 0; rd_[i] = 1; carry_[i] = 0; done_[i] = 0;
    }

private:
    int N_;
    int M_;
    std::vector<uint16_t> hm_, bm_, pm_;
    std::vector<uint8_t>  sh_, sb_, rd_, carry_, done_;
};

}  // namespace cxxgoof
