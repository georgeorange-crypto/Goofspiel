# Goofspiel Benchmark Summary

- Benchmark: goofspiel.benchmark.v1
- Profile: QUICK
- Promotion decision: REJECT_CANDIDATE

## Hard Gates
- G0_integrity: FAIL
- G1_exact_regression: PASS
- G2_exploitability: PASS
- G3_historical: PASS
- G4_regression_suite: PASS
- G5_opponent_calibration: PASS
- G6_adaptive_safety: PASS
- G7_numerical_performance: PASS

## Baselines
- Random: PRIMARY / A/B/D / goofspiel.bots.RandomBot
- Exact Nash: PRIMARY / A / goofspiel.solver.GoofspielCarrySolver
- Minimax-Q: PRIMARY / A / goofspiel.training.baseline_algorithms.create_baseline
- CFR: PRIMARY / A/C / goofspiel.reasoning.run_gt_cfr
- CFR+: PRIMARY / A/C / goofspiel.learning.game_theory.regret_matching_plus
- NeuRD: PRIMARY / A/B / goofspiel.learning.game_theory.neurd
- R-NaD: PRIMARY / B / goofspiel.training.baseline_algorithms.create_baseline
- Heuristic Suite: REFERENCE / A/B/D / goofspiel.bots.HeuristicBot
- PPO: REFERENCE / A/B / scripts.train_n5_ppo
- IPPO: REFERENCE / A/B / goofspiel.training.baseline_algorithms.create_baseline
- NFSP: REFERENCE / A / goofspiel.training.baseline_algorithms.create_baseline
- Deep CFR: REFERENCE / A/C / goofspiel.training.baseline_algorithms.create_baseline
- SM-MCTS: PRIMARY / C / goofspiel.reasoning.run_sm_mcts
- Adaptive BR: PRIMARY / D / goofspiel.reasoning.final_decision
