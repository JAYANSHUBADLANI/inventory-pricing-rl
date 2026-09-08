"""
Main experiment driver: tunes the heuristic, trains the bandit, tabular
Q-learning, and REINFORCE agents on the same environment, evaluates all four
on a held out set of episodes with a confidence interval, and saves trained
policies plus results to disk for the robustness script to reuse.

Run from the project root with the venv active:
    python src/run_experiment.py
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch

from bandit_policy import LinearContextualBandit
from environment import EnvConfig, InventoryEnv
from heuristic_policy import evaluate_policy, tune_order_up_to
from policy_gradient import ReinforceAgent
from qlearning_policy import TabularQLearning

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
N_TRAIN_EPISODES = 5000
N_EVAL_EPISODES = 300
SEED = 7


def mean_ci(rewards: np.ndarray, z: float = 1.96) -> tuple[float, float, float]:
    mean = rewards.mean()
    se = rewards.std(ddof=1) / np.sqrt(len(rewards))
    return mean, mean - z * se, mean + z * se


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    t0 = time.time()

    train_config = EnvConfig(regime_shift_day=None, seed=SEED)
    train_env = InventoryEnv(train_config)

    print("tuning heuristic on training distribution")
    heuristic = tune_order_up_to(train_config, n_tuning_episodes=40)

    print(f"training contextual bandit for {N_TRAIN_EPISODES} episodes")
    bandit = LinearContextualBandit(
        n_actions=train_env.n_actions, n_features=train_env.observation_dim, seed=SEED
    )
    bandit_train_rewards = bandit.train(train_env, N_TRAIN_EPISODES, seed_start=SEED * 1000)

    print(f"training tabular Q-learning for {N_TRAIN_EPISODES} episodes")
    qlearn = TabularQLearning(
        n_actions=train_env.n_actions, epsilon_decay_episodes=int(N_TRAIN_EPISODES * 0.6), seed=SEED
    )
    qlearn_train_rewards = qlearn.train(train_env, N_TRAIN_EPISODES, seed_start=SEED * 1000)

    print(f"training REINFORCE for {N_TRAIN_EPISODES} episodes")
    reinforce = ReinforceAgent(obs_dim=train_env.observation_dim, n_actions=train_env.n_actions, seed=SEED)
    reinforce_train_rewards = reinforce.train(train_env, N_TRAIN_EPISODES, seed_start=SEED * 1000)

    # held out evaluation, same in-sample environment, fresh seeds
    eval_config = EnvConfig(regime_shift_day=None, seed=SEED)
    eval_env = InventoryEnv(eval_config)
    eval_seed_start = SEED * 1000 + N_TRAIN_EPISODES + 5000

    policies = {
        "heuristic": lambda obs: heuristic.act(obs),
        "contextual_bandit": lambda obs: bandit.act_greedy(obs),
        "tabular_q_learning": lambda obs: qlearn.act_greedy(obs),
        "reinforce_policy_gradient": lambda obs: reinforce.act_greedy(obs),
    }

    in_sample_results = {}
    for name, fn in policies.items():
        rewards = evaluate_policy(fn, eval_env, N_EVAL_EPISODES, seed_start=eval_seed_start)
        mean, lo, hi = mean_ci(rewards)
        in_sample_results[name] = {
            "mean_reward": round(float(mean), 3),
            "ci_95_low": round(float(lo), 3),
            "ci_95_high": round(float(hi), 3),
            "n_eval_episodes": N_EVAL_EPISODES,
        }
        print(f"{name}: mean={mean:.2f}  95% CI=({lo:.2f}, {hi:.2f})")

    determinism_check = {}
    for name, fn in policies.items():
        r1 = evaluate_policy(fn, eval_env, 20, seed_start=999)
        r2 = evaluate_policy(fn, eval_env, 20, seed_start=999)
        determinism_check[name] = bool(np.allclose(r1, r2))

    training_cost = {
        "heuristic": {"environment_interactions": 40 * len(list(range(0, train_config.max_order // 2 + 1, 2))) * len(list(range(4, train_config.max_order + 1, 4))) * train_config.horizon_days, "note": "grid search tuning episodes times horizon, zero at deployment"},
        "contextual_bandit": {"environment_interactions": N_TRAIN_EPISODES * train_config.horizon_days},
        "tabular_q_learning": {"environment_interactions": N_TRAIN_EPISODES * train_config.horizon_days},
        "reinforce_policy_gradient": {"environment_interactions": N_TRAIN_EPISODES * train_config.horizon_days},
    }

    results = {
        "in_sample": in_sample_results,
        "determinism_same_seed_same_result": determinism_check,
        "training_cost": training_cost,
        "config": {
            "horizon_days": train_config.horizon_days,
            "lead_time": train_config.lead_time,
            "holding_cost_per_unit": train_config.holding_cost_per_unit,
            "lost_sale_penalty_per_unit": train_config.lost_sale_penalty_per_unit,
            "order_fixed_cost": train_config.order_fixed_cost,
            "max_order": train_config.max_order,
            "seed": SEED,
        },
        "runtime_seconds": round(time.time() - t0, 1),
    }
    with open(RESULTS_DIR / "in_sample_results.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(RESULTS_DIR / "trained_policies.pkl", "wb") as f:
        pickle.dump({"heuristic": heuristic, "bandit": bandit, "qlearn": qlearn}, f)
    torch.save(reinforce.net.state_dict(), RESULTS_DIR / "reinforce_net.pt")

    train_curves = {
        "contextual_bandit": bandit_train_rewards.tolist(),
        "tabular_q_learning": qlearn_train_rewards.tolist(),
        "reinforce_policy_gradient": reinforce_train_rewards.tolist(),
    }
    with open(RESULTS_DIR / "training_curves.json", "w") as f:
        json.dump(train_curves, f)

    print(f"done in {results['runtime_seconds']}s, results written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
