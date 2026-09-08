"""
The part that earns this project its keep. Every trained policy is
evaluated against conditions it was not trained on: higher demand, a
different lead time, and a regime shift timed differently than anything
seen in training. A policy that wins in sample and loses its edge or
performs worse than the heuristic out of sample is the honest result to
report, not something to avoid by only showing the in sample number.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import torch

from environment import EnvConfig, InventoryEnv
from heuristic_policy import evaluate_policy
from policy_gradient import PolicyNet

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
N_EVAL_EPISODES = 200
SEED = 7


def mean_ci(rewards: np.ndarray, z: float = 1.96):
    mean = rewards.mean()
    se = rewards.std(ddof=1) / np.sqrt(len(rewards))
    return float(mean), float(mean - z * se), float(mean + z * se)


def load_policies():
    with open(RESULTS_DIR / "trained_policies.pkl", "rb") as f:
        trained = pickle.load(f)
    dummy_env = InventoryEnv(EnvConfig(seed=SEED))
    net = PolicyNet(dummy_env.observation_dim, dummy_env.n_actions)
    net.load_state_dict(torch.load(RESULTS_DIR / "reinforce_net.pt"))
    net.eval()

    def reinforce_act(obs):
        normalized = obs.copy()
        normalized[0] = normalized[0] / 30.0
        with torch.no_grad():
            x = torch.as_tensor(normalized, dtype=torch.float32).unsqueeze(0)
            probs = torch.softmax(net(x), dim=-1)
            return int(torch.argmax(probs, dim=-1).item())

    return {
        "heuristic": lambda obs: trained["heuristic"].act(obs),
        "contextual_bandit": lambda obs: trained["bandit"].act_greedy(obs),
        "tabular_q_learning": lambda obs: trained["qlearn"].act_greedy(obs),
        "reinforce_policy_gradient": reinforce_act,
    }


def make_conditions() -> dict:
    base = dict(regime_shift_day=None, seed=SEED)
    return {
        "baseline_in_sample": EnvConfig(**base),
        "demand_plus_10pct": EnvConfig(**base, demand_multiplier=1.10),
        "demand_plus_25pct": EnvConfig(**base, demand_multiplier=1.25),
        "lead_time_longer_plus2": EnvConfig(**base, lead_time_override=5),
        "lead_time_shorter_minus2": EnvConfig(**base, lead_time_override=1),
        "regime_shift_day_60": EnvConfig(regime_shift_day=60, regime_shift_multiplier=1.6, seed=SEED),
        "regime_shift_day_200": EnvConfig(regime_shift_day=200, regime_shift_multiplier=1.6, seed=SEED),
        "regime_shift_down_day_120": EnvConfig(regime_shift_day=120, regime_shift_multiplier=0.5, seed=SEED),
    }


def main() -> None:
    policies = load_policies()
    conditions = make_conditions()
    eval_seed_start = SEED * 1000 + 999_000

    report = {}
    for cond_name, cfg in conditions.items():
        env = InventoryEnv(cfg)
        cond_result = {}
        for pol_name, pol_fn in policies.items():
            rewards = evaluate_policy(pol_fn, env, N_EVAL_EPISODES, seed_start=eval_seed_start)
            mean, lo, hi = mean_ci(rewards)
            cond_result[pol_name] = {
                "mean_reward": round(mean, 3),
                "ci_95_low": round(lo, 3),
                "ci_95_high": round(hi, 3),
            }
        heuristic_mean = cond_result["heuristic"]["mean_reward"]
        for pol_name in cond_result:
            cond_result[pol_name]["beats_heuristic"] = cond_result[pol_name]["mean_reward"] > heuristic_mean
        report[cond_name] = cond_result
        print(f"\n{cond_name}")
        for pol_name, r in cond_result.items():
            flag = "beats heuristic" if r["beats_heuristic"] else "does not beat heuristic"
            print(f"  {pol_name}: {r['mean_reward']:.2f} ({flag})")

    with open(RESULTS_DIR / "robustness_results.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nrobustness results written to {RESULTS_DIR / 'robustness_results.json'}")


if __name__ == "__main__":
    main()
