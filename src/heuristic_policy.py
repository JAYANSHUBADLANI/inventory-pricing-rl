"""
An order up to level (s, S) heuristic, the kind an operations team runs
today. Tuned by grid search against the training demand distribution before
any learned policy is compared against it, the same discipline as tuning a
real baseline properly rather than leaving it deliberately weak.
"""

from __future__ import annotations

import itertools

import numpy as np

from environment import EnvConfig, InventoryEnv


class OrderUpToPolicy:
    def __init__(self, reorder_point: float, order_up_to: float, action_grid: np.ndarray):
        self.s = reorder_point
        self.S = order_up_to
        self.action_grid = action_grid

    def act(self, obs: np.ndarray) -> int:
        inventory = obs[0]
        days_to_arrival_norm = obs[1]
        # only reorder if nothing is already inbound, a standard (s, S) rule
        has_pipeline = days_to_arrival_norm < 1.0
        if inventory <= self.s and not has_pipeline:
            target_order = max(self.S - inventory, 0.0)
        else:
            target_order = 0.0
        idx = int(np.argmin(np.abs(self.action_grid - target_order)))
        return idx


def evaluate_policy(policy_fn, env: InventoryEnv, n_episodes: int, seed_start: int = 0):
    rewards = []
    for ep in range(n_episodes):
        obs = env.reset(seed=seed_start + ep)
        total = 0.0
        done = False
        while not done:
            action = policy_fn(obs)
            obs, reward, done, _ = env.step(action)
            total += reward
        rewards.append(total)
    return np.array(rewards)


def tune_order_up_to(base_config: EnvConfig, n_tuning_episodes: int = 40) -> OrderUpToPolicy:
    env = InventoryEnv(base_config)
    grid = env.action_grid
    s_candidates = np.arange(0, base_config.max_order // 2 + 1, 2)
    S_candidates = np.arange(4, base_config.max_order + 1, 4)

    best = None
    best_mean = -np.inf
    for s, S in itertools.product(s_candidates, S_candidates):
        if S <= s:
            continue
        policy = OrderUpToPolicy(s, S, grid)
        rewards = evaluate_policy(policy.act, env, n_tuning_episodes, seed_start=10_000)
        mean_r = rewards.mean()
        if mean_r > best_mean:
            best_mean = mean_r
            best = (s, S)

    s, S = best
    print(f"tuned heuristic: s={s}, S={S}, mean tuning reward={best_mean:.2f}")
    return OrderUpToPolicy(s, S, grid)
