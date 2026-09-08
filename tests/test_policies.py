import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bandit_policy import LinearContextualBandit
from environment import EnvConfig, InventoryEnv
from heuristic_policy import OrderUpToPolicy
from qlearning_policy import TabularQLearning


def test_order_up_to_orders_when_below_reorder_point():
    grid = np.arange(0, 44, 4)
    policy = OrderUpToPolicy(reorder_point=4, order_up_to=40, action_grid=grid)
    obs = np.array([2.0, 2.0, 0.5, 0.1, 0.0])  # no pipeline (days_to_arrival_norm >= 1)
    action_idx = policy.act(obs)
    assert grid[action_idx] > 0


def test_order_up_to_does_not_order_above_reorder_point():
    grid = np.arange(0, 44, 4)
    policy = OrderUpToPolicy(reorder_point=4, order_up_to=40, action_grid=grid)
    obs = np.array([20.0, 2.0, 0.5, 0.1, 0.0])
    action_idx = policy.act(obs)
    assert grid[action_idx] == 0


def test_order_up_to_does_not_reorder_with_pipeline_inbound():
    grid = np.arange(0, 44, 4)
    policy = OrderUpToPolicy(reorder_point=4, order_up_to=40, action_grid=grid)
    obs = np.array([2.0, 0.5, 0.5, 0.1, 0.0])  # days_to_arrival_norm < 1, something inbound
    action_idx = policy.act(obs)
    assert grid[action_idx] == 0


def test_bandit_update_changes_action_preference():
    bandit = LinearContextualBandit(n_actions=3, n_features=2, ridge_lambda=0.1, epsilon=0.0, seed=0)
    context = np.array([1.0, 1.0])
    for _ in range(20):
        bandit.update(context, action=1, reward=10.0)
        bandit.update(context, action=0, reward=-10.0)
    assert bandit.act(context, greedy=True) == 1


def test_bandit_epsilon_zero_is_deterministic():
    bandit = LinearContextualBandit(n_actions=4, n_features=3, epsilon=0.0, seed=0)
    context = np.array([0.5, 0.2, 0.1])
    a1 = bandit.act(context)
    a2 = bandit.act(context)
    assert a1 == a2


def test_qlearning_discretize_within_bounds():
    q = TabularQLearning(n_actions=5)
    for inventory in [0, 30, 60, 100]:
        for season_log in [0.0, 1.5, 3.5, 10.0]:
            obs = np.array([inventory, 0.5, season_log, 0.5, 0.0])
            inv_idx, lead_idx, season_idx = q._discretize(obs)
            assert 0 <= inv_idx <= q.inv_bins
            assert 0 <= lead_idx <= q.lead_bins
            assert 0 <= season_idx < q.season_bins


def test_qlearning_update_moves_q_value_toward_target():
    q = TabularQLearning(n_actions=3, alpha=0.5, gamma=0.9)
    obs = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
    next_obs = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
    state = q._discretize(obs)
    before = q.Q[state][0]
    q.update(obs, action=0, reward=10.0, next_obs=next_obs, done=False)
    after = q.Q[state][0]
    assert after > before


def test_qlearning_epsilon_decays_over_episodes():
    q = TabularQLearning(n_actions=3, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay_episodes=100)
    assert q._epsilon(0) == pytest.approx(1.0)
    assert q._epsilon(100) == pytest.approx(0.05)
    assert q._epsilon(50) < q._epsilon(0)


def test_heuristic_tuning_beats_a_deliberately_bad_setting():
    from heuristic_policy import evaluate_policy, tune_order_up_to

    cfg = EnvConfig(seed=42)
    tuned = tune_order_up_to(cfg, n_tuning_episodes=15)
    env = InventoryEnv(cfg)
    bad_policy = OrderUpToPolicy(reorder_point=0, order_up_to=4, action_grid=env.action_grid)
    tuned_rewards = evaluate_policy(lambda o: tuned.act(o), env, 15, seed_start=500)
    bad_rewards = evaluate_policy(lambda o: bad_policy.act(o), env, 15, seed_start=500)
    assert tuned_rewards.mean() > bad_rewards.mean()
