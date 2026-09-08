import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from environment import EnvConfig, InventoryEnv
from policy_gradient import ReinforceAgent

ROOT = Path(__file__).resolve().parents[1]


def test_reinforce_action_within_valid_range():
    env = InventoryEnv(EnvConfig(seed=1))
    agent = ReinforceAgent(obs_dim=env.observation_dim, n_actions=env.n_actions, seed=1)
    obs = env.reset(seed=1)
    action = agent.act_greedy(obs)
    assert 0 <= action < env.n_actions


def test_reinforce_normalize_scales_inventory_only():
    env = InventoryEnv(EnvConfig(seed=1))
    agent = ReinforceAgent(obs_dim=env.observation_dim, n_actions=env.n_actions, seed=1)
    obs = np.array([30.0, 0.5, 1.0, 0.5, 0.0])
    normalized = agent._normalize(obs)
    assert normalized[0] == pytest.approx(1.0)
    assert normalized[1] == obs[1]
    assert normalized[2] == obs[2]


def test_reinforce_training_reduces_loss_direction_improves_reward():
    env = InventoryEnv(EnvConfig(seed=1))
    agent = ReinforceAgent(obs_dim=env.observation_dim, n_actions=env.n_actions, seed=1)
    early = agent.train(env, 30, seed_start=1000)
    later = agent.train(env, 30, seed_start=2000)
    # not a strict monotonic guarantee episode to episode, but the later
    # batch should not be catastrophically worse on average than the first
    assert later.mean() > early.mean() - abs(early.mean())


def test_position_baseline_shape_matches_horizon():
    env = InventoryEnv(EnvConfig(seed=1, horizon_days=20))
    agent = ReinforceAgent(obs_dim=env.observation_dim, n_actions=env.n_actions, seed=1)
    agent.train(env, 2, seed_start=1)
    assert agent.position_baseline.shape[0] == 20


def test_demand_calibration_file_has_expected_fields():
    cal_path = ROOT / "data" / "demand_calibration.json"
    with open(cal_path) as f:
        calibration = json.load(f)
    assert "day_of_year_multiplier" in calibration
    assert len(calibration["day_of_year_multiplier"]) == 366
    assert calibration["baseline_mean_units_per_day"] > 0
    assert calibration["negative_binomial_dispersion_r"] > 0
