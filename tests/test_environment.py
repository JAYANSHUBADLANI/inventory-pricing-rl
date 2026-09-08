import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from environment import EnvConfig, InventoryEnv


@pytest.fixture
def env():
    return InventoryEnv(EnvConfig(seed=1))


def test_reset_gives_starting_inventory(env):
    obs = env.reset(seed=1)
    assert obs[0] == env.cfg.starting_inventory


def test_inventory_never_negative_over_full_episode(env):
    obs = env.reset(seed=2)
    done = False
    while not done:
        obs, reward, done, info = env.step(0)
        assert info["inventory_after"] >= 0


def test_action_grid_matches_config(env):
    assert env.action_grid[0] == 0
    assert env.action_grid[-1] == env.cfg.max_order
    assert len(env.action_grid) == env.n_actions


def test_ordering_arrives_after_lead_time():
    cfg = EnvConfig(seed=3, lead_time=3, starting_inventory=0)
    env = InventoryEnv(cfg)
    obs = env.reset(seed=3)
    # order the max amount on day 0, track when inventory jumps
    order_idx = env.n_actions - 1
    order_qty = env.action_grid[order_idx]
    inventories = []
    for day in range(6):
        obs, reward, done, info = env.step(0 if day > 0 else order_idx)
        inventories.append(info["inventory_after"])
    # inventory should not jump by the full order amount before day 3
    assert inventories[0] < order_qty
    assert inventories[1] < order_qty


def test_fixed_cost_only_charged_when_ordering():
    cfg = EnvConfig(seed=4, order_fixed_cost=2.0)
    env = InventoryEnv(cfg)
    env.reset(seed=4)
    _, _, _, info_no_order = env.step(0)
    assert info_no_order["fixed_cost"] == 0.0
    env.reset(seed=4)
    _, _, _, info_order = env.step(1)
    assert info_order["fixed_cost"] == 2.0


def test_reward_equals_negative_total_cost():
    env = InventoryEnv(EnvConfig(seed=5))
    env.reset(seed=5)
    obs, reward, done, info = env.step(2)
    expected = -(info["holding_cost"] + info["stockout_cost"] + info["fixed_cost"])
    assert reward == pytest.approx(expected)


def test_episode_ends_at_horizon():
    cfg = EnvConfig(seed=6, horizon_days=10)
    env = InventoryEnv(cfg)
    env.reset(seed=6)
    for day in range(10):
        obs, reward, done, info = env.step(0)
    assert done


def test_regime_shift_increases_demand_rate():
    # compare two envs at the identical calendar day and time step, one with
    # the regime shift active and one without, so the natural change in the
    # seasonal multiplier between days does not confound the comparison.
    without_shift = InventoryEnv(EnvConfig(seed=7, regime_shift_day=None))
    with_shift = InventoryEnv(EnvConfig(seed=7, regime_shift_day=5, regime_shift_multiplier=2.0))
    without_shift.reset(seed=7, start_day_of_year=1)
    with_shift.reset(seed=7, start_day_of_year=1)
    for _ in range(6):
        without_shift.step(0)
        with_shift.step(0)
    assert with_shift.t == without_shift.t == 6
    rate_without = without_shift._demand_rate()
    rate_with = with_shift._demand_rate()
    assert rate_with == pytest.approx(rate_without * 2.0)


def test_demand_multiplier_override_scales_rate():
    base_env = InventoryEnv(EnvConfig(seed=8, demand_multiplier=1.0))
    shifted_env = InventoryEnv(EnvConfig(seed=8, demand_multiplier=1.25))
    base_env.reset(seed=8, start_day_of_year=1)
    shifted_env.reset(seed=8, start_day_of_year=1)
    assert shifted_env._demand_rate() == pytest.approx(base_env._demand_rate() * 1.25)


def test_lead_time_override_used_over_config_default():
    cfg = EnvConfig(seed=9, lead_time=3, lead_time_override=1)
    env = InventoryEnv(cfg)
    assert env._lead_time() == 1
