# SPDX-License-Identifier: MIT
"""Unit tests for PPOPolicy and reinforcement learning module."""

import os
import pytest
import numpy as np
from src.models import PPOPolicy
from src.env import ACTION_LOOKUP, KaggricultureEnv
from src.agent import KaggricultureAgent
from src.constants import PRODUCTS


@pytest.fixture
def sample_obs():
    return {
        "step": 120,
        "day": 5,
        "hour": 0,
        "player": 0,
        "farms": [
            {
                "money": 5000.0,
                "farmer": [4, 4],
                "hands": [[5, 4], [4, 5]],
                "hires_today": 2,
                "unlocked_quadrants": ["NW", "NE"],
                "tiles": [[None] * 10 for _ in range(10)]
            },
            {
                "money": 4500.0,
                "farmer": [4, 4],
                "hands": [[5, 4]],
                "hires_today": 1,
                "unlocked_quadrants": ["NW"],
                "tiles": [[None] * 10 for _ in range(10)]
            }
        ],
        "market": {
            "inventory": {p: 10000 for p in PRODUCTS},
            "prices": {p: 30.0 for p in PRODUCTS}
        },
        "town": {
            "unlocked_shops": ["Bakery"]
        },
        "private": {
            "shed": {"WHEAT": 25, "CARROT": 15},
            "seeds": {"WHEAT": 8, "TOMATO": 4},
            "inventories": [{}, {}, {}]
        }
    }


def test_ppo_policy_unloaded(sample_obs):
    policy = PPOPolicy(model_path="nonexistent_ppo.zip")
    assert policy.is_loaded is False
    assert policy.predict_action(sample_obs) == "PASS"
    advice = policy.advise(sample_obs)
    assert advice["recommended_farmer_action"] == "PASS"
    assert advice["policy_type"] == "PPO_RL"


def test_ppo_policy_feature_extraction(sample_obs):
    policy = PPOPolicy()
    vec = policy.extract_features(sample_obs, player_idx=0)
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (len(policy.feature_columns),)
    assert vec[0] == 120.0 # turn
    assert vec[1] == 5.0   # day
    assert vec[3] == 5000.0 # money


def test_ppo_policy_loaded(sample_obs):
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "ppo_agent.zip")
    if not os.path.exists(model_path):
        pytest.skip("ppo_agent.zip not yet trained")
        
    policy = PPOPolicy(model_path=model_path)
    assert policy.is_loaded is True
    
    # 1. Action prediction
    act = policy.predict_action(sample_obs)
    assert isinstance(act, str)
    assert act in ACTION_LOOKUP
    
    # 2. Advice dict
    advice = policy.advise(sample_obs)
    assert isinstance(advice, dict)
    assert "recommended_farmer_action" in advice
    assert advice["recommended_farmer_action"] in ACTION_LOOKUP
    assert advice["policy_type"] == "PPO_RL"
