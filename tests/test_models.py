# SPDX-License-Identifier: MIT
"""Unit tests for BehavioralCloningPolicy and models module."""

import os
import pytest
import numpy as np
from src.models import BehavioralCloningPolicy, BaselineModel
from src.constants import PRODUCTS, CROPS


@pytest.fixture
def sample_obs():
    return {
        "step": 72,
        "day": 3,
        "hour": 0,
        "player": 0,
        "farms": [
            {
                "money": 4200.0,
                "farmer": [4, 4],
                "hands": [[5, 4], [4, 5], [5, 5]],
                "hires_today": 3,
                "unlocked_quadrants": ["NW", "NE"],
                "tiles": [[None] * 10 for _ in range(10)]
            },
            {
                "money": 3000.0,
                "farmer": [4, 4],
                "hands": [],
                "hires_today": 0,
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
            "shed": {"WHEAT": 20, "CARROT": 10},
            "seeds": {"WHEAT": 12, "TOMATO": 6},
            "inventories": [{}, {}, {}, {}]
        }
    }


def test_baseline_model(sample_obs):
    model = BaselineModel()
    action = model.predict(sample_obs)
    assert isinstance(action, dict)
    assert "farmer" in action
    assert "hands" in action
    assert "market" in action
    
    summary = model.evaluate_summary(15000.0)
    assert summary["final_reward"] == 15000.0
    assert summary["success"] is True


def test_behavioral_cloning_policy_unloaded(sample_obs):
    policy = BehavioralCloningPolicy(model_path="nonexistent_path.joblib")
    assert policy.is_loaded is False
    assert policy.predict_farmer_action(sample_obs) == "PASS"
    assert policy.predict_hire_count(sample_obs) == 4
    assert policy.predict_should_sell(sample_obs) is True


def test_behavioral_cloning_feature_extraction(sample_obs):
    policy = BehavioralCloningPolicy()
    df = policy.extract_features(sample_obs, player_idx=0)
    assert hasattr(df, "shape")
    assert df.shape == (1, len(policy.feature_columns))
    assert df.iloc[0]["turn"] == 72.0  # turn
    assert df.iloc[0]["day"] == 3.0   # day
    assert df.iloc[0]["money"] == 4200.0 # money


def test_behavioral_cloning_predictions_loaded(sample_obs):
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "bc_model.joblib")
    if not os.path.exists(model_path):
        pytest.skip("bc_model.joblib not present")
        
    policy = BehavioralCloningPolicy(model_path=model_path)
    assert policy.is_loaded is True
    
    # 1. Action prediction
    action = policy.predict_farmer_action(sample_obs)
    assert isinstance(action, str)
    assert action in policy.classes_farmer
    
    # 2. Probability distribution
    probs = policy.predict_farmer_action_probs(sample_obs)
    assert isinstance(probs, dict)
    assert len(probs) == len(policy.classes_farmer)
    assert abs(sum(probs.values()) - 1.0) < 1e-4
    
    # 3. Labor hire prediction
    hires = policy.predict_hire_count(sample_obs)
    assert isinstance(hires, int)
    assert hires >= 0
    
    # 4. Sell trigger prediction
    should_sell = policy.predict_should_sell(sample_obs)
    assert isinstance(should_sell, bool)
    
    # 5. Full advice dict
    advice = policy.advise(sample_obs)
    assert isinstance(advice, dict)
    assert "recommended_farmer_action" in advice
    assert "action_probabilities" in advice
    assert "recommended_hire_count" in advice
    assert "should_trigger_sell" in advice

