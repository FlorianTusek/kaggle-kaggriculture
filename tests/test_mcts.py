# SPDX-License-Identifier: MIT
"""Unit tests for HybridMCTS module."""

import pytest
from src.mcts import HybridMCTS, MCTSNode, FastStateEvaluator
from src.models import PPOPolicy
from src.constants import PRODUCTS


@pytest.fixture
def sample_obs():
    return {
        "step": 48,
        "day": 2,
        "hour": 0,
        "player": 0,
        "farms": [
            {
                "money": 3500.0,
                "farmer": [4, 4],
                "hands": [[5, 4]],
                "hires_today": 1,
                "unlocked_quadrants": ["NW"],
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
            "shed": {"WHEAT": 10},
            "seeds": {"WHEAT": 4},
            "inventories": [{}, {}]
        }
    }


def test_mcts_search_basic(sample_obs):
    mcts = HybridMCTS(n_simulations=10, top_k_actions=4)
    res = mcts.search(sample_obs)
    assert "best_action" in res
    assert res["best_action"] in ("PASS", "NORTH", "SOUTH", "EAST", "WEST", "WATER", "HARVEST")
    assert "visit_distribution" in res
    assert res["num_simulations"] >= 10


def test_mcts_advise(sample_obs):
    mcts = HybridMCTS(n_simulations=5)
    advice = mcts.advise(sample_obs)
    assert advice["policy_type"] == "HYBRID_MCTS"
    assert "recommended_farmer_action" in advice
