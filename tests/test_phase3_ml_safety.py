# SPDX-License-Identifier: MIT
"""Unit and Integration Tests for Phase 3 ML Model Safety Bounds and Fallback Logic.

Ensures:
1. Safety Layer priority strictly overrides ML model recommendations for all workers.
2. Emergency feeding, emergency watering, and crop harvesting execute regardless of ML policy advice.
3. Exceptions raised during ML policy execution degrade gracefully without crashing the agent.
4. Invalid ML action recommendations are safely filtered.
"""

import pytest
from unittest.mock import MagicMock
from src.agent import KaggricultureAgent, DEFAULT_POLICY
from src.models import BehavioralCloningPolicy


class TestMLSafetyLayerPrecedence:
    """Tests ensuring Safety Layer strictly overrides ML model recommendations."""

    def test_ml_recommendation_overridden_by_emergency_watering(self):
        """Verify that emergency crop watering overrides ML farmer action recommendation."""
        agent = KaggricultureAgent(policy=DEFAULT_POLICY)

        # Mock ML policy to recommend moving EAST
        mock_bc = MagicMock()
        mock_bc.is_loaded = True
        mock_bc.advise.return_value = {
            "recommended_farmer_action": "EAST",
            "action_probabilities": {"EAST": 0.9},
        }
        agent.bc_policy = mock_bc

        # Set farmer at (4,4) standing directly on a crop requiring emergency watering
        tiles = [[None for _ in range(10)] for _ in range(10)]
        tiles[4][4] = {
            "kind": "PLANT",
            "crop": "WHEAT",
            "watered_today": False,
            "consecutive_unwatered": 1,  # Emergency!
            "planted_day": 1,
        }

        obs = {
            "player": 0,
            "day": 5,
            "hour": 2,
            "farms": [
                {
                    "farmer": (4, 4),
                    "hands": [],
                    "money": 1000,
                    "hires_today": 0,
                    "tiles": tiles,
                }
            ],
            "private": {"seeds": {}, "shed": {}},
            "market": {"prices": {"WHEAT": 25}},
            "town": {"unlocked_shops": []},
        }

        res = agent.act(obs)
        # Farmer MUST execute ["WATER"] on position (4,4), NOT ["EAST"] from ML policy
        assert res["farmer"] == ["WATER"]

    def test_ml_recommendation_overridden_by_emergency_feeding(self):
        """Verify that emergency animal feeding overrides ML recommendation."""
        agent = KaggricultureAgent(policy=DEFAULT_POLICY)

        mock_bc = MagicMock()
        mock_bc.is_loaded = True
        mock_bc.advise.return_value = {"recommended_farmer_action": "SOUTH"}
        agent.bc_policy = mock_bc

        # Set farmer at (4,4) standing directly on an animal tile requiring emergency feeding
        tiles = [[None for _ in range(10)] for _ in range(10)]
        tiles[4][4] = {
            "kind": "COOP",
            "animal": "GOOSE",
            "fed_today": False,
            "consecutive_unfed": 1,  # Emergency!
        }

        obs = {
            "player": 0,
            "day": 5,
            "hour": 2,
            "farms": [
                {
                    "farmer": (4, 4),
                    "hands": [],
                    "money": 1000,
                    "hires_today": 0,
                    "tiles": tiles,
                }
            ],
            "private": {"seeds": {"WHEAT": 5}, "shed": {}},
            "market": {"prices": {"WHEAT": 25}},
            "town": {"unlocked_shops": []},
        }

        res = agent.act(obs)
        # Farmer MUST execute ["FEED"] on position (4,4), NOT ["SOUTH"]
        assert res["farmer"] == ["FEED"]

    def test_ml_recommendation_overridden_by_ready_harvest(self):
        """Verify that ready crop harvest overrides ML recommendation."""
        agent = KaggricultureAgent(policy=DEFAULT_POLICY)

        mock_bc = MagicMock()
        mock_bc.is_loaded = True
        mock_bc.advise.return_value = {"recommended_farmer_action": "WEST"}
        agent.bc_policy = mock_bc

        # Set farmer at (4,4) standing on a crop ready to harvest
        tiles = [[None for _ in range(10)] for _ in range(10)]
        tiles[4][4] = {
            "kind": "PLANT",
            "crop": "CARROT",
            "planted_day": 1,
            "yield_units": 4,  # Ready!
        }

        obs = {
            "player": 0,
            "day": 5,
            "hour": 2,
            "farms": [
                {
                    "farmer": (4, 4),
                    "hands": [],
                    "money": 1000,
                    "hires_today": 0,
                    "tiles": tiles,
                }
            ],
            "private": {"seeds": {}, "shed": {}},
            "market": {"prices": {"CARROT": 35}},
            "town": {"unlocked_shops": []},
        }

        res = agent.act(obs)
        assert res["farmer"] == ["HARVEST"]


class TestMLPolicyRobustnessAndFallbacks:
    """Tests ensuring agent acts robustly when ML policy fails or returns invalid output."""

    def test_graceful_degradation_on_ml_exception(self):
        """Agent should not crash if bc_policy.advise raises an Exception."""
        agent = KaggricultureAgent(policy=DEFAULT_POLICY)

        mock_bc = MagicMock()
        mock_bc.is_loaded = True
        mock_bc.advise.side_effect = RuntimeError("ML inference model memory error")
        agent.bc_policy = mock_bc

        tiles = [[None for _ in range(10)] for _ in range(10)]
        obs = {
            "player": 0,
            "day": 1,
            "hour": 0,
            "farms": [
                {
                    "farmer": (4, 4),
                    "hands": [],
                    "money": 1000,
                    "hires_today": 0,
                    "tiles": tiles,
                }
            ],
            "private": {"seeds": {}, "shed": {}},
            "market": {"prices": {"WHEAT": 25}},
            "town": {"unlocked_shops": []},
        }

        # Should execute safely without raising exception
        res = agent.act(obs)
        assert "farmer" in res
        assert "market" in res

    def test_invalid_ml_action_filtering(self):
        """Agent should filter out invalid action strings returned by ML policy."""
        agent = KaggricultureAgent(policy=DEFAULT_POLICY)

        mock_bc = MagicMock()
        mock_bc.is_loaded = True
        mock_bc.advise.return_value = {"recommended_farmer_action": "INVALID_ACTION_STRING"}
        agent.bc_policy = mock_bc

        # Place farmer at shed tile (4,4) with no jobs
        tiles = [[None for _ in range(10)] for _ in range(10)]
        obs = {
            "player": 0,
            "day": 1,
            "hour": 0,
            "farms": [
                {
                    "farmer": (4, 4),
                    "hands": [],
                    "money": 1000,
                    "hires_today": 0,
                    "tiles": tiles,
                }
            ],
            "private": {"seeds": {}, "shed": {}},
            "market": {"prices": {"WHEAT": 25}},
            "town": {"unlocked_shops": []},
        }

        res = agent.act(obs)
        # Invalid action filtered out -> farmer defaults to shed DROP action
        assert res["farmer"] == ["DROP"]

    def test_unloaded_bc_policy_fallback(self):
        """Agent handles bc_policy with is_loaded = False cleanly."""
        agent = KaggricultureAgent(policy=DEFAULT_POLICY)
        agent.bc_policy = BehavioralCloningPolicy(model_path="non_existent_model.joblib")
        assert agent.bc_policy.is_loaded is False

        tiles = [[None for _ in range(10)] for _ in range(10)]
        obs = {
            "player": 0,
            "day": 1,
            "hour": 0,
            "farms": [
                {
                    "farmer": (4, 4),
                    "hands": [],
                    "money": 1000,
                    "hires_today": 0,
                    "tiles": tiles,
                }
            ],
            "private": {"seeds": {}, "shed": {}},
            "market": {"prices": {"WHEAT": 25}},
            "town": {"unlocked_shops": []},
        }

        res = agent.act(obs)
        assert res["farmer"] == ["DROP"]
