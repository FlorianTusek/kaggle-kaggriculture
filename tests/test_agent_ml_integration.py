# SPDX-License-Identifier: MIT
"""Unit tests for ML Behavioral Cloning Model integration in KaggricultureAgent."""

import unittest
from unittest.mock import MagicMock
from src.agent import KaggricultureAgent, DEFAULT_POLICY

class TestMLAgentIntegration(unittest.TestCase):

    def test_agent_initialization_with_ml_policy(self):
        agent = KaggricultureAgent()
        self.assertIsNotNone(agent.strategy_planner)
        self.assertIsNotNone(agent.market_optimizer)
        # Verify bc_policy is initialized or handled gracefully
        if agent.bc_policy is not None:
            self.assertTrue(hasattr(agent.bc_policy, 'is_loaded'))

    def test_agent_act_with_mocked_ml_advice(self):
        agent = KaggricultureAgent(policy=DEFAULT_POLICY)
        
        # Mock ML policy
        mock_bc = MagicMock()
        mock_bc.is_loaded = True
        mock_bc.advise.return_value = {
            "recommended_farmer_action": "EAST",
            "action_probabilities": {"EAST": 0.8, "PASS": 0.2},
            "recommended_hire_count": 4,
            "should_trigger_sell": True,
        }
        agent.bc_policy = mock_bc

        # Mock observation
        tiles = [[None for _ in range(10)] for _ in range(10)]
        tiles[4][4] = "UNLOCKED"
        obs = {
            "day": 1,
            "hour": 0,
            "player": 0,
            "farms": [
                {
                    "farmer": (4, 4),
                    "hands": [(5, 4)],
                    "money": 1000,
                    "hires_today": 0,
                    "tiles": tiles,
                }
            ],
            "private": {"seeds": {}, "shed": {}},
            "market": {"prices": {"CARROT": 35}},
            "town": {"unlocked_shops": []}
        }

        action_dict = agent.act(obs)
        self.assertIn("farmer", action_dict)
        self.assertIn("hands", action_dict)
        self.assertIn("market", action_dict)
        # Verify mock_bc.advise was called
        mock_bc.advise.assert_called_once_with(obs)

if __name__ == "__main__":
    unittest.main()
