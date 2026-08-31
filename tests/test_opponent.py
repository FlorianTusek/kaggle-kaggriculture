# SPDX-License-Identifier: MIT
"""Unit tests for OpponentTracker and Feed5-first Counter-Strategy."""

import unittest
from src.opponent import OpponentTracker
from src.market import MarketOptimizer
from src.agent import DEFAULT_POLICY

class TestOpponentTracker(unittest.TestCase):

    def setUp(self):
        self.tracker = OpponentTracker()

    def test_feed5_first_needed(self):
        self.assertTrue(self.tracker.is_feed5_first_needed(turn=0))
        self.assertFalse(self.tracker.is_feed5_first_needed(turn=1))

    def test_feed5_first_order_generation(self):
        obs = {"step": 0}
        me = {"money": 1000}
        priv = {"seeds": {}, "shed": {}}

        orders = self.tracker.get_counter_strategy_orders(obs, me, priv)
        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0], ["BUY_SEED", "WHEAT", 5])

    def test_wheat_denial_detection(self):
        # Step 0: Initial wheat inventory 10000
        obs0 = {
            "step": 0, "player": 0,
            "farms": [{"money": 3000, "unlocked_quadrants": ["NW"], "hires_today": 0}, {"money": 3000, "unlocked_quadrants": ["NW"], "hires_today": 0}],
            "market": {"inventory": {"WHEAT": 10000}, "prices": {"WHEAT": 25}}
        }
        self.tracker.update(obs0)

        # Step 10: Opponent buys out wheat (inventory drops by 20 units)
        obs10 = {
            "step": 10, "player": 0,
            "farms": [{"money": 3000, "unlocked_quadrants": ["NW"], "hires_today": 0}, {"money": 2500, "unlocked_quadrants": ["NW"], "hires_today": 0}],
            "market": {"inventory": {"WHEAT": 9975}, "prices": {"WHEAT": 40}}
        }
        self.tracker.update(obs10)

        self.assertTrue(self.tracker.wheat_denial_detected)
        self.assertEqual(self.tracker.archetype, "WHEAT_DENIER")

    def test_market_optimizer_feed5_first_integration(self):
        optimizer = MarketOptimizer(DEFAULT_POLICY)
        obs = {
            "step": 0, "day": 0, "hour": 0, "player": 0,
            "farms": [{"money": 3000, "unlocked_quadrants": ["NW"], "hires_today": 0}, {"money": 3000, "unlocked_quadrants": ["NW"], "hires_today": 0}],
            "private": {"seeds": {}, "shed": {}},
            "market": {"inventory": {"WHEAT": 10000}, "prices": {"WHEAT": 25}},
            "town": {"unlocked_shops": []}
        }
        me = obs["farms"][0]
        priv = obs["private"]

        orders = optimizer.plan_market_orders(obs, me, priv)
        self.assertTrue(len(orders) > 0)
        # First order must be Feed5-first wheat buy order
        self.assertEqual(orders[0], ["BUY_SEED", "WHEAT", 5])

if __name__ == "__main__":
    unittest.main()
