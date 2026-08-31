# SPDX-License-Identifier: MIT
"""Unit tests for PriceTracker and Phase 2 Sell-Timing heuristics."""

import unittest
from src.market import PriceTracker, MarketOptimizer
from src.agent import DEFAULT_POLICY

class TestPriceTracker(unittest.TestCase):

    def setUp(self):
        self.tracker = PriceTracker(history_len=48)

    def test_price_history_and_moving_average(self):
        prices_sequence = [20, 22, 24, 26, 28, 30]
        for p in prices_sequence:
            self.tracker.update({"CARROT": p})

        ma = self.tracker.get_moving_average("CARROT", window=4)
        # Average of [24, 26, 28, 30] = 27.0
        self.assertEqual(ma, 27.0)

    def test_price_momentum(self):
        # Rising price sequence
        for p in range(10, 20):
            self.tracker.update({"TOMATO": p})

        momentum = self.tracker.get_price_momentum("TOMATO", window=5)
        self.assertGreater(momentum, 0.0)

    def test_is_price_peak(self):
        for p in [30, 35, 40, 50, 48]:
            self.tracker.update({"MELON": p})

        self.assertTrue(self.tracker.is_price_peak("MELON", current_price=48))
        self.assertFalse(self.tracker.is_price_peak("MELON", current_price=30))

    def test_dynamic_lot_sizing(self):
        # Rising price -> smaller lot size (5-7) to let price climb
        for p in [10, 12, 14, 16, 18, 20]:
            self.tracker.update({"WHEAT": p})

        lot_rising = self.tracker.get_dynamic_lot_size("WHEAT", current_price=20, base_lot=15, floor_price=5)
        self.assertEqual(lot_rising, 7)  # 15 // 2 = 7

        # Price below floor -> 0 lot size
        lot_below_floor = self.tracker.get_dynamic_lot_size("WHEAT", current_price=3, base_lot=15, floor_price=5)
        self.assertEqual(lot_below_floor, 0)

class TestMarketOptimizerPhase2(unittest.TestCase):

    def test_market_optimizer_price_tracking(self):
        optimizer = MarketOptimizer(DEFAULT_POLICY)
        obs = {
            "day": 5,
            "hour": 0,
            "market": {"prices": {"CARROT": 35, "TOMATO": 60, "WHEAT": 25}}
        }
        me = {"money": 5000, "hires_today": 0}
        priv = {"seeds": {}, "shed": {"CARROT": 30}}

        orders = optimizer.plan_market_orders(obs, me, priv)
        self.assertTrue(len(orders) > 0)
        # Price tracker updated
        hist = optimizer.price_tracker.price_history.get("CARROT", [])
        self.assertIn(35.0, hist)

if __name__ == "__main__":
    unittest.main()
