# SPDX-License-Identifier: MIT
"""Unit tests for StrategyPlanner and MarketOptimizer."""

import unittest
from src.strategy import fibonacci, calculate_hire_cost, get_season_phase, StrategyPlanner
from src.market import MarketOptimizer

class TestStrategyPlanner(unittest.TestCase):

    def test_fibonacci(self):
        self.assertEqual(fibonacci(0), 1)
        self.assertEqual(fibonacci(1), 1)
        self.assertEqual(fibonacci(2), 2)
        self.assertEqual(fibonacci(3), 3)
        self.assertEqual(fibonacci(4), 5)

    def test_calculate_hire_cost(self):
        # First 4 hires cost: fib(0)+fib(1)+fib(2)+fib(3) = 1+1+2+3 = 7 coins
        self.assertEqual(calculate_hire_cost(0, 4), 7)

    def test_season_phase(self):
        policy = {"plant_until_day": 25, "liquidate_from_day": 27}

        phase_day10 = get_season_phase(10, policy)
        self.assertTrue(phase_day10["planting"])
        self.assertFalse(phase_day10["liquidating"])

        phase_day28 = get_season_phase(28, policy)
        self.assertFalse(phase_day28["planting"])
        self.assertTrue(phase_day28["liquidating"])

    def test_strategy_planner_land_purchase(self):
        planner = StrategyPlanner()

        # Low money -> no land purchase
        obs = {"day": 5}
        me = {"money": 500, "unlocked_quadrants": ["NW"]}
        self.assertIsNone(planner.evaluate_land_purchase(obs, me))

        # High money -> purchase NE quadrant ($1,000 cost * 2 reserve = $2,000)
        me_wealthy = {"money": 3000, "unlocked_quadrants": ["NW"]}
        self.assertEqual(planner.evaluate_land_purchase(obs, me_wealthy), "NE")

class TestMarketOptimizer(unittest.TestCase):

    def test_market_orders_sell_reordering(self):
        optimizer = MarketOptimizer({
            "hands": 4,
            "sell_order": ["CARROT"],
            "sell_lots": {"CARROT": 10},
            "price_floors": {"CARROT": 10},
            "plant_until_day": 25,
            "seed_stock": 10,
            "seed_batch": 5,
            "crop_share": {"CARROT": 1.0}
        })

        obs = {
            "day": 5,
            "hour": 0,
            "market": {"prices": {"CARROT": 35}}
        }
        me = {"money": 100, "hires_today": 0}
        priv = {"shed": {"CARROT": 15}, "seeds": {"CARROT": 0}}

        orders = optimizer.plan_market_orders(obs, me, priv)
        self.assertTrue(len(orders) > 0)

        # SELL order must come BEFORE HIRE / BUY_SEED orders for liquidity
        self.assertEqual(orders[0][0], "SELL")
        self.assertEqual(orders[0][1], "CARROT")
        self.assertEqual(orders[0][2], 10)

        # Order count cap
        self.assertLessEqual(len(orders), 10)

if __name__ == "__main__":
    unittest.main()
