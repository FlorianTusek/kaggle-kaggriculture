# SPDX-License-Identifier: MIT
"""Unit tests for Market and Cash Management logic in Kaggriculture Agent."""

import unittest
from src.market import MarketOptimizer
from src.agent import DEFAULT_POLICY
from src.constants import MAX_MARKET_ORDERS_PER_TURN

class TestMarketAndCashManagement(unittest.TestCase):

    def test_hire_hands_at_hour_zero(self):
        obs = {
            "day": 1,
            "hour": 0,
            "market": {"prices": {"CARROT": 35, "TOMATO": 60, "WHEAT": 25}},
        }
        policy = dict(DEFAULT_POLICY, hands=2)
        me = {"money": 5000, "hires_today": 0}
        priv = {"seeds": {}, "shed": {}}

        optimizer = MarketOptimizer(policy)
        orders = optimizer.plan_market_orders(obs, me, priv)
        hire_orders = [o for o in orders if o[0] == "HIRE"]
        self.assertEqual(len(hire_orders), 2)

    def test_buy_seed_cash_constraint(self):
        """Ensure agent does not buy seeds if money is below seed cost batch."""
        obs = {
            "day": 1,
            "hour": 5,
            "market": {"prices": {"CARROT": 35, "WHEAT": 25}},
        }
        policy = dict(DEFAULT_POLICY, crops=["CARROT"], crop_share={"CARROT": 1.0}, seed_stock=20, seed_batch=6)
        optimizer = MarketOptimizer(policy)

        # CARROT seed cost = 20 * batch(6) = 120
        # Case A: money = 50 (insufficient)
        me_poor = {"money": 50, "hires_today": 2}
        priv = {"seeds": {"CARROT": 0}, "shed": {}}
        orders_poor = optimizer.plan_market_orders(obs, me_poor, priv)
        buy_poor = [o for o in orders_poor if o[0] == "BUY_SEED"]
        self.assertEqual(len(buy_poor), 0)

        # Case B: money = 200 (sufficient)
        me_rich = {"money": 200, "hires_today": 2}
        orders_rich = optimizer.plan_market_orders(obs, me_rich, priv)
        buy_rich = [o for o in orders_rich if o[0] == "BUY_SEED"]
        self.assertEqual(len(buy_rich), 1)
        self.assertEqual(buy_rich[0], ["BUY_SEED", "CARROT", 6])

    def test_sell_produce_floor_price_and_lot_cap(self):
        """Ensure sell orders obey price floors and max lot restrictions."""
        obs = {
            "day": 10,
            "hour": 12,
            "market": {"prices": {"CARROT": 15, "WHEAT": 4}},  # WHEAT price below floor (5)
        }
        policy = dict(
            DEFAULT_POLICY,
            sell_order=["CARROT", "WHEAT"],
            price_floors={"CARROT": 10, "WHEAT": 5},
            sell_lots={"CARROT": 10, "WHEAT": 20},
            liquidate_from_day=27,
        )
        me = {"money": 1000, "hires_today": 4}
        priv = {"seeds": {}, "shed": {"CARROT": 25, "WHEAT": 50}}

        optimizer = MarketOptimizer(policy)
        orders = optimizer.plan_market_orders(obs, me, priv)
        sell_orders = [o for o in orders if o[0] == "SELL"]

        # CARROT price (15) >= floor (10): sell min(25, 10) = 10
        self.assertIn(["SELL", "CARROT", 10], sell_orders)
        # WHEAT price (4) < floor (5) and day (10) < liquidate_from_day (27): do not sell
        self.assertFalse(any(o[1] == "WHEAT" for o in sell_orders))

    def test_forced_liquidation_end_game(self):
        """Ensure forced liquidation triggers on or after liquidate_from_day regardless of floor price."""
        obs = {
            "day": 28,  # day >= liquidate_from_day (27)
            "hour": 12,
            "market": {"prices": {"WHEAT": 2}},  # Below floor (5)
        }
        policy = dict(
            DEFAULT_POLICY,
            sell_order=["WHEAT"],
            price_floors={"WHEAT": 5},
            sell_lots={"WHEAT": 20},
            liquidate_from_day=27,
        )
        me = {"money": 1000, "hires_today": 4}
        priv = {"seeds": {}, "shed": {"WHEAT": 30}}

        optimizer = MarketOptimizer(policy)
        orders = optimizer.plan_market_orders(obs, me, priv)
        sell_orders = [o for o in orders if o[0] == "SELL"]
        self.assertIn(["SELL", "WHEAT", 20], sell_orders)

    def test_max_market_orders_limit(self):
        """Ensure order count never exceeds MAX_MARKET_ORDERS_PER_TURN (10)."""
        obs = {
            "day": 1,
            "hour": 0,
            "market": {"prices": {p: 100 for p in ["CARROT", "TOMATO", "WHEAT", "MELON", "STRAWBERRY"]}},
        }
        policy = dict(
            DEFAULT_POLICY,
            hands=10,
            crops=["CARROT", "TOMATO", "WHEAT", "MELON", "STRAWBERRY"],
            crop_share={p: 0.2 for p in ["CARROT", "TOMATO", "WHEAT", "MELON", "STRAWBERRY"]},
            seed_stock=50,
            sell_order=["CARROT", "TOMATO", "WHEAT", "MELON", "STRAWBERRY"],
            price_floors={p: 0 for p in ["CARROT", "TOMATO", "WHEAT", "MELON", "STRAWBERRY"]},
        )
        me = {"money": 50000, "hires_today": 0}
        priv = {
            "seeds": {p: 0 for p in ["CARROT", "TOMATO", "WHEAT", "MELON", "STRAWBERRY"]},
            "shed": {p: 50 for p in ["CARROT", "TOMATO", "WHEAT", "MELON", "STRAWBERRY"]},
        }

        optimizer = MarketOptimizer(policy)
        orders = optimizer.plan_market_orders(obs, me, priv)
        self.assertLessEqual(len(orders), MAX_MARKET_ORDERS_PER_TURN)

if __name__ == "__main__":
    unittest.main()
