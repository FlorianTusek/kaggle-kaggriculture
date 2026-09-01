# SPDX-License-Identifier: MIT
"""Unit tests for Phase 6 Feature Expansion (Land, Animals, Crop Diversity)."""

import unittest
from src.strategy import StrategyPlanner, compute_demand_responsive_shares
from src.market import MarketOptimizer
from src.agent import KaggricultureAgent, DEFAULT_POLICY

class TestPhase6Features(unittest.TestCase):

    def setUp(self):
        self.planner = StrategyPlanner()
        self.market = MarketOptimizer()

    def _make_mock_obs(self, day=5, hour=0, money=2500, unlocked_quadrants=None, tiles=None, seeds=None, shed=None):
        if unlocked_quadrants is None:
            unlocked_quadrants = ["NW"]
        if tiles is None:
            tiles = [[None for _ in range(10)] for _ in range(10)]
            tiles[4][4] = "UNLOCKED"
        if seeds is None:
            seeds = {}
        if shed is None:
            shed = {}

        return {
            "step": day * 24 + hour,
            "day": day,
            "hour": hour,
            "player": 0,
            "farms": [
                {
                    "farmer": (4, 4),
                    "hands": [(5, 4)],
                    "money": money,
                    "hires_today": 0,
                    "unlocked_quadrants": unlocked_quadrants,
                    "tiles": tiles,
                }
            ],
            "private": {"seeds": seeds, "shed": shed},
            "market": {
                "prices": {
                    "WHEAT": 25, "CARROT": 35, "TOMATO": 60, "STRAWBERRY": 120, "MELON": 250,
                    "EGG": 40, "MILK": 100, "WOOL": 120, "FERTILIZER": 10
                }
            },
            "town": {"unlocked_shops": []}
        }

    def test_land_purchase_evaluation_and_order(self):
        obs = self._make_mock_obs(day=5, money=2500, unlocked_quadrants=["NW"])
        me = obs["farms"][0]
        priv = obs["private"]

        quad = self.planner.evaluate_land_purchase(obs, me)
        self.assertEqual(quad, "NE")

        orders = self.market.plan_market_orders(obs, me, priv)
        buy_land_orders = [o for o in orders if o[0] == "BUY_LAND"]
        self.assertTrue(len(buy_land_orders) > 0)
        self.assertEqual(buy_land_orders[0], ["BUY_LAND", "NE"])

    def test_structure_building_jobs(self):
        obs = self._make_mock_obs(day=5, money=2000)
        me = obs["farms"][0]
        priv = obs["private"]
        free_tiles = [(1, 1), (1, 2)]

        struct_jobs = self.planner.plan_structure_building_jobs(obs, me, priv, free_tiles)
        self.assertTrue(len(struct_jobs) > 0)
        self.assertIn(struct_jobs[0]["op"][0], ["BUILD_PASTURE", "BUILD_COOP"])

    def test_animal_purchase_orders(self):
        tiles = [[None for _ in range(10)] for _ in range(10)]
        tiles[4][4] = "UNLOCKED"
        tiles[1][1] = {"kind": "PASTURE", "animal": None}
        tiles[1][2] = {"kind": "COOP", "animal": None}

        obs = self._make_mock_obs(day=5, money=3000, tiles=tiles)
        me = obs["farms"][0]
        priv = obs["private"]

        orders = self.market.plan_market_orders(obs, me, priv)
        buy_animal_orders = [o for o in orders if o[0] == "BUY_ANIMAL"]
        self.assertTrue(len(buy_animal_orders) > 0)
        animal_types = [o[1] for o in buy_animal_orders]
        self.assertTrue("COW" in animal_types or "SHEEP" in animal_types or "GOOSE" in animal_types)

    def test_five_crop_diversity_shares_and_seeds(self):
        obs = self._make_mock_obs(day=5, money=2000)
        prices = obs["market"]["prices"]

        shares = compute_demand_responsive_shares(obs, prices)
        self.assertIn("STRAWBERRY", shares)
        self.assertIn("MELON", shares)
        self.assertIn("WHEAT", shares)
        self.assertIn("CARROT", shares)
        self.assertIn("TOMATO", shares)
        self.assertAlmostEqual(sum(shares.values()), 1.0, places=4)

    def test_animal_produce_sales(self):
        shed = {"MILK": 5, "WOOL": 5, "EGG": 10, "MELON": 3}
        obs = self._make_mock_obs(day=10, money=1000, shed=shed)
        me = obs["farms"][0]
        priv = obs["private"]

        orders = self.market.plan_market_orders(obs, me, priv)
        sell_orders = [o for o in orders if o[0] == "SELL"]
        sold_products = [o[1] for o in sell_orders]

        self.assertTrue(any(p in sold_products for p in ["MILK", "WOOL", "EGG", "MELON"]))

if __name__ == "__main__":
    unittest.main()
