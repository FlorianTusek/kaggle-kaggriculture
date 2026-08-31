# SPDX-License-Identifier: MIT
"""Unit tests for Demand-Responsive Planting and Town Shop Demand Boosts."""

import unittest
from src.strategy import compute_demand_responsive_shares, StrategyPlanner
from src.agent import DEFAULT_POLICY

class TestDemandResponsivePlanting(unittest.TestCase):

    def test_compute_demand_responsive_shares_price_signal(self):
        obs = {"town": {"unlocked_shops": []}}
        
        # Base prices: CARROT=35, TOMATO=60, WHEAT=25
        # High CARROT price (70 = 2.0x base) should boost CARROT share
        high_carrot_prices = {"CARROT": 70, "TOMATO": 60, "WHEAT": 25}
        shares = compute_demand_responsive_shares(obs, high_carrot_prices)

        self.assertGreater(shares["CARROT"], 0.4)
        self.assertAlmostEqual(sum(shares.values()), 1.0, places=4)

    def test_town_shop_demand_boost(self):
        # Pet Cafe boosts CARROT by 2.0x
        obs_pet_cafe = {"town": {"unlocked_shops": ["Pet Cafe"]}}
        normal_prices = {"CARROT": 35, "TOMATO": 60, "WHEAT": 25}

        shares_pet_cafe = compute_demand_responsive_shares(obs_pet_cafe, normal_prices)
        shares_no_shop = compute_demand_responsive_shares({"town": {"unlocked_shops": []}}, normal_prices)

        # CARROT share should be significantly higher with Pet Cafe unlocked
        self.assertGreater(shares_pet_cafe["CARROT"], shares_no_shop["CARROT"])

    def test_planner_planting_jobs_ordering(self):
        planner = StrategyPlanner(DEFAULT_POLICY)
        obs = {
            "day": 5,
            "town": {"unlocked_shops": ["Pizza Shop"]}  # Boosts TOMATO (1.5x)
        }
        me = {"money": 5000}
        priv = {"seeds": {"CARROT": 5, "TOMATO": 5, "WHEAT": 5}}
        free_tiles = [(0, 0), (0, 1), (0, 2)]

        jobs = planner.plan_planting_jobs(obs, me, priv, free_tiles)
        self.assertEqual(len(jobs), 3)

        # First plant job should be TOMATO because Pizza Shop boosted its share
        self.assertEqual(jobs[0]["op"][1], "TOMATO")

if __name__ == "__main__":
    unittest.main()
