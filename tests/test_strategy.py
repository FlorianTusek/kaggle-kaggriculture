# SPDX-License-Identifier: MIT
"""Unit tests for Kaggriculture StrategyPlanner module and Agent integration."""

import pytest
from src.strategy import (
    fibonacci,
    calculate_hire_cost,
    get_season_phase,
    StrategyPlanner,
)
from src.agent import KaggricultureAgent, DEFAULT_POLICY


class TestStrategyPlannerCalculations:
    """Tests for math and phase utility functions in strategy module."""

    @pytest.mark.parametrize(
        "n,expected",
        [
            (-1, 1),
            (0, 1),
            (1, 1),
            (2, 2),
            (3, 3),
            (4, 5),
            (5, 8),
            (6, 13),
        ],
    )
    def test_fibonacci_sequence(self, n, expected):
        assert fibonacci(n) == expected

    def test_calculate_hire_cost(self):
        # 0 hires -> hire 1: fib(0) = 1
        assert calculate_hire_cost(0, 1) == 1
        # 0 hires -> hire 4: fib(0)+fib(1)+fib(2)+fib(3) = 1+1+2+3 = 7
        assert calculate_hire_cost(0, 4) == 7
        # 2 existing hires -> hire 2 more: fib(2)+fib(3) = 2+3 = 5
        assert calculate_hire_cost(2, 2) == 5

    def test_get_season_phase_boundaries(self):
        policy = {"invest_until_day": 24, "plant_until_day": 25, "liquidate_from_day": 27}

        # Day 20: investing=True, planting=True, liquidating=False
        p20 = get_season_phase(20, policy)
        assert p20["investing"] is True
        assert p20["planting"] is True
        assert p20["liquidating"] is False

        # Day 25: investing=False, planting=True, liquidating=False
        p25 = get_season_phase(25, policy)
        assert p25["investing"] is False
        assert p25["planting"] is True
        assert p25["liquidating"] is False

        # Day 27: investing=False, planting=False, liquidating=True
        p27 = get_season_phase(27, policy)
        assert p27["investing"] is False
        assert p27["planting"] is False
        assert p27["liquidating"] is True


class TestStrategyPlannerLogic:
    """Tests for StrategyPlanner decision methods."""

    def test_get_target_hands_late_season(self):
        planner = StrategyPlanner()
        obs = {"day": 25}  # Past invest_until_day (24)
        me = {"money": 10000}
        assert planner.get_target_hands(obs, me) == 0

    def test_get_target_hands_capital_scaling(self):
        planner = StrategyPlanner({"hands": 4, "invest_until_day": 24})
        obs = {"day": 1}
        # Cost for 4 base hands is 7. If money < 7 + 100 = 107, scale down
        me_poor = {"money": 50}
        assert planner.get_target_hands(obs, me_poor) == max(1, 4 - 2)  # 2

        me_rich = {"money": 500}
        assert planner.get_target_hands(obs, me_rich) == 4

    def test_evaluate_land_purchase_day_cutoff(self):
        planner = StrategyPlanner()
        me = {"money": 10000, "unlocked_quadrants": ["NW"]}
        
        # Day 15: valid for purchase
        assert planner.evaluate_land_purchase({"day": 15}, me) == "NE"
        # Day 16: past cutoff
        assert planner.evaluate_land_purchase({"day": 16}, me) is None

    def test_evaluate_land_purchase_quadrant_progression(self):
        planner = StrategyPlanner()
        obs = {"day": 5}

        # Step 1: NW unlocked, needs 2 * 1000 = 2000 for NE
        me1 = {"money": 2500, "unlocked_quadrants": ["NW"]}
        assert planner.evaluate_land_purchase(obs, me1) == "NE"

        # Step 2: NW, NE unlocked, needs 2 * 2000 = 4000 for SW
        me2 = {"money": 4500, "unlocked_quadrants": ["NW", "NE"]}
        assert planner.evaluate_land_purchase(obs, me2) == "SW"

        # Step 3: NW, NE, SW unlocked, needs 2 * 4000 = 8000 for SE
        me3 = {"money": 9000, "unlocked_quadrants": ["NW", "NE", "SW"]}
        assert planner.evaluate_land_purchase(obs, me3) == "SE"

        # Step 4: All unlocked -> None
        me4 = {"money": 20000, "unlocked_quadrants": ["NW", "NE", "SW", "SE"]}
        assert planner.evaluate_land_purchase(obs, me4) is None

    def test_plan_planting_jobs(self):
        planner = StrategyPlanner({"plant_until_day": 25, "crops": ["CARROT", "WHEAT"]})
        obs = {"day": 10}
        me = {}
        priv = {"seeds": {"CARROT": 2, "WHEAT": 1}}
        free_tiles = [(0, 0), (1, 0), (2, 0), (3, 0)]

        jobs = planner.plan_planting_jobs(obs, me, priv, free_tiles)
        assert len(jobs) == 3  # 2 CARROT + 1 WHEAT
        assert jobs[0]["op"] == ["PLANT", "CARROT"]
        assert jobs[0]["pos"] == (0, 0)
        assert jobs[0]["priority"] == 9
        assert jobs[1]["op"] == ["PLANT", "CARROT"]
        assert jobs[2]["op"] == ["PLANT", "WHEAT"]


class TestAgentIntegration:
    """Integration tests for KaggricultureAgent act cycle."""

    def test_agent_act_cycle(self):
        agent = KaggricultureAgent()
        
        # Build 10x10 empty tile grid with NW unlocked (Shed at center 4,4 .. 5,5)
        tiles = [[None for _ in range(10)] for _ in range(10)]
        for y in range(10):
            for x in range(10):
                if x >= 5 or y >= 5:
                    tiles[y][x] = "LOCKED"

        obs = {
            "player": 0,
            "day": 1,
            "hour": 0,
            "farms": [
                {
                    "farmer": (4, 4),
                    "hands": [(4, 4), (4, 4)],
                    "tiles": tiles,
                    "money": 1000,
                    "hires_today": 0,
                    "unlocked_quadrants": ["NW"],
                }
            ],
            "private": {
                "seeds": {"CARROT": 4},
                "shed": {"CARROT": 10},
            },
            "market": {
                "prices": {"CARROT": 35, "TOMATO": 60, "WHEAT": 25}
            }
        }

        result = agent.act(obs)
        assert "farmer" in result
        assert "hands" in result
        assert "market" in result
        assert isinstance(result["market"], list)
        assert len(result["market"]) <= 10
        # SELL CARROT should be first market order for liquidity
        assert result["market"][0][0] == "SELL"
