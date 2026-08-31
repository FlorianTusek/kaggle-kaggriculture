# SPDX-License-Identifier: MIT
"""Comprehensive Integration & Unit Tests for Phase 2 Features.

Covers:
- PriceTracker history window pruning, edge cases, and momentum calculations.
- Dynamic lot sizing for rising, peak, and falling price momentum.
- Multi-shop town demand boosts and normalization safety in demand-responsive planting.
- Multi-turn KaggricultureAgent integration testing price tracking and market adaptation.
"""

import pytest
from src.market import PriceTracker, MarketOptimizer
from src.strategy import compute_demand_responsive_shares, StrategyPlanner, TOWN_SHOP_DEMAND_BOOST
from src.agent import KaggricultureAgent, DEFAULT_POLICY


class TestPriceTrackerEdgeCases:
    """Rigorous unit tests for PriceTracker edge cases and bounds."""

    def test_history_len_pruning(self):
        tracker = PriceTracker(history_len=5)
        for i in range(10):
            tracker.update({"CARROT": float(i)})
        # History should be pruned to max 5 elements: [5, 6, 7, 8, 9]
        hist = tracker.price_history["CARROT"]
        assert len(hist) == 5
        assert hist == [5.0, 6.0, 7.0, 8.0, 9.0]

    def test_moving_average_empty_and_short_history(self):
        tracker = PriceTracker()
        assert tracker.get_moving_average("UNKNOWN_PRODUCT") == 0.0

        tracker.update({"WHEAT": 10.0})
        # Window size 5 on 1 element -> average is 10.0
        assert tracker.get_moving_average("WHEAT", window=5) == 10.0

    def test_price_momentum_insufficient_data(self):
        tracker = PriceTracker()
        assert tracker.get_price_momentum("WHEAT") == 0.0

        tracker.update({"WHEAT": 10.0})
        assert tracker.get_price_momentum("WHEAT") == 0.0

    def test_price_peak_empty_history(self):
        tracker = PriceTracker()
        # Empty history -> returns True by default
        assert tracker.is_price_peak("TOMATO", 50.0) is True

    def test_dynamic_lot_sizing_falling_momentum(self):
        tracker = PriceTracker(history_len=48)
        # Declining price sequence: 30, 28, 25, 20, 15
        for p in [30, 28, 25, 20, 15]:
            tracker.update({"MELON": float(p)})

        # Momentum is negative (< -1.0)
        lot = tracker.get_dynamic_lot_size("MELON", current_price=15.0, base_lot=15, floor_price=10.0)
        # Should reduce lot size to base_lot // 3 = 5
        assert lot == 5


class TestDemandResponsiveEdgeCases:
    """Rigorous tests for demand-responsive calculation edge cases."""

    def test_all_town_shops_unlocked(self):
        obs = {
            "town": {
                "unlocked_shops": list(TOWN_SHOP_DEMAND_BOOST.keys())
            }
        }
        prices = {"WHEAT": 25.0, "CARROT": 35.0, "TOMATO": 60.0, "STRAWBERRY": 120.0}
        shares = compute_demand_responsive_shares(obs, prices)

        # Shares must normalize to 1.0
        assert abs(sum(shares.values()) - 1.0) < 1e-5
        # All commodities should have positive shares
        for s in shares.values():
            assert s > 0.0

    def test_extreme_low_prices_floor_safety(self):
        obs = {"town": {"unlocked_shops": []}}
        # Price is 0.1 -> base price 25.0 -> raw ratio 0.004, but price_mult floor is 0.5
        crash_prices = {"WHEAT": 0.1, "CARROT": 0.1, "TOMATO": 0.1}
        shares = compute_demand_responsive_shares(obs, crash_prices)
        assert abs(sum(shares.values()) - 1.0) < 1e-5
        assert shares["CARROT"] == pytest.approx(0.4, abs=1e-3)


class TestPhase2AgentIntegrationCycle:
    """Multi-turn integration tests verifying Phase 2 market adaptation."""

    def test_multi_turn_price_adaptation(self):
        agent = KaggricultureAgent()
        tiles = [[None for _ in range(10)] for _ in range(10)]
        for y in range(10):
            for x in range(10):
                if x >= 5 or y >= 5:
                    tiles[y][x] = "LOCKED"

        # Simulate 5 turns of rising CARROT prices (30 -> 35 -> 40 -> 45 -> 50)
        for t in range(5):
            obs = {
                "player": 0,
                "day": 1,
                "hour": t,
                "farms": [
                    {
                        "farmer": (4, 4),
                        "hands": [(4, 4)],
                        "tiles": tiles,
                        "money": 5000,
                        "hires_today": 1,
                        "unlocked_quadrants": ["NW"],
                    }
                ],
                "private": {
                    "seeds": {"CARROT": 0, "TOMATO": 0, "WHEAT": 0},
                    "shed": {"CARROT": 50},
                },
                "market": {
                    "prices": {"CARROT": 30 + t * 5, "TOMATO": 60, "WHEAT": 25}
                },
                "town": {"unlocked_shops": ["Pet Cafe"]}
            }
            res = agent.act(obs)
            assert "market" in res
            assert len(res["market"]) <= 10

        # After 5 turns of rising prices & Pet Cafe, verify price tracker updated
        hist = agent.market_optimizer.price_tracker.price_history.get("CARROT", [])
        assert len(hist) == 5
        assert hist == [30.0, 35.0, 40.0, 45.0, 50.0]
