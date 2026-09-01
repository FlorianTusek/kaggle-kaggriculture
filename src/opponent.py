# SPDX-License-Identifier: MIT
"""Kaggriculture Opponent Tracker & Counter-Strategy Module.

Identifies opponent archetypes:
- Wheat Denier: Opponent buying out market wheat stock early to starve livestock.
- Aggressive Expander: Opponent unlocking land quadrants and hiring hands aggressively.
- Market Dumper: Opponent dumping single commodities in large lots.

Executes counter-strategies:
- "Feed5-first": Buys 5 units of WHEAT at step 0 to secure early animal feed.
- Dynamic Feed Buffer: Stockpiles extra wheat when Wheat Denier activity is detected.
"""

from typing import Dict, List, Any, Optional

class OpponentTracker:
    """Tracks opponent behavior and classifies opponent strategy archetypes."""

    def __init__(self):
        self.history: List[Dict[str, Any]] = []
        self.archetype = "UNKNOWN"
        self.wheat_denial_detected = False

    def update(self, obs: Dict[str, Any]) -> None:
        """Update opponent state history from observation."""
        p = obs.get("player", 0)
        try:
            if hasattr(p, "item"): p = p.item()
            if hasattr(p, "__getitem__") and not isinstance(p, (str, bytes)): p = p[0]
            player_idx = int(p)
        except Exception:
            player_idx = 0

        opp_idx = 1 - player_idx
        farms = obs.get("farms", [])
        if not isinstance(farms, (list, tuple)) or len(farms) <= opp_idx:
            return

        opp_farm = farms[opp_idx]
        market = obs.get("market", {})
        wheat_price = market.get("prices", {}).get("WHEAT", 25)
        wheat_inventory = market.get("inventory", {}).get("WHEAT", 10000)

        snapshot = {
            "turn": obs.get("step", 0),
            "money": opp_farm.get("money", 3000),
            "quadrants": len(opp_farm.get("unlocked_quadrants", ["NW"])),
            "hires": opp_farm.get("hires_today", 0),
            "wheat_inventory": wheat_inventory,
            "wheat_price": wheat_price,
        }
        self.history.append(snapshot)

        # Detect Wheat Denial Tactic
        if len(self.history) >= 2:
            initial_wheat = self.history[0]["wheat_inventory"]
            current_wheat = snapshot["wheat_inventory"]
            # If wheat market inventory dropped by >= 15 units in early turns or price surged
            if snapshot["turn"] <= 48 and (initial_wheat - current_wheat >= 15 or snapshot["wheat_price"] > 35):
                self.wheat_denial_detected = True

        # Classify Archetype
        if self.wheat_denial_detected:
            self.archetype = "WHEAT_DENIER"
        elif snapshot["quadrants"] > 1 or snapshot["hires"] >= 3:
            self.archetype = "AGGRESSIVE_EXPANDER"
        else:
            self.archetype = "STANDARD"

    def is_feed5_first_needed(self, turn: int) -> bool:
        """Return True if step 0 'Feed5-first' wheat buy should be executed."""
        return turn == 0

    def get_counter_strategy_orders(self, obs: Dict[str, Any], me: Dict[str, Any], priv: Dict[str, Any]) -> List[List[Any]]:
        """Generate market orders to counter opponent tactics."""
        turn = obs.get("step", 0)
        money = me.get("money", 0)
        seeds = priv.get("seeds", {})
        shed = priv.get("shed", {})
        wheat_stock = seeds.get("WHEAT", 0) + shed.get("WHEAT", 0)

        orders = []

        # 1. Step 0 "Feed5-first" Counter-Strategy: Buy 5 Wheat immediately at turn 0
        if turn == 0 and wheat_stock < 5 and money >= 50:
            orders.append(["BUY_SEED", "WHEAT", 5])

        # 2. Reactive Wheat Buffer if Wheat Denial Tactic is detected
        elif self.wheat_denial_detected and wheat_stock < 10 and money >= 100:
            orders.append(["BUY_SEED", "WHEAT", 5])

        return orders
