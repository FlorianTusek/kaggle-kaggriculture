# SPDX-License-Identifier: MIT
"""Kaggriculture Market Optimizer Module.

Handles market transaction optimization:
- Order queue reordering (placing SELL orders before BUY/HIRE orders so sales fund purchases).
- Price floor enforcement & price decay mitigation (metered sales in capped lots).
- Fibonacci-aware HIRE orders.
- Seed stocking orders.
- End-of-season inventory liquidation.
"""

from typing import Dict, List, Any, Tuple, Optional
from src.constants import MAX_MARKET_ORDERS_PER_TURN, CROPS, PRODUCTS
from src.strategy import calculate_hire_cost, get_season_phase

class MarketOptimizer:
    """Encapsulates market order optimization and queuing logic."""

    def __init__(self, policy: Optional[Dict[str, Any]] = None):
        self.policy = policy if policy is not None else {}

    def plan_market_orders(self, obs: Dict[str, Any], me: Dict[str, Any], priv: Dict[str, Any]) -> List[List[Any]]:
        """Generate an optimized list of up to 10 market orders for the current turn."""
        day = obs.get("day", 0)
        hour = obs.get("hour", 0)
        money = me.get("money", 0)
        hires_today = me.get("hires_today", 0)
        seeds = priv.get("seeds", {})
        shed = priv.get("shed", {})
        prices = obs.get("market", {}).get("prices", {})
        phase = get_season_phase(day, self.policy)

        sell_orders = []
        buy_orders = []
        hire_orders = []

        # 1. SELL Orders (queued early to fund buys)
        sell_order_list = self.policy.get("sell_order", ["CARROT", "TOMATO", "WHEAT", "MELON", "STRAWBERRY"])
        sell_lots = self.policy.get("sell_lots", {"CARROT": 15, "TOMATO": 10, "WHEAT": 20, "MELON": 10})
        floors = self.policy.get("price_floors", {"CARROT": 10, "TOMATO": 20, "WHEAT": 5, "MELON": 100})

        for product in sell_order_list:
            in_shed = shed.get(product, 0)
            if in_shed <= 0:
                continue
            cur_price = prices.get(product, 0)
            floor = floors.get(product, 0)

            if phase["liquidating"] or cur_price >= floor:
                max_lot = sell_lots.get(product, 20)
                qty = min(in_shed, max_lot)
                sell_orders.append(["SELL", product, qty])

        # 2. HIRE Orders (on hour 0 of day)
        if hour == 0:
            target_hands = self.policy.get("hands", 4)
            if hires_today < target_hands:
                needed = target_hands - hires_today
                for _ in range(needed):
                    hire_orders.append(["HIRE"])

        # 3. BUY_SEED Orders
        if phase["planting"]:
            stock_target = self.policy.get("seed_stock", 12)
            batch = self.policy.get("seed_batch", 6)
            crop_shares = self.policy.get("crop_share", {"CARROT": 0.4, "TOMATO": 0.3, "WHEAT": 0.3})

            for crop, share in crop_shares.items():
                if crop not in CROPS:
                    continue
                current_stock = seeds.get(crop, 0)
                desired_stock = int(stock_target * share)
                seed_cost = CROPS[crop]["seed"] * batch
                if current_stock < desired_stock and money >= seed_cost:
                    buy_orders.append(["BUY_SEED", crop, batch])
                    money -= seed_cost

        # Reorder queue: SELL first (to generate liquidity), then HIRE, then BUY_SEED
        combined_orders = sell_orders + hire_orders + buy_orders
        return combined_orders[:MAX_MARKET_ORDERS_PER_TURN]
