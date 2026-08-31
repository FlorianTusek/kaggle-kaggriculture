# SPDX-License-Identifier: MIT
"""Kaggriculture Market Optimizer & Price Tracking Module.

Handles market transaction optimization and price dynamics:
- Price History & Momentum Tracking (PriceTracker).
- Dynamic Sell-Timing Heuristics (scaling sell lot size based on price momentum, peak windows, and decay).
- Order Queue Reordering (placing SELL orders before BUY/HIRE orders to generate immediate liquidity).
- Price floor enforcement & market order limit (max 10 orders per turn).
"""

from typing import Dict, List, Any, Tuple, Optional
from src.constants import MAX_MARKET_ORDERS_PER_TURN, CROPS, PRODUCTS
from src.strategy import calculate_hire_cost, get_season_phase

class PriceTracker:
    """Tracks price history across turns and calculates price trends, momentum, and peak windows."""

    def __init__(self, history_len: int = 48):
        self.history_len = history_len
        self.price_history: Dict[str, List[float]] = {}

    def update(self, current_prices: Dict[str, float]) -> None:
        """Update price history with current turn prices."""
        for prod, price in current_prices.items():
            if prod not in self.price_history:
                self.price_history[prod] = []
            self.price_history[prod].append(float(price))
            if len(self.price_history[prod]) > self.history_len:
                self.price_history[prod].pop(0)

    def get_moving_average(self, product: str, window: int = 12) -> float:
        """Calculate moving average price over given window."""
        hist = self.price_history.get(product, [])
        if not hist:
            return 0.0
        sample = hist[-window:]
        return sum(sample) / len(sample)

    def get_price_momentum(self, product: str, window: int = 6) -> float:
        """Calculate short-term price momentum (slope/delta).
        
        Returns positive value if price is trending upwards, negative if decaying/falling.
        """
        hist = self.price_history.get(product, [])
        if len(hist) < 2:
            return 0.0
        sample = hist[-window:]
        return sample[-1] - sample[0]

    def is_price_peak(self, product: str, current_price: float) -> bool:
        """Check if current price is at or near its recent moving peak."""
        hist = self.price_history.get(product, [])
        if not hist:
            return True
        max_recent = max(hist)
        return current_price >= max_recent * 0.95

    def get_dynamic_lot_size(self, product: str, current_price: float, base_lot: int = 15, floor_price: float = 0.0) -> int:
        """Determine dynamic sell lot size based on price momentum and price level relative to floor."""
        if current_price < floor_price:
            return 0  # Do not sell below floor price

        momentum = self.get_price_momentum(product)

        # Price is rising: sell smaller lot to allow price to climb further
        if momentum > 1.0:
            return max(5, base_lot // 2)

        # Price is at peak: sell full lot size
        if self.is_price_peak(product, current_price):
            return base_lot

        # Price is decaying/falling: reduce lot size to avoid accelerating price collapse
        if momentum < -1.0:
            return max(5, base_lot // 3)

        return base_lot

class MarketOptimizer:
    """Encapsulates market order optimization and queuing logic."""

    def __init__(self, policy: Optional[Dict[str, Any]] = None):
        self.policy = policy if policy is not None else {}
        self.price_tracker = PriceTracker(history_len=48)

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

        # Update price history tracker
        self.price_tracker.update(prices)

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

            if phase["liquidating"]:
                # End-of-season liquidation: sell all remaining stock in lots
                max_lot = sell_lots.get(product, 20)
                qty = min(in_shed, max_lot)
                sell_orders.append(["SELL", product, qty])
            elif cur_price >= floor:
                # Dynamic sell timing based on price momentum
                base_lot = sell_lots.get(product, 15)
                dynamic_lot = self.price_tracker.get_dynamic_lot_size(product, cur_price, base_lot=base_lot, floor_price=floor)
                if dynamic_lot > 0:
                    qty = min(in_shed, dynamic_lot)
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
