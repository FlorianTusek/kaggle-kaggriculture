# SPDX-License-Identifier: MIT
"""Kaggriculture Self-Contained Phase 2 Competition Agent."""

import math
from typing import Dict, List, Any, Tuple, Optional

# --- Constants ---
BOARD = 10
HALF = BOARD // 2
TURNS_PER_DAY = 24
TOTAL_DAYS = 30
SHED_CAP = 100
MAX_MARKET_ORDERS = 10

SHED_TILES = [(HALF - 1, HALF - 1), (HALF, HALF - 1), (HALF - 1, HALF), (HALF, HALF)]

CROPS = {
    "WHEAT":      {"seed": 10,  "base_price": 25,  "first": 2,  "max_day": 4,  "interval": 0, "cap": 6, "ongoing": False},
    "CARROT":     {"seed": 20,  "base_price": 35,  "first": 2,  "max_day": 3,  "interval": 0, "cap": 4, "ongoing": False},
    "TOMATO":     {"seed": 50,  "base_price": 60,  "first": 8,  "max_day": 8,  "interval": 1, "cap": 4, "ongoing": True},
    "STRAWBERRY": {"seed": 100, "base_price": 120, "first": 10, "max_day": 10, "interval": 2, "cap": 4, "ongoing": True},
    "MELON":      {"seed": 80,  "base_price": 250, "first": 10, "max_day": 12, "interval": 0, "cap": 6, "ongoing": False},
}

ANIMALS = {
    "GOOSE": {"cost": 300, "structure": "COOP",    "first": 4, "interval": 1, "cap": 4, "product": "EGG"},
    "COW":   {"cost": 400, "structure": "PASTURE", "first": 8, "interval": 2, "cap": 6, "product": "MILK"},
    "SHEEP": {"cost": 500, "structure": "PASTURE", "first": 6, "interval": 3, "cap": 6, "product": "WOOL"},
}

PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]
MOVE_OP = {(0, -1): "NORTH", (0, 1): "SOUTH", (1, 0): "EAST", (-1, 0): "WEST"}

DEFAULT_POLICY = {
    'hands': 4,
    'crops': ['CARROT', 'TOMATO', 'WHEAT', 'STRAWBERRY', 'MELON'],
    'crop_share': {'CARROT': 0.25, 'TOMATO': 0.25, 'WHEAT': 0.20, 'STRAWBERRY': 0.15, 'MELON': 0.15},
    'harvest_asap': False,
    'seed_batch': 6,
    'seed_stock': 12,
    'sell_order': ['MELON', 'STRAWBERRY', 'MILK', 'WOOL', 'EGG', 'TOMATO', 'CARROT', 'WHEAT', 'FERTILIZER'],
    'sell_lots': {'MELON': 5, 'STRAWBERRY': 10, 'MILK': 10, 'WOOL': 10, 'EGG': 15, 'TOMATO': 10, 'CARROT': 15, 'WHEAT': 20, 'FERTILIZER': 20},
    'price_floors': {'MELON': 150, 'STRAWBERRY': 80, 'MILK': 80, 'WOOL': 100, 'EGG': 30, 'TOMATO': 35, 'CARROT': 20, 'WHEAT': 10, 'FERTILIZER': 5},
    'plant_until_day': 25,
    'liquidate_from_day': 27,
    'carry': 6,
}

# --- Geometry & Math Helpers ---
def fibonacci(n: int) -> int:
    if n <= 0:
        return 1
    a, b = 1, 1
    for _ in range(n):
        a, b = b, a + b
    return a

def calculate_hire_cost(current_hires_today: int, n_additional: int = 1) -> int:
    total_cost = 0
    for i in range(n_additional):
        total_cost += fibonacci(current_hires_today + i)
    return total_cost

def get_season_phase(day: int, policy: Dict[str, Any]) -> Dict[str, bool]:
    return {
        "investing": day <= policy.get("invest_until_day", 24),
        "planting": day <= policy.get("plant_until_day", 25),
        "liquidating": day >= policy.get("liquidate_from_day", 27),
    }

def _quadrant(x: int, y: int) -> str:
    return ("N" if y < HALF else "S") + ("W" if x < HALF else "E")

def _dist(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def _step_toward(pos: Tuple[int, int], target: Tuple[int, int]) -> List[str]:
    px, py = pos
    tx, ty = target
    if px != tx:
        return [MOVE_OP[(1, 0)] if tx > px else MOVE_OP[(-1, 0)]]
    if py != ty:
        return [MOVE_OP[(0, 1)] if ty > py else MOVE_OP[(0, -1)]]
    return ["PASS"]

def _shed_tile(tiles: List[List[Any]]) -> Tuple[int, int]:
    if not tiles:
        return SHED_TILES[0]
    for (x, y) in SHED_TILES:
        if y < len(tiles) and x < len(tiles[y]) and tiles[y][x] != "LOCKED":
            return (x, y)
    return SHED_TILES[0]

def _open_tiles(tiles: List[List[Any]]) -> List[Tuple[int, int]]:
    if not tiles:
        return []
    sx, sy = _shed_tile(tiles)
    out = []
    for y in range(min(BOARD, len(tiles))):
        for x in range(min(BOARD, len(tiles[y]))):
            if tiles[y][x] is None:
                out.append((x, y))
    out.sort(key=lambda p: (_dist(p, (sx, sy)), p[1], p[0]))
    return out

# --- Safety Layer ---
def is_crop_ready_to_harvest(tile: Dict[str, Any], day: int, harvest_asap: bool = False) -> bool:
    if not isinstance(tile, dict) or tile.get("kind") != "PLANT":
        return False
    crop = tile.get("crop")
    if crop not in CROPS:
        return False
    info = CROPS[crop]
    planted_day = tile.get("planted_day", 0)
    age = day - planted_day
    yield_units = tile.get("yield_units", 0)
    if age < info["first"]:
        return False
    if yield_units <= 0:
        return False
    if info["ongoing"]:
        return True
    if harvest_asap:
        return True
    if age >= info["max_day"]:
        return True
    return yield_units >= min(info["cap"], 3)

def is_animal_unfed_emergency(tile: Dict[str, Any]) -> bool:
    if not isinstance(tile, dict) or tile.get("kind") not in ("COOP", "PASTURE"):
        return False
    if not tile.get("animal"):
        return False
    return not tile.get("fed_today", False) and tile.get("consecutive_unfed", 0) >= 1

def is_crop_unwatered_emergency(tile: Dict[str, Any]) -> bool:
    if not isinstance(tile, dict) or tile.get("kind") != "PLANT":
        return False
    return not tile.get("watered_today", False) and tile.get("consecutive_unwatered", 0) >= 1

def collect_safety_jobs(obs: Dict[str, Any], me: Dict[str, Any], priv: Dict[str, Any], policy: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    if policy is None:
        policy = {}
    day = obs.get("day", 0)
    tiles = me.get("tiles", [])
    harvest_asap = policy.get("harvest_asap", False)

    emergency_feed, emergency_water, harvest_jobs, routine_feed = [], [], [], []
    care_jobs, routine_water, fertilizer_jobs, dig_weeds = [], [], [], []

    for y in range(len(tiles)):
        for x in range(len(tiles[y])):
            tile = tiles[y][x]
            if not isinstance(tile, dict):
                continue
            kind = tile.get("kind")
            pos = (x, y)

            if kind == "WEED":
                dig_weeds.append({"pos": pos, "op": ["DIG"], "need": None, "priority": 8})
            elif kind == "PLANT":
                if is_crop_ready_to_harvest(tile, day, harvest_asap):
                    harvest_jobs.append({"pos": pos, "op": ["HARVEST"], "need": None, "priority": 3})
                elif is_crop_unwatered_emergency(tile):
                    emergency_water.append({"pos": pos, "op": ["WATER"], "need": None, "priority": 2})
                elif not tile.get("watered_today", False):
                    routine_water.append({"pos": pos, "op": ["WATER"], "need": None, "priority": 6})
            elif kind in ("COOP", "PASTURE") and tile.get("animal"):
                if tile.get("yield_units", 0) > 0:
                    harvest_jobs.append({"pos": pos, "op": ["HARVEST"], "need": None, "priority": 3})
                if is_animal_unfed_emergency(tile):
                    emergency_feed.append({"pos": pos, "op": ["FEED"], "need": ("WHEAT", 1), "priority": 1})
                elif not tile.get("fed_today", False):
                    routine_feed.append({"pos": pos, "op": ["FEED"], "need": ("WHEAT", 1), "priority": 4})
                if not tile.get("cared_today", False):
                    care_jobs.append({"pos": pos, "op": ["CARE"], "need": None, "priority": 5})
                if tile.get("fertilizer_available", 0) > 0:
                    fertilizer_jobs.append({"pos": pos, "op": ["COLLECT_FERTILIZER"], "need": None, "priority": 7})

    return emergency_feed + emergency_water + harvest_jobs + routine_feed + care_jobs + routine_water + fertilizer_jobs + dig_weeds

class SafetyLayer:
    def __init__(self, policy: Optional[Dict[str, Any]] = None):
        self.policy = policy if policy is not None else {}
    def get_jobs(self, obs: Dict[str, Any], me: Dict[str, Any], priv: Dict[str, Any]) -> List[Dict[str, Any]]:
        return collect_safety_jobs(obs, me, priv, self.policy)

TOWN_SHOP_DEMAND_BOOST = {
    "Bakery": {"WHEAT": 1.3},
    "Pizza Shop": {"WHEAT": 1.2, "TOMATO": 1.5},
    "Brunch Spot": {"WHEAT": 1.2, "STRAWBERRY": 1.8},
    "Pet Cafe": {"CARROT": 2.0},
    "Smoothie Shop": {"STRAWBERRY": 1.8},
    "Farmers Market": {"WHEAT": 1.2, "CARROT": 1.4, "TOMATO": 1.4, "STRAWBERRY": 1.4},
}

def compute_demand_responsive_shares(obs: Dict[str, Any], prices: Dict[str, float], base_shares: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    if base_shares is None:
        base_shares = {"CARROT": 0.4, "TOMATO": 0.3, "WHEAT": 0.3}

    scores = {}
    unlocked_shops = obs.get("town", {}).get("unlocked_shops", [])

    for crop, base_weight in base_shares.items():
        base_price = CROPS.get(crop, {}).get("base_price", 30)
        current_price = prices.get(crop, base_price)
        price_mult = max(0.5, current_price / max(1.0, base_price))

        town_mult = 1.0
        for shop in unlocked_shops:
            if shop in TOWN_SHOP_DEMAND_BOOST and crop in TOWN_SHOP_DEMAND_BOOST[shop]:
                town_mult *= TOWN_SHOP_DEMAND_BOOST[shop][crop]

        scores[crop] = base_weight * price_mult * town_mult

    total_score = sum(scores.values())
    if total_score <= 0:
        return base_shares

    return {crop: score / total_score for crop, score in scores.items()}

# --- Strategy Planner ---
class StrategyPlanner:
    def __init__(self, policy: Optional[Dict[str, Any]] = None):
        self.policy = policy if policy is not None else {}

    def get_target_hands(self, obs: Dict[str, Any], me: Dict[str, Any]) -> int:
        day = obs.get("day", 0)
        money = me.get("money", 0)
        phase = get_season_phase(day, self.policy)
        if not phase["investing"]:
            return 0
        base_hands = self.policy.get("hands", 4)
        cost_for_base = calculate_hire_cost(0, base_hands)
        if money < cost_for_base + 100:
            return max(1, base_hands - 2)
        return base_hands

    def evaluate_land_purchase(self, obs: Dict[str, Any], me: Dict[str, Any]) -> Optional[str]:
        day = obs.get("day", 0)
        money = me.get("money", 0)
        unlocked = me.get("unlocked_quadrants", ["NW"])
        if day > 20:
            return None
        tiles = me.get("tiles", [])
        free_tiles_cnt = len(_open_tiles(tiles))
        quad_order = [("NE", 1000), ("SW", 2000), ("SE", 4000)]
        for quad, cost in quad_order:
            if quad not in unlocked:
                if money >= cost * 1.3 or (free_tiles_cnt <= 4 and money >= cost + 200):
                    return quad
                break
        return None

    def plan_structure_building_jobs(self, obs: Dict[str, Any], me: Dict[str, Any], priv: Dict[str, Any], free_tiles: List[Tuple[int, int]]) -> List[Dict[str, Any]]:
        day = obs.get("day", 0)
        money = me.get("money", 0)
        phase = get_season_phase(day, self.policy)
        if not phase["investing"] or day > 18 or not free_tiles:
            return []
        tiles = me.get("tiles", [])
        n_structures = 0
        for row in tiles:
            for t in row:
                if isinstance(t, dict) and t.get("kind") in ("COOP", "PASTURE"):
                    n_structures += 1
        if n_structures >= 4:
            return []
        if money >= 600:
            pos = free_tiles[0]
            op = ["BUILD_PASTURE"] if n_structures % 2 == 0 else ["BUILD_COOP"]
            return [{"pos": pos, "op": op, "need": None, "priority": 8.5}]
        return []

    def plan_planting_jobs(self, obs: Dict[str, Any], me: Dict[str, Any], priv: Dict[str, Any], free_tiles: List[Tuple[int, int]]) -> List[Dict[str, Any]]:
        day = obs.get("day", 0)
        seeds = priv.get("seeds", {})
        phase = get_season_phase(day, self.policy)
        if not phase["planting"] or not free_tiles:
            return []
        base_shares = self.policy.get("crop_share", {"CARROT": 0.25, "TOMATO": 0.25, "WHEAT": 0.20, "STRAWBERRY": 0.15, "MELON": 0.15})
        prices = obs.get("market", {}).get("prices", {})
        dynamic_shares = compute_demand_responsive_shares(obs, prices, base_shares)
        crops_ordered = sorted(dynamic_shares.keys(), key=lambda c: dynamic_shares[c], reverse=True)
        available_seeds = []
        for crop in crops_ordered:
            cnt = seeds.get(crop, 0)
            if cnt > 0:
                available_seeds.extend([crop] * cnt)
        plant_jobs = []
        for pos in free_tiles:
            if not available_seeds:
                break
            crop = available_seeds.pop(0)
            plant_jobs.append({"pos": pos, "op": ["PLANT", crop], "need": None, "priority": 9})
        return plant_jobs

# --- Price Tracker ---
class PriceTracker:
    def __init__(self, history_len: int = 48):
        self.history_len = history_len
        self.price_history: Dict[str, List[float]] = {}

    def update(self, current_prices: Dict[str, float]) -> None:
        for prod, price in current_prices.items():
            if prod not in self.price_history:
                self.price_history[prod] = []
            self.price_history[prod].append(float(price))
            if len(self.price_history[prod]) > self.history_len:
                self.price_history[prod].pop(0)

    def get_moving_average(self, product: str, window: int = 12) -> float:
        hist = self.price_history.get(product, [])
        if not hist:
            return 0.0
        sample = hist[-window:]
        return sum(sample) / len(sample)

    def get_price_momentum(self, product: str, window: int = 6) -> float:
        hist = self.price_history.get(product, [])
        if len(hist) < 2:
            return 0.0
        sample = hist[-window:]
        return sample[-1] - sample[0]

    def is_price_peak(self, product: str, current_price: float) -> bool:
        hist = self.price_history.get(product, [])
        if not hist:
            return True
        max_recent = max(hist)
        return current_price >= max_recent * 0.95

    def get_dynamic_lot_size(self, product: str, current_price: float, base_lot: int = 15, floor_price: float = 0.0) -> int:
        if current_price < floor_price:
            return 0
        momentum = self.get_price_momentum(product)
        if momentum > 1.0:
            return max(5, base_lot // 2)
        if self.is_price_peak(product, current_price):
            return base_lot
        if momentum < -1.0:
            return max(5, base_lot // 3)
        return base_lot

# --- Opponent Tracker ---
class OpponentTracker:
    def __init__(self):
        self.history: List[Dict[str, Any]] = []
        self.archetype = "UNKNOWN"
        self.wheat_denial_detected = False

    def update(self, obs: Dict[str, Any]) -> None:
        player_idx = obs.get("player", 0)
        opp_idx = 1 - player_idx
        farms = obs.get("farms", [])
        if len(farms) <= opp_idx:
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

        if len(self.history) >= 2:
            initial_wheat = self.history[0]["wheat_inventory"]
            current_wheat = snapshot["wheat_inventory"]
            if snapshot["turn"] <= 48 and (initial_wheat - current_wheat >= 15 or snapshot["wheat_price"] > 35):
                self.wheat_denial_detected = True

        if self.wheat_denial_detected:
            self.archetype = "WHEAT_DENIER"
        elif snapshot["quadrants"] > 1 or snapshot["hires"] >= 3:
            self.archetype = "AGGRESSIVE_EXPANDER"
        else:
            self.archetype = "STANDARD"

    def is_feed5_first_needed(self, turn: int) -> bool:
        return turn == 0

    def get_counter_strategy_orders(self, obs: Dict[str, Any], me: Dict[str, Any], priv: Dict[str, Any]) -> List[List[Any]]:
        turn = obs.get("step", 0)
        money = me.get("money", 0)
        seeds = priv.get("seeds", {})
        shed = priv.get("shed", {})
        wheat_stock = seeds.get("WHEAT", 0) + shed.get("WHEAT", 0)

        orders = []
        if turn == 0 and wheat_stock < 5 and money >= 50:
            orders.append(["BUY_SEED", "WHEAT", 5])
        elif self.wheat_denial_detected and wheat_stock < 10 and money >= 100:
            orders.append(["BUY_SEED", "WHEAT", 5])

        return orders

# --- Market Optimizer ---
class MarketOptimizer:
    def __init__(self, policy: Optional[Dict[str, Any]] = None):
        self.policy = policy if policy is not None else {}
        self.price_tracker = PriceTracker(history_len=48)
        self.opponent_tracker = OpponentTracker()
        self.strategy_planner = StrategyPlanner(self.policy)

    def plan_market_orders(self, obs: Dict[str, Any], me: Dict[str, Any], priv: Dict[str, Any]) -> List[List[Any]]:
        day = obs.get("day", 0)
        hour = obs.get("hour", 0)
        money = me.get("money", 0)
        hires_today = me.get("hires_today", 0)
        seeds = priv.get("seeds", {})
        shed = priv.get("shed", {})
        prices = obs.get("market", {}).get("prices", {})
        phase = get_season_phase(day, self.policy)

        self.price_tracker.update(prices)
        self.opponent_tracker.update(obs)

        counter_orders = self.opponent_tracker.get_counter_strategy_orders(obs, me, priv)
        land_orders, animal_orders, sell_orders, buy_orders, hire_orders = [], [], [], [], []

        quad_to_buy = self.strategy_planner.evaluate_land_purchase(obs, me)
        if quad_to_buy:
            land_orders.append(["BUY_LAND", quad_to_buy])

        tiles = me.get("tiles", [])
        for row in tiles:
            for tile in row:
                if isinstance(tile, dict):
                    kind = tile.get("kind")
                    animal = tile.get("animal")
                    if kind == "COOP" and not animal and money >= 300:
                        animal_orders.append(["BUY_ANIMAL", "GOOSE"])
                        money -= 300
                    elif kind == "PASTURE" and not animal:
                        if money >= 500:
                            animal_orders.append(["BUY_ANIMAL", "SHEEP"])
                            money -= 500
                        elif money >= 400:
                            animal_orders.append(["BUY_ANIMAL", "COW"])
                            money -= 400

        sell_order_list = self.policy.get("sell_order", ["MELON", "STRAWBERRY", "MILK", "WOOL", "EGG", "TOMATO", "CARROT", "WHEAT", "FERTILIZER"])
        sell_lots = self.policy.get("sell_lots", {"MELON": 5, "STRAWBERRY": 10, "MILK": 10, "WOOL": 10, "EGG": 15, "TOMATO": 10, "CARROT": 15, "WHEAT": 20, "FERTILIZER": 20})
        floors = self.policy.get("price_floors", {"MELON": 150, "STRAWBERRY": 80, "MILK": 80, "WOOL": 100, "EGG": 30, "TOMATO": 35, "CARROT": 20, "WHEAT": 10, "FERTILIZER": 5})

        for product in sell_order_list:
            in_shed = shed.get(product, 0)
            if in_shed <= 0:
                continue
            cur_price = prices.get(product, 0)
            floor = floors.get(product, 0)
            if phase["liquidating"]:
                max_lot = sell_lots.get(product, 20)
                qty = min(in_shed, max_lot)
                sell_orders.append(["SELL", product, qty])
            elif cur_price >= floor:
                base_lot = sell_lots.get(product, 15)
                dynamic_lot = self.price_tracker.get_dynamic_lot_size(product, cur_price, base_lot=base_lot, floor_price=floor)
                if dynamic_lot > 0:
                    qty = min(in_shed, dynamic_lot)
                    sell_orders.append(["SELL", product, qty])

        # 2. HIRE Orders
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
            base_shares = self.policy.get("crop_share", {"CARROT": 0.25, "TOMATO": 0.25, "WHEAT": 0.20, "STRAWBERRY": 0.15, "MELON": 0.15})
            crop_shares = compute_demand_responsive_shares(obs, prices, base_shares)

            for crop, share in crop_shares.items():
                if crop not in CROPS:
                    continue
                current_stock = seeds.get(crop, 0)
                desired_stock = max(2, int(stock_target * share))
                seed_cost = CROPS[crop]["seed"] * batch
                if current_stock < desired_stock and money >= seed_cost:
                    buy_orders.append(["BUY_SEED", crop, batch])
                    money -= seed_cost

        combined = counter_orders + land_orders + animal_orders + sell_orders + hire_orders + buy_orders
        return combined[:MAX_MARKET_ORDERS]

# --- Main Agent Controller & Entrypoint ---
def _assign_worker_ops(obs: Dict[str, Any], policy: Dict[str, Any], me: Dict[str, Any], priv: Dict[str, Any], jobs: List[Dict[str, Any]], ml_advice: Optional[Dict[str, Any]] = None) -> List[List[str]]:
    positions = [me["farmer"]] + me.get("hands", [])
    n = len(positions)
    ops = [["PASS"] for _ in range(n)]
    busy = [False] * n
    tiles = me["tiles"]

    for job in jobs:
        best = None
        for i in range(n):
            if busy[i]:
                continue
            d = _dist(positions[i], job["pos"])
            if best is None or d < best[0]:
                best = (d, i)
        if best is None:
            continue
        d, i = best
        busy[i] = True
        ops[i] = job["op"] if d == 0 else _step_toward(positions[i], job["pos"])

    sx, sy = _shed_tile(tiles)
    for i in range(n):
        if not busy[i]:
            if positions[i] == (sx, sy):
                ops[i] = ["DROP"]
            elif obs.get("hour", 0) == TURNS_PER_DAY - 1:
                ops[i] = _step_toward(positions[i], (sx, sy))
            elif ml_advice and i == 0 and ml_advice.get("recommended_farmer_action"):
                ml_act = ml_advice["recommended_farmer_action"]
                if ml_act in ("NORTH", "SOUTH", "EAST", "WEST", "DROP", "DIG", "HARVEST", "WATER"):
                    ops[i] = [ml_act]

    return ops

class KaggricultureAgent:
    def __init__(self, policy: Optional[Dict[str, Any]] = None, model_path: Optional[str] = None):
        self.policy = policy if policy is not None else DEFAULT_POLICY
        self.safety_layer = SafetyLayer(self.policy)
        self.strategy_planner = StrategyPlanner(self.policy)
        self.market_optimizer = MarketOptimizer(self.policy)
        
        self.bc_policy = None
        if self.policy.get("use_ml_policy", True):
            try:
                from src.models import PPOPolicy, BehavioralCloningPolicy
                ppo = PPOPolicy()
                if ppo.is_loaded:
                    self.bc_policy = ppo
                else:
                    self.bc_policy = BehavioralCloningPolicy(model_path=model_path)
            except Exception:
                self.bc_policy = None

    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        me = obs["farms"][obs["player"]]
        priv = obs["private"]
        tiles = me["tiles"]

        ml_advice = None
        if self.bc_policy is not None and self.bc_policy.is_loaded:
            try:
                ml_advice = self.bc_policy.advise(obs)
            except Exception:
                ml_advice = None

        safety_jobs = self.safety_layer.get_jobs(obs, me, priv)
        free_tiles = _open_tiles(tiles)
        struct_jobs = self.strategy_planner.plan_structure_building_jobs(obs, me, priv, free_tiles)
        plant_jobs = self.strategy_planner.plan_planting_jobs(obs, me, priv, free_tiles)

        all_jobs = safety_jobs + struct_jobs + plant_jobs
        worker_ops = _assign_worker_ops(obs, self.policy, me, priv, all_jobs, ml_advice=ml_advice)
        market_orders = self.market_optimizer.plan_market_orders(obs, me, priv)

        return {
            "farmer": worker_ops[0],
            "hands": worker_ops[1:],
            "market": market_orders
        }

def agent(obs, config=None):
    """Main submission agent entrypoint required by Kaggle Competition."""
    agent_inst = KaggricultureAgent()
    return agent_inst.act(obs)

if __name__ == "__main__":
    print("Self-contained Kaggriculture main.py loaded successfully.")
