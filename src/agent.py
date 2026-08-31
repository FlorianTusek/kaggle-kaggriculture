# SPDX-License-Identifier: MIT
"""Kaggriculture Baseline Agent Policy Implementation."""

import math
from typing import Dict, List, Any, Tuple, Optional

BOARD = 10
HALF = BOARD // 2
TURNS_PER_DAY = 24
SHED_CAP = 100
MAX_MARKET_ORDERS = 10

SHED_TILES = [(HALF - 1, HALF - 1), (HALF, HALF - 1), (HALF - 1, HALF), (HALF, HALF)]

CROPS = {
    "WHEAT":      {"seed": 10,  "first": 2,  "max_day": 4,  "interval": 0, "cap": 6, "ongoing": False},
    "CARROT":     {"seed": 20,  "first": 2,  "max_day": 3,  "interval": 0, "cap": 4, "ongoing": False},
    "TOMATO":     {"seed": 50,  "first": 8,  "max_day": 8,  "interval": 1, "cap": 4, "ongoing": True},
    "STRAWBERRY": {"seed": 100, "first": 10, "max_day": 10, "interval": 2, "cap": 4, "ongoing": True},
    "MELON":      {"seed": 80,  "first": 10, "max_day": 12, "interval": 0, "cap": 6, "ongoing": False},
}

MOVE_OP = {(0, -1): "NORTH", (0, 1): "SOUTH", (1, 0): "EAST", (-1, 0): "WEST"}

DEFAULT_POLICY = {
    'hands': 4,
    'crops': ['CARROT', 'TOMATO', 'WHEAT'],
    'crop_share': {'CARROT': 0.4, 'TOMATO': 0.3, 'WHEAT': 0.3},
    'harvest_asap': False,
    'seed_batch': 6,
    'seed_stock': 12,
    'sell_order': ['CARROT', 'TOMATO', 'WHEAT', 'STRAWBERRY', 'MELON'],
    'sell_lots': {'CARROT': 15, 'TOMATO': 10, 'WHEAT': 20, 'MELON': 10},
    'price_floors': {'CARROT': 10, 'TOMATO': 20, 'WHEAT': 5, 'MELON': 100},
    'plant_until_day': 25,
    'liquidate_from_day': 27,
    'carry': 6,
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
    for (x, y) in SHED_TILES:
        if tiles[y][x] != "LOCKED":
            return (x, y)
    return SHED_TILES[0]

def _open_tiles(tiles: List[List[Any]]) -> List[Tuple[int, int]]:
    sx, sy = _shed_tile(tiles)
    out = []
    for y in range(BOARD):
        for x in range(BOARD):
            if tiles[y][x] is None:
                out.append((x, y))
    out.sort(key=lambda p: (_dist(p, (sx, sy)), p[1], p[0]))
    return out

def _ready_to_harvest(tile: Dict[str, Any], day: int, policy: Dict[str, Any]) -> bool:
    crop = tile["crop"]
    info = CROPS[crop]
    age = day - tile["planted_day"]
    if age < info["first"]:
        return False
    if tile.get("yield_units", 0) <= 0:
        return False
    if info["ongoing"]:
        return True
    if policy.get("harvest_asap", False):
        return True
    if age >= info["max_day"]:
        return True
    return tile.get("yield_units", 0) >= 3

def _collect_jobs(obs: Dict[str, Any], policy: Dict[str, Any], me: Dict[str, Any], priv: Dict[str, Any]):
    day = obs["day"]
    tiles = me["tiles"]
    seeds = priv["seeds"]
    
    water, harvest, plant, dig = [], [], [], []

    for y in range(BOARD):
        for x in range(BOARD):
            tile = tiles[y][x]
            if not isinstance(tile, dict):
                continue
            kind = tile.get("kind")
            pos = (x, y)

            if kind == "WEED":
                dig.append({"pos": pos, "op": ["DIG"], "need": None})
            elif kind == "PLANT":
                if _ready_to_harvest(tile, day, policy):
                    harvest.append({"pos": pos, "op": ["HARVEST"], "need": None})
                elif not tile.get("watered_today", False):
                    water.append({"pos": pos, "op": ["WATER"], "need": None})

    # Plant jobs
    planting_allowed = day <= policy.get("plant_until_day", 25)
    if planting_allowed:
        free_tiles = _open_tiles(tiles)
        available_seeds = []
        for crop in policy.get("crops", ["WHEAT"]):
            cnt = seeds.get(crop, 0)
            if cnt > 0:
                available_seeds.extend([crop] * cnt)

        for pos in free_tiles:
            if not available_seeds:
                break
            crop = available_seeds.pop(0)
            plant.append({"pos": pos, "op": ["PLANT", crop], "need": None})

    jobs = water + harvest + plant + dig
    return jobs, 0

def _plan_market(obs: Dict[str, Any], policy: Dict[str, Any], me: Dict[str, Any], priv: Dict[str, Any], animal_count: int) -> List[List[Any]]:
    day = obs["day"]
    money = me["money"]
    seeds = priv["seeds"]
    shed = priv["shed"]
    prices = obs["market"]["prices"]

    orders = []

    # 1. HIRE hands
    target_hands = policy.get("hands", 4)
    hires_today = me.get("hires_today", 0)
    if obs["hour"] == 0 and hires_today < target_hands:
        needed = target_hands - hires_today
        for _ in range(needed):
            orders.append(["HIRE"])

    # 2. BUY_SEED
    if day <= policy.get("plant_until_day", 25):
        stock_target = policy.get("seed_stock", 8)
        batch = policy.get("seed_batch", 4)
        crop_shares = policy.get("crop_share", {"WHEAT": 1.0})
        
        for crop, share in crop_shares.items():
            current_stock = seeds.get(crop, 0)
            desired_stock = int(stock_target * share)
            if current_stock < desired_stock and money >= CROPS[crop]["seed"] * batch:
                orders.append(["BUY_SEED", crop, batch])
                money -= CROPS[crop]["seed"] * batch

    # 3. SELL produce
    sell_order = policy.get("sell_order", ["CARROT", "TOMATO", "WHEAT", "MELON"])
    sell_lots = policy.get("sell_lots", {})
    floors = policy.get("price_floors", {})

    for product in sell_order:
        in_shed = shed.get(product, 0)
        if in_shed <= 0:
            continue
        cur_price = prices.get(product, 0)
        floor = floors.get(product, 0)

        # Force liquidation in final days
        if day >= policy.get("liquidate_from_day", 27) or cur_price >= floor:
            max_lot = sell_lots.get(product, 20)
            qty = min(in_shed, max_lot)
            orders.append(["SELL", product, qty])

    return orders[:MAX_MARKET_ORDERS]

def _assign(obs: Dict[str, Any], policy: Dict[str, Any], me: Dict[str, Any], priv: Dict[str, Any], jobs: List[Dict[str, Any]], animal_count: int) -> List[List[str]]:
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

    # Idle workers drop inventory at shed if adjacent
    sx, sy = _shed_tile(tiles)
    for i in range(n):
        if not busy[i]:
            if positions[i] == (sx, sy):
                ops[i] = ["DROP"]
            elif obs["hour"] == TURNS_PER_DAY - 1:
                ops[i] = _step_toward(positions[i], (sx, sy))

    return ops

class KaggricultureAgent:
    """Modular Kaggriculture Baseline Agent."""

    def __init__(self, policy: Optional[Dict[str, Any]] = None):
        self.policy = policy if policy is not None else DEFAULT_POLICY

    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        me = obs["farms"][obs["player"]]
        priv = obs["private"]
        jobs, animal_count = _collect_jobs(obs, self.policy, me, priv)
        market = _plan_market(obs, self.policy, me, priv, animal_count)
        ops = _assign(obs, self.policy, me, priv, jobs, animal_count)
        return {"farmer": ops[0], "hands": ops[1:], "market": market}

def agent_entrypoint(obs: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Kaggle environment entrypoint function."""
    agent_inst = KaggricultureAgent()
    return agent_inst.act(obs)
