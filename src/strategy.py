# SPDX-License-Identifier: MIT
"""Kaggriculture Strategy Planner Module.

Manages high-level farm decisions:
- Fibonacci labor hiring calculation.
- Land expansion evaluation (unlocking NE, SW, SE quadrants when capital reserves allow).
- Demand-responsive crop selection (balancing crop mix based on market price signals and town shop unlocks).
- Season lifecycle gating (investing, planting, liquidation phases).
"""

from typing import Dict, List, Any, Tuple, Optional
from src.constants import BOARD_SIZE, HALF_BOARD, CROPS, LAND_PRICES, TOTAL_DAYS

MOVE_OP = {(0, -1): "NORTH", (0, 1): "SOUTH", (1, 0): "EAST", (-1, 0): "WEST"}
SHED_TILES = [(HALF_BOARD - 1, HALF_BOARD - 1), (HALF_BOARD, HALF_BOARD - 1), (HALF_BOARD - 1, HALF_BOARD), (HALF_BOARD, HALF_BOARD)]

TOWN_SHOP_DEMAND_BOOST = {
    "Bakery": {"WHEAT": 1.3},
    "Pizza Shop": {"WHEAT": 1.2, "TOMATO": 1.5},
    "Brunch Spot": {"WHEAT": 1.2, "STRAWBERRY": 1.8},
    "Pet Cafe": {"CARROT": 2.0},
    "Smoothie Shop": {"STRAWBERRY": 1.8},
    "Farmers Market": {"WHEAT": 1.2, "CARROT": 1.4, "TOMATO": 1.4, "STRAWBERRY": 1.4},
}

def fibonacci(n: int) -> int:
    """Return the n-th Fibonacci number (1, 1, 2, 3, 5, 8, 13, 21...)."""
    if n <= 0:
        return 1
    a, b = 1, 1
    for _ in range(n):
        a, b = b, a + b
    return a

def calculate_hire_cost(current_hires_today: int, n_additional: int = 1) -> int:
    """Calculate total coin cost of hiring n_additional farm hands today."""
    total_cost = 0
    for i in range(n_additional):
        total_cost += fibonacci(current_hires_today + i)
    return total_cost

def get_season_phase(day: int, policy: Dict[str, Any]) -> Dict[str, bool]:
    """Determine active season phase gates."""
    return {
        "investing": day <= policy.get("invest_until_day", 24),
        "planting": day <= policy.get("plant_until_day", 25),
        "liquidating": day >= policy.get("liquidate_from_day", 27),
    }

def compute_demand_responsive_shares(obs: Dict[str, Any], prices: Dict[str, float], base_shares: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """Dynamically adjust crop planting shares based on market price signals and unlocked town shop demand."""
    if base_shares is None:
        if prices and set(prices.keys()) == {"CARROT", "TOMATO", "WHEAT"}:
            base_shares = {"CARROT": 0.4, "TOMATO": 0.3, "WHEAT": 0.3}
        else:
            base_shares = {"CARROT": 0.25, "TOMATO": 0.25, "WHEAT": 0.20, "STRAWBERRY": 0.15, "MELON": 0.15}

    scores = {}
    unlocked_shops = obs.get("town", {}).get("unlocked_shops", [])

    for crop, base_weight in base_shares.items():
        base_price = CROPS.get(crop, {}).get("base_price", 30)
        current_price = prices.get(crop, base_price)

        # Price multiplier relative to base price
        price_mult = max(0.5, current_price / max(1.0, base_price))

        # Town shop demand multiplier
        town_mult = 1.0
        for shop in unlocked_shops:
            if shop in TOWN_SHOP_DEMAND_BOOST and crop in TOWN_SHOP_DEMAND_BOOST[shop]:
                town_mult *= TOWN_SHOP_DEMAND_BOOST[shop][crop]

        scores[crop] = base_weight * price_mult * town_mult

    # Normalize shares to sum to 1.0
    total_score = sum(scores.values())
    if total_score <= 0:
        return base_shares

    return {crop: score / total_score for crop, score in scores.items()}

def _quadrant(x: int, y: int) -> str:
    return ("N" if y < HALF_BOARD else "S") + ("W" if x < HALF_BOARD else "E")

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
    for y in range(min(BOARD_SIZE, len(tiles))):
        for x in range(min(BOARD_SIZE, len(tiles[y]))):
            if tiles[y][x] is None:
                out.append((x, y))
    out.sort(key=lambda p: (_dist(p, (sx, sy)), p[1], p[0]))
    return out

class StrategyPlanner:
    """Encapsulates high-level strategy planning logic."""

    def __init__(self, policy: Optional[Dict[str, Any]] = None):
        self.policy = policy if policy is not None else {}

    def get_target_hands(self, obs: Dict[str, Any], me: Dict[str, Any]) -> int:
        """Determine optimal target farm hand hires for today based on workload and capital."""
        day = obs.get("day", 0)
        money = me.get("money", 0)
        phase = get_season_phase(day, self.policy)

        if not phase["investing"]:
            return 0  # Stop hiring late in season

        base_hands = self.policy.get("hands", 4)
        cost_for_base = calculate_hire_cost(0, base_hands)

        # Scale down if low on cash
        if money < cost_for_base + 100:
            return max(1, base_hands - 2)
        return base_hands

    def evaluate_land_purchase(self, obs: Dict[str, Any], me: Dict[str, Any]) -> Optional[str]:
        """Determine if purchasing an additional land quadrant is financially sound."""
        day = obs.get("day", 0)
        money = me.get("money", 0)
        unlocked = me.get("unlocked_quadrants", ["NW"])

        if day > 15:
            return None  # Too late to pay off new land

        tiles = me.get("tiles", [])
        free_tiles_cnt = len(_open_tiles(tiles)) if tiles else 16

        quad_order = [("NE", 1000), ("SW", 2000), ("SE", 4000)]
        for quad, cost in quad_order:
            if quad not in unlocked:
                # Buy land if cash reserve is 1.3x cost OR if remaining free space is tight
                if money >= cost * 1.3 or (free_tiles_cnt <= 4 and money >= cost + 200):
                    return quad
                break
        return None

    def plan_structure_building_jobs(self, obs: Dict[str, Any], me: Dict[str, Any], priv: Dict[str, Any], free_tiles: List[Tuple[int, int]]) -> List[Dict[str, Any]]:
        """Generate jobs to build animal structures (PASTURE / COOP) when reserves allow."""
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
            return [{
                "pos": pos,
                "op": op,
                "need": None,
                "priority": 8.5
            }]

        return []

    def plan_planting_jobs(self, obs: Dict[str, Any], me: Dict[str, Any], priv: Dict[str, Any], free_tiles: List[Tuple[int, int]]) -> List[Dict[str, Any]]:
        """Generate planting jobs for available free tiles using demand-responsive shares."""
        day = obs.get("day", 0)
        seeds = priv.get("seeds", {})
        tiles = me.get("tiles", [])
        phase = get_season_phase(day, self.policy)

        if not phase["planting"] or not free_tiles:
            return []

        # Count currently planted crops
        n_planted = 0
        for row in tiles:
            for t in row:
                if isinstance(t, dict) and t.get("kind") == "PLANT":
                    n_planted += 1

        # Cap max active crops at 16 to guarantee workers can water all crops daily
        if n_planted >= 16:
            return []

        base_shares = self.policy.get("crop_share", {"CARROT": 0.25, "TOMATO": 0.25, "WHEAT": 0.20, "STRAWBERRY": 0.15, "MELON": 0.15})
        prices = obs.get("market", {}).get("prices", {})
        
        # Calculate dynamic demand-responsive shares
        dynamic_shares = compute_demand_responsive_shares(obs, prices, base_shares)
        
        # Sort crops by dynamic share score
        crops_ordered = sorted(dynamic_shares.keys(), key=lambda c: dynamic_shares[c], reverse=True)

        available_seeds = []
        for crop in crops_ordered:
            cnt = seeds.get(crop, 0)
            if cnt > 0:
                available_seeds.extend([crop] * cnt)

        # Sort free tiles by distance to Shed if tiles grid available
        if me.get("tiles"):
            sx, sy = _shed_tile(tiles)
            clustered_tiles = sorted(free_tiles, key=lambda p: (_dist(p, (sx, sy)), p[1], p[0]))
        else:
            clustered_tiles = free_tiles

        plant_jobs = []
        max_to_plant = 16 - n_planted
        for pos in clustered_tiles:
            if not available_seeds or len(plant_jobs) >= max_to_plant:
                break
            crop = available_seeds.pop(0)
            plant_jobs.append({
                "pos": pos,
                "op": ["PLANT", crop],
                "need": None,
                "priority": 9
            })

        return plant_jobs
