# SPDX-License-Identifier: MIT
"""Kaggriculture Strategy Planner Module.

Manages high-level farm decisions:
- Fibonacci labor hiring calculation.
- Land expansion evaluation (unlocking NE, SW, SE quadrants when capital reserves allow).
- Crop selection and planting schedule planning.
- Season lifecycle gating (investing, planting, liquidation phases).
"""

from typing import Dict, List, Any, Tuple, Optional
from src.constants import BOARD_SIZE, HALF_BOARD, CROPS, LAND_PRICES, TOTAL_DAYS

MOVE_OP = {(0, -1): "NORTH", (0, 1): "SOUTH", (1, 0): "EAST", (-1, 0): "WEST"}
SHED_TILES = [(HALF_BOARD - 1, HALF_BOARD - 1), (HALF_BOARD, HALF_BOARD - 1), (HALF_BOARD - 1, HALF_BOARD), (HALF_BOARD, HALF_BOARD)]

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
    for (x, y) in SHED_TILES:
        if tiles[y][x] != "LOCKED":
            return (x, y)
    return SHED_TILES[0]

def _open_tiles(tiles: List[List[Any]]) -> List[Tuple[int, int]]:
    sx, sy = _shed_tile(tiles)
    out = []
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
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

        quad_order = [("NE", 1000), ("SW", 2000), ("SE", 4000)]
        for quad, cost in quad_order:
            if quad not in unlocked:
                # Require 2x cost reserve before buying land
                if money >= cost * 2:
                    return quad
                break
        return None

    def plan_planting_jobs(self, obs: Dict[str, Any], me: Dict[str, Any], priv: Dict[str, Any], free_tiles: List[Tuple[int, int]]) -> List[Dict[str, Any]]:
        """Generate planting jobs for available free tiles."""
        day = obs.get("day", 0)
        seeds = priv.get("seeds", {})
        phase = get_season_phase(day, self.policy)

        if not phase["planting"] or not free_tiles:
            return []

        crop_shares = self.policy.get("crop_share", {"CARROT": 0.4, "TOMATO": 0.3, "WHEAT": 0.3})
        crops_ordered = self.policy.get("crops", ["CARROT", "TOMATO", "WHEAT"])

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
            plant_jobs.append({
                "pos": pos,
                "op": ["PLANT", crop],
                "need": None,
                "priority": 9
            })

        return plant_jobs
