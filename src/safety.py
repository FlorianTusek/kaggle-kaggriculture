# SPDX-License-Identifier: MIT
"""Kaggriculture Safety Layer Module.

Guarantees:
1. All crops are watered daily (preventing weed conversion and maximizing yield).
2. All animals are fed daily with wheat (preventing escape).
3. Animals are cared for to bank yield bonuses.
4. Harvestable crops and animal produce are collected on time before decay.
5. Fertilizer is collected from animals.
6. Weeds are cleared with DIG.
"""

from typing import Dict, List, Any, Tuple, Optional
from src.constants import BOARD_SIZE, CROPS, ANIMALS

def is_crop_ready_to_harvest(tile: Dict[str, Any], day: int, harvest_asap: bool = False) -> bool:
    """Determine if a crop tile is ready to harvest according to growth cycles and decay safety."""
    if not isinstance(tile, dict) or tile.get("kind") != "PLANT":
        return False

    crop = tile.get("crop")
    if crop not in CROPS:
        return False

    info = CROPS[crop]
    planted_day = tile.get("planted_day", 0)
    age = day - planted_day
    yield_units = tile.get("yield_units", 0)

    if age < info["first_yield_day"]:
        return False
    if yield_units <= 0:
        return False

    # Ongoing crops (Tomato, Strawberry) yield periodically
    if info["ongoing"]:
        return True

    # Harvest ASAP flag overrides waiting for max yield bonus
    if harvest_asap:
        return True

    # Harvest if plant reached max yield day or max yield capacity
    if age >= info["max_yield_day"]:
        return True
    
    return yield_units >= min(info["max_yield"], 3)

def is_animal_unfed_emergency(tile: Dict[str, Any]) -> bool:
    """Check if an animal is in emergency unfed status (1 day away from escaping)."""
    if not isinstance(tile, dict) or tile.get("kind") not in ("COOP", "PASTURE"):
        return False
    if not tile.get("animal"):
        return False
    return not tile.get("fed_today", False) and tile.get("consecutive_unfed", 0) >= 1

def is_crop_unwatered_emergency(tile: Dict[str, Any]) -> bool:
    """Check if a crop is in emergency unwatered status (1 day away from becoming a weed)."""
    if not isinstance(tile, dict) or tile.get("kind") != "PLANT":
        return False
    return not tile.get("watered_today", False) and tile.get("consecutive_unwatered", 0) >= 1

def collect_safety_jobs(obs: Dict[str, Any], me: Dict[str, Any], priv: Dict[str, Any], policy: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Scan the farm board and collect all safety jobs ordered strictly by safety priority.
    
    Priority Order:
    1. Emergency Animal Feeding (consecutive_unfed >= 1)
    2. Emergency Crop Watering (consecutive_unwatered >= 1)
    3. Harvesting ready crops & animal produce before decay
    4. Routine Animal Feeding (fed_today == False)
    5. Animal Care (cared_today == False)
    6. Routine Crop Watering (watered_today == False)
    7. Fertilizer Collection (fertilizer_available > 0)
    8. Weed Digging (kind == "WEED")
    """
    if policy is None:
        policy = {}

    day = obs.get("day", 0)
    tiles = me.get("tiles", [])
    harvest_asap = policy.get("harvest_asap", False)

    emergency_feed = []
    emergency_water = []
    harvest_jobs = []
    routine_feed = []
    care_jobs = []
    routine_water = []
    fertilizer_jobs = []
    dig_weeds = []

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
                animal = tile.get("animal")

                # Animal harvesting (produce)
                if tile.get("yield_units", 0) > 0:
                    harvest_jobs.append({"pos": pos, "op": ["HARVEST"], "need": None, "priority": 3})

                # Feeding
                if is_animal_unfed_emergency(tile):
                    emergency_feed.append({"pos": pos, "op": ["FEED"], "need": ("WHEAT", 1), "priority": 1})
                elif not tile.get("fed_today", False):
                    routine_feed.append({"pos": pos, "op": ["FEED"], "need": ("WHEAT", 1), "priority": 4})

                # Care
                if not tile.get("cared_today", False) and (tile.get("fed_today", False) or is_animal_unfed_emergency(tile) or not tile.get("fed_today", False)):
                    care_jobs.append({"pos": pos, "op": ["CARE"], "need": None, "priority": 5})

                # Fertilizer collection
                if tile.get("fertilizer_available", 0) > 0:
                    fertilizer_jobs.append({"pos": pos, "op": ["COLLECT_FERTILIZER"], "need": None, "priority": 7})

    # Combine in strict priority order
    all_safety_jobs = (
        emergency_feed +
        emergency_water +
        harvest_jobs +
        routine_feed +
        care_jobs +
        routine_water +
        fertilizer_jobs +
        dig_weeds
    )

    return all_safety_jobs

class SafetyLayer:
    """Encapsulates Safety Layer logic for Kaggriculture Agents."""

    def __init__(self, policy: Optional[Dict[str, Any]] = None):
        self.policy = policy if policy is not None else {}

    def get_jobs(self, obs: Dict[str, Any], me: Dict[str, Any], priv: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Return safety jobs for the current turn."""
        return collect_safety_jobs(obs, me, priv, self.policy)
