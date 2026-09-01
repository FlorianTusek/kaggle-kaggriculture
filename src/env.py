# SPDX-License-Identifier: MIT
"""Kaggriculture OpenAI Gym / Gymnasium Environment Wrapper.

Provides a realistic simulation environment reflecting official Kaggle competition mechanics:
- Dynamic market supply & demand price curves (glut price floor crashes & scarcity spikes).
- Progressive town shop unlocks (every 3 days up to 8 shop instances) and town center consumption.
- Realistic crop growth timelines, watering requirements, and unwatered weed degradation.
- Livestock feeding/care mechanics and escape risks.
- Physical worker movement tracking on board and tile-specific operations.
"""

import math
import random
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    try:
        import gym
        from gym import spaces
    except ImportError:
        gym = None
        spaces = None

from src.constants import BOARD_SIZE, TURNS_PER_DAY, TOTAL_DAYS, CROPS, ANIMALS, PRODUCTS, SHED_CAPACITY
from src.replay_parser import extract_state_features
from src.train_bc import FEATURE_COLUMNS
from src.agent import KaggricultureAgent

ACTION_LOOKUP = [
    "PASS", "NORTH", "SOUTH", "EAST", "WEST", "WATER", "HARVEST",
    "FEED", "CARE", "COLLECT_FERTILIZER", "DIG", "DROP", "PLANT",
    "FERTILIZE", "PLACE", "BUILD_PASTURE", "PICKUP"
]

PRICE_CURVES = {
    "WHEAT": {"base": 25, "I0": 10000, "T": 400, "glut": ("log", 0.20), "scarcity": ("sqrt", 0.80)},
    "CARROT": {"base": 35, "I0": 10000, "T": 450, "glut": ("sqrt", 0.70), "scarcity": ("hinge", 1.00)},
    "TOMATO": {"base": 60, "I0": 10000, "T": 200, "glut": ("sqrt", 0.60), "scarcity": ("hinge", 0.40)},
    "STRAWBERRY": {"base": 120, "I0": 10000, "T": 100, "glut": ("linear", 1.60), "scarcity": ("sqrt", 0.70)},
    "MELON": {"base": 250, "I0": 10000, "T": 300, "glut": ("sq", 3.60), "scarcity": ("log", 0.20)},
    "EGG": {"base": 50, "I0": 10000, "T": 332, "glut": ("log", 0.20), "scarcity": ("hinge", 0.40)},
    "MILK": {"base": 160, "I0": 10000, "T": 122, "glut": ("linear", 1.60), "scarcity": ("sqrt", 0.60)},
    "WOOL": {"base": 200, "I0": 10000, "T": 105, "glut": ("sq", 3.20), "scarcity": ("log", 0.20)},
    "FERTILIZER": {"base": 100, "I0": 10000, "T": 200, "glut": ("linear", 0.40), "scarcity": ("linear", 0.40)},
}

TOWN_SHOPS = {
    "Bakery": ["EGG", "WHEAT"],
    "Pizza Shop": ["MILK", "TOMATO", "WHEAT"],
    "Brunch Spot": ["EGG", "WHEAT", "STRAWBERRY"],
    "Yarn Store": ["WOOL", "WOOL"],
    "Ice Cream Shop": ["STRAWBERRY", "MILK", "WHEAT"],
    "Pet Cafe": ["CARROT", "CARROT"],
    "Smoothie Shop": ["STRAWBERRY", "MILK"],
    "Farmers Market": ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY"],
}

def _eval_shape(shape: str, x: float, T: float) -> float:
    u = x / T
    if shape == "linear":
        return u
    elif shape == "sq":
        return u ** 2
    elif shape == "sqrt":
        return math.sqrt(max(0.0, u))
    elif shape == "log":
        return math.log(1.0 + u)
    elif shape == "hinge":
        return u + 8.0 * (max(0.0, u - 1.0) ** 2)
    return u

def calculate_market_price(product: str, inventory: int) -> float:
    cfg = PRICE_CURVES.get(product)
    if not cfg:
        return 30.0
    base = float(cfg["base"])
    I0 = float(cfg["I0"])
    T = float(cfg["T"])
    inv = float(inventory)

    if inv == I0:
        return base
    elif inv > I0:
        shape, target = cfg["glut"]
        x = inv - I0
        f_T = _eval_shape(shape, T, T)
        amp = target * base / f_T
        price = base - amp * _eval_shape(shape, x, T)
    else:
        shape, target = cfg["scarcity"]
        x = I0 - inv
        f_T = _eval_shape(shape, T, T)
        amp = target * base / f_T
        price = base + amp * _eval_shape(shape, x, T)

    return float(max(1, round(price)))


class KaggricultureEnv(gym.Env if gym is not None else object):
    """OpenAI Gym / Gymnasium Environment for Kaggriculture."""

    metadata = {"render_modes": ["human"], "render_fps": 24}

    def __init__(self, max_turns: int = 720, opponent_agent: Optional[Any] = None):
        super().__init__()
        self.max_turns = max_turns
        self.opponent_agent = opponent_agent if opponent_agent is not None else KaggricultureAgent()

        self.n_features = len(FEATURE_COLUMNS)
        if spaces is not None:
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(self.n_features,), dtype=np.float32
            )
            self.action_space = spaces.Discrete(len(ACTION_LOOKUP))

        self.reset()

    def _init_state(self) -> None:
        self.current_turn = 0
        self.money = 3000.0
        self.opponent_money = 3000.0
        self.unlocked_quadrants = ["NW"]
        self.opponent_unlocked_quadrants = ["NW"]

        # Grid setup (10x10): NW is unlocked (0..4, 0..4), rest locked
        self.tiles = [[None for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                if x >= 5 or y >= 5:
                    self.tiles[y][x] = "LOCKED"

        self.opponent_tiles = [[None for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                if x >= 5 or y >= 5:
                    self.opponent_tiles[y][x] = "LOCKED"

        self.farmer_pos = [4, 4]
        self.hands_pos = []
        self.opp_farmer_pos = [4, 4]
        self.opp_hands_pos = []

        self.shed = {"WHEAT": 10, "CARROT": 10}
        self.seeds = {"CARROT": 6, "TOMATO": 6, "WHEAT": 6, "STRAWBERRY": 4, "MELON": 4}
        self.opponent_shed = {"WHEAT": 10, "CARROT": 10}
        self.opponent_seeds = {"CARROT": 6, "TOMATO": 6, "WHEAT": 6, "STRAWBERRY": 4, "MELON": 4}

        self.market_inventory = {p: 10000 for p in PRODUCTS}
        self.market_prices = {p: calculate_market_price(p, 10000) for p in PRODUCTS}

        self.unlocked_shops = []

    def _get_obs_dict(self, player_idx: int = 0) -> Dict[str, Any]:
        day = self.current_turn // TURNS_PER_DAY
        hour = self.current_turn % TURNS_PER_DAY

        if player_idx == 0:
            my_money, opp_money = self.money, self.opponent_money
            my_quads, opp_quads = self.unlocked_quadrants, self.opponent_unlocked_quadrants
            my_tiles, opp_tiles = self.tiles, self.opponent_tiles
            my_shed, my_seeds = self.shed, self.seeds
            my_farmer, my_hands = self.farmer_pos, self.hands_pos
            opp_farmer, opp_hands = self.opp_farmer_pos, self.opp_hands_pos
        else:
            my_money, opp_money = self.opponent_money, self.money
            my_quads, opp_quads = self.opponent_unlocked_quadrants, self.unlocked_quadrants
            my_tiles, opp_tiles = self.opponent_tiles, self.tiles
            my_shed, my_seeds = self.opponent_shed, self.seeds
            my_farmer, my_hands = self.opp_farmer_pos, self.opp_hands_pos
            opp_farmer, opp_hands = self.farmer_pos, self.hands_pos

        return {
            "player": player_idx,
            "step": self.current_turn,
            "day": day,
            "hour": hour,
            "farms": [
                {
                    "money": my_money,
                    "tiles": my_tiles,
                    "farmer": list(my_farmer),
                    "hands": [list(h) for h in my_hands],
                    "unlocked_quadrants": my_quads,
                    "hires_today": len(my_hands),
                },
                {
                    "money": opp_money,
                    "tiles": opp_tiles,
                    "farmer": list(opp_farmer),
                    "hands": [list(h) for h in opp_hands],
                    "unlocked_quadrants": opp_quads,
                    "hires_today": len(opp_hands),
                }
            ],
            "private": {
                "shed": my_shed,
                "seeds": my_seeds,
                "inventories": [[] for _ in range(1 + len(my_hands))]
            },
            "market": {
                "inventory": self.market_inventory,
                "prices": self.market_prices
            },
            "town": {
                "unlocked_shops": list(self.unlocked_shops)
            }
        }

    def _get_obs_vector(self, obs_dict: Dict[str, Any]) -> np.ndarray:
        feat_dict = extract_state_features(obs_dict, player_idx=0)
        vec = [float(feat_dict.get(col, 0)) for col in FEATURE_COLUMNS]
        return np.array(vec, dtype=np.float32)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        self._init_state()
        self.obs = self._get_obs_dict(0)
        obs_vec = self._get_obs_vector(self.obs)
        info = {"turn": self.current_turn, "money": self.money}
        return obs_vec, info

    def _unlock_quadrant(self, quad: str, player_idx: int = 0) -> None:
        quad_bounds = {
            "NE": (range(5, 10), range(0, 5)),
            "SW": (range(0, 5), range(5, 10)),
            "SE": (range(5, 10), range(5, 10)),
        }
        if quad not in quad_bounds:
            return

        target_tiles = self.tiles if player_idx == 0 else self.opponent_tiles
        target_quads = self.unlocked_quadrants if player_idx == 0 else self.opponent_unlocked_quadrants

        if quad not in target_quads:
            target_quads.append(quad)
            x_range, y_range = quad_bounds[quad]
            for y in y_range:
                for x in x_range:
                    if target_tiles[y][x] == "LOCKED":
                        target_tiles[y][x] = None

    def execute_agent_turn(self, action_dict: Dict[str, Any], player_idx: int = 0) -> float:
        """Execute full action dictionary (market orders + worker operations) for player."""
        is_me = (player_idx == 0)
        money = self.money if is_me else self.opponent_money
        shed = self.shed if is_me else self.opponent_shed
        seeds = self.seeds if is_me else self.opponent_seeds
        tiles = self.tiles if is_me else self.opponent_tiles

        start_money = money

        # Reset hands at start of day (hour 0)
        if self.current_turn % TURNS_PER_DAY == 0:
            if is_me:
                self.hands_pos = []
                self.farmer_pos = [4, 4]
            else:
                self.opp_hands_pos = []
                self.opp_farmer_pos = [4, 4]

        # 1. Process Market Orders
        market_orders = action_dict.get("market", [])
        for order in market_orders:
            if not isinstance(order, list) or not order:
                continue
            op = order[0]

            if op == "HIRE":
                if is_me:
                    self.hands_pos.append([4, 4])
                else:
                    self.opp_hands_pos.append([4, 4])

            elif op == "BUY_LAND" and len(order) >= 2:
                quad = order[1]
                cost_map = {"NE": 1000, "SW": 2000, "SE": 4000}
                cost = cost_map.get(quad, 1000)
                if money >= cost:
                    money -= cost
                    self._unlock_quadrant(quad, player_idx=player_idx)

            elif op == "BUY_ANIMAL" and len(order) >= 2:
                animal = order[1]
                cost_map = {"GOOSE": 300, "COW": 400, "SHEEP": 500}
                cost = cost_map.get(animal, 300)
                target_kind = "COOP" if animal == "GOOSE" else "PASTURE"
                if money >= cost:
                    for row in tiles:
                        for tile in row:
                            if isinstance(tile, dict) and tile.get("kind") == target_kind and not tile.get("animal"):
                                tile["animal"] = animal
                                tile["placed_day"] = self.current_turn // TURNS_PER_DAY
                                tile["fed_today"] = True
                                tile["cared_today"] = True
                                tile["consecutive_unfed"] = 0
                                tile["yield_units"] = 0
                                money -= cost
                                break
                        else:
                            continue
                        break

            elif op == "BUY_SEED" and len(order) >= 3:
                crop, qty = order[1], order[2]
                cost_per_seed = CROPS.get(crop, {}).get("seed", 20)
                total_cost = cost_per_seed * qty
                if money >= total_cost:
                    money -= total_cost
                    seeds[crop] = seeds.get(crop, 0) + qty

            elif op == "BUY_PRODUCT" and len(order) >= 3:
                product, qty = order[1], order[2]
                for _ in range(qty):
                    cur_price = self.market_prices.get(product, 30)
                    if money >= cur_price:
                        money -= cur_price
                        self.market_inventory[product] = max(0, self.market_inventory.get(product, 10000) - 1)
                        self.market_prices[product] = calculate_market_price(product, self.market_inventory[product])
                        shed[product] = shed.get(product, 0) + 1
                    else:
                        break

            elif op == "SELL" and len(order) >= 3:
                product, qty = order[1], order[2]
                avail = shed.get(product, 0)
                sell_qty = min(avail, qty)
                for _ in range(sell_qty):
                    shed[product] -= 1
                    self.market_inventory[product] = self.market_inventory.get(product, 10000) + 1
                    cur_price = calculate_market_price(product, self.market_inventory[product])
                    self.market_prices[product] = cur_price
                    money += cur_price

        # 2. Process Worker Ops (Movement, Planting, Building, Harvesting, Watering, Feeding)
        farmer_op = action_dict.get("farmer", [])
        if farmer_op and isinstance(farmer_op, list):
            if isinstance(farmer_op[0], list):
                farmer_op = farmer_op[0]
        else:
            farmer_op = ["PASS"]

        worker_ops = [farmer_op]
        for hand in action_dict.get("hands", []):
            if isinstance(hand, list):
                if hand and isinstance(hand[0], list):
                    worker_ops.append(hand[0])
                else:
                    worker_ops.append(hand)

        worker_positions = [self.farmer_pos] + self.hands_pos if is_me else [self.opp_farmer_pos] + self.opp_hands_pos
        cur_day = self.current_turn // TURNS_PER_DAY

        for idx, op in enumerate(worker_ops):
            if idx >= len(worker_positions):
                break
            if not isinstance(op, list) or not op:
                continue

            pos = worker_positions[idx]
            wx, wy = pos[0], pos[1]
            cmd = op[0]

            # Movement
            if cmd == "NORTH":
                wy = max(0, wy - 1)
                pos[1] = wy
            elif cmd == "SOUTH":
                wy = min(BOARD_SIZE - 1, wy + 1)
                pos[1] = wy
            elif cmd == "WEST":
                wx = max(0, wx - 1)
                pos[0] = wx
            elif cmd == "EAST":
                wx = min(BOARD_SIZE - 1, wx + 1)
                pos[0] = wx

            # Tile Actions at current worker position (wx, wy)
            elif cmd == "PLANT" and len(op) >= 2:
                crop = op[1]
                if seeds.get(crop, 0) > 0 and tiles[wy][wx] is None:
                    seeds[crop] -= 1
                    tiles[wy][wx] = {
                        "kind": "PLANT",
                        "crop": crop,
                        "planted_day": cur_day,
                        "watered_today": True,
                        "consecutive_unwatered": 0,
                        "yield_units": 0,
                    }

            elif cmd == "WATER":
                t = tiles[wy][wx]
                if isinstance(t, dict) and t.get("kind") == "PLANT":
                    t["watered_today"] = True
                    t["consecutive_unwatered"] = 0

            elif cmd == "FEED":
                t = tiles[wy][wx]
                if isinstance(t, dict) and t.get("kind") in ("COOP", "PASTURE") and t.get("animal"):
                    if shed.get("WHEAT", 0) > 0:
                        shed["WHEAT"] -= 1
                        t["fed_today"] = True
                        t["consecutive_unfed"] = 0

            elif cmd == "BUILD_PASTURE":
                if tiles[wy][wx] is None:
                    tiles[wy][wx] = {"kind": "PASTURE", "animal": None}

            elif cmd == "BUILD_COOP":
                if tiles[wy][wx] is None:
                    tiles[wy][wx] = {"kind": "COOP", "animal": None}

            elif cmd == "HARVEST":
                t = tiles[wy][wx]
                if isinstance(t, dict):
                    if t.get("kind") == "PLANT" and t.get("yield_units", 0) > 0:
                        crop = t.get("crop", "CARROT")
                        harvested = t["yield_units"]
                        room = max(0, SHED_CAPACITY - sum(shed.values()))
                        actual_add = min(harvested, room)
                        if actual_add > 0:
                            shed[crop] = shed.get(crop, 0) + actual_add
                        if CROPS.get(crop, {}).get("ongoing"):
                            t["yield_units"] = 0
                        else:
                            tiles[wy][wx] = None
                    elif t.get("kind") in ("COOP", "PASTURE") and t.get("animal") and t.get("yield_units", 0) > 0:
                        anim = t["animal"]
                        prod = ANIMALS.get(anim, {}).get("product", "EGG")
                        harvested = t["yield_units"]
                        room = max(0, SHED_CAPACITY - sum(shed.values()))
                        actual_add = min(harvested, room)
                        if actual_add > 0:
                            shed[prod] = shed.get(prod, 0) + actual_add
                        t["yield_units"] = 0

        if is_me:
            self.money = money
        else:
            self.opponent_money = money

        return money - start_money

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.current_turn += 1
        day = self.current_turn // TURNS_PER_DAY
        hour = self.current_turn % TURNS_PER_DAY

        # 0. Unlock town shop every 3 days (days 3, 6, 9, 12, 15, 18, 21, 24)
        if hour == 0 and day > 0 and day % 3 == 0 and len(self.unlocked_shops) < 8:
            all_shops = list(TOWN_SHOPS.keys())
            new_shop = all_shops[(day // 3) % len(all_shops)]
            self.unlocked_shops.append(new_shop)

        if isinstance(action, dict):
            agent_action = action
            act_repr = str(agent_action.get("farmer", [["PASS"]])[0])
        elif isinstance(action, (int, np.integer)):
            act_name = ACTION_LOOKUP[action] if 0 <= action < len(ACTION_LOOKUP) else "PASS"
            agent_action = {"farmer": [[act_name]], "hands": [], "market": []}
            act_repr = act_name
        else:
            agent_action = {"farmer": [["PASS"]], "hands": [], "market": []}
            act_repr = "PASS"

        # 1. Run player agent turn
        earned_this_turn = self.execute_agent_turn(agent_action, player_idx=0)

        # 2. Run opponent agent turn
        opp_obs = self._get_obs_dict(1)
        opp_action = self.opponent_agent.act(opp_obs)
        self.execute_agent_turn(opp_action, player_idx=1)

        # 3. Town Shops consumption every 4 turns
        if hour % 4 == 0:
            for shop in self.unlocked_shops:
                demands = TOWN_SHOPS.get(shop, [])
                for item in demands:
                    self.market_inventory[item] = max(0, self.market_inventory.get(item, 10000) - 1)
                    self.market_prices[item] = calculate_market_price(item, self.market_inventory[item])

        # 4. End-of-Day refresh on hour 23
        if hour == 23:
            # Town center daily flat consumption
            for p in PRODUCTS:
                self.market_inventory[p] = max(0, self.market_inventory.get(p, 10000) - 1)
                self.market_prices[p] = calculate_market_price(p, self.market_inventory[p])

            # Process plant maturation & animal feeding status
            for tiles in (self.tiles, self.opponent_tiles):
                for y in range(BOARD_SIZE):
                    for x in range(BOARD_SIZE):
                        t = tiles[y][x]
                        if not isinstance(t, dict):
                            continue

                        if t.get("kind") == "PLANT":
                            if not t.get("watered_today"):
                                t["consecutive_unwatered"] = t.get("consecutive_unwatered", 0) + 1
                                if t["consecutive_unwatered"] >= 2:
                                    tiles[y][x] = {"kind": "WEED"}
                                    continue
                            else:
                                t["consecutive_unwatered"] = 0
                            t["watered_today"] = False

                            # Crop growth and yield maturation
                            crop = t.get("crop", "CARROT")
                            info = CROPS.get(crop, {})
                            planted_day = t.get("planted_day", day)
                            age = day - planted_day
                            first_day = info.get("first_yield_day", 2)
                            max_units = info.get("max_yield", 4)
                            if age >= first_day:
                                t["yield_units"] = min(max_units, t.get("yield_units", 0) + 1)

                        elif t.get("kind") in ("COOP", "PASTURE") and t.get("animal"):
                            if not t.get("fed_today"):
                                t["consecutive_unfed"] = t.get("consecutive_unfed", 0) + 1
                                if t["consecutive_unfed"] >= 2:
                                    t["animal"] = None  # Animal escapes!
                                    continue
                            else:
                                t["consecutive_unfed"] = 0
                            t["fed_today"] = False

                            # Livestock product maturation
                            anim = t["animal"]
                            info = ANIMALS.get(anim, {})
                            placed_day = t.get("placed_day", day)
                            age = day - placed_day
                            first_day = info.get("first_yield_day", 4)
                            max_units = info.get("max_yield", 4)
                            if age >= first_day:
                                t["yield_units"] = min(max_units, t.get("yield_units", 0) + 1)

        reward = earned_this_turn
        terminated = self.current_turn >= self.max_turns
        truncated = False

        self.obs = self._get_obs_dict(0)
        obs_vec = self._get_obs_vector(self.obs)

        info = {
            "turn": self.current_turn,
            "money": self.money,
            "action_executed": act_repr,
            "earned": earned_this_turn
        }

        return obs_vec, reward, terminated, truncated, info

    def render(self, mode: str = "human") -> None:
        print(f"[Turn {self.current_turn}/720] Money: ${self.money:,.2f} | Opponent: ${self.opponent_money:,.2f}")
