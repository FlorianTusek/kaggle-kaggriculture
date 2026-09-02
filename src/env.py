# SPDX-License-Identifier: MIT
"""Kaggriculture OpenAI Gym / Gymnasium Environment Wrapper.

Provides a standard Gymnasium interface for RL training (PPO, A2C, DQN)
and evaluation against baseline agents in Kaggriculture simulation.
"""

import math
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

from src.constants import BOARD_SIZE, TURNS_PER_DAY, TOTAL_DAYS, CROPS, PRODUCTS
from src.replay_parser import extract_state_features
from src.train_bc import FEATURE_COLUMNS
from src.agent import KaggricultureAgent

ACTION_LOOKUP = [
    "PASS", "NORTH", "SOUTH", "EAST", "WEST", "WATER", "HARVEST",
    "FEED", "CARE", "COLLECT_FERTILIZER", "DIG", "DROP", "PLANT",
    "FERTILIZE", "PLACE", "BUILD_PASTURE", "PICKUP"
]

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

        self.shed = {"WHEAT": 10, "CARROT": 10}
        self.seeds = {"CARROT": 6, "TOMATO": 6, "WHEAT": 6, "STRAWBERRY": 4, "MELON": 4}
        self.opponent_shed = {"WHEAT": 10, "CARROT": 10}
        self.opponent_seeds = {"CARROT": 6, "TOMATO": 6, "WHEAT": 6}

        self.market_prices = {
            "WHEAT": 25, "CARROT": 35, "TOMATO": 60, "STRAWBERRY": 120, "MELON": 250,
            "EGG": 40, "MILK": 100, "WOOL": 120, "FERTILIZER": 10
        }

    def _get_obs_dict(self, player_idx: int = 0) -> Dict[str, Any]:
        day = self.current_turn // TURNS_PER_DAY
        hour = self.current_turn % TURNS_PER_DAY

        if player_idx == 0:
            my_money, opp_money = self.money, self.opponent_money
            my_quads, opp_quads = self.unlocked_quadrants, self.opponent_unlocked_quadrants
            my_tiles, opp_tiles = self.tiles, self.opponent_tiles
            my_shed, my_seeds = self.shed, self.seeds
        else:
            my_money, opp_money = self.opponent_money, self.money
            my_quads, opp_quads = self.opponent_unlocked_quadrants, self.unlocked_quadrants
            my_tiles, opp_tiles = self.opponent_tiles, self.tiles
            my_shed, my_seeds = self.opponent_shed, self.opponent_seeds

        return {
            "player": player_idx,
            "step": self.current_turn,
            "day": day,
            "hour": hour,
            "farms": [
                {
                    "money": my_money,
                    "tiles": my_tiles,
                    "farmer": [4, 4],
                    "hands": [],
                    "unlocked_quadrants": my_quads,
                    "hires_today": 0,
                },
                {
                    "money": opp_money,
                    "tiles": opp_tiles,
                    "farmer": [4, 4],
                    "hands": [],
                    "unlocked_quadrants": opp_quads,
                    "hires_today": 0,
                }
            ],
            "private": {
                "shed": my_shed,
                "seeds": my_seeds,
                "inventories": [[], [], [], [], []]
            },
            "market": {
                "inventory": {"WHEAT": 10000, "CARROT": 10000, "TOMATO": 10000, "STRAWBERRY": 10000, "MELON": 10000},
                "prices": self.market_prices
            },
            "town": {
                "unlocked_shops": ["Bakery", "Pizza Shop", "Brunch Spot", "Pet Cafe", "Smoothie Shop", "Farmers Market"]
            }
        }

    def _get_obs_vector(self, obs_dict: Dict[str, Any]) -> np.ndarray:
        feat_dict = extract_state_features(obs_dict, player_idx=0)
        vec = [float(feat_dict.get(col, 0)) for col in FEATURE_COLUMNS]
        return np.array(vec, dtype=np.float32)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            np.random.seed(seed)
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

        # 1. Process Market Orders
        market_orders = action_dict.get("market", [])
        for order in market_orders:
            if not isinstance(order, list) or not order:
                continue
            op = order[0]

            if op == "HIRE":
                hire_cost = 50
                if money >= hire_cost:
                    money -= hire_cost

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
                                tile["fed_today"] = True
                                tile["cared_today"] = True
                                tile["yield_units"] = 1
                                money -= cost
                                break

            elif op == "BUY_SEED" and len(order) >= 3:
                crop, qty = order[1], order[2]
                cost_per_seed = CROPS.get(crop, {}).get("seed", 20)
                total_cost = cost_per_seed * qty
                if money >= total_cost:
                    money -= total_cost
                    seeds[crop] = seeds.get(crop, 0) + qty

            elif op == "SELL" and len(order) >= 3:
                product, qty = order[1], order[2]
                avail = shed.get(product, 0)
                sell_qty = min(avail, qty)
                if sell_qty > 0:
                    shed[product] -= sell_qty
                    price = self.market_prices.get(product, 30)
                    money += sell_qty * price

        # 2. Process Worker Ops (Planting, Building, Harvesting, Watering, Caring)
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

        for op in worker_ops:
            if not isinstance(op, list) or not op:
                continue
            cmd = op[0]

            if cmd == "PLANT":
                crop = op[1] if len(op) >= 2 and op[1] in seeds else None
                if not crop:
                    for c in ["MELON", "STRAWBERRY", "TOMATO", "CARROT", "WHEAT"]:
                        if seeds.get(c, 0) > 0:
                            crop = c
                            break
                if crop and seeds.get(crop, 0) > 0:
                    for y in range(BOARD_SIZE):
                        for x in range(BOARD_SIZE):
                            if tiles[y][x] is None:
                                seeds[crop] -= 1
                                tiles[y][x] = {
                                    "kind": "PLANT",
                                    "crop": crop,
                                    "planted_day": self.current_turn // TURNS_PER_DAY,
                                    "watered_today": True,
                                    "yield_units": 2,
                                }
                                break
                        else:
                            continue
                        break

            elif cmd == "WATER":
                for y in range(BOARD_SIZE):
                    for x in range(BOARD_SIZE):
                        t = tiles[y][x]
                        if isinstance(t, dict) and t.get("kind") == "PLANT" and not t.get("watered_today"):
                            t["watered_today"] = True
                            break

            elif cmd in ("FEED", "CARE"):
                for y in range(BOARD_SIZE):
                    for x in range(BOARD_SIZE):
                        t = tiles[y][x]
                        if isinstance(t, dict) and t.get("kind") in ("COOP", "PASTURE") and t.get("animal"):
                            t["fed_today"] = True
                            t["cared_today"] = True
                            break

            elif cmd == "BUILD_PASTURE":
                for y in range(BOARD_SIZE):
                    for x in range(BOARD_SIZE):
                        if tiles[y][x] is None:
                            tiles[y][x] = {"kind": "PASTURE", "animal": None}
                            break
                    else:
                        continue
                    break

            elif cmd == "BUILD_COOP":
                for y in range(BOARD_SIZE):
                    for x in range(BOARD_SIZE):
                        if tiles[y][x] is None:
                            tiles[y][x] = {"kind": "COOP", "animal": None}
                            break
                    else:
                        continue
                    break

            elif cmd == "HARVEST":
                for y in range(BOARD_SIZE):
                    for x in range(BOARD_SIZE):
                        t = tiles[y][x]
                        if isinstance(t, dict):
                            if t.get("kind") == "PLANT":
                                crop = t.get("crop", "CARROT")
                                shed[crop] = shed.get(crop, 0) + max(2, t.get("yield_units", 2))
                                info = CROPS.get(crop, {})
                                if info.get("ongoing"):
                                    t["yield_units"] = 0
                                else:
                                    tiles[y][x] = None
                                break
                            elif t.get("kind") in ("COOP", "PASTURE") and t.get("animal"):
                                prod_map = {"GOOSE": "EGG", "COW": "MILK", "SHEEP": "WOOL"}
                                prod = prod_map.get(t["animal"], "EGG")
                                shed[prod] = shed.get(prod, 0) + max(2, t.get("yield_units", 2))
                                t["yield_units"] = 0
                                break

        if is_me:
            self.money = money
        else:
            self.opponent_money = money

        return money - start_money

    def _calculate_net_wealth(self, player_idx: int = 0) -> float:
        """Calculate total net wealth including liquid bank, shed goods, seeds, and farm infrastructure."""
        money = self.money if player_idx == 0 else self.opponent_money
        shed = self.shed if player_idx == 0 else self.opponent_shed
        seeds = self.seeds if player_idx == 0 else self.opponent_seeds
        tiles = self.tiles if player_idx == 0 else self.opponent_tiles

        shed_val = sum(shed.get(crop, 0) * self.market_prices.get(crop, 30) for crop in shed)
        seed_val = sum(seeds.get(crop, 0) * CROPS.get(crop, {}).get("seed", 20) for crop in seeds)

        tile_val = 0
        for row in tiles:
            for t in row:
                if isinstance(t, dict):
                    if t.get("kind") == "PLANT":
                        tile_val += 50
                    elif t.get("kind") in ("COOP", "PASTURE"):
                        tile_val += 300 if t.get("animal") else 100

        return float(money + shed_val + seed_val + tile_val)

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        current_step = self.current_turn
        hour = current_step % TURNS_PER_DAY

        start_wealth_0 = self._calculate_net_wealth(0)

        if isinstance(action, dict):
            agent_action = action
            act_repr = str(agent_action.get("farmer", [["PASS"]])[0])
        else:
            if isinstance(action, np.ndarray):
                act_int = int(action.item() if action.ndim == 0 else action[0])
            elif isinstance(action, (int, np.integer)):
                act_int = int(action)
            else:
                try:
                    act_int = int(action)
                except Exception:
                    act_int = 0

            act_name = ACTION_LOOKUP[act_int] if 0 <= act_int < len(ACTION_LOOKUP) else "PASS"
            if not hasattr(self, "_base_executor"):
                self._base_executor = KaggricultureAgent(policy={"use_ensemble": False, "use_ml_policy": False})
            obs_0 = self._get_obs_dict(0)
            agent_action = self._base_executor.act(obs_0)
            if act_name != "PASS":
                agent_action["farmer"] = [[act_name]]
            act_repr = act_name

        # Run agent turn ONCE
        earned_this_turn = self.execute_agent_turn(agent_action, player_idx=0)

        # Execute opponent turn ONCE
        if self.opponent_agent is not None:
            opp_obs = self._get_obs_dict(1)
            if hasattr(self.opponent_agent, "act"):
                opp_action = self.opponent_agent.act(opp_obs)
            elif callable(self.opponent_agent):
                try:
                    opp_action = self.opponent_agent(opp_obs, None)
                except TypeError:
                    opp_action = self.opponent_agent(opp_obs)
            else:
                opp_action = {"farmer": [["PASS"]], "hands": [], "market": []}
            self.execute_agent_turn(opp_action, player_idx=1)

        # Daily yield accumulation & tile refresh (on hour 23)
        if hour == 23:
            for tiles, shed in [(self.tiles, self.shed), (self.opponent_tiles, self.opponent_shed)]:
                for row in tiles:
                    for t in row:
                        if isinstance(t, dict):
                            if t.get("kind") == "PLANT":
                                crop = t.get("crop", "CARROT")
                                yield_mult = 5 if crop in ("MELON", "STRAWBERRY") else 3
                                shed[crop] = shed.get(crop, 0) + yield_mult
                            elif t.get("kind") in ("COOP", "PASTURE") and t.get("animal"):
                                prod_map = {"GOOSE": "EGG", "COW": "MILK", "SHEEP": "WOOL"}
                                prod = prod_map.get(t["animal"], "EGG")
                                yield_mult = 4 if prod in ("MILK", "WOOL") else 3
                                shed[prod] = shed.get(prod, 0) + yield_mult

        # Advance step counter
        self.current_turn += 1

        end_wealth_0 = self._calculate_net_wealth(0)
        wealth_delta = end_wealth_0 - start_wealth_0

        # Step reward: incremental net wealth gain normalized
        reward = float(wealth_delta / 100.0)

        terminated = self.current_turn >= self.max_turns
        truncated = False

        # Terminal competitive margin bonus
        if terminated:
            end_wealth_1 = self._calculate_net_wealth(1)
            margin = end_wealth_0 - end_wealth_1
            reward += float(margin / 500.0)

        self.obs = self._get_obs_dict(0)
        obs_vec = self._get_obs_vector(self.obs)

        if isinstance(agent_action, dict):
            farmer_ops = agent_action.get("farmer", [["PASS"]])
            act_repr = farmer_ops[0][0] if (farmer_ops and isinstance(farmer_ops[0], list) and farmer_ops[0]) else str(farmer_ops[0])
        else:
            act_repr = str(agent_action)

        info = {
            "turn": self.current_turn,
            "money": self.money,
            "net_wealth": end_wealth_0,
            "action_executed": act_repr,
            "earned": earned_this_turn
        }

        return obs_vec, reward, terminated, truncated, info

    def render(self, mode: str = "human") -> None:
        print(f"[Turn {self.current_turn}/720] Money: ${self.money:,.2f} | Opponent: ${self.opponent_money:,.2f}")

