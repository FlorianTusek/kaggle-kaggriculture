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

        self.current_turn = 0
        self.money = 3000.0
        self.opponent_money = 3000.0
        self.last_score = 3000.0
        self.obs = None

    def _get_obs_dict(self) -> Dict[str, Any]:
        day = self.current_turn // TURNS_PER_DAY
        hour = self.current_turn % TURNS_PER_DAY
        tiles = [[None for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                if x >= 5 or y >= 5:
                    tiles[y][x] = "LOCKED"

        return {
            "player": 0,
            "step": self.current_turn,
            "day": day,
            "hour": hour,
            "farms": [
                {
                    "money": self.money,
                    "tiles": tiles,
                    "farmer": [4, 4],
                    "hands": [],
                    "unlocked_quadrants": ["NW"],
                    "hires_today": 0,
                },
                {
                    "money": self.opponent_money,
                    "tiles": tiles,
                    "farmer": [4, 4],
                    "hands": [],
                    "unlocked_quadrants": ["NW"],
                    "hires_today": 0,
                }
            ],
            "private": {
                "shed": {"WHEAT": 10, "CARROT": 10},
                "seeds": {"CARROT": 4, "WHEAT": 4},
                "inventories": [[], [], [], [], []]
            },
            "market": {
                "inventory": {"WHEAT": 10000, "CARROT": 10000, "TOMATO": 10000},
                "prices": {"WHEAT": 25, "CARROT": 35, "TOMATO": 60, "MELON": 250}
            },
            "town": {
                "unlocked_shops": ["Bakery"]
            }
        }

    def _get_obs_vector(self, obs_dict: Dict[str, Any]) -> np.ndarray:
        feat_dict = extract_state_features(obs_dict, player_idx=0)
        vec = [float(feat_dict.get(col, 0)) for col in FEATURE_COLUMNS]
        return np.array(vec, dtype=np.float32)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            np.random.seed(seed)
        self.current_turn = 0
        self.money = 3000.0
        self.opponent_money = 3000.0
        self.last_score = 3000.0
        self.obs = self._get_obs_dict()
        obs_vec = self._get_obs_vector(self.obs)
        info = {"turn": self.current_turn, "money": self.money}
        return obs_vec, info

    def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.current_turn += 1
        day = self.current_turn // TURNS_PER_DAY
        hour = self.current_turn % TURNS_PER_DAY

        # Decode discrete action index
        act_name = ACTION_LOOKUP[action_idx] if 0 <= action_idx < len(ACTION_LOOKUP) else "PASS"

        # Execute opponent turn
        opp_obs = self._get_obs_dict()
        opp_obs["player"] = 1
        opp_action = self.opponent_agent.act(opp_obs)

        # Economic simulation update
        earned_this_turn = 0.0
        if hour == 23 and day > 2:
            earned_this_turn = 450.0 + (50.0 if act_name != "PASS" else 0.0)
            self.money += earned_this_turn
            self.opponent_money += 400.0

        reward = earned_this_turn
        terminated = self.current_turn >= self.max_turns
        truncated = False

        self.obs = self._get_obs_dict()
        obs_vec = self._get_obs_vector(self.obs)
        info = {
            "turn": self.current_turn,
            "money": self.money,
            "action_executed": act_name,
            "earned": earned_this_turn
        }

        return obs_vec, reward, terminated, truncated, info

    def render(self, mode: str = "human") -> None:
        print(f"[Turn {self.current_turn}/720] Money: ${self.money:,.2f} | Opponent: ${self.opponent_money:,.2f}")
