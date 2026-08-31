# SPDX-License-Identifier: MIT
"""Hybrid Monte Carlo Tree Search (MCTS) with Neural Network Policy Prior.

Implements MCTS for Kaggriculture that uses the Phase 4 PPO policy network (or BC model)
as a prior to prune the search tree and guide exploration via the PUCT formula:
    PUCT(s, a) = Q(s, a) + c_puct * P(s, a) * (sqrt(N(s)) / (1 + N(s, a)))
"""

import math
import copy
import numpy as np
from typing import Dict, List, Any, Optional, Tuple

from src.constants import BOARD_SIZE, TURNS_PER_DAY, PRODUCTS, CROPS
from src.env import ACTION_LOOKUP, KaggricultureEnv


class MCTSNode:
    """Represents a node in the Monte Carlo search tree."""

    def __init__(self, state_obs: Dict[str, Any], parent: Optional['MCTSNode'] = None, action: Optional[str] = None, prior: float = 1.0):
        self.state_obs = state_obs
        self.parent = parent
        self.action = action
        self.prior = float(prior)
        self.visit_count = 0
        self.value_sum = 0.0
        self.children: Dict[str, 'MCTSNode'] = {}
        self.is_expanded = False

    @property
    def q_value(self) -> float:
        """Mean action value Q(s, a)."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def puct_score(self, c_puct: float = 1.414) -> float:
        """Compute the PUCT upper confidence bound."""
        parent_visits = self.parent.visit_count if self.parent is not None else 1
        exploration = c_puct * self.prior * (math.sqrt(parent_visits) / (1 + self.visit_count))
        return self.q_value + exploration

    def select_child(self, c_puct: float = 1.414) -> Tuple[str, 'MCTSNode']:
        """Select the child node maximizing the PUCT score."""
        best_score = -float('inf')
        best_action = None
        best_child = None

        for action, child in self.children.items():
            score = child.puct_score(c_puct=c_puct)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def expand(self, action_priors: Dict[str, float], env_simulator: Any) -> None:
        """Expand node by creating child nodes for candidate actions with positive priors."""
        for action_name, prior_prob in action_priors.items():
            if action_name not in self.children and prior_prob > 0.0:
                next_obs = env_simulator.simulate_step(self.state_obs, action_name)
                self.children[action_name] = MCTSNode(
                    state_obs=next_obs,
                    parent=self,
                    action=action_name,
                    prior=prior_prob
                )
        self.is_expanded = True

    def backpropagate(self, value: float) -> None:
        """Backpropagate leaf node evaluation up the search tree."""
        self.visit_count += 1
        self.value_sum += value
        if self.parent is not None:
            self.parent.backpropagate(value)


class FastStateEvaluator:
    """Fast value function and transition simulator for Kaggriculture states."""

    def __init__(self):
        pass

    def evaluate(self, obs: Dict[str, Any]) -> float:
        """Heuristic leaf evaluation estimating net total farm equity and profit delta."""
        player_idx = obs.get("player", 0)
        farms = obs.get("farms", [])
        if len(farms) <= player_idx:
            return 0.0

        me = farms[player_idx]
        opp = farms[1 - player_idx] if len(farms) > 1 else {}

        # 1. Liquid cash
        my_money = float(me.get("money", 3000.0))
        opp_money = float(opp.get("money", 3000.0))

        # 2. Inventory asset value
        priv = obs.get("private", {})
        shed = priv.get("shed", {})
        seeds = priv.get("seeds", {})
        prices = obs.get("market", {}).get("prices", {})

        inv_value = 0.0
        for prod, qty in shed.items():
            price = float(prices.get(prod, 20.0))
            inv_value += qty * price

        for crop, qty in seeds.items():
            seed_cost = float(CROPS.get(crop, {}).get("seed", 10.0))
            inv_value += qty * seed_cost

        # 3. Active tile valuation
        tile_value = 0.0
        tiles = me.get("tiles", [])
        for row in tiles:
            for tile in row:
                if isinstance(tile, dict):
                    kind = tile.get("kind")
                    if kind == "PLANT":
                        crop = tile.get("crop", "WHEAT")
                        base_p = float(CROPS.get(crop, {}).get("base_price", 25.0))
                        yields = float(tile.get("yield_units", 1))
                        tile_value += yields * base_p * 0.8
                    elif kind in ("COOP", "PASTURE") and tile.get("animal"):
                        tile_value += 300.0

        total_equity = my_money + inv_value + tile_value
        relative_advantage = total_equity - opp_money

        # Normalized value between -1.0 and 1.0
        return float(np.tanh(relative_advantage / 5000.0))

    def simulate_step(self, obs: Dict[str, Any], action_name: str) -> Dict[str, Any]:
        """Fast forward-model state transition approximation."""
        next_obs = copy.deepcopy(obs)
        step = next_obs.get("step", 0) + 1
        next_obs["step"] = step
        next_obs["day"] = step // TURNS_PER_DAY
        next_obs["hour"] = step % TURNS_PER_DAY

        player_idx = next_obs.get("player", 0)
        farm = next_obs["farms"][player_idx]

        # Movement approximation
        fx, fy = farm.get("farmer", [4, 4])
        if action_name == "NORTH":
            fy = max(0, fy - 1)
        elif action_name == "SOUTH":
            fy = min(BOARD_SIZE - 1, fy + 1)
        elif action_name == "EAST":
            fx = min(BOARD_SIZE - 1, fx + 1)
        elif action_name == "WEST":
            fx = max(0, fx - 1)
        farm["farmer"] = [fx, fy]

        # End of day reward tick
        if next_obs["hour"] == 23 and next_obs["day"] > 2:
            earned = 450.0 + (50.0 if action_name != "PASS" else 0.0)
            farm["money"] = farm.get("money", 3000.0) + earned

        return next_obs


class HybridMCTS:
    """Monte Carlo Tree Search with PPO Policy Priors and Leaf Evaluation."""

    def __init__(self, policy_prior: Optional[Any] = None, c_puct: float = 1.414, n_simulations: int = 25, top_k_actions: int = 5):
        self.policy_prior = policy_prior
        self.c_puct = c_puct
        self.n_simulations = n_simulations
        self.top_k_actions = top_k_actions
        self.evaluator = FastStateEvaluator()

    def get_action_priors(self, obs: Dict[str, Any]) -> Dict[str, float]:
        """Compute pruned prior probability distribution over candidate actions."""
        if self.policy_prior is not None and hasattr(self.policy_prior, "predict_farmer_action_probs"):
            probs = self.policy_prior.predict_farmer_action_probs(obs)
        elif self.policy_prior is not None and hasattr(self.policy_prior, "predict_action"):
            top_act = self.policy_prior.predict_action(obs)
            probs = {a: 0.05 for a in ACTION_LOOKUP}
            probs[top_act] = 0.60
        else:
            probs = {a: 1.0 / len(ACTION_LOOKUP) for a in ACTION_LOOKUP}

        # Filter to top K actions
        sorted_actions = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:self.top_k_actions]
        total_p = sum(p for _, p in sorted_actions)
        if total_p <= 0:
            return {ACTION_LOOKUP[0]: 1.0}

        return {act: p / total_p for act, p in sorted_actions}

    def search(self, root_obs: Dict[str, Any]) -> Dict[str, Any]:
        """Run MCTS simulations starting from the root observation."""
        root = MCTSNode(state_obs=root_obs)
        priors = self.get_action_priors(root_obs)
        root.expand(priors, self.evaluator)

        for _ in range(self.n_simulations):
            node = root
            search_path = [node]

            # 1. Selection
            while node.is_expanded and node.children:
                act, child = node.select_child(self.c_puct)
                if child is None:
                    break
                node = child
                search_path.append(node)

            # 2. Expansion
            if not node.is_expanded:
                child_priors = self.get_action_priors(node.state_obs)
                node.expand(child_priors, self.evaluator)

            # 3. Evaluation
            leaf_value = self.evaluator.evaluate(node.state_obs)

            # 4. Backpropagation
            node.backpropagate(leaf_value)

        # Select most visited action
        visit_counts = {act: child.visit_count for act, child in root.children.items()}
        if not visit_counts:
            best_action = "PASS"
        else:
            best_action = max(visit_counts.keys(), key=lambda a: visit_counts[a])

        total_visits = sum(visit_counts.values())
        visit_dist = {act: count / max(1, total_visits) for act, count in visit_counts.items()}

        return {
            "best_action": best_action,
            "visit_counts": visit_counts,
            "visit_distribution": visit_dist,
            "root_value": root.q_value,
            "num_simulations": total_visits,
        }

    def advise(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Advisory hook for agent controller."""
        search_res = self.search(obs)
        return {
            "recommended_farmer_action": search_res["best_action"],
            "action_probabilities": search_res["visit_distribution"],
            "mcts_root_value": search_res["root_value"],
            "policy_type": "HYBRID_MCTS",
        }
