# SPDX-License-Identifier: MIT
"""Strategy Ensembling Meta-Controller for Kaggriculture.

Combines multiple policy sources into a single coherent action selection:
  1. **Safety Layer** — absolute veto authority; any safety job overrides all other advice.
  2. **Hybrid MCTS** — tree-search recommendations using PPO/BC priors (high quality, slow).
  3. **PPO RL Policy** — fast single-step neural network action selection.
  4. **Behavioral Cloning** — expert imitation policy from replay data.
  5. **Opponent Modeling** — reactive market counter-orders from OpponentTracker.

The ensemble uses a context-aware weighting scheme:
  - Early game (days 0–8): heavier BC weight (proven expert imitation).
  - Mid game (days 9–20): heavier MCTS weight (deeper search matters most).
  - Late game / liquidation (days 21+): heavier PPO weight (fast reactive play).

The Safety Layer is NOT a voting participant — it is a hard override that runs first.
"""

from typing import Dict, List, Any, Optional, Tuple
from collections import Counter

from src.safety import SafetyLayer
from src.strategy import StrategyPlanner, _open_tiles
from src.market import MarketOptimizer
from src.opponent import OpponentTracker


# Valid worker actions for ensemble voting
VALID_WORKER_ACTIONS = frozenset([
    "NORTH", "SOUTH", "EAST", "WEST",
    "PASS", "DIG", "DROP", "PICKUP", "PLACE",
    "PLANT", "WATER", "HARVEST", "FEED", "CARE",
    "FERTILIZE", "COLLECT_FERTILIZER", "BUILD_PASTURE",
])

# Phase-dependent ensemble weights: (bc_weight, ppo_weight, mcts_weight)
PHASE_WEIGHTS = {
    "early":  {"bc": 0.40, "ppo": 0.25, "mcts": 0.35},
    "mid":    {"bc": 0.20, "ppo": 0.25, "mcts": 0.55},
    "late":   {"bc": 0.15, "ppo": 0.50, "mcts": 0.35},
}


def _get_game_phase(day: int) -> str:
    """Determine the current game phase for weight selection."""
    if day <= 8:
        return "early"
    elif day <= 20:
        return "mid"
    else:
        return "late"


def _weighted_vote(candidates: Dict[str, Dict[str, float]], weights: Dict[str, float]) -> Tuple[str, Dict[str, float]]:
    """Aggregate action votes from multiple policies using weighted scores.

    Args:
        candidates: {policy_name: {action: probability, ...}}
        weights: {policy_name: weight}

    Returns:
        (best_action, merged_scores_dict)
    """
    merged: Dict[str, float] = {}

    for policy_name, action_probs in candidates.items():
        w = weights.get(policy_name, 0.0)
        if w <= 0.0:
            continue
        for action, prob in action_probs.items():
            if action not in VALID_WORKER_ACTIONS:
                continue
            merged[action] = merged.get(action, 0.0) + w * float(prob)

    if not merged:
        return "PASS", {"PASS": 1.0}

    # Normalize
    total = sum(merged.values())
    if total > 0:
        merged = {a: s / total for a, s in merged.items()}

    best_action = max(merged, key=merged.get)
    return best_action, merged


class StrategyEnsemble:
    """Unified meta-controller ensembling Safety + MCTS + PPO + BC + Opponent policies.

    Usage:
        ensemble = StrategyEnsemble(policy_config)
        action_dict = ensemble.act(obs)
    """

    def __init__(self, policy: Optional[Dict[str, Any]] = None):
        self.policy = policy if policy is not None else {}

        # Core layers
        self.safety_layer = SafetyLayer(self.policy)
        self.strategy_planner = StrategyPlanner(self.policy)
        self.market_optimizer = MarketOptimizer(self.policy)

        # ML / search policies (lazy loaded)
        self._ppo_policy = None
        self._bc_policy = None
        self._mcts_policy = None
        self._policies_initialized = False

        # Ensemble configuration
        self.use_mcts = self.policy.get("use_mcts", True)
        self.mcts_simulations = self.policy.get("mcts_simulations", 25)
        self.mcts_top_k = self.policy.get("mcts_top_k", 5)

    def _init_policies(self) -> None:
        """Lazy-initialize ML policies to avoid import-time overhead."""
        if self._policies_initialized:
            return
        self._policies_initialized = True

        # PPO Policy
        try:
            from src.models import PPOPolicy
            ppo = PPOPolicy()
            if ppo.is_loaded:
                self._ppo_policy = ppo
        except Exception:
            pass

        # Behavioral Cloning Policy
        try:
            from src.models import BehavioralCloningPolicy
            bc = BehavioralCloningPolicy()
            if bc.is_loaded:
                self._bc_policy = bc
        except Exception:
            pass

        # Hybrid MCTS (using best available policy prior)
        if self.use_mcts:
            try:
                from src.mcts import HybridMCTS
                prior = self._ppo_policy or self._bc_policy
                self._mcts_policy = HybridMCTS(
                    policy_prior=prior,
                    n_simulations=self.mcts_simulations,
                    top_k_actions=self.mcts_top_k,
                )
            except Exception:
                pass

    def _collect_policy_votes(self, obs: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Query each policy for its action distribution or recommendation."""
        votes: Dict[str, Dict[str, float]] = {}

        # BC policy
        if self._bc_policy is not None and self._bc_policy.is_loaded:
            try:
                probs = self._bc_policy.predict_farmer_action_probs(obs)
                if probs:
                    votes["bc"] = probs
            except Exception:
                pass

        # PPO policy (single-point recommendation spread into soft distribution)
        if self._ppo_policy is not None and self._ppo_policy.is_loaded:
            try:
                action = self._ppo_policy.predict_action(obs)
                # Create a soft distribution: 70% on predicted, 30% uniform over rest
                n_actions = len(VALID_WORKER_ACTIONS)
                uniform_share = 0.30 / max(1, n_actions - 1)
                ppo_dist = {a: uniform_share for a in VALID_WORKER_ACTIONS}
                if action in VALID_WORKER_ACTIONS:
                    ppo_dist[action] = 0.70
                votes["ppo"] = ppo_dist
            except Exception:
                pass

        # MCTS policy
        if self._mcts_policy is not None:
            try:
                result = self._mcts_policy.search(obs)
                visit_dist = result.get("visit_distribution", {})
                if visit_dist:
                    votes["mcts"] = visit_dist
            except Exception:
                pass

        return votes

    def _assign_ensemble_worker_ops(
        self,
        obs: Dict[str, Any],
        me: Dict[str, Any],
        priv: Dict[str, Any],
        jobs: List[Dict[str, Any]],
        ensemble_action: Optional[str] = None,
    ) -> List[List[str]]:
        """Assign worker operations: safety jobs first, ensemble recommendation for idle farmer."""
        from src.strategy import _dist, _step_toward, _shed_tile
        from src.constants import TURNS_PER_DAY

        positions = [me["farmer"]] + me.get("hands", [])
        n = len(positions)
        ops: List[List[str]] = [["PASS"] for _ in range(n)]
        busy = [False] * n
        tiles = me["tiles"]

        # Assign safety + strategy jobs by proximity
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

        # Idle workers: farmer (index 0) gets ensemble recommendation, others go to shed
        sx, sy = _shed_tile(tiles)
        for i in range(n):
            if not busy[i]:
                if i == 0 and ensemble_action and ensemble_action in VALID_WORKER_ACTIONS:
                    # Only allow safe idle actions for the farmer from the ensemble
                    if ensemble_action in ("NORTH", "SOUTH", "EAST", "WEST", "DROP",
                                            "DIG", "HARVEST", "WATER", "CARE", "FEED",
                                            "COLLECT_FERTILIZER"):
                        ops[i] = [ensemble_action]
                    else:
                        ops[i] = _step_toward(positions[i], (sx, sy))
                elif positions[i] == (sx, sy):
                    ops[i] = ["DROP"]
                elif obs.get("hour", 0) == TURNS_PER_DAY - 1:
                    ops[i] = _step_toward(positions[i], (sx, sy))

        return ops

    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the full ensemble pipeline and return agent action dict.

        Pipeline order:
          1. Initialize ML policies (lazy, first call only).
          2. Safety Layer collects hard-priority jobs.
          3. Strategy Planner collects planting jobs.
          4. Ensemble votes on idle-farmer action from PPO + BC + MCTS.
          5. Market Optimizer plans market orders (with opponent counter-strategy).
          6. Assemble final action dict.
        """
        self._init_policies()

        me = obs["farms"][obs["player"]]
        priv = obs["private"]
        tiles = me["tiles"]
        day = obs.get("day", 0)

        # 1. Safety Layer (absolute priority)
        safety_jobs = self.safety_layer.get_jobs(obs, me, priv)

        # 2. Strategy Planner (planting)
        free_tiles = _open_tiles(tiles)
        plant_jobs = self.strategy_planner.plan_planting_jobs(obs, me, priv, free_tiles)

        all_jobs = safety_jobs + plant_jobs

        # 3. Ensemble voting for idle farmer action
        ensemble_action = None
        ensemble_scores = {}
        if not safety_jobs:
            # Only run expensive ensemble when safety isn't urgent
            votes = self._collect_policy_votes(obs)
            if votes:
                phase = _get_game_phase(day)
                weights = PHASE_WEIGHTS[phase]
                ensemble_action, ensemble_scores = _weighted_vote(votes, weights)

        # 4. Worker assignment (safety overrides ensemble)
        worker_ops = self._assign_ensemble_worker_ops(
            obs, me, priv, all_jobs, ensemble_action=ensemble_action
        )

        # 5. Market orders (includes opponent counter-strategy)
        market_orders = self.market_optimizer.plan_market_orders(obs, me, priv)

        return {
            "farmer": worker_ops[0],
            "hands": worker_ops[1:],
            "market": market_orders,
        }

    def advise(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Return detailed advisory dict for debugging / logging."""
        self._init_policies()

        day = obs.get("day", 0)
        phase = _get_game_phase(day)
        weights = PHASE_WEIGHTS[phase]
        votes = self._collect_policy_votes(obs)
        ensemble_action, ensemble_scores = _weighted_vote(votes, weights) if votes else ("PASS", {"PASS": 1.0})

        return {
            "game_phase": phase,
            "phase_weights": weights,
            "policy_votes": {k: dict(v) for k, v in votes.items()},
            "ensemble_scores": ensemble_scores,
            "recommended_action": ensemble_action,
            "opponent_archetype": self.market_optimizer.opponent_tracker.archetype,
            "policies_loaded": {
                "ppo": self._ppo_policy is not None and self._ppo_policy.is_loaded,
                "bc": self._bc_policy is not None and self._bc_policy.is_loaded,
                "mcts": self._mcts_policy is not None,
            },
        }
