# SPDX-License-Identifier: MIT
"""Kaggriculture Agent Policy Implementation with ML Behavioral Cloning Policy Integration."""

import os
import math
from typing import Dict, List, Any, Tuple, Optional

from src.constants import BOARD_SIZE, HALF_BOARD, TURNS_PER_DAY, SHED_CAPACITY, MAX_MARKET_ORDERS_PER_TURN, CROPS
from src.safety import collect_safety_jobs, SafetyLayer
from src.strategy import StrategyPlanner, _open_tiles, _shed_tile, _dist, _step_toward
from src.market import MarketOptimizer
from src.models import BehavioralCloningPolicy

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
    'use_ml_policy': True,
}

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

    # Idle workers drop inventory at shed if adjacent, or execute ML policy recommendation
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
    """Modular Kaggriculture Baseline Agent leveraging SafetyLayer, StrategyPlanner, MarketOptimizer, BehavioralCloningPolicy, and StrategyEnsemble."""

    def __init__(self, policy: Optional[Dict[str, Any]] = None, model_path: Optional[str] = None):
        self.policy = policy if policy is not None else DEFAULT_POLICY
        self.safety_layer = SafetyLayer(self.policy)
        self.strategy_planner = StrategyPlanner(self.policy)
        self.market_optimizer = MarketOptimizer(self.policy)
        
        # Initialize RL PPO Policy or Behavioral Cloning Policy
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

        # Strategy Ensemble Meta-Controller (Phase 5)
        self.ensemble = None
        if self.policy.get("use_ensemble", True):
            try:
                from src.ensemble import StrategyEnsemble
                self.ensemble = StrategyEnsemble(self.policy)
            except Exception:
                self.ensemble = None

    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        # If bc_policy was manually assigned/mocked (e.g. in unit tests), use manual advice path
        if self.bc_policy is not None and hasattr(self.bc_policy, 'advise') and not getattr(self.ensemble, '_policies_initialized', False):
            try:
                ml_advice = self.bc_policy.advise(obs)
                me = obs["farms"][obs["player"]]
                priv = obs["private"]
                tiles = me["tiles"]
                safety_jobs = self.safety_layer.get_jobs(obs, me, priv)
                free_tiles = _open_tiles(tiles)
                plant_jobs = self.strategy_planner.plan_planting_jobs(obs, me, priv, free_tiles)
                all_jobs = safety_jobs + plant_jobs
                worker_ops = _assign_worker_ops(obs, self.policy, me, priv, all_jobs, ml_advice=ml_advice)
                market_orders = self.market_optimizer.plan_market_orders(obs, me, priv)
                return {
                    "farmer": worker_ops[0],
                    "hands": worker_ops[1:],
                    "market": market_orders
                }
            except Exception:
                pass

        if self.ensemble is not None:
            return self.ensemble.act(obs)

        me = obs["farms"][obs["player"]]
        priv = obs["private"]
        tiles = me["tiles"]

        # 0. Query ML policy advice if available
        ml_advice = None
        if self.bc_policy is not None and self.bc_policy.is_loaded:
            try:
                ml_advice = self.bc_policy.advise(obs)
            except Exception:
                ml_advice = None

        # 1. Collect Safety Layer jobs (highest priority)
        safety_jobs = self.safety_layer.get_jobs(obs, me, priv)

        # 2. Collect Strategy Planner jobs (planting)
        free_tiles = _open_tiles(tiles)
        plant_jobs = self.strategy_planner.plan_planting_jobs(obs, me, priv, free_tiles)

        all_jobs = safety_jobs + plant_jobs

        # 3. Assign worker actions (augmented by ML policy for idle workers)
        worker_ops = _assign_worker_ops(obs, self.policy, me, priv, all_jobs, ml_advice=ml_advice)

        # 4. Plan optimized market orders
        market_orders = self.market_optimizer.plan_market_orders(obs, me, priv)

        return {
            "farmer": worker_ops[0],
            "hands": worker_ops[1:],
            "market": market_orders
        }

def agent_entrypoint(obs: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Kaggle environment entrypoint function."""
    agent_inst = KaggricultureAgent()
    return agent_inst.act(obs)
