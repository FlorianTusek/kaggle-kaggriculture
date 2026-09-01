# SPDX-License-Identifier: MIT
"""NSGA-II Multi-Objective Evolutionary Optimization for Kaggriculture Champion Policy.

Optimizes conflicting multi-objective criteria using Non-dominated Sorting Genetic Algorithm II:
1. Maximize Net Bank Balance (Mean revenue extracted across both Player 0 and Player 1 seats).
2. Maximize Profit Margin vs Top Meta Agent (>2900 Elo Kaggle benchmark).
3. Maximize Win Rate & Minimize Variance across seats.

Parameterizes the high-ceiling distilled tactical controller with Pareto-optimal hyperparameters.
"""

import os
import sys
import json
import time
import copy
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

# Prioritize site-packages
sys.path.insert(0, r"C:\Python310\lib\site-packages")
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize

from src.env import KaggricultureEnv
from submissions.meta_agent import agent as meta_agent_fn
import main as main_module

# 10-Dimensional Hyperparameter Optimization Vector:
# [0] glut_weight_melon (1.5 to 6.0)
# [1] glut_weight_strawberry (1.0 to 4.5)
# [2] glut_weight_wool (1.5 to 5.5)
# [3] glut_weight_milk (1.0 to 4.5)
# [4] split_cap_melon (16 to 96)
# [5] split_cap_strawberry (16 to 96)
# [6] split_cap_milk (8 to 48)
# [7] split_cap_wool (8 to 48)
# [8] split_start_turn (0 to 48)
# [9] race_weight (0.2 to 3.0)

PARAM_BOUNDS_LOWER = np.array([
    1.5, 1.0, 1.5, 1.0,        # glut weights: melon, strawberry, wool, milk
    16.0, 16.0, 8.0, 8.0,      # split caps: melon, strawberry, milk, wool
    0.0,                       # split start turn
    0.2                        # race weight
], dtype=np.float32)

PARAM_BOUNDS_UPPER = np.array([
    6.0, 4.5, 5.5, 4.5,        # glut weights: melon, strawberry, wool, milk
    96.0, 96.0, 48.0, 48.0,    # split caps: melon, strawberry, milk, wool
    48.0,                      # split start turn
    3.0                        # race weight
], dtype=np.float32)


def decode_chromosome(x: np.ndarray) -> Dict[str, Any]:
    """Decodes continuous 10-parameter chromosome into executable tactical agent configuration."""
    return {
        "glut_weights": {
            "MELON": float(x[0]),
            "STRAWBERRY": float(x[1]),
            "WOOL": float(x[2]),
            "MILK": float(x[3]),
            "EGG": 1.0,
            "WHEAT": 1.0,
            "TOMATO": 1.0,
            "CARROT": 1.0,
            "FERTILIZER": 1.0
        },
        "split_caps": {
            "MELON": int(round(x[4])),
            "STRAWBERRY": int(round(x[5])),
            "MILK": int(round(x[6])),
            "WOOL": int(round(x[7]))
        },
        "split_start_turn": int(round(x[8])),
        "race_weight": float(x[9])
    }


def create_agent_from_chromosome(config: Dict[str, Any]):
    """Returns an isolated agent function executing the champion controller with chromosome hyperparameters."""
    glut_weights = config.get("glut_weights", main_module._GLUT_WEIGHT)
    split_caps = config.get("split_caps", main_module._C94_SPLIT_CAPS)
    split_start = config.get("split_start_turn", 24)
    race_weight = config.get("race_weight", 1.0)

    def parameterized_agent(obs, conf=None):
        orig_glut = main_module._GLUT_WEIGHT
        orig_caps = main_module._C94_SPLIT_CAPS
        orig_start = main_module._C94_SPLIT_START
        orig_race = main_module._RACE_WEIGHT

        try:
            main_module._GLUT_WEIGHT = glut_weights
            main_module._C94_SPLIT_CAPS = split_caps
            main_module._C94_SPLIT_START = split_start
            main_module._RACE_WEIGHT = race_weight

            return main_module.agent(obs, conf)
        finally:
            main_module._GLUT_WEIGHT = orig_glut
            main_module._C94_SPLIT_CAPS = orig_caps
            main_module._C94_SPLIT_START = orig_start
            main_module._RACE_WEIGHT = orig_race

    return parameterized_agent


def evaluate_candidate_match(candidate_fn, opponent_fn, seat: int = 0) -> Tuple[float, float]:
    """Run a single 720-turn match between candidate and opponent on KaggricultureEnv."""
    env = KaggricultureEnv(max_turns=720, opponent_agent=None)
    env.reset()

    for turn in range(720):
        obs0 = env._get_obs_dict(0)
        obs1 = env._get_obs_dict(1)

        if seat == 0:
            act0 = candidate_fn(obs0, None)
            act1 = opponent_fn(obs1, None)
        else:
            act0 = opponent_fn(obs0, None)
            act1 = candidate_fn(obs1, None)

        env.execute_agent_turn(act0, player_idx=0)
        env.execute_agent_turn(act1, player_idx=1)

        env.current_turn += 1
        hour = env.current_turn % 24
        if hour == 23:
            for tiles, shed in [(env.tiles, env.shed), (env.opponent_tiles, env.opponent_shed)]:
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

    if seat == 0:
        return float(env.money), float(env.opponent_money)
    else:
        return float(env.opponent_money), float(env.money)


class KaggricultureNSGA2Problem(Problem):
    """Pymoo Problem Definition for Multi-Objective Kaggriculture Hyperparameter Evolution."""

    def __init__(self, n_matches_per_eval: int = 2):
        super().__init__(
            n_var=len(PARAM_BOUNDS_LOWER),
            n_obj=3,     # 3 Objectives: -MeanBank, -MarginVsMeta, -WinRate
            n_ieq_constr=0,
            xl=PARAM_BOUNDS_LOWER,
            xu=PARAM_BOUNDS_UPPER
        )
        self.n_matches_per_eval = n_matches_per_eval

    def _evaluate(self, X: np.ndarray, out: Dict[str, Any], *args, **kwargs):
        n_candidates = len(X)
        F = np.zeros((n_candidates, 3), dtype=np.float64)

        for i in range(n_candidates):
            config = decode_chromosome(X[i])
            candidate_fn = create_agent_from_chromosome(config)

            my_banks = []
            opp_banks = []
            margins = []
            wins = 0
            total_matches = 0

            # Matches vs Top Meta across Seat 0 & Seat 1
            for seat in [0, 1]:
                for _ in range(self.n_matches_per_eval):
                    my_b, opp_b = evaluate_candidate_match(candidate_fn, meta_agent_fn, seat=seat)
                    my_banks.append(my_b)
                    opp_banks.append(opp_b)
                    margins.append(my_b - opp_b)
                    if my_b > opp_b:
                        wins += 1
                    total_matches += 1

            mean_bank = np.mean(my_banks)
            mean_margin = np.mean(margins)
            win_rate = wins / max(1, total_matches)
            bank_std = np.std(my_banks)

            # Objectives to minimize:
            # f1: -Mean Bank ($)
            # f2: -Profit Margin ($)
            # f3: -Win Rate (%) + Downside Variance Penalty
            F[i, 0] = -float(mean_bank) / 10000.0
            F[i, 1] = -float(mean_margin) / 10000.0
            F[i, 2] = -float(win_rate * 100.0) + float(bank_std / 50000.0)

        out["F"] = F


def run_nsga2(
    pop_size: int = 16,
    n_gen: int = 8,
    n_matches_per_eval: int = 2,
    output_dir: str = "models/nsga2",
    seed: int = 42
) -> Dict[str, Any]:
    """Run NSGA-II Evolutionary Optimization and export the Pareto-optimal champion policy."""
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("  STARTING NSGA-II MULTI-OBJECTIVE HYPERPARAMETER EVOLUTION")
    print(f"  Population Size: {pop_size} | Generations: {n_gen} | Matches/Seat: {n_matches_per_eval} | Seed: {seed}")
    print("  Objectives: [1] Maximize Bank ($), [2] Maximize Margin vs Top Meta ($), [3] Maximize Win Rate (%)")
    print("=" * 70)

    problem = KaggricultureNSGA2Problem(n_matches_per_eval=n_matches_per_eval)

    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(prob=0.15, eta=20),
        eliminate_duplicates=True
    )

    start_time = time.time()
    res = minimize(
        problem,
        algorithm,
        ("n_gen", n_gen),
        seed=seed,
        verbose=True
    )
    elapsed = time.time() - start_time

    # Process Pareto Front
    pareto_X = res.X
    pareto_F = res.F

    if pareto_X.ndim == 1:
        pareto_X = np.expand_dims(pareto_X, axis=0)
        pareto_F = np.expand_dims(pareto_F, axis=0)

    pareto_solutions = []
    best_composite_idx = 0
    best_composite_score = -float("inf")

    for idx, (x_vec, f_vec) in enumerate(zip(pareto_X, pareto_F)):
        cfg = decode_chromosome(x_vec)
        real_bank = -f_vec[0] * 10000.0
        real_margin = -f_vec[1] * 10000.0

        sol_record = {
            "solution_id": idx + 1,
            "mean_bank": float(real_bank),
            "mean_margin": float(real_margin),
            "score_composite": float(real_bank + real_margin),
            "chromosome": [float(v) for v in x_vec],
            "config": cfg
        }
        pareto_solutions.append(sol_record)

        if (real_bank + real_margin) > best_composite_score:
            best_composite_score = real_bank + real_margin
            best_composite_idx = idx

    champion_sol = pareto_solutions[best_composite_idx]
    champion_config = champion_sol["config"]

    print("\n" + "=" * 70)
    print(f"  NSGA-II OPTIMIZATION COMPLETE in {elapsed:.1f}s!")
    print(f"  Pareto Front Size: {len(pareto_solutions)} trade-off solutions discovered.")
    print(f"  Top Champion: Mean Bank = ${champion_sol['mean_bank']:,.2f} | Margin = ${champion_sol['mean_margin']:+,.2f}")
    print("=" * 70)

    # Save Pareto Front & Champion Config
    pareto_path = os.path.join(output_dir, "pareto_front.json")
    with open(pareto_path, "w", encoding="utf-8") as f:
        json.dump(pareto_solutions, f, indent=2)

    champion_path = os.path.join(output_dir, "champion_policy.json")
    with open(champion_path, "w", encoding="utf-8") as f:
        json.dump(champion_config, f, indent=2)

    # Generate deployable submission script based on champion policy
    generate_nsga2_champion_submission(champion_config, "submissions/nsga2_champion_agent.py")

    return {
        "elapsed_seconds": elapsed,
        "n_pareto": len(pareto_solutions),
        "champion": champion_sol,
        "pareto_front": pareto_solutions
    }


def generate_nsga2_champion_submission(config: Dict[str, Any], output_file: str = "submissions/nsga2_champion_agent.py"):
    """Export the Pareto-optimal NSGA-II policy as an autonomous Kaggle submission agent."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    glut_weights = config.get("glut_weights", main_module._GLUT_WEIGHT)
    split_caps = config.get("split_caps", main_module._C94_SPLIT_CAPS)
    split_start = config.get("split_start_turn", 24)
    race_weight = config.get("race_weight", 1.0)

    code = f'''# SPDX-License-Identifier: MIT
"""Kaggle Submission Agent: NSGA-II Multi-Objective Optimized Champion.
Generated by NSGA-II Pareto Multi-Objective Evolution.
"""

import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

import main as main_module

# NSGA-II Discovered Pareto-Optimal Hyperparameters
NSGA2_GLUT_WEIGHTS = {json.dumps(glut_weights, indent=4)}
NSGA2_SPLIT_CAPS = {json.dumps(split_caps, indent=4)}
NSGA2_SPLIT_START = {split_start}
NSGA2_RACE_WEIGHT = {race_weight}

def agent(obs, configuration=None):
    orig_glut = main_module._GLUT_WEIGHT
    orig_caps = main_module._C94_SPLIT_CAPS
    orig_start = main_module._C94_SPLIT_START
    orig_race = main_module._RACE_WEIGHT

    try:
        main_module._GLUT_WEIGHT = NSGA2_GLUT_WEIGHTS
        main_module._C94_SPLIT_CAPS = NSGA2_SPLIT_CAPS
        main_module._C94_SPLIT_START = NSGA2_SPLIT_START
        main_module._RACE_WEIGHT = NSGA2_RACE_WEIGHT

        return main_module.agent(obs, configuration)
    finally:
        main_module._GLUT_WEIGHT = orig_glut
        main_module._C94_SPLIT_CAPS = orig_caps
        main_module._C94_SPLIT_START = orig_start
        main_module._RACE_WEIGHT = orig_race
'''
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(code)
    print(f"Exported NSGA-II Champion agent to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NSGA-II Multi-Objective Evolutionary Optimization")
    parser.add_argument("--pop-size", type=int, default=16, help="Population size per generation")
    parser.add_argument("--generations", type=int, default=8, help="Number of evolutionary generations")
    parser.add_argument("--matches", type=int, default=2, help="Matches per evaluation per seat")
    parser.add_argument("--output-dir", default="models/nsga2", help="Output directory for Pareto front")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    run_nsga2(
        pop_size=args.pop_size,
        n_gen=args.generations,
        n_matches_per_eval=args.matches,
        output_dir=args.output_dir,
        seed=args.seed
    )
