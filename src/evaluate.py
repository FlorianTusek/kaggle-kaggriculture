# SPDX-License-Identifier: MIT
"""Local Benchmarking & Evaluation Suite for Kaggriculture.

Runs head-to-head simulations across multiple 720-turn seasons to evaluate
and compare Heuristic Baseline, Phase 3 Behavioral Cloning (Hybrid), and Phase 4 PPO RL agents.
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.env import KaggricultureEnv, ACTION_LOOKUP
from src.agent import KaggricultureAgent
from src.models import BehavioralCloningPolicy, PPOPolicy


def evaluate_agent_matchup(
    agent_type: str = "bc",
    n_episodes: int = 10,
    max_turns: int = 720,
    seed: int = 42
) -> Dict[str, Any]:
    """Run head-to-head simulation benchmarking against baseline opponent."""
    print(f"\n=======================================================")
    print(f"  Evaluating {agent_type.upper()} Policy vs. Baseline Opponent")
    print(f"  Matches: {n_episodes} | Episode Length: {max_turns} turns")
    print(f"=======================================================")
    
    np.random.seed(seed)
    env = KaggricultureEnv(max_turns=max_turns, opponent_agent=KaggricultureAgent())
    
    # Load policy
    if agent_type == "bc":
        policy = BehavioralCloningPolicy()
        print(f"Loaded Behavioral Cloning Policy (model loaded: {policy.is_loaded})")
    elif agent_type == "ppo":
        policy = PPOPolicy()
        print(f"Loaded PPO RL Policy (model loaded: {policy.is_loaded})")
    else:
        policy = None
        print(f"Using default random/pass baseline")
        
    scores_agent = []
    scores_opponent = []
    wins = 0
    draws = 0
    
    for ep in range(n_episodes):
        obs_vec, info = env.reset(seed=seed + ep)
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            if agent_type == "bc" and policy and policy.is_loaded:
                act_str = policy.predict_farmer_action(env.obs)
                act_idx = ACTION_LOOKUP.index(act_str) if act_str in ACTION_LOOKUP else 0
            elif agent_type == "ppo" and policy and policy.is_loaded:
                act_str = policy.predict_action(env.obs)
                act_idx = ACTION_LOOKUP.index(act_str) if act_str in ACTION_LOOKUP else 0
            else:
                act_idx = 0
                
            obs_vec, reward, terminated, truncated, step_info = env.step(act_idx)
            
        final_agent_money = env.money
        final_opp_money = env.opponent_money
        
        scores_agent.append(final_agent_money)
        scores_opponent.append(final_opp_money)
        
        if final_agent_money > final_opp_money:
            wins += 1
            result_str = "WIN"
        elif final_agent_money == final_opp_money:
            draws += 1
            result_str = "DRAW"
        else:
            result_str = "LOSS"
            
        print(f"  Episode {ep+1:02d}: Agent = ${final_agent_money:,.2f} | Opponent = ${final_opp_money:,.2f} -> {result_str}")
        
    win_rate = (wins / n_episodes) * 100
    mean_agent = float(np.mean(scores_agent))
    mean_opp = float(np.mean(scores_opponent))
    profit_margin = mean_agent - mean_opp
    
    print(f"\n--- Benchmark Summary for {agent_type.upper()} ---")
    print(f"  Win Rate: {win_rate:.1f}% ({wins}/{n_episodes} wins, {draws} draws, {n_episodes - wins - draws} losses)")
    print(f"  Mean Agent Bank: ${mean_agent:,.2f}")
    print(f"  Mean Opponent Bank: ${mean_opp:,.2f}")
    print(f"  Net Profit Advantage: ${profit_margin:,.2f}")
    
    results = {
        "agent_type": agent_type,
        "n_episodes": n_episodes,
        "win_rate_pct": win_rate,
        "wins": wins,
        "draws": draws,
        "losses": n_episodes - wins - draws,
        "mean_agent_money": mean_agent,
        "mean_opponent_money": mean_opp,
        "profit_advantage": profit_margin,
        "scores_agent": [float(s) for s in scores_agent],
        "scores_opponent": [float(s) for s in scores_opponent]
    }
    
    return results


def run_full_comparative_benchmark(
    n_episodes: int = 10,
    output_path: str = "models/eval_comparison.json"
) -> Dict[str, Any]:
    """Run comparative benchmark across Baseline, Phase 3 BC, and Phase 4 PPO."""
    print("\n=======================================================")
    print("  PHASE 4 COMPARATIVE BENCHMARK: BC vs. PPO vs. Baseline")
    print("=======================================================")
    
    bc_results = evaluate_agent_matchup(agent_type="bc", n_episodes=n_episodes)
    ppo_results = evaluate_agent_matchup(agent_type="ppo", n_episodes=n_episodes)
    
    # Comparison summary
    bc_win_rate = bc_results["win_rate_pct"]
    ppo_win_rate = ppo_results["win_rate_pct"]
    bc_profit = bc_results["mean_agent_money"]
    ppo_profit = ppo_results["mean_agent_money"]
    
    # Decision recommendation per ROADMAP Section 4 / Phase 4
    # "If RL agent does not consistently beat the Phase 3 baseline, skip Phase 4 submission."
    ppo_beats_bc = (ppo_win_rate > bc_win_rate) or (ppo_profit > bc_profit and ppo_win_rate >= bc_win_rate)
    
    recommendation = {
        "benchmark_passed": bool(ppo_beats_bc),
        "preferred_model": "Phase 4 PPO" if ppo_beats_bc else "Phase 3 Hybrid BC",
        "rationale": (
            f"Phase 4 PPO achieved {ppo_win_rate:.1f}% win rate (${ppo_profit:,.2f} mean bank) vs "
            f"Phase 3 BC {bc_win_rate:.1f}% win rate (${bc_profit:,.2f} mean bank). "
            + ("PPO demonstrates superior performance over BC baseline." if ppo_beats_bc else
               "Phase 3 Hybrid BC agent maintains superior or equivalent performance; as per ROADMAP, Phase 3 Hybrid agent remains the primary competitive submission.")
        )
    }
    
    comparison = {
        "timestamp": "2026-08-31",
        "n_episodes": n_episodes,
        "phase3_bc": bc_results,
        "phase4_ppo": ppo_results,
        "comparison": {
            "bc_win_rate": bc_win_rate,
            "ppo_win_rate": ppo_win_rate,
            "bc_mean_profit": bc_profit,
            "ppo_mean_profit": ppo_profit,
            "profit_delta_ppo_minus_bc": float(ppo_profit - bc_profit),
        },
        "recommendation": recommendation
    }
    
    print("\n=======================================================")
    print("  FINAL COMPARISON & RECOMMENDATION")
    print("=======================================================")
    print(f"  Phase 3 Hybrid BC: Win Rate = {bc_win_rate:.1f}% | Mean Bank = ${bc_profit:,.2f}")
    print(f"  Phase 4 PPO RL:    Win Rate = {ppo_win_rate:.1f}% | Mean Bank = ${ppo_profit:,.2f}")
    print(f"  Benchmark Passed:  {recommendation['benchmark_passed']}")
    print(f"  Preferred Model:   {recommendation['preferred_model']}")
    print(f"  Rationale:         {recommendation['rationale']}")
    print("=======================================================")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"Full comparison report saved to: {output_path}")
    
    return comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Kaggriculture Agents")
    parser.add_argument("--mode", choices=["single", "compare"], default="compare", help="Benchmark mode")
    parser.add_argument("--agent", choices=["bc", "ppo", "baseline"], default="bc", help="Agent type for single mode")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--output", default="models/eval_comparison.json", help="Output results file")
    args = parser.parse_args()
    
    if args.mode == "compare":
        run_full_comparative_benchmark(n_episodes=args.episodes, output_path=args.output)
    else:
        res = evaluate_agent_matchup(agent_type=args.agent, n_episodes=args.episodes)
        with open(args.output, "w") as f:
            json.dump(res, f, indent=2)
        print(f"\nResults saved to {args.output}")
