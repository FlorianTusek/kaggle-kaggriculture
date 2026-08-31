# SPDX-License-Identifier: MIT
"""Local Benchmarking & Evaluation Suite for Kaggriculture.

Runs head-to-head simulations across multiple 720-turn seasons to evaluate
and compare Heuristic Baseline, Behavioral Cloning, and PPO RL agents.
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.env import KaggricultureEnv
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
        print(f"Loaded Behavioral Cloning Policy (model exists: {policy.is_loaded})")
    elif agent_type == "ppo":
        policy = PPOPolicy()
        print(f"Loaded PPO RL Policy (model exists: {policy.is_loaded})")
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
                from src.env import ACTION_LOOKUP
                act_idx = ACTION_LOOKUP.index(act_str) if act_str in ACTION_LOOKUP else 0
            elif agent_type == "ppo" and policy and policy.is_loaded:
                act_str = policy.predict_action(env.obs)
                from src.env import ACTION_LOOKUP
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
    
    print(f"\n--- Benchmark Summary ---")
    print(f"  Win Rate: {win_rate:.1f}% ({wins}/{n_episodes} wins, {draws} draws)")
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Kaggriculture Agents")
    parser.add_argument("--agent", choices=["bc", "ppo", "baseline"], default="bc", help="Agent type to evaluate")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--output", default="models/eval_results.json", help="Output results file")
    args = parser.parse_args()
    
    res = evaluate_agent_matchup(agent_type=args.agent, n_episodes=args.episodes)
    with open(args.output, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nResults saved to {args.output}")
