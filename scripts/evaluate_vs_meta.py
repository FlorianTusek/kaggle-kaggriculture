# SPDX-License-Identifier: MIT
"""Phase 12: Head-to-Head Evaluation script comparing Phase 11 main.py agent against Zero to Top Meta Agent (submissions/meta_agent.py)."""

import sys
import json
import os
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from main import agent as main_agent_fn
from submissions.meta_agent import agent as meta_agent_fn
from src.env import KaggricultureEnv


def run_single_match(p0_fn, p1_fn, max_turns: int = 720):
    """Run a single 720-turn match between p0_fn and p1_fn on KaggricultureEnv."""
    env = KaggricultureEnv(max_turns=max_turns, opponent_agent=None)
    obs, _ = env.reset()

    for turn in range(max_turns):
        obs0 = env._get_obs_dict(0)
        obs1 = env._get_obs_dict(1)

        action0 = p0_fn(obs0, None)
        action1 = p1_fn(obs1, None)

        env.execute_agent_turn(action0, player_idx=0)
        env.execute_agent_turn(action1, player_idx=1)

        env.current_turn += 1
        hour = env.current_turn % 24

        # Hour 23: un-nerfed daily yield accumulation
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

    return {
        "p0_money": float(env.money),
        "p1_money": float(env.opponent_money),
        "winner": "P0" if env.money > env.opponent_money else ("P1" if env.opponent_money > env.money else "DRAW"),
        "margin": abs(float(env.money) - float(env.opponent_money))
    }


def run_head_to_head_evaluation(n_matches: int = 5):
    """Run a comprehensive 10-match evaluation (5 as P0, 5 as P1)."""
    print("=" * 70)
    print("  PHASE 12 HEAD-TO-HEAD BENCHMARK: Phase 11 main.py vs Top Meta Opponent")
    print("=" * 70)

    p0_results = []
    print(f"\n--- Seat 1: Phase 11 Agent (Player 0) vs. Top Meta Agent (Player 1) [{n_matches} Matches] ---")
    for m in range(n_matches):
        res = run_single_match(main_agent_fn, meta_agent_fn)
        p0_results.append(res)
        print(f"  Match {m+1:02d}: Phase 11 = ${res['p0_money']:,.2f} | Top Meta = ${res['p1_money']:,.2f} -> {res['winner']} (Margin: ${res['margin']:,.2f})")

    p1_results = []
    print(f"\n--- Seat 2: Top Meta Agent (Player 0) vs. Phase 11 Agent (Player 1) [{n_matches} Matches] ---")
    for m in range(n_matches):
        res = run_single_match(meta_agent_fn, main_agent_fn)
        p1_results.append(res)
        p1_winner = "Phase 11" if res['p1_money'] > res['p0_money'] else ("Top Meta" if res['p0_money'] > res['p1_money'] else "DRAW")
        print(f"  Match {m+1:02d}: Phase 11 = ${res['p1_money']:,.2f} | Top Meta = ${res['p0_money']:,.2f} -> {p1_winner} (Margin: ${res['margin']:,.2f})")

    # Aggregate Statistics
    phase11_scores = [r['p0_money'] for r in p0_results] + [r['p1_money'] for r in p1_results]
    meta_scores = [r['p1_money'] for r in p0_results] + [r['p0_money'] for r in p1_results]

    phase11_wins = sum(1 for r in p0_results if r['winner'] == 'P0') + sum(1 for r in p1_results if r['winner'] == 'P1')
    total_matches = n_matches * 2
    win_rate = (phase11_wins / total_matches) * 100.0

    mean_phase11 = sum(phase11_scores) / total_matches
    mean_meta = sum(meta_scores) / total_matches
    net_advantage = mean_phase11 - mean_meta

    print("\n" + "=" * 70)
    print("  HEAD-TO-HEAD BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"  Total Matches Evaluated: {total_matches}")
    print(f"  Phase 11 Agent Win Rate: {win_rate:.1f}% ({phase11_wins}/{total_matches} wins)")
    print(f"  Phase 11 Agent Mean Bank Balance: ${mean_phase11:,.2f}")
    print(f"  Top Meta Opponent Mean Bank Balance: ${mean_meta:,.2f}")
    print(f"  Net Profit Advantage: ${net_advantage:,.2f}")
    print(f"  Overall Match Winner: {'Phase 11 Agent' if mean_phase11 > mean_meta else 'Top Meta Opponent'}")
    print("=" * 70)

    summary_data = {
        "total_matches": total_matches,
        "phase11_wins": phase11_wins,
        "win_rate_percent": win_rate,
        "mean_phase11_bank": mean_phase11,
        "mean_meta_bank": mean_meta,
        "net_advantage": net_advantage,
        "overall_winner": "Phase 11 Agent" if mean_phase11 > mean_meta else "Top Meta Opponent",
        "seat0_results": p0_results,
        "seat1_results": p1_results
    }

    out_file = root_dir / "models" / "meta_eval_results.json"
    with open(out_file, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Full benchmark results saved to: {out_file}\n")

    return summary_data


if __name__ == "__main__":
    run_head_to_head_evaluation(n_matches=5)
