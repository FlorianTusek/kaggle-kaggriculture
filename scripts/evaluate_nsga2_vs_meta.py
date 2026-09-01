# SPDX-License-Identifier: MIT
"""Head-to-head 10-match evaluation between NSGA-II Champion and Top Meta Agent."""

import sys
import numpy as np
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from submissions.nsga2_champion_agent import agent as nsga2_fn
from submissions.meta_agent import agent as meta_fn
from scripts.evaluate_vs_meta import run_single_match

def run_benchmark():
    print("=" * 70)
    print("  RUNNING 10-MATCH BENCHMARK: NSGA-II CHAMPION vs TOP META (>2900 Elo)")
    print("=" * 70)

    # Seat 1: NSGA-II P0 vs Meta P1
    print("\n--- Seat 1: NSGA-II (Player 0) vs Top Meta (Player 1) ---")
    s1_nsga = []
    s1_meta = []
    s1_wins = 0
    for i in range(5):
        res = run_single_match(nsga2_fn, meta_fn)
        s1_nsga.append(res["p0_money"])
        s1_meta.append(res["p1_money"])
        if res["p0_money"] > res["p1_money"]:
            s1_wins += 1
        print(f"  Match {i+1}: NSGA-II = ${res['p0_money']:,.2f} | Meta = ${res['p1_money']:,.2f} | Margin: +${res['p0_money'] - res['p1_money']:,.2f}")

    # Seat 2: Meta P0 vs NSGA-II P1
    print("\n--- Seat 2: Top Meta (Player 0) vs NSGA-II (Player 1) ---")
    s2_nsga = []
    s2_meta = []
    s2_wins = 0
    for i in range(5):
        res = run_single_match(meta_fn, nsga2_fn)
        s2_meta.append(res["p0_money"])
        s2_nsga.append(res["p1_money"])
        if res["p1_money"] > res["p0_money"]:
            s2_wins += 1
        print(f"  Match {i+1}: Meta = ${res['p0_money']:,.2f} | NSGA-II = ${res['p1_money']:,.2f} | Margin: +${res['p1_money'] - res['p0_money']:,.2f}")

    all_nsga = s1_nsga + s2_nsga
    all_meta = s1_meta + s2_meta

    mean_nsga = float(np.mean(all_nsga))
    mean_meta = float(np.mean(all_meta))
    total_wins = s1_wins + s2_wins

    print("\n" + "=" * 70)
    print("  OVERALL BENCHMARK RESULTS (10 Full Matches)")
    print(f"  NSGA-II Champion Mean Bank: ${mean_nsga:,.2f} (+/- ${np.std(all_nsga):,.2f})")
    print(f"  Top Meta Agent Mean Bank:   ${mean_meta:,.2f} (+/- ${np.std(all_meta):,.2f})")
    print(f"  Net Profit Margin:          ${mean_nsga - mean_meta:+,.2f}")
    print(f"  Seat 1 Win Rate:            {s1_wins}/5 ({s1_wins/5*100:.0f}%)")
    print(f"  Seat 2 Win Rate:            {s2_wins}/5 ({s2_wins/5*100:.0f}%)")
    print(f"  Overall Match Win Rate:     {total_wins}/10 ({total_wins/10*100:.0f}%)")
    print("=" * 70)

if __name__ == "__main__":
    run_benchmark()
