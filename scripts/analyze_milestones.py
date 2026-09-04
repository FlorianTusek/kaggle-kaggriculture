# SPDX-License-Identifier: MIT
"""Performance and Milestone Analysis for 100k Generation League Run."""

import json
import numpy as np
from datetime import datetime, timedelta

def main():
    eval_data = []
    with open("models/league/league_history.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("eval") and "vs_baseline" in d["eval"]:
                gen = d["gen"]
                steps = d["timesteps"]
                base_bank = float(d["eval"]["vs_baseline"].get("mean_bank", 0))
                meta_bank = float(d["eval"].get("vs_top_meta", {}).get("mean_bank", 0))
                eval_data.append((gen, steps, base_bank, meta_bank))

    print(f"Total evaluation records parsed: {len(eval_data)}")

    # Epoch Summary
    bins = [
        (5600, 10000, "Early League Convergence"),
        (10000, 15000, "Mid-League Counter-Play"),
        (15000, 20000, "20k Stabilization"),
        (20000, 22500, "20k-22.5k Expansion"),
        (22500, 24500, "Current Active Window (Gen 22.5k-24.2k)")
    ]

    print("\n" + "=" * 80)
    print("  HISTORICAL PERFORMANCE BY LEAGUE GENERATION EPOCHS")
    print("=" * 80)
    print(f"{'Epoch Window':<35} | {'Mean vs Base':<15} | {'Max vs Base':<14} | {'Mean vs Meta':<14} | {'Max vs Meta':<14}")
    print("-" * 80)

    for low, high, label in bins:
        subset = [r for r in eval_data if low <= r[0] < high]
        if subset:
            avg_base = np.mean([r[2] for r in subset])
            max_base = np.max([r[2] for r in subset])
            avg_meta = np.mean([r[3] for r in subset])
            max_meta = np.max([r[3] for r in subset])
            print(f"{label} (Gen {low}-{high})".ljust(35) + f" | ${avg_base:>12,.2f}  | ${max_base:>11,.2f}  | ${avg_meta:>11,.2f}  | ${max_meta:>11,.2f}")

    # Recent evaluation window
    print("\n" + "=" * 80)
    print("  LATEST 10 DETAILED EVALUATION RUNS (Current Generation 23,600 - 24,100)")
    print("=" * 80)
    for r in eval_data[-10:]:
        print(f"  Gen {r[0]:05d} ({r[1]:,} steps): vs Baseline = ${r[2]:>10,.2f} | vs Top Meta = ${r[3]:>10,.2f}")

    # Read current summary
    with open("models/league/league_summary.json", "r", encoding="utf-8") as f:
        summary = json.load(f)

    current_gen = int(summary.get("current_generation", 24126))
    fps = float(summary.get("fps", 358.1))
    steps_per_gen = 2048
    now = datetime.fromisoformat("2026-09-03T20:35:44")

    milestones = [
        (25000, "Gen 25,000 (Quarter Way)"),
        (50000, "Gen 50,000 (Halfway Milestone)"),
        (75000, "Gen 75,000 (Three-Quarter Mark)"),
        (100000, "Gen 100,000 (Final 100k League Completion)")
    ]

    print("\n" + "=" * 80)
    print("  RECALCULATED MILESTONE TIME ESTIMATES")
    print(f"  Current Status: Generation {current_gen:,} / 100,000 | Speed: {fps:.1f} FPS ({steps_per_gen/fps:.2f}s/gen)")
    print("=" * 80)

    for m_gen, m_label in milestones:
        rem_gens = max(0, m_gen - current_gen)
        rem_sec = (rem_gens * steps_per_gen) / fps
        rem_hours = rem_sec / 3600.0
        est_time = now + timedelta(seconds=rem_sec)
        print(f"  {m_label:<42}: in {rem_hours:>5.1f}h -> {est_time.strftime('%A, %b %d at %H:%M')}")

if __name__ == "__main__":
    main()
