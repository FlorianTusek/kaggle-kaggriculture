"""Analyze score distribution across episodes to set a good filter threshold."""
import json
import os
import random

REPLAY_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "replays", "kaggle_kaggriculture-episodes-2026-08-09")
files = sorted([f for f in os.listdir(REPLAY_DIR) if f.endswith(".json") and not f.endswith(".zip")])

# Sample 50 episodes for speed
sample = random.sample(files, min(50, len(files)))

scores = []
for fname in sample:
    try:
        with open(os.path.join(REPLAY_DIR, fname), "r") as f:
            ep = json.load(f)
        rewards = ep.get("rewards", [0, 0])
        for r in rewards:
            if r is not None:
                scores.append(r)
    except:
        pass

scores.sort()
print(f"Sampled {len(scores)} player scores from {len(sample)} episodes")
print(f"Min: {min(scores):.0f}")
print(f"Max: {max(scores):.0f}")
print(f"Median: {scores[len(scores)//2]:.0f}")
print(f"Mean: {sum(scores)/len(scores):.0f}")

# Percentiles
for p in [10, 25, 50, 75, 90, 95]:
    idx = int(len(scores) * p / 100)
    print(f"  P{p}: {scores[idx]:.0f}")

# Count above thresholds
for thresh in [5000, 10000, 20000, 50000, 75000, 100000]:
    count = sum(1 for s in scores if s >= thresh)
    print(f"  >= ${thresh:,}: {count}/{len(scores)} ({100*count/len(scores):.0f}%)")
