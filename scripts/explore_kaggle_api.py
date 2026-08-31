"""Explore Kaggle API for Kaggriculture replays and leaderboard."""
import json
import os
import sys

from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

COMP = "kaggriculture"

# 1. Leaderboard
print("=== LEADERBOARD (top 20) ===")
try:
    lb = api.competition_leaderboard_view(COMP)
    print(f"Total entries: {len(lb)}")
    for entry in lb[:20]:
        print(f"  Rank {entry.rank} - {entry.teamName} (Score: {entry.score})")
except Exception as e:
    print(f"Leaderboard error: {e}")

# 2. Check for episode/replay endpoints
print("\n=== EXPLORING REPLAY API ===")

# Try the competitions submissions endpoint
try:
    subs = api.competitions_submissions_list(COMP)
    print(f"Our submissions: {len(subs)}")
    for s in subs[:5]:
        print(f"  {s}")
except Exception as e:
    print(f"Submissions error: {e}")

# Try dataset search for replays
print("\n=== SEARCHING FOR REPLAY DATASETS ===")
try:
    datasets = api.dataset_list(search="kaggriculture replays")
    print(f"Found {len(datasets)} replay datasets")
    for ds in datasets[:10]:
        print(f"  {ds.ref} - {ds.title}")
except Exception as e:
    print(f"Dataset search error: {e}")

# Also check the kaggle_environments API
print("\n=== CHECKING KAGGLE_ENVIRONMENTS ===")
try:
    import kaggle_environments
    print(f"kaggle_environments version: {kaggle_environments.__version__}")
    env = kaggle_environments.make("kaggriculture")
    print(f"Environment created: {env.name}")
    print(f"Specification keys: {list(env.specification.keys()) if hasattr(env, 'specification') else 'N/A'}")
except Exception as e:
    print(f"kaggle_environments error: {e}")

# Check if there's a replay download endpoint
print("\n=== CHECKING API METHODS ===")
api_methods = [m for m in dir(api) if 'replay' in m.lower() or 'episode' in m.lower()]
print(f"Replay/Episode related methods: {api_methods}")

# Also check competition-related methods
comp_methods = [m for m in dir(api) if 'competition' in m.lower()]
print(f"Competition methods: {comp_methods}")
