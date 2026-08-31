"""Quick test: parse one episode to validate the format and feature extraction."""
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.replay_parser import parse_episode, extract_state_features, extract_action_features

# Pick the first episode file
REPLAY_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "replays", "kaggle_kaggriculture-episodes-2026-08-09")
files = sorted([f for f in os.listdir(REPLAY_DIR) if f.endswith(".json") and not f.endswith(".zip")])
print(f"Total JSON files: {len(files)}")

test_file = os.path.join(REPLAY_DIR, files[0])
print(f"\nParsing: {files[0]} ({os.path.getsize(test_file) / 1e6:.1f} MB)")

# Load and inspect structure
with open(test_file, "r", encoding="utf-8") as f:
    ep = json.load(f)

print(f"\nTop-level keys: {list(ep.keys())}")
print(f"Info: {ep.get('info', {})}")

# Rewards
rewards = ep.get("rewards", [])
print(f"Rewards: {rewards}")

# Steps structure
steps = ep.get("steps", [])
print(f"Number of steps: {len(steps)}")

if steps:
    # Look at first step structure
    step0 = steps[0]
    print(f"\nStep 0 type: {type(step0)}")
    if isinstance(step0, list):
        print(f"Step 0 length: {len(step0)}")
        for seat in range(min(2, len(step0))):
            agent_step = step0[seat]
            if isinstance(agent_step, dict):
                print(f"  Seat {seat} keys: {list(agent_step.keys())}")
                obs = agent_step.get("observation", {})
                action = agent_step.get("action")
                if obs:
                    print(f"  Seat {seat} obs keys: {list(obs.keys())[:20]}")
                    print(f"  Seat {seat} obs.player: {obs.get('player')}")
                    print(f"  Seat {seat} obs.day: {obs.get('day')}")
                    print(f"  Seat {seat} obs.hour: {obs.get('hour')}")
                    farms = obs.get("farms", [])
                    print(f"  Seat {seat} farms type: {type(farms)}, len: {len(farms) if isinstance(farms, (list, dict)) else 'N/A'}")
                    if isinstance(farms, list) and farms:
                        farm0 = farms[0]
                        if isinstance(farm0, dict):
                            print(f"  Seat {seat} farm[0] keys: {list(farm0.keys())}")
                    market = obs.get("market", {})
                    print(f"  Seat {seat} market keys: {list(market.keys()) if isinstance(market, dict) else market}")
                    priv = obs.get("private", {})
                    print(f"  Seat {seat} private keys: {list(priv.keys()) if isinstance(priv, dict) else 'N/A'}")
                print(f"  Seat {seat} action: {action}")
            else:
                print(f"  Seat {seat}: {type(agent_step)}")
    elif isinstance(step0, dict):
        print(f"Step 0 keys: {list(step0.keys())}")

    # Try a mid-game step
    mid = len(steps) // 2
    step_mid = steps[mid]
    if isinstance(step_mid, list) and len(step_mid) > 0:
        agent_mid = step_mid[0]
        if isinstance(agent_mid, dict):
            obs_mid = agent_mid.get("observation", {})
            action_mid = agent_mid.get("action", {})
            print(f"\nMid-game step {mid}:")
            print(f"  day={obs_mid.get('day')}, hour={obs_mid.get('hour')}")
            if isinstance(action_mid, dict):
                print(f"  action keys: {list(action_mid.keys())}")
                print(f"  farmer: {action_mid.get('farmer')}")
                hands = action_mid.get('hands', [])
                print(f"  hands ({len(hands)}): {hands[:2]}...")
                market = action_mid.get('market', [])
                print(f"  market ({len(market)}): {market[:3]}...")

# Now test the parser
print("\n=== Testing parse_episode ===")
pairs = parse_episode(test_file, min_final_score=0)
print(f"Total pairs extracted: {len(pairs)}")

if pairs:
    p = pairs[0]
    print(f"\nFirst pair:")
    print(f"  episode_id: {p['episode_id']}")
    print(f"  player_idx: {p['player_idx']}")
    print(f"  turn: {p['turn']}")
    print(f"  final_score: {p['final_score']}")
    print(f"  state keys ({len(p['state'])}): {list(p['state'].keys())[:15]}...")
    print(f"  action keys: {list(p['action'].keys())}")

    # Sample a mid-game pair
    mid_pair = pairs[len(pairs) // 2]
    print(f"\nMid-game pair (turn {mid_pair['turn']}):")
    print(f"  money: {mid_pair['state']['money']}")
    print(f"  tiles_plant: {mid_pair['state'].get('tiles_plant', 'N/A')}")
    print(f"  farmer_action: {mid_pair['action'].get('farmer_action')}")
    print(f"  market orders: {mid_pair['action'].get('num_market_orders')}")

print("\n=== DONE ===")
