# SPDX-License-Identifier: MIT
"""Parse Kaggle Kaggriculture episode replays into (state, action) training pairs.

Reads raw JSON episode files and extracts per-turn (observation, action) pairs
for behavioral cloning. Filters for high-scoring players only.

Output: a list of dicts with flattened state features and structured actions,
serialized as JSON Lines (.jsonl) for efficient streaming during training.
"""

import json
import os
import glob
import sys
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

# Import game constants for feature extraction
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.constants import (
    BOARD_SIZE, HALF_BOARD, TURNS_PER_DAY, TOTAL_TURNS,
    CROPS, ANIMALS, PRODUCTS, MAX_MARKET_ORDERS_PER_TURN
)


def extract_state_features(obs: Dict[str, Any], player_idx: int) -> Dict[str, Any]:
    """Extract a flat feature dict from a raw observation.
    
    Features:
    - turn/day/hour context
    - economic state: money, shed inventory counts
    - farm state: tile counts by type, crop counts, animal counts
    - market prices for all products
    - labor: number of hands
    - opponent signals: opponent farm size, money (if visible)
    """
    features = {}
    
    # Turn context
    features["turn"] = obs.get("step", obs.get("turn", 0))
    features["day"] = obs.get("day", features["turn"] // TURNS_PER_DAY)
    features["hour"] = obs.get("hour", features["turn"] % TURNS_PER_DAY)
    
    # Player farm
    farms = obs.get("farms", [])
    if isinstance(farms, list):
        me = farms[player_idx] if player_idx < len(farms) else {}
    elif isinstance(farms, dict):
        me = farms.get(str(player_idx), farms.get(player_idx, {}))
    else:
        me = {}
    
    # Economic state
    features["money"] = me.get("money", 0)
    
    # Private data (may not always be available in replays)
    priv = obs.get("private", {})
    shed = priv.get("shed", {})
    seeds = priv.get("seeds", {})
    
    for prod in PRODUCTS:
        features[f"shed_{prod.lower()}"] = shed.get(prod, 0)
    for crop in CROPS:
        features[f"seeds_{crop.lower()}"] = seeds.get(crop, 0)
    
    # Farm tile analysis
    tiles = me.get("tiles", [])
    tile_counts = {"empty": 0, "plant": 0, "coop": 0, "pasture": 0, "weed": 0, "locked": 0, "other": 0}
    crop_counts = {c.lower(): 0 for c in CROPS}
    animal_counts = {a.lower(): 0 for a in ANIMALS}
    unwatered = 0
    unfed = 0
    harvestable = 0
    
    for y, row in enumerate(tiles):
        for x, tile in enumerate(row):
            if tile is None:
                tile_counts["empty"] += 1
            elif tile == "LOCKED":
                tile_counts["locked"] += 1
            elif isinstance(tile, dict):
                kind = tile.get("kind", "").upper()
                if kind == "PLANT":
                    tile_counts["plant"] += 1
                    crop = tile.get("crop", "").upper()
                    if crop.lower() in crop_counts:
                        crop_counts[crop.lower()] += 1
                    if not tile.get("watered_today", False):
                        unwatered += 1
                    if tile.get("yield_units", 0) > 0:
                        harvestable += 1
                elif kind in ("COOP", "PASTURE"):
                    tile_counts[kind.lower()] += 1
                    animal = tile.get("animal", "")
                    if isinstance(animal, str) and animal.lower() in animal_counts:
                        animal_counts[animal.lower()] += 1
                    if not tile.get("fed_today", False) and tile.get("animal"):
                        unfed += 1
                    if tile.get("yield_units", 0) > 0:
                        harvestable += 1
                elif kind == "WEED":
                    tile_counts["weed"] += 1
                else:
                    tile_counts["other"] += 1
            else:
                tile_counts["other"] += 1
    
    for k, v in tile_counts.items():
        features[f"tiles_{k}"] = v
    for k, v in crop_counts.items():
        features[f"crop_{k}"] = v
    for k, v in animal_counts.items():
        features[f"animal_{k}"] = v
    
    features["unwatered_crops"] = unwatered
    features["unfed_animals"] = unfed
    features["harvestable_tiles"] = harvestable
    
    # Labor
    features["num_hands"] = len(me.get("hands", []))
    
    # Market prices
    market = obs.get("market", {})
    prices = market.get("prices", {})
    for prod in PRODUCTS:
        features[f"price_{prod.lower()}"] = prices.get(prod, 0)
    
    # Opponent info (if available)
    opp_idx = 1 - player_idx
    if isinstance(farms, list) and opp_idx < len(farms):
        opp = farms[opp_idx]
        features["opp_money"] = opp.get("money", 0)
        features["opp_num_hands"] = len(opp.get("hands", []))
    elif isinstance(farms, dict):
        opp = farms.get(str(opp_idx), farms.get(opp_idx, {}))
        features["opp_money"] = opp.get("money", 0)
        features["opp_num_hands"] = len(opp.get("hands", []))
    
    # Town shops
    town = obs.get("town", {})
    unlocked = town.get("unlocked_shops", [])
    features["num_unlocked_shops"] = len(unlocked)
    
    return features


def extract_action_features(action: Any) -> Dict[str, Any]:
    """Extract structured action features from a raw action.
    
    The action dict has: farmer (list of ops), hands (list of list of ops), market (list of orders)
    """
    if action is None:
        return {"valid": False}
    
    if not isinstance(action, dict):
        return {"valid": False}
    
    features = {"valid": True}
    
    # Farmer action
    farmer_ops = action.get("farmer", ["PASS"])
    features["farmer_action"] = farmer_ops[0] if farmer_ops else "PASS"
    
    # Hands actions
    hands = action.get("hands", [])
    features["num_hands_actions"] = len(hands)
    hand_actions = []
    for h in hands:
        if isinstance(h, list) and h:
            hand_actions.append(h[0])
        else:
            hand_actions.append("PASS")
    features["hand_actions"] = hand_actions
    
    # Market orders
    market = action.get("market", [])
    features["num_market_orders"] = len(market)
    
    sell_orders = []
    buy_orders = []
    hire_count = 0
    
    for order in market:
        if not isinstance(order, list) or not order:
            continue
        order_type = order[0]
        if order_type == "SELL" and len(order) >= 3:
            sell_orders.append({"product": order[1], "qty": order[2]})
        elif order_type == "BUY_SEED" and len(order) >= 3:
            buy_orders.append({"crop": order[1], "qty": order[2]})
        elif order_type == "HIRE":
            hire_count += 1
        elif order_type == "BUY_LAND" and len(order) >= 2:
            features["buy_land"] = order[1]
    
    features["sell_orders"] = sell_orders
    features["buy_orders"] = buy_orders
    features["hire_count"] = hire_count
    features["raw_market"] = market
    
    return features


def parse_episode(episode_path: str, min_final_score: float = 0) -> List[Dict[str, Any]]:
    """Parse a single episode JSON file into (state, action) pairs.
    
    Args:
        episode_path: Path to the episode JSON file
        min_final_score: Only include players with final score >= this value
        
    Returns:
        List of dicts with 'state', 'action', 'player', 'episode_id', 'final_score', 'turn'
    """
    with open(episode_path, "r", encoding="utf-8") as f:
        episode = json.load(f)
    
    # Extract metadata
    episode_id = episode.get("info", {}).get("EpisodeId", 
                 episode.get("id", os.path.basename(episode_path)))
    
    # Get rewards/final scores
    rewards = episode.get("rewards", [0, 0])
    if not rewards:
        rewards = [0, 0]
    
    # Get steps
    steps = episode.get("steps", [])
    if not steps:
        return []
    
    pairs = []
    
    for player_idx in range(2):
        # Filter by score
        final_score = rewards[player_idx] if player_idx < len(rewards) else 0
        if final_score is None:
            final_score = 0
        if final_score < min_final_score:
            continue
        
        for t, step in enumerate(steps):
            if not isinstance(step, list) or player_idx >= len(step):
                continue
            
            agent_step = step[player_idx]
            if not isinstance(agent_step, dict):
                continue
            
            obs = agent_step.get("observation", {})
            action = agent_step.get("action", None)
            
            if obs is None or action is None:
                continue
            
            state_features = extract_state_features(obs, player_idx)
            action_features = extract_action_features(action)
            
            if not action_features.get("valid", False):
                continue
            
            pairs.append({
                "episode_id": str(episode_id),
                "player_idx": player_idx,
                "turn": t,
                "final_score": final_score,
                "state": state_features,
                "action": action_features,
            })
    
    return pairs


def parse_episodes_directory(
    replay_dir: str,
    output_path: str,
    min_final_score: float = 5000,
    max_episodes: int = 0,
) -> Dict[str, Any]:
    """Parse all episode files in a directory into a JSONL training dataset.
    
    Args:
        replay_dir: Directory containing episode JSON files
        output_path: Output JSONL file path
        min_final_score: Only include players with final_score >= this value
        max_episodes: Max episodes to process (0 = all)
        
    Returns:
        Stats dict with counts
    """
    # Find all JSON files
    patterns = [
        os.path.join(replay_dir, "**", "*.json"),
        os.path.join(replay_dir, "**", "*.jsonl"),
    ]
    
    json_files = []
    for pattern in patterns:
        json_files.extend(glob.glob(pattern, recursive=True))
    
    # De-duplicate
    json_files = sorted(set(json_files))
    
    if max_episodes > 0:
        json_files = json_files[:max_episodes]
    
    print(f"Found {len(json_files)} episode files in {replay_dir}")
    
    stats = {
        "total_files": len(json_files),
        "parsed_episodes": 0,
        "skipped_episodes": 0,
        "total_pairs": 0,
        "errors": 0,
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as out:
        for i, fpath in enumerate(json_files):
            if i % 50 == 0:
                print(f"  Processing {i+1}/{len(json_files)}...")
            
            try:
                pairs = parse_episode(fpath, min_final_score=min_final_score)
                if pairs:
                    stats["parsed_episodes"] += 1
                    stats["total_pairs"] += len(pairs)
                    for pair in pairs:
                        out.write(json.dumps(pair, separators=(",", ":")) + "\n")
                else:
                    stats["skipped_episodes"] += 1
            except Exception as e:
                stats["errors"] += 1
                if stats["errors"] <= 5:
                    print(f"  Error parsing {fpath}: {e}")
    
    print(f"\nParsing complete:")
    print(f"  Episodes parsed: {stats['parsed_episodes']}")
    print(f"  Episodes skipped (low score): {stats['skipped_episodes']}")
    print(f"  Errors: {stats['errors']}")
    print(f"  Total (state, action) pairs: {stats['total_pairs']}")
    print(f"  Output: {output_path}")
    
    return stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Parse Kaggriculture replays")
    parser.add_argument("--replay-dir", default="data/replays",
                        help="Directory containing replay JSON files")
    parser.add_argument("--output", default="data/processed/training_pairs.jsonl",
                        help="Output JSONL file path")
    parser.add_argument("--min-score", type=float, default=5000,
                        help="Minimum final score to include a player")
    parser.add_argument("--max-episodes", type=int, default=0,
                        help="Max episodes to process (0 = all)")
    
    args = parser.parse_args()
    
    stats = parse_episodes_directory(
        replay_dir=args.replay_dir,
        output_path=args.output,
        min_final_score=args.min_score,
        max_episodes=args.max_episodes,
    )
    
    # Save stats
    stats_path = args.output.replace(".jsonl", "_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Stats saved to {stats_path}")
