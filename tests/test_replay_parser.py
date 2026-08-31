# SPDX-License-Identifier: MIT
"""Unit tests for replay parser and feature extraction module."""

import pytest
import os
import json
import tempfile
from src.replay_parser import (
    extract_state_features,
    extract_action_features,
    parse_episode
)
from src.constants import PRODUCTS, CROPS, ANIMALS


@pytest.fixture
def sample_observation():
    return {
        "step": 48,
        "day": 2,
        "hour": 0,
        "player": 0,
        "farms": [
            {
                "money": 3500.0,
                "farmer": [4, 4],
                "hands": [[5, 4], [4, 5]],
                "hires_today": 2,
                "unlocked_quadrants": ["NW"],
                "tiles": [
                    [
                        {"kind": "PLANT", "crop": "WHEAT", "planted_day": 0, "watered_today": True, "consecutive_unwatered": 0, "yield_units": 0},
                        {"kind": "PLANT", "crop": "CARROT", "planted_day": 0, "watered_today": False, "consecutive_unwatered": 1, "yield_units": 2},
                        {"kind": "COOP", "animal": "GOOSE", "placed_day": 0, "fed_today": False, "consecutive_unfed": 1, "yield_units": 1, "cared_today": True, "fertilizer_available": 1},
                        {"kind": "WEED"},
                        None
                    ] + [None] * 5
                ] + [[None] * 10 for _ in range(9)]
            },
            {
                "money": 2800.0,
                "farmer": [4, 4],
                "hands": [[5, 4]],
                "hires_today": 1,
                "unlocked_quadrants": ["NW"],
                "tiles": [[None] * 10 for _ in range(10)]
            }
        ],
        "market": {
            "inventory": {p: 10000 for p in PRODUCTS},
            "prices": {"WHEAT": 28.0, "CARROT": 40.0, "TOMATO": 60.0, "STRAWBERRY": 120.0, "MELON": 250.0, "EGG": 50.0, "MILK": 160.0, "WOOL": 200.0, "FERTILIZER": 100.0}
        },
        "town": {
            "unlocked_shops": ["Bakery", "Pet Cafe"]
        },
        "private": {
            "shed": {"WHEAT": 10, "CARROT": 5},
            "seeds": {"WHEAT": 6, "CARROT": 12},
            "inventories": [{}, {}]
        }
    }


def test_extract_state_features(sample_observation):
    feat = extract_state_features(sample_observation, player_idx=0)
    
    # Context
    assert feat["turn"] == 48
    assert feat["day"] == 2
    assert feat["hour"] == 0
    assert feat["money"] == 3500.0
    
    # Inventory
    assert feat["shed_wheat"] == 10
    assert feat["shed_carrot"] == 5
    assert feat["shed_tomato"] == 0
    assert feat["seeds_wheat"] == 6
    assert feat["seeds_carrot"] == 12
    
    # Tiles & Crops
    assert feat["tiles_plant"] == 2
    assert feat["tiles_coop"] == 1
    assert feat["tiles_weed"] == 1
    assert feat["crop_wheat"] == 1
    assert feat["crop_carrot"] == 1
    assert feat["animal_goose"] == 1
    assert feat["unwatered_crops"] == 1
    assert feat["unfed_animals"] == 1
    assert feat["harvestable_tiles"] == 2
    assert feat["num_hands"] == 2
    
    # Market & Opponent
    assert feat["price_wheat"] == 28.0
    assert feat["price_carrot"] == 40.0
    assert feat["opp_money"] == 2800.0
    assert feat["opp_num_hands"] == 1
    assert feat["num_unlocked_shops"] == 2


def test_extract_action_features():
    action = {
        "farmer": ["WATER"],
        "hands": [["PLANT", "WHEAT"], ["PASS"]],
        "market": [
            ["SELL", "WHEAT", 15],
            ["BUY_SEED", "CARROT", 6],
            ["HIRE"],
            ["BUY_LAND", "NE"]
        ]
    }
    
    feat = extract_action_features(action)
    assert feat["valid"] is True
    assert feat["farmer_action"] == "WATER"
    assert feat["num_hands_actions"] == 2
    assert feat["hand_actions"] == ["PLANT", "PASS"]
    assert feat["num_market_orders"] == 4
    assert feat["hire_count"] == 1
    assert len(feat["sell_orders"]) == 1
    assert feat["sell_orders"][0] == {"product": "WHEAT", "qty": 15}
    assert len(feat["buy_orders"]) == 1
    assert feat["buy_orders"][0] == {"crop": "CARROT", "qty": 6}
    assert feat.get("buy_land") == "NE"


def test_extract_action_invalid():
    assert extract_action_features(None)["valid"] is False
    assert extract_action_features("INVALID")["valid"] is False


def test_parse_episode_filtering(sample_observation):
    mock_action = {
        "farmer": ["PASS"],
        "hands": [],
        "market": []
    }
    mock_episode = {
        "info": {"EpisodeId": 12345678},
        "rewards": [85000.0, 42000.0],
        "steps": [
            [
                {"observation": sample_observation, "action": mock_action},
                {"observation": sample_observation, "action": mock_action}
            ]
        ]
    }
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(mock_episode, f)
        temp_path = f.name
        
    try:
        # High threshold: only player 0 should pass
        pairs_high = parse_episode(temp_path, min_final_score=75000)
        assert len(pairs_high) == 1
        assert pairs_high[0]["player_idx"] == 0
        assert pairs_high[0]["final_score"] == 85000.0
        
        # Low threshold: both players should pass
        pairs_low = parse_episode(temp_path, min_final_score=40000)
        assert len(pairs_low) == 2
        
        # Very high threshold: no players should pass
        pairs_none = parse_episode(temp_path, min_final_score=100000)
        assert len(pairs_none) == 0
    finally:
        os.remove(temp_path)
