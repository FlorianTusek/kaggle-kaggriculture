# SPDX-License-Identifier: MIT
"""Unit tests for Kaggriculture Safety Layer module."""

import pytest
from src.safety import (
    is_crop_ready_to_harvest,
    is_animal_unfed_emergency,
    is_crop_unwatered_emergency,
    collect_safety_jobs,
    SafetyLayer,
)
from src.constants import CROPS, ANIMALS


class TestCropHarvestReadiness:
    """Tests for crop harvest readiness logic."""

    def test_non_plant_tile(self):
        tile = {"kind": "EMPTY"}
        assert is_crop_ready_to_harvest(tile, day=5) is False

    def test_invalid_crop_type(self):
        tile = {"kind": "PLANT", "crop": "MAGIC_BEAN", "planted_day": 1, "yield_units": 5}
        assert is_crop_ready_to_harvest(tile, day=5) is False

    def test_crop_too_young(self):
        # WHEAT first_yield_day is 2
        tile = {"kind": "PLANT", "crop": "WHEAT", "planted_day": 5, "yield_units": 1}
        assert is_crop_ready_to_harvest(tile, day=6) is False  # age = 1

    def test_zero_yield_units(self):
        tile = {"kind": "PLANT", "crop": "WHEAT", "planted_day": 1, "yield_units": 0}
        assert is_crop_ready_to_harvest(tile, day=5) is False

    def test_ongoing_crop_harvest(self):
        # TOMATO is ongoing, first_yield_day = 8
        tile = {"kind": "PLANT", "crop": "TOMATO", "planted_day": 1, "yield_units": 2}
        assert is_crop_ready_to_harvest(tile, day=9) is True  # age = 8 >= 8

    def test_harvest_asap_override(self):
        # WHEAT: first_yield_day = 2, max_yield_day = 4
        tile = {"kind": "PLANT", "crop": "WHEAT", "planted_day": 1, "yield_units": 1}
        assert is_crop_ready_to_harvest(tile, day=3, harvest_asap=False) is False
        assert is_crop_ready_to_harvest(tile, day=3, harvest_asap=True) is True

    def test_max_yield_day_reached(self):
        tile = {"kind": "PLANT", "crop": "WHEAT", "planted_day": 1, "yield_units": 1}
        assert is_crop_ready_to_harvest(tile, day=5) is True  # age = 4 >= max_yield_day (4)

    def test_yield_threshold_reached(self):
        tile = {"kind": "PLANT", "crop": "WHEAT", "planted_day": 1, "yield_units": 3}
        assert is_crop_ready_to_harvest(tile, day=3) is True  # age = 2 >= 2 and yield >= 3


class TestEmergencyStatus:
    """Tests for emergency unfed and unwatered detection."""

    def test_animal_emergency_unfed(self):
        tile_fed = {"kind": "COOP", "animal": "GOOSE", "fed_today": True, "consecutive_unfed": 1}
        assert is_animal_unfed_emergency(tile_fed) is False

        tile_unfed_safe = {"kind": "COOP", "animal": "GOOSE", "fed_today": False, "consecutive_unfed": 0}
        assert is_animal_unfed_emergency(tile_unfed_safe) is False

        tile_unfed_emergency = {"kind": "COOP", "animal": "GOOSE", "fed_today": False, "consecutive_unfed": 1}
        assert is_animal_unfed_emergency(tile_unfed_emergency) is True

    def test_animal_emergency_non_animal_tile(self):
        tile = {"kind": "EMPTY", "fed_today": False, "consecutive_unfed": 2}
        assert is_animal_unfed_emergency(tile) is False

    def test_crop_emergency_unwatered(self):
        tile_watered = {"kind": "PLANT", "crop": "WHEAT", "watered_today": True, "consecutive_unwatered": 1}
        assert is_crop_unwatered_emergency(tile_watered) is False

        tile_unwatered_safe = {"kind": "PLANT", "crop": "WHEAT", "watered_today": False, "consecutive_unwatered": 0}
        assert is_crop_unwatered_emergency(tile_unwatered_safe) is False

        tile_unwatered_emergency = {"kind": "PLANT", "crop": "WHEAT", "watered_today": False, "consecutive_unwatered": 1}
        assert is_crop_unwatered_emergency(tile_unwatered_emergency) is True


class TestSafetyJobCollectionAndPriority:
    """Tests for collect_safety_jobs prioritization and ordering."""

    def test_strict_priority_ordering(self):
        """Ensure safety jobs are ordered 1 -> 8 strictly by safety priority."""
        tiles = [
            [
                {"kind": "WEED"},  # Weed (Priority 8) at (0,0)
                {"kind": "PLANT", "crop": "WHEAT", "watered_today": False, "consecutive_unwatered": 0, "planted_day": 10},  # Routine water (Priority 6) at (1,0)
                {"kind": "COOP", "animal": "GOOSE", "fed_today": False, "cared_today": True, "consecutive_unfed": 0},  # Routine feed (Priority 4) at (2,0)
            ],
            [
                {"kind": "PLANT", "crop": "WHEAT", "watered_today": False, "consecutive_unwatered": 1, "planted_day": 10},  # Emergency water (Priority 2) at (0,1)
                {"kind": "COOP", "animal": "COW", "fed_today": False, "cared_today": True, "consecutive_unfed": 1},  # Emergency feed (Priority 1) at (1,1)
                {"kind": "PLANT", "crop": "WHEAT", "planted_day": 1, "yield_units": 5},  # Harvest (Priority 3) at (2,1)
            ],
            [
                {"kind": "COOP", "animal": "SHEEP", "fed_today": True, "cared_today": False, "consecutive_unfed": 0},  # Care (Priority 5) at (0,2)
                {"kind": "COOP", "animal": "GOOSE", "fertilizer_available": 3, "fed_today": True, "cared_today": True},  # Fertilizer (Priority 7) at (1,2)
                None,  # Empty
            ]
        ]
        me = {"tiles": tiles}
        obs = {"day": 10}
        priv = {}

        jobs = collect_safety_jobs(obs, me, priv)
        priorities = [j["priority"] for j in jobs]

        # Verify exact priorities present in correct sorted order
        assert priorities == [1, 2, 3, 4, 5, 6, 7, 8]
        assert jobs[0]["op"] == ["FEED"]
        assert jobs[0]["pos"] == (1, 1)  # Emergency feed COW
        assert jobs[1]["op"] == ["WATER"]
        assert jobs[1]["pos"] == (0, 1)  # Emergency water crop
        assert jobs[2]["op"] == ["HARVEST"]
        assert jobs[2]["pos"] == (2, 1)  # Ready harvest WHEAT
        assert jobs[3]["op"] == ["FEED"]
        assert jobs[3]["pos"] == (2, 0)  # Routine feed GOOSE
        assert jobs[4]["op"] == ["CARE"]
        assert jobs[4]["pos"] == (0, 2)  # Care SHEEP
        assert jobs[5]["op"] == ["WATER"]
        assert jobs[5]["pos"] == (1, 0)  # Routine water WHEAT
        assert jobs[6]["op"] == ["COLLECT_FERTILIZER"]
        assert jobs[6]["pos"] == (1, 2)  # Fertilizer GOOSE
        assert jobs[7]["op"] == ["DIG"]
        assert jobs[7]["pos"] == (0, 0)  # Dig Weed

    def test_safety_layer_class_wrapper(self):
        tiles = [[{"kind": "PLANT", "crop": "WHEAT", "watered_today": False, "consecutive_unwatered": 0, "planted_day": 10}]]
        me = {"tiles": tiles}
        obs = {"day": 10}
        priv = {}

        safety = SafetyLayer()
        jobs = safety.get_jobs(obs, me, priv)
        assert len(jobs) == 1
        assert jobs[0]["priority"] == 6
        assert jobs[0]["op"] == ["WATER"]
