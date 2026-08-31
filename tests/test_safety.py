# SPDX-License-Identifier: MIT
"""Unit tests for Kaggriculture Safety Layer."""

import unittest
from src.safety import (
    is_crop_ready_to_harvest,
    is_animal_unfed_emergency,
    is_crop_unwatered_emergency,
    collect_safety_jobs,
    SafetyLayer,
)

class TestSafetyLayer(unittest.TestCase):

    def test_crop_ready_to_harvest(self):
        # Wheat at day 4 (max yield day)
        wheat_tile = {
            "kind": "PLANT",
            "crop": "WHEAT",
            "planted_day": 0,
            "yield_units": 4,
            "watered_today": True
        }
        self.assertTrue(is_crop_ready_to_harvest(wheat_tile, day=4))

        # Wheat at day 1 (too young)
        self.assertFalse(is_crop_ready_to_harvest(wheat_tile, day=1))

        # Tomato ongoing crop with produce
        tomato_tile = {
            "kind": "PLANT",
            "crop": "TOMATO",
            "planted_day": 0,
            "yield_units": 2,
            "watered_today": True
        }
        self.assertTrue(is_crop_ready_to_harvest(tomato_tile, day=8))

    def test_animal_unfed_emergency(self):
        coop_tile = {
            "kind": "COOP",
            "animal": "GOOSE",
            "fed_today": False,
            "consecutive_unfed": 1
        }
        self.assertTrue(is_animal_unfed_emergency(coop_tile))

        coop_fed = {
            "kind": "COOP",
            "animal": "GOOSE",
            "fed_today": True,
            "consecutive_unfed": 0
        }
        self.assertFalse(is_animal_unfed_emergency(coop_fed))

    def test_crop_unwatered_emergency(self):
        plant_tile = {
            "kind": "PLANT",
            "crop": "WHEAT",
            "watered_today": False,
            "consecutive_unwatered": 1
        }
        self.assertTrue(is_crop_unwatered_emergency(plant_tile))

    def test_collect_safety_jobs_priority(self):
        tiles = [[None for _ in range(10)] for _ in range(10)]
        
        # Tile (0,0) Weed
        tiles[0][0] = {"kind": "WEED"}

        # Tile (0,1) Emergency unfed Cow
        tiles[0][1] = {
            "kind": "PASTURE",
            "animal": "COW",
            "fed_today": False,
            "consecutive_unfed": 1,
            "cared_today": False,
            "fertilizer_available": 1
        }

        # Tile (0,2) Crop needing routine watering
        tiles[0][2] = {
            "kind": "PLANT",
            "crop": "CARROT",
            "planted_day": 0,
            "watered_today": False,
            "consecutive_unwatered": 0,
            "yield_units": 0
        }

        obs = {"day": 1}
        me = {"tiles": tiles}
        priv = {"shed": {"WHEAT": 10}}

        jobs = collect_safety_jobs(obs, me, priv)
        self.assertTrue(len(jobs) >= 3)

        # First job must be emergency feeding (priority 1)
        self.assertEqual(jobs[0]["op"], ["FEED"])
        self.assertEqual(jobs[0]["pos"], (1, 0))

        # Check safety layer wrapper
        layer = SafetyLayer()
        layer_jobs = layer.get_jobs(obs, me, priv)
        self.assertEqual(len(layer_jobs), len(jobs))

if __name__ == "__main__":
    unittest.main()
