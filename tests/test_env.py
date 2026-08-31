# SPDX-License-Identifier: MIT
"""Unit tests for Kaggriculture OpenAI Gym / Gymnasium Environment Wrapper."""

import unittest
import numpy as np
from src.env import KaggricultureEnv, ACTION_LOOKUP

class TestKaggricultureEnv(unittest.TestCase):

    def setUp(self):
        self.env = KaggricultureEnv(max_turns=24)

    def test_reset(self):
        obs, info = self.env.reset(seed=42)
        self.assertIsInstance(obs, np.ndarray)
        self.assertEqual(obs.shape, (49,))
        self.assertEqual(info["turn"], 0)
        self.assertEqual(info["money"], 3000.0)

    def test_step_execution(self):
        self.env.reset(seed=42)
        # Execute action 3 ("EAST")
        obs, reward, terminated, truncated, info = self.env.step(3)
        self.assertEqual(info["turn"], 1)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["action_executed"], "EAST")

    def test_episode_completion(self):
        self.env.reset()
        done = False
        step_count = 0
        while not done:
            action = np.random.randint(0, len(ACTION_LOOKUP))
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            step_count += 1

        self.assertEqual(step_count, 24)
        self.assertTrue(done)

if __name__ == "__main__":
    unittest.main()
