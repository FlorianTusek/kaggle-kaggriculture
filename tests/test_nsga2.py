# SPDX-License-Identifier: MIT
"""Unit tests for NSGA-II Multi-Objective Optimization pipeline."""

import unittest
import numpy as np
from src.nsga2 import (
    decode_chromosome,
    create_agent_from_chromosome,
    evaluate_candidate_match,
    PARAM_BOUNDS_LOWER,
    PARAM_BOUNDS_UPPER
)
from submissions.meta_agent import agent as meta_fn


class TestNSGA2Pipeline(unittest.TestCase):

    def test_chromosome_decoding(self):
        mid_vec = (PARAM_BOUNDS_LOWER + PARAM_BOUNDS_UPPER) / 2.0
        config = decode_chromosome(mid_vec)
        
        self.assertIn("glut_weights", config)
        self.assertIn("MELON", config["glut_weights"])
        self.assertIn("split_caps", config)
        self.assertIn("split_start_turn", config)
        self.assertIn("race_weight", config)
        
        self.assertGreater(config["glut_weights"]["MELON"], 0.0)
        self.assertGreater(config["split_caps"]["MELON"], 0)

    def test_create_agent_from_chromosome(self):
        mid_vec = (PARAM_BOUNDS_LOWER + PARAM_BOUNDS_UPPER) / 2.0
        config = decode_chromosome(mid_vec)
        agent_fn = create_agent_from_chromosome(config)
        self.assertTrue(callable(agent_fn))

    def test_evaluate_candidate_match_fast(self):
        mid_vec = (PARAM_BOUNDS_LOWER + PARAM_BOUNDS_UPPER) / 2.0
        config = decode_chromosome(mid_vec)
        candidate_fn = create_agent_from_chromosome(config)
        
        my_bank, opp_bank = evaluate_candidate_match(candidate_fn, meta_fn, seat=0)
        self.assertGreater(my_bank, 0.0)
        self.assertGreater(opp_bank, 0.0)


if __name__ == "__main__":
    unittest.main()
