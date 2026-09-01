# SPDX-License-Identifier: MIT
"""Launcher script for NSGA-II Multi-Objective Evolutionary Optimization."""

import sys
import os
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, r"C:\Python310\lib\site-packages")

from src.nsga2 import run_nsga2

if __name__ == "__main__":
    print("=" * 70)
    print("  LAUNCHING NSGA-II PARETO MULTI-OBJECTIVE EVOLUTION")
    print("=" * 70)
    
    run_nsga2(
        pop_size=16,
        n_gen=8,
        n_matches_per_eval=1,
        output_dir="models/nsga2",
        seed=42
    )
