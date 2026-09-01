# SPDX-License-Identifier: MIT
"""Launcher script for 100,000 Generation Continuous League Self-Play Training."""

import sys
import os

# Prioritize CUDA 12.1 PyTorch
sys.path.insert(0, r"C:\Python310\lib\site-packages")
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

# Ensure real-time unbuffered stdout logging
sys.stdout.reconfigure(line_buffering=True)

from src.train_ppo import train_ppo_league

if __name__ == "__main__":
    print("=" * 70)
    print("  LAUNCHING 100,000 GENERATION CONTINUOUS LEAGUE SELF-PLAY TRAINING")
    print("=" * 70)
    
    train_ppo_league(
        n_generations=100000,
        timesteps_per_gen=2048,
        output_dir="models/league",
        init_with_bc=True,
        eval_episodes=2,
        eval_freq=50,
        save_freq=100,
        keep_recent_checkpoints=5,
        resume=True
    )
