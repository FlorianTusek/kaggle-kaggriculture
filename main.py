# SPDX-License-Identifier: MIT
"""Kaggriculture Competition Submission Entrypoint."""

import sys
from pathlib import Path

# Add repo root to path if running locally
repo_root = str(Path(__file__).resolve().parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.agent import agent_entrypoint

def agent(obs, config=None):
    """Main submission agent entrypoint required by Kaggle Competition."""
    return agent_entrypoint(obs, config)

if __name__ == "__main__":
    print("Kaggriculture submission entrypoint loaded successfully.")
