# SPDX-License-Identifier: MIT
"""Utilities for logging, random seed control, and submission packaging."""

import os
import random
import logging
from pathlib import Path

def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_logger(name: str = "kaggriculture") -> logging.Logger:
    """Configure and return a standard logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

def save_submission_csv(output_path: Path = Path("submissions/submission.csv"), agent_file: str = "main.py") -> None:
    """Generate submission tracking csv file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("id,submission_file,model_name,version,status\n")
        f.write(f"1,{agent_file},Baseline_Rotation_Agent,v1.0,READY\n")
