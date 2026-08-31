# SPDX-License-Identifier: MIT
"""Data loading utilities for Kaggriculture."""

import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional

def load_crop_economics(data_dir: Path = Path("data/external/reference_agents")) -> Optional[pd.DataFrame]:
    """Load crop economics parameters from reference agents dataset."""
    path = data_dir / "crop_economics.csv"
    if path.exists():
        return pd.read_csv(path)
    return None

def load_agents_manifest(data_dir: Path = Path("data/external/reference_agents")) -> Optional[pd.DataFrame]:
    """Load reference agents manifest."""
    path = data_dir / "agents_manifest.csv"
    if path.exists():
        return pd.read_csv(path)
    return None

def load_raw_data(data_dir: Path = Path("data/raw")) -> Dict[str, Path]:
    """Inspect data/raw files directory."""
    raw_files = {}
    if data_dir.exists():
        for p in data_dir.glob("*"):
            if p.is_file():
                raw_files[p.name] = p
    return raw_files
