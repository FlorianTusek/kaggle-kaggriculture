import pandas as pd
from pathlib import Path
from typing import Tuple, Optional

def load_raw_data(data_dir: Path = Path("data/raw")) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Load train and test datasets from data/raw directory."""
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    
    train = pd.read_csv(train_path) if train_path.exists() else None
    test = pd.read_csv(test_path) if test_path.exists() else None
    
    return train, test
