# SPDX-License-Identifier: MIT
"""Behavioral Cloning Model Training for Kaggriculture.

Trains supervised learning models (LightGBM multi-output / classifiers)
on expert (state, action) pairs extracted from top Kaggle replay trajectories.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.constants import PRODUCTS, CROPS, ANIMALS


FEATURE_COLUMNS = [
    "turn", "day", "hour", "money",
    "shed_wheat", "shed_carrot", "shed_tomato", "shed_strawberry", "shed_melon",
    "shed_egg", "shed_milk", "shed_wool", "shed_fertilizer",
    "seeds_wheat", "seeds_carrot", "seeds_tomato", "seeds_strawberry", "seeds_melon",
    "tiles_empty", "tiles_plant", "tiles_coop", "tiles_pasture", "tiles_weed", "tiles_locked", "tiles_other",
    "crop_wheat", "crop_carrot", "crop_tomato", "crop_strawberry", "crop_melon",
    "animal_goose", "animal_cow", "animal_sheep",
    "unwatered_crops", "unfed_animals", "harvestable_tiles", "num_hands",
    "price_wheat", "price_carrot", "price_tomato", "price_strawberry", "price_melon",
    "price_egg", "price_milk", "price_wool", "price_fertilizer",
    "opp_money", "opp_num_hands", "num_unlocked_shops"
]


def load_dataset(jsonl_path: str, max_samples: int = 150000) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load streaming JSONL dataset into feature and target DataFrames."""
    print(f"Loading up to {max_samples} samples from {jsonl_path}...")
    
    rows_x = []
    rows_y = []
    
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            record = json.loads(line)
            state = record.get("state", {})
            action = record.get("action", {})
            
            # Extract state vector
            feat_vec = [state.get(col, 0) for col in FEATURE_COLUMNS]
            rows_x.append(feat_vec)
            
            # Target labels
            farmer_op = action.get("farmer_action", "PASS")
            # Normalize complex actions (e.g. ['PLANT', 'WHEAT'] -> 'PLANT')
            if isinstance(farmer_op, list):
                farmer_op = farmer_op[0] if farmer_op else "PASS"
            farmer_op = str(farmer_op).upper()
            
            hire_count = action.get("hire_count", 0)
            num_market_orders = action.get("num_market_orders", 0)
            
            rows_y.append({
                "farmer_action": farmer_op,
                "hire_count": hire_count,
                "num_market_orders": num_market_orders
            })
            
            if (i + 1) % 50000 == 0:
                print(f"  Loaded {i+1} rows...")
                
    df_x = pd.DataFrame(rows_x, columns=FEATURE_COLUMNS)
    df_y = pd.DataFrame(rows_y)
    
    print(f"Dataset loaded: {len(df_x)} samples, {len(FEATURE_COLUMNS)} features.")
    return df_x, df_y


def train_behavioral_cloning(
    data_path: str,
    output_dir: str = "models",
    max_samples: int = 150000
) -> Dict[str, Any]:
    """Train LightGBM Behavioral Cloning policy models."""
    os.makedirs(output_dir, exist_ok=True)
    
    df_x, df_y = load_dataset(data_path, max_samples=max_samples)
    
    # 1. Train Farmer Action Classifier
    print("\n--- Training Farmer Action Classifier ---")
    y_farmer = df_y["farmer_action"]
    
    # Encode target
    classes, y_encoded = np.unique(y_farmer, return_inverse=True)
    class_map = {int(i): str(c) for i, c in enumerate(classes)}
    print(f"Action classes ({len(classes)}): {class_map}")
    
    X_train, X_val, y_train, y_val = train_test_split(
        df_x, y_encoded, test_size=0.15, random_state=42, stratify=y_encoded
    )
    
    clf = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        num_leaves=31,
        random_state=42,
        verbosity=-1
    )
    clf.fit(X_train, y_train)
    
    val_preds = clf.predict(X_val)
    acc = accuracy_score(y_val, val_preds)
    print(f"Farmer Action Validation Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    
    # 2. Train Market Hire Predictor (Regressor / Classifier)
    print("\n--- Training Market Hire Predictor ---")
    y_hire = df_y["hire_count"]
    hire_reg = lgb.LGBMRegressor(
        n_estimators=50,
        learning_rate=0.1,
        num_leaves=15,
        random_state=42,
        verbosity=-1
    )
    hire_reg.fit(X_train, y_hire.iloc[X_train.index])
    
    # 3. Save artifacts
    artifacts = {
        "feature_columns": FEATURE_COLUMNS,
        "class_map": class_map,
        "classes": list(classes),
        "farmer_classifier": clf,
        "hire_regressor": hire_reg,
        "accuracy": float(acc),
    }
    
    model_path = os.path.join(output_dir, "bc_model.joblib")
    joblib.dump(artifacts, model_path)
    print(f"\nModel artifacts saved successfully to {model_path}")
    
    # Also save metadata json
    meta = {
        "model_type": "LightGBM Behavioral Cloning Policy",
        "num_samples": len(df_x),
        "num_features": len(FEATURE_COLUMNS),
        "action_classes": [str(c) for c in classes],
        "validation_accuracy": float(acc),
    }
    meta_path = os.path.join(output_dir, "bc_model_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved to {meta_path}")
    
    return meta


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Behavioral Cloning Policy")
    parser.add_argument("--data", default="data/processed/training_pairs.jsonl", help="Dataset path")
    parser.add_argument("--output", default="models", help="Output model directory")
    parser.add_argument("--max-samples", type=int, default=150000, help="Max training samples")
    args = parser.parse_args()
    
    train_behavioral_cloning(args.data, args.output, args.max_samples)
