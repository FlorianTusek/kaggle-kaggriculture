# SPDX-License-Identifier: MIT
"""Behavioral Cloning Model Training for Kaggriculture.

Trains multi-head supervised learning models (LightGBM) on expert (state, action)
pairs extracted from top Kaggle replay trajectories:
1. Farmer Action Classifier (predicts worker operation: PASS, WATER, HARVEST, FEED, CARE, etc.)
2. Planting Crop Recommender (predicts optimal crop: WHEAT, CARROT, TOMATO, STRAWBERRY, MELON)
3. Labor Hiring Predictor (predicts target farmhand hires for the day)
4. Market Sell Decision Predictor (predicts whether to trigger batch sales)
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error
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


def load_dataset(jsonl_path: str, max_samples: int = 200000) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
            feat_vec = [float(state.get(col, 0)) for col in FEATURE_COLUMNS]
            rows_x.append(feat_vec)
            
            # Target labels
            farmer_op = action.get("farmer_action", "PASS")
            planted_crop = "NONE"
            if isinstance(farmer_op, list):
                if len(farmer_op) > 1:
                    planted_crop = str(farmer_op[1]).upper()
                farmer_op = farmer_op[0] if farmer_op else "PASS"
            farmer_op = str(farmer_op).upper()
            
            hire_count = int(action.get("hire_count", 0))
            num_market_orders = int(action.get("num_market_orders", 0))
            sell_orders = action.get("sell_orders", [])
            has_sell = 1 if len(sell_orders) > 0 else 0
            
            rows_y.append({
                "farmer_action": farmer_op,
                "planted_crop": planted_crop,
                "hire_count": hire_count,
                "num_market_orders": num_market_orders,
                "has_sell": has_sell,
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
    max_samples: int = 200000
) -> Dict[str, Any]:
    """Train multi-head LightGBM Behavioral Cloning policy models."""
    os.makedirs(output_dir, exist_ok=True)
    
    df_x, df_y = load_dataset(data_path, max_samples=max_samples)
    
    # 1. Train Farmer Action Classifier
    print("\n--- 1. Training Farmer Action Classifier ---")
    y_farmer = df_y["farmer_action"]
    classes_farmer, y_farmer_enc = np.unique(y_farmer, return_inverse=True)
    class_map_farmer = {int(i): str(c) for i, c in enumerate(classes_farmer)}
    print(f"Farmer action classes ({len(classes_farmer)}): {class_map_farmer}")
    
    X_train_f, X_val_f, y_train_f, y_val_f = train_test_split(
        df_x, y_farmer_enc, test_size=0.15, random_state=42, stratify=y_farmer_enc
    )
    
    clf_farmer = lgb.LGBMClassifier(
        n_estimators=120,
        learning_rate=0.08,
        num_leaves=31,
        random_state=42,
        verbosity=-1,
        n_jobs=-1
    )
    clf_farmer.fit(X_train_f, y_train_f)
    val_preds_f = clf_farmer.predict(X_val_f)
    acc_farmer = accuracy_score(y_val_f, val_preds_f)
    print(f"Farmer Action Validation Accuracy: {acc_farmer:.4f} ({acc_farmer*100:.2f}%)")
    
    # 2. Train Market Hire Predictor (predict target hires)
    print("\n--- 2. Training Market Hire Predictor ---")
    y_hire = df_y["hire_count"]
    hire_reg = lgb.LGBMRegressor(
        n_estimators=60,
        learning_rate=0.1,
        num_leaves=15,
        random_state=42,
        verbosity=-1,
        n_jobs=-1
    )
    hire_reg.fit(X_train_f, y_hire.iloc[X_train_f.index])
    val_preds_hire = hire_reg.predict(X_val_f)
    mse_hire = mean_squared_error(y_hire.iloc[X_val_f.index], val_preds_hire)
    print(f"Labor Hire Regressor Validation MSE: {mse_hire:.4f}")
    
    # 3. Train Market Sell Decision Classifier
    print("\n--- 3. Training Market Sell Trigger Classifier ---")
    y_sell = df_y["has_sell"]
    clf_sell = lgb.LGBMClassifier(
        n_estimators=80,
        learning_rate=0.1,
        num_leaves=20,
        random_state=42,
        verbosity=-1,
        n_jobs=-1
    )
    clf_sell.fit(X_train_f, y_sell.iloc[X_train_f.index])
    val_preds_sell = clf_sell.predict(X_val_f)
    acc_sell = accuracy_score(y_sell.iloc[X_val_f.index], val_preds_sell)
    print(f"Market Sell Trigger Validation Accuracy: {acc_sell:.4f} ({acc_sell*100:.2f}%)")
    
    # 4. Feature Importance Analysis
    print("\n--- Top Feature Importances (Farmer Action Model) ---")
    importances = clf_farmer.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    top_features = []
    for rank in range(min(10, len(FEATURE_COLUMNS))):
        idx = sorted_idx[rank]
        print(f"  #{rank+1} {FEATURE_COLUMNS[idx]}: {importances[idx]}")
        top_features.append({"feature": FEATURE_COLUMNS[idx], "importance": int(importances[idx])})
        
    # 5. Save Artifacts
    artifacts = {
        "feature_columns": FEATURE_COLUMNS,
        "class_map_farmer": class_map_farmer,
        "classes_farmer": list(classes_farmer),
        "clf_farmer": clf_farmer,
        "hire_regressor": hire_reg,
        "clf_sell": clf_sell,
        "acc_farmer": float(acc_farmer),
        "acc_sell": float(acc_sell),
        "mse_hire": float(mse_hire),
    }
    
    model_path = os.path.join(output_dir, "bc_model.joblib")
    joblib.dump(artifacts, model_path)
    print(f"\nModel artifacts saved successfully to {model_path}")
    
    # 6. Save Metadata Summary
    meta = {
        "model_type": "LightGBM Multi-Head Behavioral Cloning Policy",
        "num_training_samples": len(df_x),
        "num_features": len(FEATURE_COLUMNS),
        "feature_columns": FEATURE_COLUMNS,
        "action_classes": [str(c) for c in classes_farmer],
        "metrics": {
            "farmer_action_accuracy": float(acc_farmer),
            "market_sell_accuracy": float(acc_sell),
            "hire_regressor_mse": float(mse_hire),
        },
        "top_features": top_features,
        "model_file": model_path
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
    parser.add_argument("--max-samples", type=int, default=200000, help="Max training samples")
    args = parser.parse_args()
    
    train_behavioral_cloning(args.data, args.output, args.max_samples)
