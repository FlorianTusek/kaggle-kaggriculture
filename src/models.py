# SPDX-License-Identifier: MIT
"""Behavioral Cloning Model Policy & Evaluation for Kaggriculture."""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

from src.constants import PRODUCTS, CROPS, ANIMALS
from src.replay_parser import extract_state_features
from src.train_bc import FEATURE_COLUMNS


class BehavioralCloningPolicy:
    """Multi-head Behavioral Cloning Policy loaded from trained LightGBM models.
    
    Serves as an expert ML policy or warm-start advisor for Kaggriculture agents:
    - Predicts farmer action probabilities & top recommended action
    - Predicts optimal labor hiring count
    - Predicts market sell timing triggers
    """

    def __init__(self, model_path: Optional[str] = None):
        if model_path is None:
            model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "bc_model.joblib")
        
        self.model_path = model_path
        self.artifacts = None
        self.is_loaded = False
        
        if os.path.exists(model_path):
            self.load_model(model_path)

    def load_model(self, model_path: str) -> None:
        """Load serialized model artifacts from disk."""
        self.artifacts = joblib.load(model_path)
        self.feature_columns = self.artifacts.get("feature_columns", FEATURE_COLUMNS)
        self.class_map_farmer = self.artifacts.get("class_map_farmer", {})
        self.classes_farmer = self.artifacts.get("classes_farmer", [])
        self.clf_farmer = self.artifacts.get("clf_farmer")
        self.hire_regressor = self.artifacts.get("hire_regressor")
        self.clf_sell = self.artifacts.get("clf_sell")
        self.is_loaded = True

    def extract_features(self, obs: Dict[str, Any], player_idx: Optional[int] = None) -> pd.DataFrame:
        """Extract state feature DataFrame from live game observation."""
        if player_idx is None:
            player_idx = obs.get("player", 0)
        feat_dict = extract_state_features(obs, player_idx)
        vec = [float(feat_dict.get(col, 0)) for col in self.feature_columns]
        return pd.DataFrame([vec], columns=self.feature_columns)

    def predict_farmer_action(self, obs: Dict[str, Any]) -> str:
        """Predict the most likely expert farmer action given the observation."""
        if not self.is_loaded:
            return "PASS"
        X = self.extract_features(obs)
        pred_idx = self.clf_farmer.predict(X)[0]
        return self.class_map_farmer.get(int(pred_idx), "PASS")

    def predict_farmer_action_probs(self, obs: Dict[str, Any]) -> Dict[str, float]:
        """Predict probability distribution over worker actions."""
        if not self.is_loaded:
            return {"PASS": 1.0}
        X = self.extract_features(obs)
        probs = self.clf_farmer.predict_proba(X)[0]
        return {str(cls_name): float(probs[i]) for i, cls_name in enumerate(self.classes_farmer)}

    def predict_hire_count(self, obs: Dict[str, Any]) -> int:
        """Predict target farmhands to hire today."""
        if not self.is_loaded or self.hire_regressor is None:
            return 4
        X = self.extract_features(obs)
        pred = self.hire_regressor.predict(X)[0]
        return max(0, int(round(pred)))

    def predict_should_sell(self, obs: Dict[str, Any]) -> bool:
        """Predict whether expert strategy would trigger market sales this turn."""
        if not self.is_loaded or self.clf_sell is None:
            return True
        X = self.extract_features(obs)
        pred = self.clf_sell.predict(X)[0]
        return bool(pred == 1)

    def advise(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Provide a complete strategic advice dictionary to augment heuristic planners."""
        return {
            "recommended_farmer_action": self.predict_farmer_action(obs),
            "action_probabilities": self.predict_farmer_action_probs(obs),
            "recommended_hire_count": self.predict_hire_count(obs),
            "should_trigger_sell": self.predict_should_sell(obs),
        }


class BaselineModel:
    """Wrapper around baseline heuristic agent policy."""

    def __init__(self, policy: Optional[Dict[str, Any]] = None):
        from src.agent import KaggricultureAgent, DEFAULT_POLICY
        self.policy = policy if policy is not None else DEFAULT_POLICY
        self.agent = KaggricultureAgent(self.policy)

    def predict(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Return action dict for a given observation."""
        return self.agent.act(obs)

    def evaluate_summary(self, final_reward: float) -> Dict[str, Any]:
        """Generate evaluation summary from an episode run."""
        return {
            "policy": self.policy,
            "final_reward": final_reward,
            "success": final_reward > 3000,
        }
