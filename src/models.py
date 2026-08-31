# SPDX-License-Identifier: MIT
"""Model evaluation and policy benchmarking for Kaggriculture."""

from typing import Dict, Any, List, Optional
import numpy as np
from src.agent import KaggricultureAgent, DEFAULT_POLICY

class BaselineModel:
    """Wrapper around baseline heuristic agent policy."""

    def __init__(self, policy: Optional[Dict[str, Any]] = None):
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
