# SPDX-License-Identifier: MIT
"""PPO Reinforcement Learning Training for Kaggriculture.

Trains a Proximal Policy Optimization (PPO) agent using stable-baselines3
on the KaggricultureEnv Gymnasium environment against heuristic opponent agents.
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.env import KaggricultureEnv, ACTION_LOOKUP
from src.agent import KaggricultureAgent


def train_ppo(
    total_timesteps: int = 25000,
    output_path: str = "models/ppo_agent.zip",
    eval_episodes: int = 5,
    seed: int = 42
) -> Dict[str, Any]:
    """Train PPO agent in KaggricultureEnv."""
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.evaluation import evaluate_policy
    except ImportError as e:
        raise ImportError("stable-baselines3 is required for PPO training.") from e

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"\n=== Initializing KaggricultureEnv for PPO Training ===")
    env = KaggricultureEnv(max_turns=720, opponent_agent=KaggricultureAgent())
    eval_env = KaggricultureEnv(max_turns=720, opponent_agent=KaggricultureAgent())
    
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space} ({len(ACTION_LOOKUP)} discrete actions)")
    
    print(f"\n=== Initializing PPO Model (MlpPolicy, seed={seed}) ===")
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        seed=seed
    )
    
    print(f"\n=== Training PPO Policy for {total_timesteps} timesteps ===")
    model.learn(total_timesteps=total_timesteps)
    
    # Save trained model
    model.save(output_path)
    print(f"\nPPO model saved to: {output_path}")
    
    # Evaluation
    print(f"\n=== Evaluating PPO Agent over {eval_episodes} episodes ===")
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=eval_episodes, deterministic=True
    )
    print(f"Mean episode reward: ${mean_reward:,.2f} (+/- ${std_reward:,.2f})")
    
    # Save training metadata
    meta = {
        "algorithm": "PPO (Proximal Policy Optimization)",
        "framework": "stable-baselines3",
        "policy": "MlpPolicy",
        "total_timesteps": total_timesteps,
        "n_features": env.n_features,
        "n_actions": len(ACTION_LOOKUP),
        "action_lookup": ACTION_LOOKUP,
        "hyperparameters": {
            "learning_rate": 3e-4,
            "n_steps": 512,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
        },
        "evaluation": {
            "mean_reward": float(mean_reward),
            "std_reward": float(std_reward),
            "n_eval_episodes": eval_episodes
        },
        "model_file": output_path
    }
    
    meta_path = output_path.replace(".zip", "_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved to: {meta_path}")
    
    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO Agent on Kaggriculture")
    parser.add_argument("--timesteps", type=int, default=25000, help="Total training timesteps")
    parser.add_argument("--output", default="models/ppo_agent.zip", help="Output model path")
    parser.add_argument("--eval-episodes", type=int, default=5, help="Number of eval episodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    train_ppo(
        total_timesteps=args.timesteps,
        output_path=args.output,
        eval_episodes=args.eval_episodes,
        seed=args.seed
    )
