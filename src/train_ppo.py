# SPDX-License-Identifier: MIT
"""PPO Reinforcement Learning Training & BC Weight Initialization for Kaggriculture.

Supports:
1. Behavioral Cloning Weight Pre-Training: Initializes PPO actor-critic PyTorch neural
   network weights from the expert $118k+ (state, action) dataset before RL training.
2. PPO Reinforcement Learning: Trains the warm-started policy on KaggricultureEnv
   against competitive opponent baselines.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.env import KaggricultureEnv, ACTION_LOOKUP
from src.agent import KaggricultureAgent
from src.train_bc import FEATURE_COLUMNS


def load_bc_dataset(
    dataset_path: str = "data/processed/training_pairs.jsonl",
    max_samples: int = 200000
) -> Tuple[np.ndarray, np.ndarray]:
    """Load expert (state, action) pairs into feature matrix X and target action indices y."""
    print(f"Loading up to {max_samples} BC samples from {dataset_path}...")
    action_to_idx = {act: i for i, act in enumerate(ACTION_LOOKUP)}
    
    rows_x = []
    rows_y = []
    
    with open(dataset_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            record = json.loads(line)
            state = record.get("state", {})
            action = record.get("action", {})
            
            # Feature vector
            feat_vec = [float(state.get(col, 0)) for col in FEATURE_COLUMNS]
            
            # Action target index
            farmer_op = action.get("farmer_action", "PASS")
            if isinstance(farmer_op, list):
                farmer_op = farmer_op[0] if farmer_op else "PASS"
            farmer_op = str(farmer_op).upper()
            
            act_idx = action_to_idx.get(farmer_op, 0)
            
            rows_x.append(feat_vec)
            rows_y.append(act_idx)
            
            if (i + 1) % 50000 == 0:
                print(f"  Loaded {i+1} rows...")
                
    X = np.array(rows_x, dtype=np.float32)
    y = np.array(rows_y, dtype=np.int64)
    print(f"Dataset loaded: {len(X)} samples, {X.shape[1]} features.")
    return X, y


def pretrain_ppo_with_bc(
    model: Any,
    dataset_path: str = "data/processed/training_pairs.jsonl",
    max_samples: int = 200000,
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 1e-3,
    val_split: float = 0.15,
    seed: int = 42
) -> Dict[str, Any]:
    """Supervised pre-training (Behavioral Cloning) directly into PPO PyTorch policy network."""
    print("\n=== Initializing PPO PyTorch Policy with Behavioral Cloning Weights ===")
    
    X, y = load_bc_dataset(dataset_path, max_samples=max_samples)
    
    # Train / Val Split
    np.random.seed(seed)
    indices = np.random.permutation(len(X))
    val_size = int(len(X) * val_split)
    train_idx, val_idx = indices[val_size:], indices[:val_size]
    
    X_train, y_train = torch.tensor(X[train_idx]), torch.tensor(y[train_idx])
    X_val, y_val = torch.tensor(X[val_idx]), torch.tensor(y[val_idx])
    
    # Policy network & optimizer
    policy_net = model.policy
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    n_train = len(X_train)
    history = []
    
    print(f"Training on {n_train} samples, validating on {len(X_val)} samples for {epochs} epochs...")
    
    for epoch in range(epochs):
        policy_net.train()
        perm = torch.randperm(n_train)
        running_loss = 0.0
        correct = 0
        total = 0
        
        for start_idx in range(0, n_train, batch_size):
            batch_indices = perm[start_idx:start_idx + batch_size]
            batch_x = X_train[batch_indices]
            batch_y = y_train[batch_indices]
            
            optimizer.zero_grad()
            dist = policy_net.get_distribution(batch_x)
            logits = dist.distribution.logits
            
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * len(batch_y)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == batch_y).sum().item()
            total += len(batch_y)
            
        train_loss = running_loss / max(1, total)
        train_acc = correct / max(1, total)
        
        # Validation evaluation
        policy_net.eval()
        with torch.no_grad():
            val_dist = policy_net.get_distribution(X_val)
            val_logits = val_dist.distribution.logits
            val_loss = criterion(val_logits, y_val).item()
            val_preds = torch.argmax(val_logits, dim=1)
            val_acc = (val_preds == y_val).sum().item() / max(1, len(y_val))
            
        print(f"  Epoch {epoch+1:02d}/{epochs:02d} - Loss: {train_loss:.4f}, Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")
        history.append({
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "val_loss": float(val_loss),
            "val_acc": float(val_acc)
        })
        
    final_val_acc = history[-1]["val_acc"] if history else 0.0
    print(f"\nBC Pre-Training Complete! Final Policy Network Validation Accuracy: {final_val_acc*100:.2f}%\n")
    
    return {
        "epochs": epochs,
        "final_val_acc": float(final_val_acc),
        "history": history
    }


def train_ppo(
    total_timesteps: int = 25000,
    output_path: str = "models/ppo_agent.zip",
    init_with_bc: bool = True,
    bc_dataset_path: str = "data/processed/training_pairs.jsonl",
    bc_epochs: int = 10,
    bc_max_samples: int = 200000,
    eval_episodes: int = 5,
    seed: int = 42
) -> Dict[str, Any]:
    """Train PPO agent in KaggricultureEnv, optionally initializing with BC weights."""
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.evaluation import evaluate_policy
    except ImportError as e:
        raise ImportError("stable-baselines3 is required for PPO training.") from e

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"\n=== Initializing KaggricultureEnv for PPO Training ===")
    training_opp = KaggricultureAgent(policy={"use_ensemble": False, "use_ml_policy": False})
    eval_opp = KaggricultureAgent(policy={"use_ensemble": False, "use_ml_policy": False})
    env = KaggricultureEnv(max_turns=720, opponent_agent=training_opp)
    eval_env = KaggricultureEnv(max_turns=720, opponent_agent=eval_opp)
    
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
    
    bc_metrics = {}
    if init_with_bc and os.path.exists(bc_dataset_path):
        bc_metrics = pretrain_ppo_with_bc(
            model=model,
            dataset_path=bc_dataset_path,
            max_samples=bc_max_samples,
            epochs=bc_epochs,
            seed=seed
        )
        # Save BC-initialized checkpoint
        bc_init_path = output_path.replace(".zip", "_bc_initialized.zip")
        model.save(bc_init_path)
        print(f"Saved BC-initialized PPO checkpoint to: {bc_init_path}")

    if total_timesteps > 0:
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
        "initialized_with_bc": bool(init_with_bc and os.path.exists(bc_dataset_path)),
        "bc_metrics": bc_metrics,
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
    parser.add_argument("--timesteps", type=int, default=25000, help="Total RL training timesteps")
    parser.add_argument("--output", default="models/ppo_agent.zip", help="Output model path")
    parser.add_argument("--init-with-bc", action="store_true", default=True, help="Pre-train with BC weights")
    parser.add_argument("--no-bc", action="store_false", dest="init_with_bc", help="Do not pre-train with BC")
    parser.add_argument("--bc-data", default="data/processed/training_pairs.jsonl", help="BC dataset path")
    parser.add_argument("--bc-epochs", type=int, default=10, help="BC pretraining epochs")
    parser.add_argument("--bc-samples", type=int, default=200000, help="Max BC samples for pretraining")
    parser.add_argument("--eval-episodes", type=int, default=5, help="Number of eval episodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    train_ppo(
        total_timesteps=args.timesteps,
        output_path=args.output,
        init_with_bc=args.init_with_bc,
        bc_dataset_path=args.bc_data,
        bc_epochs=args.bc_epochs,
        bc_max_samples=args.bc_samples,
        eval_episodes=args.eval_episodes,
        seed=args.seed
    )
