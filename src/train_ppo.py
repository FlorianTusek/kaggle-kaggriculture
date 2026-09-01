# SPDX-License-Identifier: MIT
"""PPO Reinforcement Learning & Multi-Generational League Training for Kaggriculture.

Features:
1. GPU-Accelerated PyTorch CUDA Backend (auto-detects NVIDIA RTX 4070 / CUDA 12.1).
2. Behavioral Cloning Policy Pre-Training: Initializes PPO ActorCriticPolicy weights from expert dataset.
3. Multi-Generational Self-Play (League Training): Trains agent across successive generations
   against a dynamic pool of evolving self-checkpoints and elite opponents (Top Meta & Heuristic Baselines).
4. Automated League Evaluation & Checkpointing.
"""

import os
import sys
import json
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List, Union

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.env import KaggricultureEnv, ACTION_LOOKUP
from src.agent import KaggricultureAgent
from src.train_bc import FEATURE_COLUMNS
from src.replay_parser import extract_state_features


def get_torch_device(preferred_device: Optional[str] = None) -> torch.device:
    """Resolve and log the best available PyTorch compute device."""
    if preferred_device and preferred_device.lower() in ("cuda", "gpu"):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            print("[DeviceManager] WARNING: CUDA requested but not available; falling back to CPU.")
            device = torch.device("cpu")
    elif preferred_device and preferred_device.lower() == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print("=" * 70)
        print(f"  [GPU ACCELERATION] Device: {device} | {gpu_name} ({vram_gb:.1f} GB VRAM)")
        print(f"  CUDA Version: {torch.version.cuda} | PyTorch Backend: {torch.__version__}")
        print("=" * 70)
    else:
        print("=" * 70)
        print(f"  [COMPUTE DEVICE] Running on CPU (CUDA unavailable or not requested)")
        print("=" * 70)

    return device


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
            
            feat_vec = [float(state.get(col, 0)) for col in FEATURE_COLUMNS]
            
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
    device: Optional[torch.device] = None,
    seed: int = 42
) -> Dict[str, Any]:
    """Supervised pre-training (Behavioral Cloning) directly into PPO PyTorch policy network on GPU/CPU."""
    if device is None:
        device = get_torch_device()
        
    print(f"\n=== Initializing PPO Policy with Behavioral Cloning Weights on {device} ===")
    X, y = load_bc_dataset(dataset_path, max_samples=max_samples)
    
    # Train / Val Split
    np.random.seed(seed)
    indices = np.random.permutation(len(X))
    val_size = int(len(X) * val_split)
    train_idx, val_idx = indices[val_size:], indices[:val_size]
    
    X_train = torch.tensor(X[train_idx], device=device)
    y_train = torch.tensor(y[train_idx], device=device)
    X_val = torch.tensor(X[val_idx], device=device)
    y_val = torch.tensor(y[val_idx], device=device)
    
    # Policy network & optimizer
    policy_net = model.policy.to(device)
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss().to(device)
    
    n_train = len(X_train)
    history = []
    
    print(f"Training on {n_train} samples, validating on {len(X_val)} samples for {epochs} epochs on {device}...")
    
    for epoch in range(epochs):
        policy_net.train()
        perm = torch.randperm(n_train, device=device)
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
    print(f"\nBC Pre-Training Complete! Final Validation Accuracy: {final_val_acc*100:.2f}%\n")
    
    return {
        "epochs": epochs,
        "final_val_acc": float(final_val_acc),
        "history": history
    }


class LeagueOpponentPool:
    """Dynamic League Opponent Pool for Multi-Generational Self-Play.
    
    Maintains a weighted population of opponents:
    - Top Meta Agent (`submissions/meta_agent.py`)
    - Fast Baseline Heuristic Agent
    - Past Generation Checkpoints (`models/league/gen_*.zip`)
    """
    
    def __init__(self, include_top_meta: bool = True, include_baseline: bool = True):
        self.opponents: List[Dict[str, Any]] = []
        self.weights: List[float] = []
        self.current_opponent: Any = None
        self.current_name: str = "Unknown"
        
        if include_baseline:
            self.add_opponent(
                name="HeuristicBaseline",
                agent_obj=KaggricultureAgent(policy={"use_ensemble": False, "use_ml_policy": False}),
                weight=0.30
            )
            
        if include_top_meta:
            try:
                from submissions.meta_agent import agent as meta_agent_fn
                self.add_opponent(
                    name="TopMetaAgent",
                    agent_obj=meta_agent_fn,
                    weight=0.35
                )
            except Exception as e:
                print(f"[LeaguePool] Notice: Could not import TopMetaAgent: {e}")
                
        self.sample_new_match_opponent()

    def add_opponent(self, name: str, agent_obj: Any, weight: float = 0.25):
        """Register a new opponent in the active league pool."""
        self.opponents.append({"name": name, "agent": agent_obj})
        self.weights.append(weight)
        print(f"  [LeaguePool] Registered opponent: '{name}' (Weight: {weight:.2f})")

    def add_generation_checkpoint(self, checkpoint_path: str, gen: int):
        """Load a trained generation model checkpoint and add it to the opponent pool."""
        try:
            from stable_baselines3 import PPO
            model = PPO.load(checkpoint_path, device="cpu")
            
            class PPOGenWrapper:
                def __init__(self, ppo_model):
                    self.ppo_model = ppo_model
                    self.base_agent = KaggricultureAgent(policy={"use_ensemble": False, "use_ml_policy": False})
                
                def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
                    p = obs.get("player", 1)
                    feat_dict = extract_state_features(obs, player_idx=p)
                    feat_vec = np.array([float(feat_dict.get(c, 0)) for c in FEATURE_COLUMNS], dtype=np.float32)
                    act_idx, _ = self.ppo_model.predict(feat_vec, deterministic=True)
                    act_name = ACTION_LOOKUP[int(act_idx)] if 0 <= int(act_idx) < len(ACTION_LOOKUP) else "PASS"
                    action = self.base_agent.act(obs)
                    action["farmer"] = [[act_name]]
                    return action

            wrapper = PPOGenWrapper(model)
            weight = max(0.15, 0.40 / max(1, gen))
            self.add_opponent(name=f"Gen_{gen:02d}", agent_obj=wrapper, weight=weight)
        except Exception as e:
            print(f"[LeaguePool] Notice: Failed to add generation checkpoint {checkpoint_path}: {e}")

    def sample_new_match_opponent(self):
        """Sample a new opponent according to normalized league weights."""
        if not self.opponents:
            self.current_opponent = KaggricultureAgent(policy={"use_ensemble": False, "use_ml_policy": False})
            self.current_name = "DefaultFallback"
            return
        w = np.array(self.weights, dtype=np.float32)
        p = w / w.sum()
        chosen = np.random.choice(len(self.opponents), p=p)
        self.current_opponent = self.opponents[chosen]["agent"]
        self.current_name = self.opponents[chosen]["name"]

    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Callable/object interface for KaggricultureEnv opponent step."""
        step = obs.get("step", 0)
        if step == 0:
            self.sample_new_match_opponent()
            
        if hasattr(self.current_opponent, "act"):
            return self.current_opponent.act(obs)
        elif callable(self.current_opponent):
            try:
                return self.current_opponent(obs, None)
            except TypeError:
                return self.current_opponent(obs)
        return {"farmer": [["PASS"]], "hands": [], "market": []}


def evaluate_against_benchmarks(
    model: Any,
    n_episodes: int = 5,
    include_top_meta: bool = True
) -> Dict[str, Any]:
    """Comprehensive evaluation against Baseline and Top Meta opponents measuring final bank balances."""
    results = {}
    
    # 1. Evaluate vs Baseline
    baseline_opp = KaggricultureAgent(policy={"use_ensemble": False, "use_ml_policy": False})
    base_scores = []
    for _ in range(n_episodes):
        env = KaggricultureEnv(max_turns=720, opponent_agent=baseline_opp)
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        base_scores.append(float(env.money))
        
    mean_rew_base = float(np.mean(base_scores))
    std_rew_base = float(np.std(base_scores))
    results["vs_baseline"] = {"mean_bank": mean_rew_base, "std_bank": std_rew_base}
    print(f"  Evaluation vs Baseline: Mean Bank = ${mean_rew_base:,.2f} (+/- ${std_rew_base:,.2f})")
    
    # 2. Evaluate vs Top Meta (if available)
    if include_top_meta:
        try:
            from submissions.meta_agent import agent as meta_fn
            meta_scores = []
            for _ in range(n_episodes):
                env = KaggricultureEnv(max_turns=720, opponent_agent=meta_fn)
                obs, _ = env.reset()
                done = False
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                meta_scores.append(float(env.money))
                
            mean_rew_meta = float(np.mean(meta_scores))
            std_rew_meta = float(np.std(meta_scores))
            results["vs_top_meta"] = {"mean_bank": mean_rew_meta, "std_bank": std_rew_meta}
            print(f"  Evaluation vs Top Meta: Mean Bank = ${mean_rew_meta:,.2f} (+/- ${std_rew_meta:,.2f})")
        except Exception as e:
            print(f"  Evaluation vs Top Meta skipped: {e}")
            
    return results


def train_ppo_league(
    n_generations: int = 3,
    timesteps_per_gen: int = 20000,
    output_dir: str = "models/league",
    init_with_bc: bool = True,
    bc_dataset_path: str = "data/processed/training_pairs.jsonl",
    bc_epochs: int = 5,
    eval_episodes: int = 5,
    seed: int = 42,
    preferred_device: Optional[str] = None
) -> Dict[str, Any]:
    """Execute Multi-Generational League Training (Self-Play & Elite Opponents) on GPU/CPU."""
    try:
        from stable_baselines3 import PPO
    except ImportError as e:
        raise ImportError("stable-baselines3 is required for PPO training.") from e

    device = get_torch_device(preferred_device)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    print("\n" + "=" * 70)
    print(f"  STARTING MULTI-GENERATIONAL PPO LEAGUE TRAINING")
    print(f"  Generations: {n_generations} | Timesteps/Gen: {timesteps_per_gen:,} | Device: {device}")
    print("=" * 70)

    # Initialize League Opponent Pool
    league_pool = LeagueOpponentPool(include_top_meta=True, include_baseline=True)

    # Training Environment wrapped with League Opponent Pool
    env = KaggricultureEnv(max_turns=720, opponent_agent=league_pool)

    # Initialize PPO Model on selected device
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
        device=str(device),
        seed=seed
    )

    bc_metrics = {}
    if init_with_bc and os.path.exists(bc_dataset_path):
        bc_metrics = pretrain_ppo_with_bc(
            model=model,
            dataset_path=bc_dataset_path,
            epochs=bc_epochs,
            device=device,
            seed=seed
        )
        bc_checkpoint_path = os.path.join(output_dir, "gen_00_bc_init.zip")
        model.save(bc_checkpoint_path)
        print(f"Saved initial BC checkpoint to: {bc_checkpoint_path}")

    league_history = []

    for gen in range(1, n_generations + 1):
        print("\n" + "#" * 70)
        print(f"  >>> LEAGUE GENERATION {gen:02d} / {n_generations:02d} <<<")
        print(f"  Training for {timesteps_per_gen:,} timesteps on {device} against League Pool ({len(league_pool.opponents)} opponents)...")
        print("#" * 70)

        # Train Generation
        model.learn(total_timesteps=timesteps_per_gen, reset_num_timesteps=False)

        # Save Generation Checkpoint
        gen_checkpoint_path = os.path.join(output_dir, f"gen_{gen:02d}.zip")
        model.save(gen_checkpoint_path)
        model.save("models/ppo_agent.zip")  # Update active champion model
        print(f"\n[Generation {gen:02d}] Saved checkpoint to: {gen_checkpoint_path}")

        # Add this generation to the League Pool for subsequent generations
        league_pool.add_generation_checkpoint(gen_checkpoint_path, gen=gen)

        # Evaluate against benchmarks
        print(f"\n--- Evaluating Generation {gen:02d} ---")
        eval_metrics = evaluate_against_benchmarks(model, n_episodes=eval_episodes, include_top_meta=True)

        gen_record = {
            "generation": gen,
            "timesteps": gen * timesteps_per_gen,
            "device": str(device),
            "checkpoint": gen_checkpoint_path,
            "eval_metrics": eval_metrics
        }
        league_history.append(gen_record)

        # Save cumulative league history
        history_path = os.path.join(output_dir, "league_history.json")
        with open(history_path, "w") as f:
            json.dump(league_history, f, indent=2)

    print("\n" + "=" * 70)
    print(f"  LEAGUE TRAINING COMPLETE! {n_generations} Generations Successfully Trained.")
    print(f"  Champion model updated at: models/ppo_agent.zip")
    print(f"  Full league history saved to: {os.path.join(output_dir, 'league_history.json')}")
    print("=" * 70)

    return {
        "n_generations": n_generations,
        "timesteps_per_gen": timesteps_per_gen,
        "device": str(device),
        "bc_metrics": bc_metrics,
        "league_history": league_history
    }


def train_ppo(
    total_timesteps: int = 25000,
    output_path: str = "models/ppo_agent.zip",
    init_with_bc: bool = True,
    bc_dataset_path: str = "data/processed/training_pairs.jsonl",
    bc_epochs: int = 10,
    bc_max_samples: int = 200000,
    eval_episodes: int = 5,
    preferred_device: Optional[str] = None,
    seed: int = 42
) -> Dict[str, Any]:
    """Single-stage PPO training (legacy interface)."""
    device = get_torch_device(preferred_device)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    training_opp = KaggricultureAgent(policy={"use_ensemble": False, "use_ml_policy": False})
    eval_opp = KaggricultureAgent(policy={"use_ensemble": False, "use_ml_policy": False})
    env = KaggricultureEnv(max_turns=720, opponent_agent=training_opp)
    eval_env = KaggricultureEnv(max_turns=720, opponent_agent=eval_opp)
    
    from stable_baselines3 import PPO
    from stable_baselines3.common.evaluation import evaluate_policy
    
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
        device=str(device),
        seed=seed
    )
    
    bc_metrics = {}
    if init_with_bc and os.path.exists(bc_dataset_path):
        bc_metrics = pretrain_ppo_with_bc(
            model=model,
            dataset_path=bc_dataset_path,
            max_samples=bc_max_samples,
            epochs=bc_epochs,
            device=device,
            seed=seed
        )
        bc_init_path = output_path.replace(".zip", "_bc_initialized.zip")
        model.save(bc_init_path)

    if total_timesteps > 0:
        model.learn(total_timesteps=total_timesteps)
    
    model.save(output_path)
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=eval_episodes, deterministic=True)
    
    meta = {
        "algorithm": "PPO",
        "device": str(device),
        "total_timesteps": total_timesteps,
        "evaluation": {"mean_reward": float(mean_reward), "std_reward": float(std_reward)},
        "model_file": output_path
    }
    meta_path = output_path.replace(".zip", "_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
        
    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Generational PPO League Training for Kaggriculture")
    parser.add_argument("--league", action="store_true", default=True, help="Run multi-generational league self-play")
    parser.add_argument("--generations", type=int, default=3, help="Number of self-play generations")
    parser.add_argument("--timesteps-per-gen", type=int, default=20000, help="Timesteps per generation")
    parser.add_argument("--device", default="auto", help="Compute device (auto, cuda, cpu)")
    parser.add_argument("--output-dir", default="models/league", help="Output directory for league checkpoints")
    parser.add_argument("--init-with-bc", action="store_true", default=True, help="Pre-train with BC weights")
    parser.add_argument("--no-bc", action="store_false", dest="init_with_bc", help="Do not pre-train with BC")
    parser.add_argument("--bc-data", default="data/processed/training_pairs.jsonl", help="BC dataset path")
    parser.add_argument("--bc-epochs", type=int, default=5, help="BC pretraining epochs")
    parser.add_argument("--eval-episodes", type=int, default=3, help="Number of eval episodes per generation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    if args.league or args.generations > 1:
        train_ppo_league(
            n_generations=args.generations,
            timesteps_per_gen=args.timesteps_per_gen,
            output_dir=args.output_dir,
            init_with_bc=args.init_with_bc,
            bc_dataset_path=args.bc_data,
            bc_epochs=args.bc_epochs,
            eval_episodes=args.eval_episodes,
            seed=args.seed,
            preferred_device=args.device
        )
    else:
        train_ppo(
            total_timesteps=args.timesteps_per_gen,
            output_path=os.path.join(args.output_dir, "ppo_agent.zip"),
            init_with_bc=args.init_with_bc,
            bc_dataset_path=args.bc_data,
            bc_epochs=args.bc_epochs,
            eval_episodes=args.eval_episodes,
            preferred_device=args.device,
            seed=args.seed
        )
