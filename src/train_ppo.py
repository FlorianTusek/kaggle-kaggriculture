# SPDX-License-Identifier: MIT
"""PPO Reinforcement Learning & 100,000 Generation League Training for Kaggriculture.

Features:
1. GPU-Accelerated PyTorch Backend (auto-detects CUDA 12.1 / RTX 4070 with CPU fallback).
2. Behavioral Cloning Policy Pre-Training: Initializes PPO ActorCriticPolicy weights from expert dataset.
3. Multi-Generational Self-Play (League Training): Continuous multi-generation self-play loop
   designed to run up to 100,000 generations.
4. Constant Memory & Disk Management:
   - Rolling Opponent Pool: In-memory sliding window of recent generation policies.
   - Smart Checkpoint Pruning: Retains milestone checkpoints every N generations while keeping disk footprint small.
   - Streaming JSONL Telemetry: Appends records continuously to `league_history.jsonl`.
5. Automatic Resume & Fault Tolerance.
"""

import os
import sys
import glob
import json
import time
import shutil
import argparse

# Ensure real-time unbuffered logging
sys.stdout.reconfigure(line_buffering=True)

# Prioritize base Python site-packages with CUDA 12.1 PyTorch
sys.path.insert(0, r"C:\Python310\lib\site-packages")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional, List, Union

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
    epochs: int = 5,
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
    
    np.random.seed(seed)
    indices = np.random.permutation(len(X))
    val_size = int(len(X) * val_split)
    train_idx, val_idx = indices[val_size:], indices[:val_size]
    
    X_train = torch.tensor(X[train_idx], device=device)
    y_train = torch.tensor(y[train_idx], device=device)
    X_val = torch.tensor(X[val_idx], device=device)
    y_val = torch.tensor(y[val_idx], device=device)
    
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
    """Dynamic Constant-Memory League Opponent Pool for Multi-Generational Self-Play.
    
    Maintains a sliding window of generation opponents along with elite anchors:
    - Top Meta Agent (`submissions/meta_agent.py`)
    - Fast Baseline Heuristic Agent
    - Sliding window of past generation checkpoints (`PPOGenWrapper`)
    """
    
    def __init__(self, include_top_meta: bool = True, include_baseline: bool = True, max_gen_opponents: int = 10):
        self.base_opponents: List[Dict[str, Any]] = []
        self.base_weights: List[float] = []
        self.gen_opponents: List[Dict[str, Any]] = []
        self.gen_weights: List[float] = []
        self.max_gen_opponents = max_gen_opponents
        self.current_opponent: Any = None
        self.current_name: str = "Unknown"
        
        if include_baseline:
            self.base_opponents.append({
                "name": "HeuristicBaseline",
                "agent": KaggricultureAgent(policy={"use_ensemble": False, "use_ml_policy": False})
            })
            self.base_weights.append(0.30)
            
        if include_top_meta:
            try:
                from submissions.meta_agent import agent as meta_agent_fn
                self.base_opponents.append({
                    "name": "TopMetaAgent",
                    "agent": meta_agent_fn
                })
                self.base_weights.append(0.35)
            except Exception as e:
                print(f"[LeaguePool] Notice: Could not import TopMetaAgent: {e}")
                
        self.sample_new_match_opponent()

    def add_generation_checkpoint(self, checkpoint_path: str, gen: int):
        """Load a trained generation checkpoint into the sliding memory pool."""
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
            
            # Maintain sliding window to prevent OOM
            if len(self.gen_opponents) >= self.max_gen_opponents:
                self.gen_opponents.pop(0)
                self.gen_weights.pop(0)
                
            self.gen_opponents.append({
                "name": f"Gen_{gen:05d}",
                "agent": wrapper
            })
            self.gen_weights.append(0.35 / max(1, len(self.gen_opponents)))
        except Exception as e:
            print(f"[LeaguePool] Notice: Failed to add generation checkpoint {checkpoint_path}: {e}")

    def sample_new_match_opponent(self):
        """Sample a new opponent according to normalized league weights."""
        all_opps = self.base_opponents + self.gen_opponents
        all_weights = self.base_weights + self.gen_weights
        if not all_opps:
            self.current_opponent = KaggricultureAgent(policy={"use_ensemble": False, "use_ml_policy": False})
            self.current_name = "DefaultFallback"
            return
        w = np.array(all_weights, dtype=np.float32)
        p = w / w.sum()
        chosen = np.random.choice(len(all_opps), p=p)
        self.current_opponent = all_opps[chosen]["agent"]
        self.current_name = all_opps[chosen]["name"]

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
    n_episodes: int = 2,
    include_top_meta: bool = True
) -> Dict[str, Any]:
    """Evaluation against Baseline and Top Meta measuring final bank balances."""
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
        except Exception as e:
            results["vs_top_meta"] = {"error": str(e)}
            
    return results


def prune_old_checkpoints(output_dir: str, keep_recent: int = 5):
    """Remove older non-milestone generation zip files to save disk space."""
    pattern = os.path.join(output_dir, "gen_*.zip")
    files = sorted(glob.glob(pattern))
    if len(files) > keep_recent:
        for f in files[:-keep_recent]:
            try:
                os.remove(f)
            except Exception:
                pass


def train_ppo_league(
    n_generations: int = 100000,
    timesteps_per_gen: int = 2048,
    output_dir: str = "models/league",
    init_with_bc: bool = True,
    bc_dataset_path: str = "data/processed/training_pairs.jsonl",
    bc_epochs: int = 5,
    eval_episodes: int = 2,
    eval_freq: int = 100,
    save_freq: int = 100,
    keep_recent_checkpoints: int = 5,
    seed: int = 42,
    preferred_device: Optional[str] = None,
    resume: bool = True
) -> Dict[str, Any]:
    """Execute Multi-Generational League Training targeting up to 100,000 generations."""
    try:
        from stable_baselines3 import PPO
    except ImportError as e:
        raise ImportError("stable-baselines3 is required for PPO training.") from e

    device = get_torch_device(preferred_device)
    os.makedirs(output_dir, exist_ok=True)
    milestones_dir = os.path.join(output_dir, "milestones")
    os.makedirs(milestones_dir, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    print("\n" + "=" * 70)
    print(f"  STARTING 100,000 GENERATION PPO LEAGUE TRAINING")
    print(f"  Target Generations: {n_generations:,} | Timesteps/Gen: {timesteps_per_gen:,} | Device: {device}")
    print(f"  Eval Frequency: every {eval_freq} gens | Milestone Save: every {save_freq} gens")
    print("=" * 70)

    league_pool = LeagueOpponentPool(include_top_meta=True, include_baseline=True, max_gen_opponents=10)
    env = KaggricultureEnv(max_turns=720, opponent_agent=league_pool)

    start_gen = 1
    champion_path = "models/ppo_agent.zip"
    model = None

    # Check for resume
    summary_path = os.path.join(output_dir, "league_summary.json")
    if resume and os.path.exists(summary_path) and os.path.exists(champion_path):
        try:
            with open(summary_path, "r") as f:
                summary = json.load(f)
            start_gen = summary.get("current_generation", 0) + 1
            print(f"\n[Resume] Found active league state. Resuming from Generation {start_gen:,}...")
            model = PPO.load(champion_path, env=env, device=str(device))
        except Exception as e:
            print(f"[Resume] Could not resume from checkpoint ({e}). Starting fresh.")
            model = None

    if model is None:
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
            verbose=0,
            device=str(device),
            seed=seed
        )
        if init_with_bc and os.path.exists(bc_dataset_path):
            pretrain_ppo_with_bc(
                model=model,
                dataset_path=bc_dataset_path,
                epochs=bc_epochs,
                device=device,
                seed=seed
            )
            model.save(champion_path)

    jsonl_path = os.path.join(output_dir, "league_history.jsonl")
    start_time = time.time()
    total_timesteps_trained = (start_gen - 1) * timesteps_per_gen
    target_end_gen = max(n_generations, start_gen + n_generations - 1) if start_gen > n_generations else n_generations
    gen = start_gen

    summary = {
        "current_generation": start_gen,
        "target_generations": target_end_gen,
        "total_timesteps": total_timesteps_trained,
        "device": str(device),
        "fps": 0.0,
        "uptime_seconds": 0.0,
        "last_eval": {},
        "champion_model": champion_path
    }

    try:
        for gen in range(start_gen, target_end_gen + 1):
            gen_start_time = time.time()

            # Train generation
            model.learn(total_timesteps=timesteps_per_gen, reset_num_timesteps=False)
            total_timesteps_trained += timesteps_per_gen
            gen_fps = timesteps_per_gen / max(0.001, (time.time() - gen_start_time))

            # Save rolling checkpoint & champion
            gen_checkpoint_path = os.path.join(output_dir, f"gen_{gen:05d}.zip")
            model.save(gen_checkpoint_path)
            model.save(champion_path)

            # Save milestone
            if gen % save_freq == 0:
                milestone_path = os.path.join(milestones_dir, f"gen_{gen:05d}.zip")
                shutil.copyfile(gen_checkpoint_path, milestone_path)

            # Prune old zip files to preserve disk space
            prune_old_checkpoints(output_dir, keep_recent=keep_recent_checkpoints)

            # Add to sliding league pool
            league_pool.add_generation_checkpoint(gen_checkpoint_path, gen=gen)

            # Periodic benchmark evaluation
            eval_metrics = {}
            if gen % eval_freq == 0 or gen <= 5:
                eval_metrics = evaluate_against_benchmarks(model, n_episodes=eval_episodes, include_top_meta=True)
                base_b = eval_metrics.get("vs_baseline", {}).get("mean_bank", 0)
                meta_b = eval_metrics.get("vs_top_meta", {}).get("mean_bank", 0)
                print(f"[Gen {gen:05d}/{target_end_gen:,}] FPS: {gen_fps:.0f} | Eval vs Base: ${base_b:,.0f} | vs Top Meta: ${meta_b:,.0f} | Total Steps: {total_timesteps_trained:,}")
            elif gen % 10 == 0:
                elapsed = time.time() - start_time
                print(f"[Gen {gen:05d}/{target_end_gen:,}] FPS: {gen_fps:.0f} | Total Steps: {total_timesteps_trained:,} | Uptime: {elapsed/60:.1f}m")

            # Stream telemetry to JSONL
            record = {
                "gen": gen,
                "timesteps": total_timesteps_trained,
                "fps": float(gen_fps),
                "timestamp": time.time(),
                "eval": eval_metrics
            }
            with open(jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

            # Update summary JSON
            summary = {
                "current_generation": gen,
                "target_generations": target_end_gen,
                "total_timesteps": total_timesteps_trained,
                "device": str(device),
                "fps": float(gen_fps),
                "uptime_seconds": time.time() - start_time,
                "last_eval": eval_metrics,
                "champion_model": champion_path
            }
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)

    except KeyboardInterrupt:
        print("\n[League Training] Training interrupted by user. State saved successfully.")
    except Exception as e:
        print(f"\n[League Training] Exception encountered: {e}")
        raise

    print("\n" + "=" * 70)
    print(f"  LEAGUE TRAINING COMPLETE / SUSPENDED at Generation {gen:,}.")
    print(f"  Summary saved to: {summary_path}")
    print("=" * 70)

    return summary


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
    """Single-stage PPO training."""
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
    parser.add_argument("--generations", type=int, default=100000, help="Number of self-play generations")
    parser.add_argument("--timesteps-per-gen", type=int, default=2048, help="Timesteps per generation")
    parser.add_argument("--eval-freq", type=int, default=100, help="Evaluation frequency in generations")
    parser.add_argument("--save-freq", type=int, default=100, help="Milestone save frequency in generations")
    parser.add_argument("--device", default="auto", help="Compute device (auto, cuda, cpu)")
    parser.add_argument("--output-dir", default="models/league", help="Output directory for league checkpoints")
    parser.add_argument("--init-with-bc", action="store_true", default=True, help="Pre-train with BC weights")
    parser.add_argument("--no-bc", action="store_false", dest="init_with_bc", help="Do not pre-train with BC")
    parser.add_argument("--bc-data", default="data/processed/training_pairs.jsonl", help="BC dataset path")
    parser.add_argument("--bc-epochs", type=int, default=5, help="BC pretraining epochs")
    parser.add_argument("--eval-episodes", type=int, default=2, help="Number of eval episodes per generation")
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
            eval_freq=args.eval_freq,
            save_freq=args.save_freq,
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
