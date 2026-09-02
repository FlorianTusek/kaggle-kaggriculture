# SPDX-License-Identifier: MIT
import sys
sys.path.insert(0, ".")
sys.path.insert(0, r"C:\Python310\lib\site-packages")
from src.env import KaggricultureEnv
from src.agent import KaggricultureAgent
from stable_baselines3 import PPO

model = PPO.load("models/ppo_agent.zip", device="cpu")
opp = KaggricultureAgent(policy={"use_ensemble": False, "use_ml_policy": False})
env = KaggricultureEnv(max_turns=720, opponent_agent=opp)
obs, info = env.reset()

print(f"Turn 0: Money=${env.money}, Seeds={env.seeds}, Shed={env.shed}")
for t in range(720):
    action, _ = model.predict(obs, deterministic=True)
    obs, r, term, trunc, info = env.step(action)
    if (t + 1) % 24 == 0 or info.get("earned", 0) != 0:
        day = (t + 1) // 24
        print(f"Turn {t+1} (Day {day}): Money=${env.money} | NetWealth=${info.get('net_wealth')} | Shed={env.shed} | Seeds={env.seeds} | OppMoney=${env.opponent_money}")
