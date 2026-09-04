import sys
import os
import importlib.util

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.env import KaggricultureEnv

def load_agent(filepath):
    spec = importlib.util.spec_from_file_location("opponent", filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.agent

def run_match():
    print("Loading agents...")
    agent_main = load_agent("main.py")
    opponent_func = load_agent("submissions/meta_agent.py")
    
    env = KaggricultureEnv(max_turns=720, opponent_agent=opponent_func)
    
    print(f"Running match...")
    try:
        results = env.run_match(agent_main)
        
        r1 = results["agent_reward"]
        r2 = results["opponent_reward"]
        
        print("\n--- RESULTS ---")
        print(f"Our Phase 11 Agent: ${r1}")
        print(f"Meta Opponent: ${r2}")
        
        if r1 > r2:
            print(f"Our Agent WINS by ${r1 - r2}")
        elif r2 > r1:
            print(f"Meta Opponent WINS by ${r2 - r1}")
        else:
            print("TIE!")
            
    except Exception as e:
        print(f"Match failed: {e}")

if __name__ == "__main__":
    run_match()
