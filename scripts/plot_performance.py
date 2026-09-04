import json
import matplotlib.pyplot as plt
import os

# Load history
with open('models/league/league_history.json', 'r') as f:
    history = json.load(f)

generations = [entry['generation'] for entry in history]
vs_baseline = [entry['eval_metrics']['vs_baseline']['mean_bank'] for entry in history]
vs_top_meta = [entry['eval_metrics']['vs_top_meta']['mean_bank'] for entry in history]

plt.figure(figsize=(10, 6))
plt.plot(generations, vs_baseline, label='vs Baseline (Self/Previous)', marker='o', color='blue', linewidth=2)
plt.plot(generations, vs_top_meta, label='vs Top Meta', marker='s', color='red', linewidth=2)

plt.title('Agent Performance Across Generations')
plt.xlabel('Generation')
plt.ylabel('Mean Bank Balance (Coins)')
plt.legend()
plt.grid(True, alpha=0.3)

# Save to artifact directory
artifact_dir = r"C:\Users\Florian\.gemini\antigravity-cli\brain\dacf9574-0214-4c01-9676-192ff662e022"
os.makedirs(artifact_dir, exist_ok=True)
output_path = os.path.join(artifact_dir, 'performance_chart.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Saved plot to {output_path}")
