import json
import matplotlib.pyplot as plt
import os
from datetime import datetime

# Read scoring steps data
with open('scoring_steps.json') as f:
    data = json.load(f)

steps = [step['step'] for step in data['steps']]
scores = [step['cumulative_score'] for step in data['steps']]
deltas = [step['delta'] for step in data['steps']]

# Create figures directory if it doesn't exist
figures_dir = 'figures'
if not os.path.exists(figures_dir):
    os.makedirs(figures_dir)

# Generate timestamp for the figure filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
figure_filename = f"scoring_evolution_{timestamp}.png"
figure_path = os.path.join(figures_dir, figure_filename)

# Generate and save the figure
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(steps, scores, 'b-', linewidth=2)
plt.title('Cumulative Score Evolution')
plt.xlabel('Step')
plt.ylabel('Score')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(steps, deltas, 'r-', linewidth=2)
plt.title('Delta per Step')
plt.xlabel('Step')
plt.ylabel('Delta')
plt.grid(True)

plt.tight_layout()
plt.savefig(figure_path)

print(f"Figure saved to {figure_path}")