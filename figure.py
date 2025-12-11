import json
import matplotlib.pyplot as plt

with open('poem_scoring_steps.json') as f:
    data = json.load(f)

steps = [step['step'] for step in data['steps']]
scores = [step['cumulative_score'] for step in data['steps']]
deltas = [step['delta'] for step in data['steps']]

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
plt.savefig('scoring_evolution.png')