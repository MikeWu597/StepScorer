import json
import matplotlib.pyplot as plt
import os
from datetime import datetime
import sys
import argparse

def generate_figure(input_file='scoring_steps.json', output_file=None):
    """Generate figure from scoring steps data"""
    # Read scoring steps data
    with open(input_file) as f:
        data = json.load(f)

    steps = [step['step'] for step in data['steps']]
    scores = [step['cumulative_score'] for step in data['steps']]
    deltas = [step['delta'] for step in data['steps']]

    # Create figures directory if it doesn't exist
    figures_dir = 'figures'
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)

    # Determine output file path
    if output_file:
        figure_path = output_file
    else:
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
    plt.close()  # Close the figure to free memory

    print(f"Figure saved to {figure_path}")
    return figure_path

def main():
    parser = argparse.ArgumentParser(description='Generate figure from scoring steps data')
    parser.add_argument('--input', default='scoring_steps.json', help='Input JSON file with scoring steps')
    parser.add_argument('--output', help='Output PNG file for the figure')
    
    args = parser.parse_args()
    generate_figure(args.input, args.output)

if __name__ == "__main__":
    main()