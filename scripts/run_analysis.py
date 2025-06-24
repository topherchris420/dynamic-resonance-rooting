import argparse
import numpy as np
import pandas as pd
import sys
import os

# Add the project root to the Python path to allow for module imports
# This is necessary so you can run this script from the project root
# (e.g., `python scripts/run_analysis.py`)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from drr_framework.analysis import DynamicResonanceRooting
from drr_framework.models import BenchmarkSystems

def main():
    """
    Main function to run the DRR analysis from the command line.
    """
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(description="Run Dynamic Resonance Rooting (DRR) analysis.")
    parser.add_argument(
        '--system',
        type=str,
        default='lorenz',
        choices=['lorenz', 'rossler', 'physio'],
        help='The system to analyze (default: lorenz).'
    )
    parser.add_argument(
        '--embedding_dim',
        type=int,
        default=3,
        help='Embedding dimension for phase space reconstruction.'
    )
    parser.add_argument(
        '--tau',
        type=int,
        default=10,
        help='Time delay for embedding.'
    )
    parser.add_argument(
        '--sampling_rate',
        type=int,
        default=100,
        help='Sampling rate of the data in Hz.'
    )
    args = parser.parse_args()

    print(f"--- Starting DRR analysis for the {args.system.capitalize()} system ---")

    # 1. Generate or load data based on the chosen system
    if args.system == 'lorenz':
        _, data = BenchmarkSystems.generate_lorenz_data(duration=30, dt=1.0/args.sampling_rate)
        is_multivariate = True
    elif args.system == 'rossler':
        _, data = BenchmarkSystems.generate_rossler_data(duration=30, dt=1.0/args.sampling_rate)
        is_multivariate = True
    else:
        # For simplicity, we'll use a placeholder for custom data.
        # In a real scenario, you would load this from a file.
        print("Generating synthetic physiological signal...")
        t = np.linspace(0, 30, 30 * args.sampling_rate)
        data = (np.sin(2 * np.pi * 1.2 * t) + 0.5 * np.sin(2 * np.pi * 0.25 * t))
        is_multivariate = False


    # 2. Initialize the DRR framework with parameters from command line
    drr_analyzer = DynamicResonanceRooting(
        embedding_dim=args.embedding_dim,
        tau=args.tau,
        sampling_rate=args.sampling_rate
    )

    # 3. Run the analysis
    results = drr_analyzer.analyze_system(data, multivariate=is_multivariate)

    # 4. Print the results
    print("\n--- Analysis Complete ---")
    print("Resonance Depths:")
    if results and 'resonance_depths' in results:
        for key, depth in results['resonance_depths'].items():
            print(f"  - {key}: {depth:.6f}")
    else:
        print("  - No resonance depths found.")

    print("\nDominant Frequencies:")
    if results and 'resonances' in results:
        for key, resonance_data in results['resonances'].items():
            freq = resonance_data.get('dominant_freq', 'N/A')
            print(f"  - {key}: {freq} Hz")
    else:
        print("  - No dominant frequencies found.")

if __name__ == "__main__":
    main()
