import argparse
import yaml
import numpy as np
import pandas as pd
import sys
import os
from typing import Dict, Any

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from drr_framework.analysis import DynamicResonanceRooting
from drr_framework.models import BenchmarkSystems

def load_data(config: Dict[str, Any]) -> np.ndarray:
    """
    Loads or generates data based on the configuration.

    Args:
        config (Dict[str, Any]): The configuration dictionary.

    Returns:
        np.ndarray: The loaded or generated data.
    """
    data_config = config['data']
    if data_config['source'] == 'generate':
        system = config['system']
        print(f"Generating data for the {system} system...")
        if system == 'lorenz':
            _, data = BenchmarkSystems.generate_lorenz_data(
                duration=data_config['duration'], dt=data_config['dt']
            )
        elif system == 'rossler':
            _, data = BenchmarkSystems.generate_rossler_data(
                duration=data_config['duration'], dt=data_config['dt']
            )
        else: # physio
            t = np.linspace(0, data_config['duration'], int(data_config['duration'] * config['drr_parameters']['sampling_rate']))
            data = (np.sin(2 * np.pi * 1.2 * t) + 0.5 * np.sin(2 * np.pi * 0.25 * t))
        return data
    elif data_config['source'] == 'file':
        try:
            print(f"Loading data from {data_config['file_path']}...")
            return pd.read_csv(data_config['file_path']).values
        except FileNotFoundError:
            print(f"Error: The file {data_config['file_path']} was not found.")
            sys.exit(1)
        except Exception as e:
            print(f"An error occurred while loading the data: {e}")
            sys.exit(1)
    else:
        raise ValueError("Invalid data source in config. Choose 'generate' or 'file'.")


def main():
    """
    Main function to run the DRR analysis from the command line.
    """
    parser = argparse.ArgumentParser(description="Run Dynamic Resonance Rooting (DRR) analysis.")
    parser.add_argument(
        '--config',
        type=str,
        default='config.yml',
        help='Path to the configuration file (default: config.yml).'
    )
    args = parser.parse_args()

    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: The config file {args.config} was not found.")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing the YAML config file: {e}")
        sys.exit(1)


    print(f"--- Starting DRR analysis for the {config['system'].capitalize()} system ---")

    # 1. Load data
    data = load_data(config)
    
    # 2. Initialize the DRR framework
    drr_analyzer = DynamicResonanceRooting(**config['drr_parameters'])

    # 3. Run the analysis
    results = drr_analyzer.analyze_system(data, multivariate=config['is_multivariate'])

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
