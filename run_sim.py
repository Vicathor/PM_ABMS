#!/usr/bin/env python3
"""
Main simulation runner for football process mining simulation.

This script runs the football simulation with configurable parameters
and outputs event logs for process mining analysis.
"""

import argparse
import yaml
import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from sim.env import FootballModel


def load_config(config_path: str) -> dict:
    """
    Load configuration from YAML file.
    
    Parameters
    ----------
    config_path : str
        Path to configuration file
        
    Returns
    -------
    dict
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Configuration file not found: {config_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing configuration file: {e}")
        sys.exit(1)


def run_simulation(config: dict, num_loops: int = 1) -> dict:
    """
    Run the football simulation.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
    num_loops : int
        Number of simulation loops to run
        
    Returns
    -------
    dict
        Simulation results and statistics
    """
    print("Starting football simulation...")
    print(f"Loops: {num_loops}")
    print(f"Realism level: {config.get('realism_level', 'toy')}")
    print(f"Pitch dimensions: {config.get('pitch_width', 34)}m x {config.get('pitch_length', 52)}m")
    print("-" * 50)
    
    all_stats = []
    
    for loop in range(num_loops):
        print(f"Running simulation {loop + 1}/{num_loops}...")
        
        # Create and run model
        model = FootballModel(config)
        
        # Run until completion
        step_count = 0
        max_steps = 10000  # Safety limit
        
        while model.running and step_count < max_steps:
            model.step()
            step_count += 1
        
        # Get results
        stats = model.get_match_summary()
        stats['loop_number'] = loop + 1
        stats['total_steps'] = step_count
        all_stats.append(stats)
        
        print(f"  Completed in {step_count} steps ({stats['match_duration']:.1f}s match time)")
        print(f"  Events: {stats['total_events']}, Possessions: {stats['total_possessions']}")
        print(f"  Goals: {stats['goals']}, Shots: {stats['shots']}")
        print(f"  Mean xThreat/possession: {stats['mean_xThreat_per_possession']:.4f}")
        
        # Save logs (overwrite for single run, append suffix for multiple)
        if num_loops == 1:
            csv_path = config.get('logging', {}).get('csv_output', 'match_log.csv')
            xes_path = config.get('logging', {}).get('xes_output', 'match_log.xes')
        else:
            csv_path = f"match_log_{loop + 1}.csv"
            xes_path = f"match_log_{loop + 1}.xes"
        
        model.save_logs(csv_path, xes_path)
        print(f"  Logs saved to {csv_path} and {xes_path}")
        print()
    
    # Calculate summary statistics across all runs
    if len(all_stats) > 1:
        summary_stats = calculate_summary_statistics(all_stats)
        print("=" * 50)
        print("SUMMARY ACROSS ALL RUNS")
        print("=" * 50)
        print(f"Total runs: {len(all_stats)}")
        print(f"Average events per match: {summary_stats['avg_events']:.1f}")
        print(f"Average possessions per match: {summary_stats['avg_possessions']:.1f}")
        print(f"Average goals per match: {summary_stats['avg_goals']:.2f}")
        print(f"Average shots per match: {summary_stats['avg_shots']:.2f}")
        print(f"Average xThreat per possession: {summary_stats['avg_xThreat_per_possession']:.4f}")
        print(f"Average turnovers per match: {summary_stats['avg_turnovers']:.1f}")
        print(f"Average match duration: {summary_stats['avg_match_duration']:.1f}s")
        
        return summary_stats
    else:
        return all_stats[0]


def calculate_summary_statistics(all_stats: list) -> dict:
    """
    Calculate summary statistics across multiple simulation runs.
    
    Parameters
    ----------
    all_stats : list
        List of statistics dictionaries from each run
        
    Returns
    -------
    dict
        Summary statistics
    """
    import numpy as np
    
    return {
        'avg_events': np.mean([s['total_events'] for s in all_stats]),
        'avg_possessions': np.mean([s['total_possessions'] for s in all_stats]),
        'avg_goals': np.mean([s['goals'] for s in all_stats]),
        'avg_shots': np.mean([s['shots'] for s in all_stats]),
        'avg_xThreat_per_possession': np.mean([s['mean_xThreat_per_possession'] for s in all_stats]),
        'avg_turnovers': np.mean([s['turnovers'] for s in all_stats]),
        'avg_match_duration': np.mean([s['match_duration'] for s in all_stats]),
        'std_events': np.std([s['total_events'] for s in all_stats]),
        'std_possessions': np.std([s['total_possessions'] for s in all_stats]),
        'std_goals': np.std([s['goals'] for s in all_stats]),
        'total_runs': len(all_stats)
    }


def main():
    """Main entry point for the simulation."""
    parser = argparse.ArgumentParser(
        description='Run football simulation for process mining analysis'
    )
    parser.add_argument(
        '--loops', '-l',
        type=int,
        default=1,
        help='Number of simulation loops to run (default: 1)'
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='configs/base.yaml',
        help='Path to configuration file (default: configs/base.yaml)'
    )
    parser.add_argument(
        '--realism',
        choices=['toy', 'empirical'],
        help='Override realism level from config'
    )
    parser.add_argument(
        '--time',
        type=int,
        help='Override max game time in seconds'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Apply command line overrides
    if args.realism:
        config['realism_level'] = args.realism
    if args.time:
        config['max_game_time'] = args.time
    
    # Ensure output directory exists
    os.makedirs('outputs', exist_ok=True)
    
    # Update output paths to use outputs directory
    if 'logging' in config:
        config['logging']['csv_output'] = 'outputs/' + config['logging'].get('csv_output', 'match_log.csv')
        config['logging']['xes_output'] = 'outputs/' + config['logging'].get('xes_output', 'match_log.xes')
    
    try:
        # Run simulation
        run_simulation(config, args.loops)
        
        print("\nSimulation completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
        return 1
    except Exception as e:
        print(f"\nError running simulation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
