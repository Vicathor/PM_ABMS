#!/usr/bin/env python3
"""
Football Simulation Process Mining Demo
=====================================

This script demonstrates process mining analysis of football simulation data using PM4Py.
It loads match logs, discovers process models, and visualizes patterns.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Process mining libraries
try:
    import pm4py
    from pm4py.objects.log.importer.xes import importer as xes_importer
    from pm4py.algo.discovery.inductive import algorithm as inductive_miner
    from pm4py.visualization.petri_net import visualizer as pn_visualizer
    from pm4py.objects.conversion.log import converter as log_converter
    print("✓ PM4Py successfully imported")
except ImportError as e:
    print(f"✗ PM4Py import error: {e}")
    print("Please install PM4Py: pip install pm4py")
    exit(1)

def load_match_data():
    """Load match log from available formats."""
    
    # Try to load XES format first
    xes_path = Path('outputs/match_log.xes')
    csv_path = Path('outputs/match_log.csv')
    
    if xes_path.exists():
        print(f"✓ Loading XES format: {xes_path}")
        try:
            log = xes_importer.apply(str(xes_path))
            print(f"✓ Loaded {len(log)} traces from XES")
            return log, 'xes'
        except Exception as e:
            print(f"✗ XES loading failed: {e}")
    
    if csv_path.exists():
        print(f"✓ Loading CSV format: {csv_path}")
        try:
            df = pd.read_csv(csv_path)
            print(f"✓ Loaded {len(df)} events from CSV")
            
            # Convert to PM4Py event log format
            df = df.rename(columns={
                'case_id': 'case:concept:name',
                'timestamp': 'time:timestamp',
                'action_type': 'concept:name'
            })
            
            # Convert timestamp to datetime
            df['time:timestamp'] = pd.to_datetime(df['time:timestamp'])
            
            # Create event log
            log = log_converter.apply(df)
            print(f"✓ Converted to PM4Py log with {len(log)} traces")
            return log, 'csv'
        except Exception as e:
            print(f"✗ CSV loading failed: {e}")
    
    print("✗ No valid match log found")
    return None, None

def analyze_events(log, source_format):
    """Analyze event patterns in the log."""
    print("\n" + "="*50)
    print("EVENT ANALYSIS")
    print("="*50)
    
    if source_format == 'csv':
        # For CSV, we need to extract data differently
        df = pd.read_csv('outputs/match_log.csv')
        
        print(f"Total events: {len(df)}")
        print(f"Unique possessions: {df['case_id'].nunique()}")
        print(f"Event types: {df['action_type'].value_counts().to_dict()}")
        
        # xThreat analysis
        if 'xThreat_delta' in df.columns:
            threat_stats = df['xThreat_delta'].describe()
            print(f"\nxThreat Statistics:")
            print(f"  Mean delta: {threat_stats['mean']:.4f}")
            print(f"  Max delta: {threat_stats['max']:.4f}")
            print(f"  Min delta: {threat_stats['min']:.4f}")
            
            # Positive threat events
            positive_threat = df[df['xThreat_delta'] > 0]
            print(f"  Positive threat events: {len(positive_threat)} ({len(positive_threat)/len(df)*100:.1f}%)")
    
    else:
        # For XES format
        print(f"Total traces: {len(log)}")
        total_events = sum(len(trace) for trace in log)
        print(f"Total events: {total_events}")

def discover_process_model(log):
    """Discover process model using Inductive Miner."""
    print("\n" + "="*50)
    print("PROCESS MODEL DISCOVERY")
    print("="*50)
    
    try:
        # Apply Inductive Miner
        print("Applying Inductive Miner...")
        net, initial_marking, final_marking = inductive_miner.apply(log)
        
        print(f"✓ Discovered Petri net with {len(net.places)} places and {len(net.transitions)} transitions")
        
        # Visualize the Petri net
        print("Generating Petri net visualization...")
        gviz = pn_visualizer.apply(net, initial_marking, final_marking)
        
        # Save visualization
        output_path = Path('outputs/petri_net.png')
        pn_visualizer.save(gviz, str(output_path))
        print(f"✓ Petri net saved to: {output_path}")
        
        return net, initial_marking, final_marking
        
    except Exception as e:
        print(f"✗ Process discovery failed: {e}")
        return None, None, None

def analyze_possession_patterns(log, source_format):
    """Analyze possession patterns and transitions."""
    print("\n" + "="*50)
    print("POSSESSION PATTERN ANALYSIS")
    print("="*50)
    
    if source_format == 'csv':
        df = pd.read_csv('outputs/match_log.csv')
        
        # Analyze possession lengths
        possession_lengths = df.groupby('case_id').size()
        print(f"Possession lengths (events per possession):")
        print(f"  Mean: {possession_lengths.mean():.1f}")
        print(f"  Median: {possession_lengths.median():.1f}")
        print(f"  Max: {possession_lengths.max()}")
        print(f"  Min: {possession_lengths.min()}")
        
        # Analyze action sequences
        print(f"\nMost common action sequences:")
        sequences = df.groupby('case_id')['action_type'].apply(lambda x: ' -> '.join(x)).value_counts()
        for seq, count in sequences.head(5).items():
            print(f"  {seq}: {count} times")
        
        # Team analysis
        if 'team' in df.columns:
            team_actions = df.groupby(['team', 'action_type']).size().unstack(fill_value=0)
            print(f"\nActions by team:")
            print(team_actions)

def main():
    """Main demo function."""
    print("Football Simulation Process Mining Demo")
    print("="*50)
    
    # Load match data
    log, source_format = load_match_data()
    if log is None:
        print("No match data available. Please run the simulation first.")
        return
    
    # Analyze events
    analyze_events(log, source_format)
    
    # Analyze possession patterns
    analyze_possession_patterns(log, source_format)
    
    # Discover process model
    net, initial_marking, final_marking = discover_process_model(log)
    
    if net is not None:
        print("\n" + "="*50)
        print("PROCESS MINING COMPLETE")
        print("="*50)
        print("✓ Event log loaded and analyzed")
        print("✓ Possession patterns identified")
        print("✓ Process model discovered")
        print("✓ Petri net visualization saved")
        print("\nCheck outputs/ directory for results!")
    else:
        print("\n⚠ Process model discovery failed, but analysis completed")

if __name__ == "__main__":
    main()
