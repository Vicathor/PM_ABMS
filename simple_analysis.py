#!/usr/bin/env python3
"""Simple process mining analysis of football simulation data."""

import pandas as pd
import pm4py

def main():
    print("Football Simulation Process Mining Analysis")
    print("=" * 50)
    
    # Load CSV data
    try:
        df = pd.read_csv('outputs/match_log.csv')
        print(f"✓ Loaded {len(df)} events from CSV")
        print(f"✓ Found {df['case_id'].nunique()} possessions")
        
        # Show event type distribution
        print("\nEvent Distribution:")
        event_counts = df['action_type'].value_counts()
        for action, count in event_counts.items():
            print(f"  {action}: {count}")
        
        # Show possession statistics
        possession_lengths = df.groupby('case_id').size()
        print(f"\nPossession Statistics:")
        print(f"  Average events per possession: {possession_lengths.mean():.1f}")
        print(f"  Max events in possession: {possession_lengths.max()}")
        print(f"  Min events in possession: {possession_lengths.min()}")
        
        # Show xThreat analysis
        if 'xThreat_delta' in df.columns:
            positive_threat = df[df['xThreat_delta'] > 0]
            print(f"\nThreat Analysis:")
            print(f"  Events with positive xThreat: {len(positive_threat)} ({len(positive_threat)/len(df)*100:.1f}%)")
            print(f"  Average xThreat delta: {df['xThreat_delta'].mean():.4f}")
            print(f"  Max xThreat gain: {df['xThreat_delta'].max():.4f}")
        
        # Try simple process discovery
        try:
            # Prepare data for PM4Py
            df_pm4py = df.rename(columns={
                'case_id': 'case:concept:name',
                'timestamp': 'time:timestamp', 
                'action_type': 'concept:name'
            })
            df_pm4py['time:timestamp'] = pd.to_datetime(df_pm4py['time:timestamp'])
            
            # Convert to event log
            log = pm4py.convert_to_event_log(df_pm4py)
            print(f"✓ Converted to PM4Py event log")
            
            # Discover process model
            net, im, fm = pm4py.discover_petri_net_inductive(log)
            print(f"✓ Discovered process model with {len(net.places)} places and {len(net.transitions)} transitions")
            
            # Save visualization
            pm4py.save_vis_petri_net(net, im, fm, 'outputs/process_model.png')
            print("✓ Process model saved to outputs/process_model.png")
            
        except Exception as e:
            print(f"⚠ Process discovery failed: {e}")
        
        print("\n" + "=" * 50)
        print("ANALYSIS COMPLETE!")
        print("=" * 50)
        
    except Exception as e:
        print(f"✗ Error: {e}")

if __name__ == "__main__":
    main()
