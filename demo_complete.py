#!/usr/bin/env python3
"""
Football Simulation Process Mining Complete Demo
==============================================

This script demonstrates the complete workflow of football simulation
and process mining analysis using PM4Py.
"""

import pandas as pd
import pm4py
import numpy as np
from pathlib import Path

def analyze_match_data():
    """Complete analysis of football simulation match data."""
    
    print("🏈 FOOTBALL SIMULATION PROCESS MINING ANALYSIS")
    print("=" * 60)
    
    # Load match data
    csv_path = Path('outputs/match_log.csv')
    if not csv_path.exists():
        print("❌ No match log found. Please run the simulation first:")
        print("   python3 run_sim.py")
        return
    
    # Load and analyze basic statistics
    df = pd.read_csv(csv_path)
    print(f"📊 MATCH STATISTICS")
    print(f"   Total events: {len(df)}")
    print(f"   Possessions: {df['case_id'].nunique()}")
    print(f"   Duration: {df['tick'].max()} ticks ({df['tick'].max() * 0.066:.1f} seconds)")
    
    # Event type analysis
    print(f"\n🎯 EVENT DISTRIBUTION")
    action_counts = df['action_type'].value_counts()
    for action, count in action_counts.items():
        percentage = (count / len(df)) * 100
        print(f"   {action:15s} {count:3d} ({percentage:4.1f}%)")
    
    # xThreat analysis
    print(f"\n⚡ EXPECTED THREAT (xT) ANALYSIS")
    xt_stats = df['xThreat_delta'].describe()
    print(f"   Mean xT delta:     {xt_stats['mean']:6.4f}")
    print(f"   Max xT gain:       {xt_stats['max']:6.4f}")
    print(f"   Min xT loss:       {xt_stats['min']:6.4f}")
    
    positive_xt = df[df['xThreat_delta'] > 0]
    print(f"   Positive xT events: {len(positive_xt):2d} ({len(positive_xt)/len(df)*100:4.1f}%)")
    
    # Team analysis
    print(f"\n👥 TEAM PERFORMANCE")
    team_stats = df.groupby('team').agg({
        'action_type': 'count',
        'xThreat_delta': ['mean', 'sum']
    }).round(4)
    team_stats.columns = ['Total_Actions', 'Avg_xThreat', 'Total_xThreat']
    print(team_stats)
    
    # Possession flow analysis
    print(f"\n🔄 POSSESSION FLOW")
    possession_lengths = df.groupby('case_id').size()
    print(f"   Average events per possession: {possession_lengths.mean():4.1f}")
    print(f"   Longest possession: {possession_lengths.max()} events")
    print(f"   Shortest possession: {possession_lengths.min()} events")
    
    # Action sequences
    print(f"\n📋 MOST COMMON ACTION SEQUENCES")
    sequences = df.groupby('case_id')['action_type'].apply(
        lambda x: ' → '.join(x)
    ).value_counts()
    
    for i, (seq, count) in enumerate(sequences.head(3).items()):
        print(f"   {i+1}. {seq}")
        print(f"      Occurred {count} time(s)")
    
    return df

def discover_process_model(df):
    """Discover and visualize process model using PM4Py."""
    
    print(f"\n🔍 PROCESS MODEL DISCOVERY")
    print("=" * 60)
    
    # Convert to PM4Py format
    df_pm4py = df.rename(columns={
        'case_id': 'case:concept:name',
        'timestamp': 'time:timestamp', 
        'action_type': 'concept:name'
    })
    df_pm4py['time:timestamp'] = pd.to_datetime(df_pm4py['time:timestamp'])
    
    # Create event log
    log = pm4py.convert_to_event_log(df_pm4py)
    print(f"✅ Converted to PM4Py event log with {len(log)} trace(s)")
    
    # Apply Inductive Miner
    net, initial_marking, final_marking = pm4py.discover_petri_net_inductive(log)
    print(f"✅ Discovered Petri net:")
    print(f"   Places: {len(net.places)}")
    print(f"   Transitions: {len(net.transitions)}")
    
    # Save process model visualization
    output_path = Path('outputs/process_model.png')
    pm4py.save_vis_petri_net(net, initial_marking, final_marking, str(output_path))
    print(f"✅ Process model saved to: {output_path}")
    
    # Generate process statistics
    print(f"\n📈 PROCESS MODEL STATISTICS")
    transitions = [t.label for t in net.transitions if t.label]
    print(f"   Unique activities: {len(set(transitions))}")
    print(f"   Process complexity: {len(net.places) + len(net.transitions)} nodes")
    
    return log, net, initial_marking, final_marking

def generate_insights(df):
    """Generate actionable insights from the analysis."""
    
    print(f"\n💡 KEY INSIGHTS")
    print("=" * 60)
    
    # Goal scoring analysis
    goals = df[df['action_type'] == 'GOAL']
    shots = df[df['action_type'].str.contains('SHOT')]
    
    if len(goals) > 0:
        print(f"🥅 GOAL SCORING:")
        print(f"   Goals scored: {len(goals)}")
        print(f"   Total shots: {len(shots)}")
        if len(shots) > 0:
            conversion_rate = len(goals) / len(shots) * 100
            print(f"   Conversion rate: {conversion_rate:.1f}%")
    
    # Ball progression analysis
    passes = df[df['action_type'] == 'PASS']
    if len(passes) > 0:
        avg_pass_xt = passes['xThreat_delta'].mean()
        print(f"\n⚽ BALL PROGRESSION:")
        print(f"   Successful passes: {len(passes)}")
        print(f"   Average xT gain per pass: {avg_pass_xt:.4f}")
    
    # Defensive actions
    defensive_actions = df[df['action_type'].isin(['CLEAR', 'TACKLE'])]
    if len(defensive_actions) > 0:
        print(f"\n🛡️  DEFENSIVE PLAY:")
        print(f"   Defensive actions: {len(defensive_actions)}")
        avg_defensive_xt = defensive_actions['xThreat_delta'].mean()
        print(f"   Average xT impact: {avg_defensive_xt:.4f}")

def main():
    """Main demo function."""
    
    # Check if outputs directory exists
    Path('outputs').mkdir(exist_ok=True)
    
    try:
        # Analyze match data
        df = analyze_match_data()
        if df is None:
            return
        
        # Discover process model
        log, net, im, fm = discover_process_model(df)
        
        # Generate insights
        generate_insights(df)
        
        print(f"\n🎉 ANALYSIS COMPLETE!")
        print("=" * 60)
        print("📁 Generated files:")
        print("   outputs/match_log.csv     - Raw event data")
        print("   outputs/match_log.xes     - XES format for PM4Py")
        print("   outputs/process_model.png - Process model visualization")
        print()
        print("🔗 Next steps:")
        print("   • Open process_model.png to view the discovered process")
        print("   • Load match_log.xes in ProM or other process mining tools")
        print("   • Run additional PM4Py analyses (conformance, enhancement)")
        print("   • Generate more match data with different configurations")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
