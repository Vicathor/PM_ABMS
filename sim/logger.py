"""
Event logging system for process mining analysis.

This module handles the collection and formatting of simulation events
for later analysis with PM4Py and other process mining tools.
"""

import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET
import xml.dom.minidom


class EventLogger:
    """
    Collects and formats simulation events for process mining.
    
    This logger captures all player actions and game events in a format
    suitable for process mining analysis, supporting both CSV and XES output.
    
    Attributes
    ----------
    events : list
        List of event dictionaries
    case_id : str
        Unique identifier for this simulation run
    start_time : datetime
        Simulation start timestamp
    """
    
    def __init__(self, case_id: Optional[str] = None):
        """
        Initialize the event logger.
        
        Parameters
        ----------
        case_id : str, optional
            Unique case identifier. If None, generates timestamp-based ID.
        """
        self.events = []
        self.case_id = case_id or f"match_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.start_time = datetime.now()
        self.tick_count = 0
    
    def log_event(self, player: int, team: int, action_type: str, 
                  dest_x: float, dest_y: float, xThreat_delta: float):
        """
        Log a single event during simulation.
        
        Parameters
        ----------
        player : int
            Player ID who performed the action
        team : int
            Team ID (0 or 1)
        action_type : str
            Type of action performed
        dest_x, dest_y : float
            Destination coordinates
        xThreat_delta : float
            Change in expected threat value
        """
        timestamp = self.start_time + timedelta(seconds=self.tick_count * 0.066)
        
        event = {
            'case_id': self.case_id,
            'timestamp': timestamp.isoformat(),
            'player': player,
            'team': team,
            'action_type': action_type,
            'dest_x': round(dest_x, 2),
            'dest_y': round(dest_y, 2),
            'xThreat_delta': round(xThreat_delta, 4),
            'tick': self.tick_count
        }
        
        self.events.append(event)
    
    def increment_tick(self):
        """Increment the simulation tick counter."""
        self.tick_count += 1
    
    def get_dataframe(self) -> pd.DataFrame:
        """
        Get events as a pandas DataFrame.
        
        Returns
        -------
        pd.DataFrame
            DataFrame containing all logged events
        """
        if not self.events:
            return pd.DataFrame(columns=['case_id', 'timestamp', 'player', 'team', 
                                       'action_type', 'dest_x', 'dest_y', 'xThreat_delta'])
        
        df = pd.DataFrame(self.events)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    
    def save_csv(self, filepath: str):
        """
        Save events to CSV file.
        
        Parameters
        ----------
        filepath : str
            Path to save CSV file
        """
        df = self.get_dataframe()
        df.to_csv(filepath, index=False)
        print(f"Events saved to CSV: {filepath}")
    
    def save_xes(self, filepath: str):
        """
        Save events to XES format for PM4Py.
        
        Parameters
        ----------
        filepath : str
            Path to save XES file
        """
        # Create XES root element
        root = ET.Element('log')
        root.set('xes.version', '1.0')
        root.set('xes.features', 'nested-attributes')
        
        # Add extensions
        extension = ET.SubElement(root, 'extension')
        extension.set('name', 'Lifecycle')
        extension.set('prefix', 'lifecycle')
        extension.set('uri', 'http://www.xes-standard.org/lifecycle.xesext')
        
        extension2 = ET.SubElement(root, 'extension')
        extension2.set('name', 'Organizational')
        extension2.set('prefix', 'org')
        extension2.set('uri', 'http://www.xes-standard.org/org.xesext')
        
        extension3 = ET.SubElement(root, 'extension')
        extension3.set('name', 'Time')
        extension3.set('prefix', 'time')
        extension3.set('uri', 'http://www.xes-standard.org/time.xesext')
        
        extension4 = ET.SubElement(root, 'extension')
        extension4.set('name', 'Concept')
        extension4.set('prefix', 'concept')
        extension4.set('uri', 'http://www.xes-standard.org/concept.xesext')
        
        # Add global attributes
        global_trace = ET.SubElement(root, 'global')
        global_trace.set('scope', 'trace')
        
        string_attr = ET.SubElement(global_trace, 'string')
        string_attr.set('key', 'concept:name')
        string_attr.set('value', 'name')
        
        global_event = ET.SubElement(root, 'global')
        global_event.set('scope', 'event')
        
        string_attr2 = ET.SubElement(global_event, 'string')
        string_attr2.set('key', 'concept:name')
        string_attr2.set('value', 'name')
        
        date_attr = ET.SubElement(global_event, 'date')
        date_attr.set('key', 'time:timestamp')
        date_attr.set('value', '1970-01-01T00:00:00.000+00:00')
        
        # Add classifier
        classifier = ET.SubElement(root, 'classifier')
        classifier.set('name', 'Event Name')
        classifier.set('keys', 'concept:name')
        
        # Group events by case (match)
        df = self.get_dataframe()
        if df.empty:
            # Create empty trace
            trace = ET.SubElement(root, 'trace')
            string_attr = ET.SubElement(trace, 'string')
            string_attr.set('key', 'concept:name')
            string_attr.set('value', self.case_id)
        else:
            for case_id, case_events in df.groupby('case_id'):
                trace = ET.SubElement(root, 'trace')
                
                # Trace attributes
                string_attr = ET.SubElement(trace, 'string')
                string_attr.set('key', 'concept:name')
                string_attr.set('value', str(case_id))
                
                # Sort events by timestamp
                case_events = case_events.sort_values('timestamp')
                
                # Add events
                for _, event in case_events.iterrows():
                    event_elem = ET.SubElement(trace, 'event')
                    
                    # Event name
                    string_attr = ET.SubElement(event_elem, 'string')
                    string_attr.set('key', 'concept:name')
                    string_attr.set('value', str(event['action_type']))
                    
                    # Timestamp
                    date_attr = ET.SubElement(event_elem, 'date')
                    date_attr.set('key', 'time:timestamp')
                    date_attr.set('value', event['timestamp'].strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + '+00:00')
                    
                    # Player
                    int_attr = ET.SubElement(event_elem, 'int')
                    int_attr.set('key', 'player')
                    int_attr.set('value', str(event['player']))
                    
                    # Team
                    int_attr = ET.SubElement(event_elem, 'int')
                    int_attr.set('key', 'team')
                    int_attr.set('value', str(event['team']))
                    
                    # Position
                    float_attr = ET.SubElement(event_elem, 'float')
                    float_attr.set('key', 'dest_x')
                    float_attr.set('value', str(event['dest_x']))
                    
                    float_attr = ET.SubElement(event_elem, 'float')
                    float_attr.set('key', 'dest_y')
                    float_attr.set('value', str(event['dest_y']))
                    
                    # xThreat delta
                    float_attr = ET.SubElement(event_elem, 'float')
                    float_attr.set('key', 'xThreat_delta')
                    float_attr.set('value', str(event['xThreat_delta']))
        
        # Pretty print and save
        rough_string = ET.tostring(root, 'unicode')
        reparsed = xml.dom.minidom.parseString(rough_string)
        pretty_xml = reparsed.toprettyxml(indent='  ')
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(pretty_xml)
        
        print(f"Events saved to XES: {filepath}")
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Calculate summary statistics for the logged events.
        
        Returns
        -------
        dict
            Summary statistics including KPIs
        """
        if not self.events:
            return {
                'total_events': 0,
                'total_possessions': 0,
                'mean_xThreat_per_possession': 0.0,
                'turnovers': 0,
                'goals': 0,
                'shots': 0,
                'passes': 0,
                'dribbles': 0
            }
        
        df = self.get_dataframe()
        
        # Count possessions (POSSESSION events)
        possessions = len(df[df['action_type'] == 'POSSESSION'])
        
        # Calculate mean xThreat per possession
        if possessions > 0:
            total_xThreat = df['xThreat_delta'].sum()
            mean_xThreat_per_possession = total_xThreat / possessions
        else:
            mean_xThreat_per_possession = 0.0
        
        # Count different action types
        action_counts = df['action_type'].value_counts().to_dict()
        
        # Count turnovers (failed actions)
        turnovers = len(df[df['action_type'].str.contains('FAILED', na=False)])
        
        return {
            'total_events': len(df),
            'total_possessions': possessions,
            'mean_xThreat_per_possession': round(mean_xThreat_per_possession, 4),
            'turnovers': turnovers,
            'goals': action_counts.get('GOAL', 0),
            'shots': action_counts.get('SHOT_MISSED', 0) + action_counts.get('GOAL', 0),
            'passes': action_counts.get('PASS', 0),
            'dribbles': action_counts.get('DRIBBLE', 0),
            'clears': action_counts.get('CLEAR', 0),
            'action_distribution': action_counts
        }
    
    def reset(self):
        """Reset the logger for a new simulation run."""
        self.events.clear()
        self.tick_count = 0
        self.start_time = datetime.now()
        self.case_id = f"match_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


class ProcessMiningConverter:
    """
    Utility class for converting event logs to different process mining formats.
    
    This class provides methods to transform the raw event data into formats
    suitable for various process mining tools and analyses.
    """
    
    @staticmethod
    def to_pm4py_format(events: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Convert events to PM4Py standard format.
        
        Parameters
        ----------
        events : list
            List of event dictionaries
            
        Returns
        -------
        pd.DataFrame
            DataFrame in PM4Py format
        """
        if not events:
            return pd.DataFrame()
        
        df = pd.DataFrame(events)
        
        # Rename columns to PM4Py standard
        pm4py_df = df.rename(columns={
            'case_id': 'case:concept:name',
            'action_type': 'concept:name',
            'timestamp': 'time:timestamp'
        })
        
        # Ensure timestamp is datetime
        pm4py_df['time:timestamp'] = pd.to_datetime(pm4py_df['time:timestamp'])
        
        return pm4py_df
    
    @staticmethod
    def calculate_possession_sequences(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Group events into possession sequences for analysis.
        
        Parameters
        ----------
        events : list
            List of event dictionaries
            
        Returns
        -------
        list
            List of possession sequence dictionaries
        """
        sequences = []
        current_sequence = []
        current_team = None
        sequence_id = 0
        
        for event in events:
            if event['action_type'] == 'POSSESSION':
                # Start new sequence
                if current_sequence:
                    sequences.append({
                        'sequence_id': sequence_id,
                        'team': current_team,
                        'events': current_sequence.copy(),
                        'duration': len(current_sequence),
                        'total_xThreat': sum(e['xThreat_delta'] for e in current_sequence),
                        'outcome': ProcessMiningConverter._determine_outcome(current_sequence)
                    })
                    sequence_id += 1
                
                current_sequence = [event]
                current_team = event['team']
            else:
                if current_team == event['team']:
                    current_sequence.append(event)
                else:
                    # Possession changed - end current sequence
                    if current_sequence:
                        sequences.append({
                            'sequence_id': sequence_id,
                            'team': current_team,
                            'events': current_sequence.copy(),
                            'duration': len(current_sequence),
                            'total_xThreat': sum(e['xThreat_delta'] for e in current_sequence),
                            'outcome': ProcessMiningConverter._determine_outcome(current_sequence)
                        })
                        sequence_id += 1
                    
                    current_sequence = [event]
                    current_team = event['team']
        
        # Add final sequence
        if current_sequence:
            sequences.append({
                'sequence_id': sequence_id,
                'team': current_team,
                'events': current_sequence.copy(),
                'duration': len(current_sequence),
                'total_xThreat': sum(e['xThreat_delta'] for e in current_sequence),
                'outcome': ProcessMiningConverter._determine_outcome(current_sequence)
            })
        
        return sequences
    
    @staticmethod
    def _determine_outcome(sequence: List[Dict[str, Any]]) -> str:
        """
        Determine the outcome of a possession sequence.
        
        Parameters
        ----------
        sequence : list
            List of events in the sequence
            
        Returns
        -------
        str
            Outcome classification
        """
        if not sequence:
            return 'UNKNOWN'
        
        last_action = sequence[-1]['action_type']
        
        if last_action == 'GOAL':
            return 'GOAL'
        elif last_action in ['SHOT_MISSED']:
            return 'SHOT'
        elif 'FAILED' in last_action:
            return 'TURNOVER'
        elif last_action == 'CLEAR':
            return 'CLEAR'
        else:
            return 'ONGOING'
