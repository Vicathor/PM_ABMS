"""
Mesa environment and model for football simulation.

This module contains the main FootballModel class that orchestrates the simulation,
including pitch setup, agent management, and game flow control.
"""

from mesa import Model, DataCollector
from mesa.space import ContinuousSpace
import json
import os
from typing import List, Dict, Any

from .agents_enhanced import PlayerAgent, BallAgent, RefAgent
from .logger import EventLogger


class FootballModel(Model):
    """
    Main football simulation model using Mesa framework.
    
    This class manages the entire simulation including the pitch, players, ball,
    time progression, and data collection for process mining analysis.
    
    Attributes
    ----------
    pitch_width : float
        Pitch width in meters (34m for half-pitch)
    pitch_length : float  
        Pitch length in meters (52m for half-pitch)
    dt : float
        Time step in seconds (0.066s ≈ 15 FPS)
    players : list
        List of all player agents
    ball : BallAgent
        The ball agent
    referee : RefAgent
        The referee agent
    logger : EventLogger
        Event logger for process mining
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the football simulation model.
        
        Parameters
        ----------
        config : dict
            Configuration dictionary with simulation parameters
        """
        super().__init__()
        
        # Agent ID counter for unique identification
        self._agent_id_counter = 0
        
        # Simple agent list for activation (Mesa compatibility)
        self.game_agents = []
        
        # Pitch dimensions (half FIFA pitch)
        self.width = config.get('pitch_width', 34.0)  # m - also add as width
        self.height = config.get('pitch_length', 52.0)  # m - also add as height
        self.pitch_width = self.width
        self.pitch_length = self.height
        self.dt = config.get('dt', 0.066)  # s (≈15 FPS)
        
        # Game state
        self.running = True
        self.ticks_since_possession_change = 0
        self.max_ticks_without_change = config.get('max_ticks_without_change', 1200)  # 20x longer
        self.max_game_time = config.get('max_game_time', 900)  # 15 minutes for 2000+ events
        self.last_possession_change = 0  # Track for referee
        
        # Realism settings
        self.realism_level = config.get('realism_level', 'toy')  # 'toy' or 'empirical'
        
        # Initialize space 
        self.space = ContinuousSpace(self.pitch_width, self.pitch_length, False)
        
        # Load data files
        self.xT_lookup = self._load_xT_lookup()
        self.pass_distributions = self._load_pass_distributions()
        self.dribble_durations = self._load_dribble_durations()
        
        # Initialize logger
        self.logger = EventLogger()
        
        # Create agents
        self.players = []
        self.ball = None
        self.referee = None
        
        self._create_agents(config)
        
        # Data collector for Mesa's built-in data collection
        self.datacollector = DataCollector(
            model_reporters={
                "Ball_X": lambda m: m.ball.x if m.ball else 0,
                "Ball_Y": lambda m: m.ball.y if m.ball else 0,
                "Possession_Team": lambda m: m._get_possessing_team_id(),
                "Match_Time": lambda m: m.referee.match_time if m.referee else 0
            },
            agent_reporters={
                "X": "x",
                "Y": "y", 
                "Team": "team_id",
                "Stamina": "stamina"
            }
        )
    
    def next_id(self) -> int:
        """Generate next unique agent ID."""
        self._agent_id_counter += 1
        return self._agent_id_counter
    
    def register_agent(self, agent):
        """Register agent with the model."""
        self.game_agents.append(agent)
    
    def step(self):
        """Execute one step of the simulation."""
        if not self.running:
            return
            
        # Increment logger tick
        self.logger.increment_tick()
        
        # Check for possession changes
        previous_possessor = getattr(self, '_previous_possessor', None)
        current_possessor = self.ball.possessing_player if self.ball else None
        
        if previous_possessor != current_possessor:
            self.ticks_since_possession_change = 0
        else:
            self.ticks_since_possession_change += 1
        
        self._previous_possessor = current_possessor
        
        # Step all agents
        for agent in self.game_agents:
            agent.step()
        
        # Collect data
        self.datacollector.collect(self)
        
        # Check end conditions
        if (self.referee and self.referee.match_time >= self.max_game_time) or \
           (self.ticks_since_possession_change >= self.max_ticks_without_change):
            self.running = False
    
    def get_xThreat(self, x: float, y: float) -> float:
        """
        Get expected threat value for a position on the pitch.
        
        Parameters
        ----------
        x, y : float
            Position coordinates
            
        Returns
        -------
        float
            Expected threat value (0-1)
        """
        # Normalize coordinates to grid indices
        grid_x = int(min(11, max(0, (x / self.pitch_width) * 12)))
        grid_y = int(min(7, max(0, (y / self.pitch_length) * 8)))
        
        # Look up xThreat value
        return self.xT_lookup.get(f"{grid_x}_{grid_y}", 0.0)
    
    def end_possession(self):
        """End current possession (e.g., after goal)."""
        if self.ball:
            self.ball.possessing_player = None
            # Reset ball to center
            self.ball.x = self.pitch_width / 2
            self.ball.y = self.pitch_length / 2
        self.ticks_since_possession_change = 0
    
    def get_match_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics for the completed match.
        
        Returns
        -------
        dict
            Match summary with KPIs
        """
        stats = self.logger.get_summary_stats()
        
        # Add model-specific stats
        stats.update({
            'total_ticks': self.logger.tick_count,
            'match_duration': self.referee.match_time if self.referee else 0,
            'pitch_dimensions': f"{self.pitch_width}x{self.pitch_length}m",
            'realism_level': self.realism_level
        })
        
        return stats
    
    def save_logs(self, csv_path: str, xes_path: str):
        """
        Save event logs to files.
        
        Parameters
        ----------
        csv_path : str
            Path for CSV output
        xes_path : str
            Path for XES output
        """
        self.logger.save_csv(csv_path)
        self.logger.save_xes(xes_path)
    
    def _create_agents(self, config: Dict[str, Any]):
        """Create and position all agents."""
        # Create ball at center
        self.ball = BallAgent(self, self.pitch_width/2, self.pitch_length/2)
        self.register_agent(self.ball)
        self.space.place_agent(self.ball, (self.ball.x, self.ball.y))
        
        # Create referee
        self.referee = RefAgent(self)
        self.register_agent(self.referee)
        
        # Create players for both teams
        team_formations = config.get('formations', {
            'team_0': self._get_default_formation(0),
            'team_1': self._get_default_formation(1)
        })
        
        for team_id in [0, 1]:
            formation = team_formations[f'team_{team_id}']
            for position in formation:
                player = PlayerAgent(
                    self, 
                    team_id, 
                    position['role'],
                    position['x'], 
                    position['y']
                )
                self.players.append(player)
                self.register_agent(player)
                self.space.place_agent(player, (player.x, player.y))
        
        # Give initial possession to team 0
        if self.players:
            center_player = min(self.players, 
                              key=lambda p: abs(p.x - self.pitch_width/2) + abs(p.y - self.pitch_length/2))
            self.ball.possessing_player = center_player.unique_id
    
    def _get_default_formation(self, team_id: int) -> List[Dict[str, Any]]:
        """
        Get default 7-a-side formation for a team.
        
        Parameters
        ----------
        team_id : int
            Team identifier (0 or 1)
            
        Returns
        -------
        list
            List of player position dictionaries
        """
        if team_id == 0:  # Team 0 - attacks towards y=52
            return [
                {'role': 'goalkeeper', 'x': 17, 'y': 5},
                {'role': 'defender', 'x': 10, 'y': 15},
                {'role': 'defender', 'x': 24, 'y': 15},
                {'role': 'midfielder', 'x': 17, 'y': 25},
                {'role': 'midfielder', 'x': 8, 'y': 30},
                {'role': 'midfielder', 'x': 26, 'y': 30},
                {'role': 'forward', 'x': 17, 'y': 40}
            ]
        else:  # Team 1 - attacks towards y=0
            return [
                {'role': 'goalkeeper', 'x': 17, 'y': 47},
                {'role': 'defender', 'x': 10, 'y': 37},
                {'role': 'defender', 'x': 24, 'y': 37},
                {'role': 'midfielder', 'x': 17, 'y': 27},
                {'role': 'midfielder', 'x': 8, 'y': 22},
                {'role': 'midfielder', 'x': 26, 'y': 22},
                {'role': 'forward', 'x': 17, 'y': 12}
            ]
    
    def _load_xT_lookup(self) -> Dict[str, float]:
        """Load expected threat lookup table."""
        xT_path = os.path.join('data', 'xT_lookup.json')
        
        if os.path.exists(xT_path) and self.realism_level == 'empirical':
            try:
                with open(xT_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                pass
        
        # Return default/toy xT values
        return self._generate_toy_xT_lookup()
    
    def _generate_toy_xT_lookup(self) -> Dict[str, float]:
        """Generate simplified xThreat lookup for toy mode."""
        xT_lookup = {}
        
        # Create gradient from defensive to attacking end
        for x in range(12):
            for y in range(8):
                # Higher threat closer to opponent goal
                base_threat = y / 8.0  # 0 to 1 from defensive to attacking
                
                # Central positions slightly higher
                central_bonus = 1.0 - abs(x - 6) / 6.0 * 0.2
                
                # Penalty area very high threat
                if y >= 6 and 3 <= x <= 9:
                    base_threat += 0.3
                
                xT_lookup[f"{x}_{y}"] = min(1.0, base_threat * central_bonus)
        
        return xT_lookup
    
    def _load_pass_distributions(self) -> Dict[str, Any]:
        """Load empirical pass length distributions."""
        pass_path = os.path.join('data', 'pass_dist.json')
        
        if os.path.exists(pass_path) and self.realism_level == 'empirical':
            try:
                with open(pass_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                pass
        
        # Return toy distributions
        return {
            'short_pass': {'mean': 8.0, 'std': 2.0, 'weight': 0.6},
            'medium_pass': {'mean': 18.0, 'std': 4.0, 'weight': 0.3},
            'long_pass': {'mean': 35.0, 'std': 8.0, 'weight': 0.1}
        }
    
    def _load_dribble_durations(self) -> Dict[str, Any]:
        """Load empirical dribble duration distributions."""
        dribble_path = os.path.join('data', 'dribble_duration.json')
        
        if os.path.exists(dribble_path) and self.realism_level == 'empirical':
            try:
                with open(dribble_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                pass
        
        # Return toy distributions  
        return {
            'short_dribble': {'distance': 2.0, 'duration': 1.0, 'weight': 0.7},
            'medium_dribble': {'distance': 4.0, 'duration': 2.0, 'weight': 0.2},
            'long_dribble': {'distance': 6.0, 'duration': 3.0, 'weight': 0.1}
        }
    
    def _get_possessing_team_id(self) -> int:
        """Get the team ID of the player currently possessing the ball."""
        if not self.ball or self.ball.possessing_player is None:
            return -1
        
        for player in self.players:
            if player.unique_id == self.ball.possessing_player:
                return player.team_id
        return -1
