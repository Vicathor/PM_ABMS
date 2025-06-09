"""
Enhanced football simulation agents with dynamic roles and context-dependent probabilities.

This module contains the improved agent classes with type hints, dynamic role switching,
and context-dependent success probabilities for more realistic behavior.
"""

import numpy as np
import random
import json
from mesa import Agent
from typing import TYPE_CHECKING, Optional, Tuple, List, Dict

if TYPE_CHECKING:
    from .env import FootballModel
from typing import Dict, Any, List, Optional, Tuple, Protocol, Union
from enum import Enum
from dataclasses import dataclass
from .actions import ActionType, ActionPlanner, ActionValidator


class PossessionState(Enum):
    """Player states based on team possession."""
    IN_POSSESSION = "in_possession"
    OUT_OF_POSSESSION = "out_of_possession"
    TRANSITIONING = "transitioning"


class PlayerRole(Enum):
    """Base player roles that can change dynamically."""
    GOALKEEPER = "goalkeeper"
    DEFENDER = "defender"
    MIDFIELDER = "midfielder"
    FORWARD = "forward"


@dataclass
class PlayerStats:
    """Player performance and physical statistics."""
    stamina: float
    speed: float
    passing_accuracy: float
    defensive_pressure: float
    distance_to_ball: float
    nearest_opponent_distance: float


class FootballAgent(Protocol):
    """Protocol for football simulation agents."""
    unique_id: int
    x: float
    y: float
    
    def step(self) -> None:
        """Execute one simulation step."""
        ...


class PositionInfo:
    """Helper class for position-based calculations."""
    
    @staticmethod
    def calculate_pressure(player_pos: Tuple[float, float], 
                          opponents: List[Tuple[float, float]]) -> float:
        """Calculate defensive pressure on a player."""
        x, y = player_pos
        if not opponents:
            return 0.0
        
        distances = [np.sqrt((x - ox)**2 + (y - oy)**2) for ox, oy in opponents]
        min_distance = min(distances)
        
        # Pressure decreases exponentially with distance
        # Max pressure at 0m, minimal at 10m+
        return max(0.0, 1.0 - (min_distance / 10.0))


class PlayerAgent(Agent):
    """
    Enhanced football player agent with dynamic roles and context-dependent behavior.
    
    Players can switch between attacking and defensive mindsets based on possession,
    with success probabilities that depend on fatigue and opponent pressure.
    """
    
    def __init__(self, model: 'FootballModel', team_id: int, role: str, 
                 x: float, y: float) -> None:
        """Initialize enhanced player agent."""
        super().__init__(model)
        self.unique_id = model.next_id()
        self.team_id: int = team_id
        self.base_role: PlayerRole = PlayerRole(role)
        self.current_role: PlayerRole = self.base_role
        self.possession_state: PossessionState = PossessionState.OUT_OF_POSSESSION
        
        # Position and movement
        self.x: float = x
        self.y: float = y
        self.vx: float = 0.0
        self.vy: float = 0.0
        self.max_speed: float = 7.0
        
        # Physical attributes
        self.stamina: float = 100.0
        self.max_stamina: float = 100.0
        self.fatigue_rate: float = 0.1
        self.recovery_rate: float = 0.05
        
        # Performance attributes
        self.passing_accuracy: float = 0.8
        self.dribbling_skill: float = 0.7
        self.defensive_ability: float = 0.6
        
        # Dynamic action weights that change with role
        self.base_action_weights: Dict[str, float] = self._get_role_weights()
        self.action_weights: Dict[str, float] = self.base_action_weights.copy()
        
        # Tactical awareness
        self.pressure_tolerance: float = 0.7
        self.decision_time: float = 0.0
        
    def step(self) -> None:
        """Execute enhanced simulation step with dynamic role adaptation."""
        # Update possession state and role
        self._update_possession_state()
        self._adapt_role()
        
        # Update stats for context-dependent decisions
        self._update_player_stats()
        
        if self.model.ball.possessing_player == self.unique_id:
            # Player has possession
            self.possession_state = PossessionState.IN_POSSESSION
            action = self.choose_action()
            self._execute_action(action)
        else:
            # Move according to current role and possession state
            self._move_intelligent()
            
        # Update physical state
        self._update_stamina()
        self._update_position()
    
    def choose_action(self) -> Dict[str, Any]:
        """
        Enhanced action selection with context-dependent evaluation.
        
        Returns
        -------
        Dict[str, Any]
            Action dictionary with type and parameters
        """
        # Adjust epsilon based on pressure and fatigue
        base_epsilon = 0.15
        pressure = self._calculate_current_pressure()
        fatigue_factor = 1.0 - (self.stamina / self.max_stamina)
        
        epsilon = base_epsilon + 0.1 * pressure + 0.05 * fatigue_factor
        epsilon = min(0.4, epsilon)  # Cap at 40% randomness
        
        if np.random.random() < epsilon:
            return self._random_action()
        
        # Evaluate actions with enhanced context
        legal_actions = self._get_legal_actions()
        best_action = None
        best_value = float('-inf')
        
        for action in legal_actions:
            expected_value = self._evaluate_action_enhanced(action)
            if expected_value > best_value:
                best_value = expected_value
                best_action = action
                
        return best_action if best_action else self._random_action()
    
    def _update_possession_state(self) -> None:
        """Update possession state based on team ball control."""
        ball_possessor = self.model.ball.possessing_player
        
        if ball_possessor is None:
            self.possession_state = PossessionState.TRANSITIONING
        else:
            possessor_player = self._find_player_by_id(ball_possessor)
            if possessor_player and possessor_player.team_id == self.team_id:
                if ball_possessor == self.unique_id:
                    self.possession_state = PossessionState.IN_POSSESSION
                else:
                    self.possession_state = PossessionState.IN_POSSESSION  # Team has ball
            else:
                self.possession_state = PossessionState.OUT_OF_POSSESSION
    
    def _adapt_role(self) -> None:
        """Dynamically adapt role based on game situation."""
        if self.possession_state == PossessionState.OUT_OF_POSSESSION:
            # Defensive mindset - all outfield players become defenders/midfielders
            if self.base_role == PlayerRole.FORWARD:
                self.current_role = PlayerRole.MIDFIELDER  # First defender
            elif self.base_role == PlayerRole.MIDFIELDER:
                self.current_role = PlayerRole.DEFENDER  # Drop back
            else:
                self.current_role = self.base_role
        
        elif self.possession_state == PossessionState.IN_POSSESSION:
            # Attacking mindset - defenders can become midfielders, midfielders forwards
            if self.base_role == PlayerRole.DEFENDER and self.stamina > 70:
                self.current_role = PlayerRole.MIDFIELDER  # Join attack
            elif self.base_role == PlayerRole.MIDFIELDER and self.stamina > 60:
                self.current_role = PlayerRole.FORWARD  # Advanced position
            else:
                self.current_role = self.base_role
        
        else:  # TRANSITIONING
            self.current_role = self.base_role
        
        # Update action weights based on current role
        self._update_action_weights()
    
    def _get_role_weights(self) -> Dict[str, float]:
        """Get base action weights for player's role."""
        weights_by_role = {
            PlayerRole.GOALKEEPER: {
                'PASS': 1.2, 'DRIBBLE': 0.2, 'SHOOT': 0.1, 
                'CLEAR': 2.0, 'MOVE_OFFBALL': 0.5
            },
            PlayerRole.DEFENDER: {
                'PASS': 1.0, 'DRIBBLE': 0.4, 'SHOOT': 0.3, 
                'CLEAR': 1.5, 'MOVE_OFFBALL': 0.7
            },
            PlayerRole.MIDFIELDER: {
                'PASS': 1.3, 'DRIBBLE': 1.0, 'SHOOT': 0.8, 
                'CLEAR': 0.6, 'MOVE_OFFBALL': 1.0
            },
            PlayerRole.FORWARD: {
                'PASS': 0.9, 'DRIBBLE': 1.2, 'SHOOT': 1.8, 
                'CLEAR': 0.3, 'MOVE_OFFBALL': 0.8
            }
        }
        return weights_by_role[self.base_role]
    
    def _update_action_weights(self) -> None:
        """Update action weights based on current role and possession state."""
        role_weights = {
            PlayerRole.GOALKEEPER: {
                'PASS': 1.2, 'DRIBBLE': 0.2, 'SHOOT': 0.1, 
                'CLEAR': 2.0, 'MOVE_OFFBALL': 0.5
            },
            PlayerRole.DEFENDER: {
                'PASS': 1.0, 'DRIBBLE': 0.4, 'SHOOT': 0.3, 
                'CLEAR': 1.5, 'MOVE_OFFBALL': 0.7
            },
            PlayerRole.MIDFIELDER: {
                'PASS': 1.3, 'DRIBBLE': 1.0, 'SHOOT': 0.8, 
                'CLEAR': 0.6, 'MOVE_OFFBALL': 1.0
            },
            PlayerRole.FORWARD: {
                'PASS': 0.9, 'DRIBBLE': 1.2, 'SHOOT': 1.8, 
                'CLEAR': 0.3, 'MOVE_OFFBALL': 0.8
            }
        }
        
        self.action_weights = role_weights[self.current_role].copy()
        
        # Modify based on possession state
        if self.possession_state == PossessionState.OUT_OF_POSSESSION:
            # More defensive
            self.action_weights['CLEAR'] *= 1.5
            self.action_weights['SHOOT'] *= 0.5
        elif self.possession_state == PossessionState.IN_POSSESSION:
            # More attacking
            self.action_weights['SHOOT'] *= 1.3
            self.action_weights['PASS'] *= 1.2
    
    def _calculate_current_pressure(self) -> float:
        """Calculate current defensive pressure on this player."""
        opponents = [(p.x, p.y) for p in self.model.players 
                    if p.team_id != self.team_id]
        return PositionInfo.calculate_pressure((self.x, self.y), opponents)
    
    def _calculate_success_probability(self, action_type: str) -> float:
        """
        Calculate context-dependent success probability for an action.
        
        Considers fatigue, pressure, and skill level.
        """
        base_probs = {
            'PASS': self.passing_accuracy,
            'DRIBBLE': self.dribbling_skill,
            'SHOOT': 0.3,
            'CLEAR': 0.9
        }
        
        base_prob = base_probs.get(action_type, 0.5)
        
        # Fatigue penalty (linear decrease)
        fatigue_factor = self.stamina / self.max_stamina
        fatigue_penalty = 1.0 - (1.0 - fatigue_factor) * 0.3  # Up to 30% penalty
        
        # Pressure penalty
        pressure = self._calculate_current_pressure()
        pressure_penalty = 1.0 - pressure * 0.2  # Up to 20% penalty
        
        # Distance penalty for passes (longer = harder)
        distance_penalty = 1.0
        if action_type == 'PASS':
            # This would be calculated per target, simplified here
            distance_penalty = 0.9  # 10% penalty for now
        
        final_prob = base_prob * fatigue_penalty * pressure_penalty * distance_penalty
        return max(0.05, min(0.95, final_prob))  # Clamp between 5% and 95%
    
    def _move_intelligent(self) -> None:
        """Enhanced off-ball movement based on role and game state."""
        target_x, target_y = self._get_tactical_position()
        
        # Move towards target with some variation
        dx = target_x - self.x
        dy = target_y - self.y
        distance = np.sqrt(dx**2 + dy**2)
        
        if distance > 2.0:  # Don't micro-adjust
            speed_factor = min(1.0, self.stamina / 50.0)  # Slower when tired
            self.vx = (dx / distance) * self.max_speed * speed_factor
            self.vy = (dy / distance) * self.max_speed * speed_factor
    
    def _get_tactical_position(self) -> Tuple[float, float]:
        """Get tactical position based on current role and possession state."""
        # Basic tactical positioning
        if self.current_role == PlayerRole.GOALKEEPER:
            return (self.model.width / 2, 5.0 if self.team_id == 0 else self.model.height - 5.0)
        
        elif self.current_role == PlayerRole.DEFENDER:
            base_y = 15.0 if self.team_id == 0 else self.model.height - 15.0
            spread_x = self.model.width / 4 + (self.unique_id % 3) * (self.model.width / 4)
            
        elif self.current_role == PlayerRole.MIDFIELDER:
            base_y = self.model.height / 2
            spread_x = self.model.width / 4 + (self.unique_id % 3) * (self.model.width / 4)
            
        else:  # FORWARD
            base_y = self.model.height - 15.0 if self.team_id == 0 else 15.0
            spread_x = self.model.width / 4 + (self.unique_id % 3) * (self.model.width / 4)
        
        # Add some randomness for realistic movement
        variation_x = np.random.normal(0, 3.0)
        variation_y = np.random.normal(0, 2.0)
        
        target_x = np.clip(spread_x + variation_x, 5, self.model.width - 5)
        target_y = np.clip(base_y + variation_y, 5, self.model.height - 5)
        
        return (target_x, target_y)
    
    def _update_player_stats(self) -> None:
        """Update player statistics for decision making."""
        # Calculate distance to ball
        ball_distance = np.sqrt((self.x - self.model.ball.x)**2 + 
                               (self.y - self.model.ball.y)**2)
        
        # Calculate nearest opponent distance
        opponent_distances = [
            np.sqrt((self.x - p.x)**2 + (self.y - p.y)**2)
            for p in self.model.players if p.team_id != self.team_id
        ]
        nearest_opponent = min(opponent_distances) if opponent_distances else 100.0
        
        # Update current pressure
        pressure = self._calculate_current_pressure()
        
        # Store for use in decision making
        self.current_stats = PlayerStats(
            stamina=self.stamina,
            speed=self.max_speed,
            passing_accuracy=self.passing_accuracy,
            defensive_pressure=pressure,
            distance_to_ball=ball_distance,
            nearest_opponent_distance=nearest_opponent
        )
    
    def _update_stamina(self) -> None:
        """Enhanced stamina system with role-based recovery."""
        # Stamina cost based on activity
        speed = np.sqrt(self.vx**2 + self.vy**2)
        base_cost = self.fatigue_rate
        movement_cost = 0.05 * speed
        
        # Additional cost for ball possession
        possession_cost = 0.02 if self.model.ball.possessing_player == self.unique_id else 0.0
        
        total_cost = base_cost + movement_cost + possession_cost
        self.stamina = max(0, self.stamina - total_cost)
        
        # Recovery when not sprinting
        if speed < 3.0:
            self.stamina = min(self.max_stamina, self.stamina + self.recovery_rate)
        
        # Adjust max speed based on stamina
        stamina_factor = max(0.5, self.stamina / self.max_stamina)
        self.max_speed = 7.0 * stamina_factor
    
    def _evaluate_action_enhanced(self, action: Dict[str, Any]) -> float:
        """Enhanced action evaluation with context awareness."""
        action_type = action['type']
        base_weight = self.action_weights.get(action_type, 1.0)
        
        # Get success probability for this action
        success_prob = self._calculate_success_probability(action_type)
        
        # Calculate expected xThreat
        if action_type == 'PASS':
            target = action['target']
            xT_delta = self._calculate_xT_delta(target.x, target.y)
            expected_value = success_prob * xT_delta + (1 - success_prob) * (-0.1)
            
        elif action_type == 'DRIBBLE':
            direction = action['direction']
            distance = action['distance']
            target_x = self.x + distance * np.cos(direction)
            target_y = self.y + distance * np.sin(direction)
            xT_delta = self._calculate_xT_delta(target_x, target_y)
            expected_value = success_prob * xT_delta + (1 - success_prob) * (-0.15)
            
        elif action_type == 'SHOOT':
            expected_value = success_prob * 1.0 + (1 - success_prob) * (-0.2)
            
        else:  # CLEAR
            expected_value = success_prob * (-0.05) + (1 - success_prob) * (-0.1)
        
        return base_weight * expected_value
    
    # Include the rest of the methods from the original PlayerAgent
    def _get_legal_actions(self) -> List[Dict[str, Any]]:
        """Get list of legal actions in current state."""
        actions = []
        
        # PASS actions to teammates
        teammates = [p for p in self.model.players if p.team_id == self.team_id and p.unique_id != self.unique_id]
        for teammate in teammates:
            if self._can_pass_to(teammate):
                actions.append({'type': 'PASS', 'target': teammate})
        
        # DRIBBLE actions
        for angle in np.linspace(0, 2*np.pi, 8):
            for dist in [2.0, 4.0, 6.0]:
                actions.append({'type': 'DRIBBLE', 'direction': angle, 'distance': dist})
        
        # SHOOT (only in opponent half)
        if self._in_opponent_half():
            actions.append({'type': 'SHOOT'})
        
        # CLEAR
        actions.append({'type': 'CLEAR'})
        
        return actions
    
    def _random_action(self) -> Dict[str, Any]:
        """Choose random legal action."""
        legal_actions = self._get_legal_actions()
        return random.choice(legal_actions) if legal_actions else {'type': 'CLEAR'}
    
    def _calculate_xT_delta(self, target_x: float, target_y: float) -> float:
        """Calculate expected threat delta for position change."""
        current_xT = self.model.get_xThreat(self.x, self.y)
        target_xT = self.model.get_xThreat(target_x, target_y)
        return target_xT - current_xT
    
    def _can_pass_to(self, teammate: 'PlayerAgent') -> bool:
        """Check if pass to teammate is feasible."""
        distance = np.sqrt((teammate.x - self.x)**2 + (teammate.y - self.y)**2)
        return distance < 30.0 and distance > 1.0
    
    def _in_opponent_half(self) -> bool:
        """Check if player is in opponent half."""
        if self.team_id == 0:
            return self.y > self.model.height / 2
        else:
            return self.y < self.model.height / 2
    
    def _execute_action(self, action: Dict[str, Any]) -> None:
        """Execute the chosen action with enhanced success probabilities."""
        action_type = action['type']
        success_prob = self._calculate_success_probability(action_type)
        
        if action_type == 'PASS':
            self._execute_pass(action['target'], success_prob)
        elif action_type == 'DRIBBLE':
            self._execute_dribble(action['direction'], action['distance'], success_prob)
        elif action_type == 'SHOOT':
            self._execute_shoot(success_prob)
        elif action_type == 'CLEAR':
            self._execute_clear(success_prob)
    
    def _execute_pass(self, target: 'PlayerAgent', success_prob: float) -> None:
        """Execute pass with context-dependent success probability."""
        if np.random.random() < success_prob:
            # Successful pass
            pass_speed = min(20.0, np.random.normal(15.0, 2.0))
            dx = target.x - self.x
            dy = target.y - self.y
            distance = np.sqrt(dx**2 + dy**2)
            
            if distance > 0:
                noise_x = np.random.normal(0, 0.3)
                noise_y = np.random.normal(0, 0.3)
                target_x = target.x + noise_x
                target_y = target.y + noise_y
                
                self.model.ball.kick(target_x, target_y, pass_speed)
                
                xT_delta = self._calculate_xT_delta(target_x, target_y)
                self.model.logger.log_event(
                    player=self.unique_id,
                    team=self.team_id,
                    action_type='PASS',
                    dest_x=target_x,
                    dest_y=target_y,
                    xThreat_delta=xT_delta
                )
        else:
            # Failed pass
            self.model.ball.possessing_player = None
            self.model.logger.log_event(
                player=self.unique_id,
                team=self.team_id,
                action_type='PASS_FAILED',
                dest_x=self.x,
                dest_y=self.y,
                xThreat_delta=-0.1
            )
    
    def _execute_dribble(self, direction: float, distance: float, success_prob: float) -> None:
        """Execute dribble with context-dependent success."""
        distance = min(6.0, distance)
        
        if np.random.random() < success_prob:
            # Successful dribble
            target_x = self.x + distance * np.cos(direction)
            target_y = self.y + distance * np.sin(direction)
            
            target_x = np.clip(target_x, 0, self.model.width)
            target_y = np.clip(target_y, 0, self.model.height)
            
            self.x = target_x
            self.y = target_y
            self.model.ball.x = target_x
            self.model.ball.y = target_y
            
            xT_delta = self._calculate_xT_delta(target_x, target_y)
            self.model.logger.log_event(
                player=self.unique_id,
                team=self.team_id,
                action_type='DRIBBLE',
                dest_x=target_x,
                dest_y=target_y,
                xThreat_delta=xT_delta
            )
        else:
            # Failed dribble
            self.model.ball.possessing_player = None
            self.model.logger.log_event(
                player=self.unique_id,
                team=self.team_id,
                action_type='DRIBBLE_FAILED',
                dest_x=self.x,
                dest_y=self.y,
                xThreat_delta=-0.15
            )
    
    def _execute_shoot(self, success_prob: float) -> None:
        """Execute shot with context-dependent probability."""
        roll = np.random.random()
        
        # More realistic shooting distribution:
        # 60% shots on target, 40% missed
        # Of shots on target: ~15% become goals, 85% saved/blocked
        
        if roll < 0.6:  # 60% chance shot is on target
            if np.random.random() < 0.15:  # 15% of on-target shots become goals
                self.model.logger.log_event(
                    player=self.unique_id,
                    team=self.team_id,
                    action_type='GOAL',
                    dest_x=self.x,
                    dest_y=self.y,
                    xThreat_delta=1.0
                )
                self.model.end_possession()
            else:  # 85% of on-target shots are saved/blocked
                self.model.logger.log_event(
                    player=self.unique_id,
                    team=self.team_id,
                    action_type='SHOT',
                    dest_x=self.x,
                    dest_y=self.y,
                    xThreat_delta=0.1
                )
                self.model.ball.possessing_player = None
        else:  # 40% chance shot misses the target entirely
            self.model.logger.log_event(
                player=self.unique_id,
                team=self.team_id,
                action_type='SHOT_MISSED',
                dest_x=self.x,
                dest_y=self.y,
                xThreat_delta=-0.2
            )
            self.model.ball.possessing_player = None
    
    def _execute_clear(self, success_prob: float) -> None:
        """Execute clearance."""
        if self.x < self.model.width / 2:
            target_x = max(0, self.x - 25)
        else:
            target_x = min(self.model.width, self.x + 25)
        
        target_y = self.y
        self.model.ball.kick(target_x, target_y, 15.0)
        
        self.model.logger.log_event(
            player=self.unique_id,
            team=self.team_id,
            action_type='CLEAR',
            dest_x=target_x,
            dest_y=target_y,
            xThreat_delta=-0.05
        )
    
    def _update_position(self) -> None:
        """Update position based on velocity."""
        dt = self.model.dt
        self.x += self.vx * dt
        self.y += self.vy * dt
        
        self.x = np.clip(self.x, 0, self.model.width)
        self.y = np.clip(self.y, 0, self.model.height)
        
        # Apply friction
        self.vx *= 0.9
        self.vy *= 0.9
    
    def _find_player_by_id(self, player_id: int) -> Optional['PlayerAgent']:
        """Find a player by their unique_id."""
        for player in self.model.players:
            if player.unique_id == player_id:
                return player
        return None


class BallAgent(Agent):
    """Enhanced ball agent with improved physics and possession mechanics."""
    
    def __init__(self, model: 'FootballModel', x: float, y: float) -> None:
        """Initialize the ball agent."""
        super().__init__(model)
        self.unique_id = model.next_id()
        self.x: float = x
        self.y: float = y
        self.vx: float = 0.0
        self.vy: float = 0.0
        self.possessing_player: Optional[int] = None
        self.last_touch_player: Optional[int] = None
        self.possession_time: float = 0.0
    
    def step(self) -> None:
        """Execute one simulation step for the ball."""
        if self.possessing_player is None:
            self._check_possession()
            if abs(self.vx) > 0.1 or abs(self.vy) > 0.1:
                self._update_position()
        else:
            # Ball follows possessing player
            possessor = self._find_player_by_id(self.possessing_player)
            if possessor:
                self.x = possessor.x
                self.y = possessor.y
                self.vx = 0.0
                self.vy = 0.0
                self.possession_time += self.model.dt
    
    def kick(self, target_x: float, target_y: float, speed: float = 15.0) -> None:
        """Kick ball towards target position."""
        dx = target_x - self.x
        dy = target_y - self.y
        distance = np.sqrt(dx**2 + dy**2)
        
        if distance > 0:
            self.vx = (dx / distance) * speed
            self.vy = (dy / distance) * speed
        
        self.last_touch_player = self.possessing_player
        self.possessing_player = None
        self.possession_time = 0.0
    
    def _check_possession(self) -> None:
        """Enhanced possession checking with team priorities."""
        candidates = []
        
        for player in self.model.players:
            distance = np.sqrt((player.x - self.x)**2 + (player.y - self.y)**2)
            if distance < 3.0:  # Increased possession radius
                candidates.append((player, distance))
        
        if candidates:
            # Sort by distance, with slight preference for defending team after ball is kicked
            candidates.sort(key=lambda x: x[1])
            
            # Check if there's a clear winner or if we need to consider team priority
            closest_player, closest_distance = candidates[0]
            
            # If ball was recently kicked and multiple players nearby, 
            # slightly favor the team that didn't kick it
            if (len(candidates) > 1 and 
                self.last_touch_player is not None and 
                closest_distance < 1.5):
                
                last_touch_player = self._find_player_by_id(self.last_touch_player)
                if last_touch_player:
                    last_touch_team = last_touch_player.team_id
                    for player, distance in candidates:
                        if (player.team_id != last_touch_team and 
                            distance < closest_distance + 0.5):
                            closest_player = player
                            break
            
            self.possessing_player = closest_player.unique_id
            self.possession_time = 0.0
            
            # Log possession change
            self.model.logger.log_event(
                player=closest_player.unique_id,
                team=closest_player.team_id,
                action_type='POSSESSION',
                dest_x=self.x,
                dest_y=self.y,
                xThreat_delta=0.0
            )
            
            # Update model's possession tracking
            self.model.last_possession_change = self.model.steps
    
    def _update_position(self) -> None:
        """Update ball position with realistic physics."""
        dt = self.model.dt
        
        # Update position
        self.x += self.vx * dt
        self.y += self.vy * dt
        
        # Bounce off boundaries with energy loss
        if self.x <= 0 or self.x >= self.model.width:
            self.vx *= -0.6
            self.x = np.clip(self.x, 0, self.model.width)
        
        if self.y <= 0 or self.y >= self.model.height:
            self.vy *= -0.6
            self.y = np.clip(self.y, 0, self.model.height)
        
        # Apply friction - realistic grass friction
        friction = 0.92
        self.vx *= friction
        self.vy *= friction
        
        # Stop if moving very slowly
        if np.sqrt(self.vx**2 + self.vy**2) < 0.3:
            self.vx = 0.0
            self.vy = 0.0
    
    def _find_player_by_id(self, player_id: int) -> Optional['PlayerAgent']:
        """Find a player by their unique_id."""
        for player in self.model.players:
            if player.unique_id == player_id:
                return player
        return None


class RefAgent(Agent):
    """Enhanced referee agent with better match management."""
    
    def __init__(self, model: 'FootballModel') -> None:
        """Initialize referee agent."""
        super().__init__(model)
        self.unique_id = model.next_id()
        self.match_time: float = 0.0
        self.half: int = 1
        self.goals_team_0: int = 0
        self.goals_team_1: int = 0
        self.last_event_time: float = 0.0
    
    def step(self) -> None:
        """Update match time and check conditions."""
        self.match_time += self.model.dt
        
        # Check for match end conditions
        # End after longer time OR if no significant events for extended period
        max_match_time = 900.0  # 15 minutes for longer games
        max_inactive_time = 300.0  # 5 minutes without events to maximize event generation
        
        time_since_last_possession = (self.model.steps - 
                                    self.model.last_possession_change) * self.model.dt
        
        if (self.match_time >= max_match_time or 
            time_since_last_possession >= max_inactive_time):
            self.model.running = False
