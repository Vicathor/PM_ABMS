"""
Action definitions and mechanics for football simulation.

This module defines the available actions, their parameters, and execution mechanics.
"""

from enum import Enum
import numpy as np


class ActionType(Enum):
    """Enumeration of available player actions."""
    PASS = "PASS"
    DRIBBLE = "DRIBBLE"
    SHOOT = "SHOOT"
    CLEAR = "CLEAR"
    MOVE_OFFBALL = "MOVE_OFFBALL"


class ActionConstraints:
    """
    Action constraints and parameters.
    
    This class defines the physical and logical constraints for each action type,
    including speed limits, success probabilities, and range limitations.
    """
    
    # Pass constraints
    MAX_PASS_SPEED = 20.0  # m/s
    PASS_NOISE_SIGMA = 0.3  # m
    
    # Dribble constraints
    MAX_DRIBBLE_DISTANCE = 6.0  # m
    BASE_DRIBBLE_SUCCESS = 0.8
    PRESSURE_PENALTY = 0.02  # per nearby opponent
    
    # Shooting constraints
    BASE_SHOT_CONVERSION = 0.3  # 30% success rate
    
    # Clearance constraints
    CLEAR_DISTANCE = 25.0  # m
    
    # Movement constraints
    MAX_OFFBALL_SPEED = 7.0  # m/s


class ActionValidator:
    """
    Validates whether actions are legal in the current game state.
    
    This class provides methods to check if specific actions can be performed
    given the current positions of players, ball state, and game rules.
    """
    
    @staticmethod
    def can_pass(player, target_player, model) -> bool:
        """
        Check if a pass to target player is valid.
        
        Parameters
        ----------
        player : PlayerAgent
            The player attempting the pass
        target_player : PlayerAgent
            The intended recipient
        model : FootballModel
            The simulation model
            
        Returns
        -------
        bool
            True if pass is valid
        """
        # Same team check
        if player.team_id != target_player.team_id:
            return False
        
        # Distance check
        distance = np.sqrt((target_player.x - player.x)**2 + 
                          (target_player.y - player.y)**2)
        
        # Must be reasonable passing distance (1-40m)
        return 1.0 <= distance <= 40.0
    
    @staticmethod
    def can_dribble(player, direction: float, distance: float, model) -> bool:
        """
        Check if a dribble in the specified direction is valid.
        
        Parameters
        ----------
        player : PlayerAgent
            The player attempting to dribble
        direction : float
            Direction in radians
        distance : float
            Distance to dribble
        model : FootballModel
            The simulation model
            
        Returns
        -------
        bool
            True if dribble is valid
        """
        # Distance constraint
        if distance > ActionConstraints.MAX_DRIBBLE_DISTANCE:
            return False
        
        # Check if target position would be in bounds
        target_x = player.x + distance * np.cos(direction)
        target_y = player.y + distance * np.sin(direction)
        
        return (0 <= target_x <= model.pitch_width and 
                0 <= target_y <= model.pitch_length)
    
    @staticmethod
    def can_shoot(player, model) -> bool:
        """
        Check if player can shoot (must be in opponent half).
        
        Parameters
        ----------
        player : PlayerAgent
            The player attempting to shoot
        model : FootballModel
            The simulation model
            
        Returns
        -------
        bool
            True if shoot is valid
        """
        if player.team_id == 0:
            return player.y > model.pitch_length / 2
        else:
            return player.y < model.pitch_length / 2


class ActionExecutor:
    """
    Executes validated actions and applies their effects to the game state.
    
    This class handles the physics and game logic for each action type,
    including success/failure determination and state updates.
    """
    
    @staticmethod
    def execute_pass(player, target_player, model):
        """
        Execute a pass action.
        
        Parameters
        ----------
        player : PlayerAgent
            The passing player
        target_player : PlayerAgent
            The receiving player
        model : FootballModel
            The simulation model
        """
        # Determine pass speed with variation
        base_speed = 15.0
        pass_speed = min(ActionConstraints.MAX_PASS_SPEED, 
                        np.random.normal(base_speed, 2.0))
        
        # Add noise to target position
        noise_x = np.random.normal(0, ActionConstraints.PASS_NOISE_SIGMA)
        noise_y = np.random.normal(0, ActionConstraints.PASS_NOISE_SIGMA)
        
        target_x = target_player.x + noise_x
        target_y = target_player.y + noise_y
        
        # Execute the kick
        model.ball.kick(target_x, target_y, pass_speed)
        
        # Calculate xThreat delta
        current_xT = model.get_xThreat(player.x, player.y)
        target_xT = model.get_xThreat(target_x, target_y)
        xT_delta = target_xT - current_xT
        
        # Log the action
        model.logger.log_event(
            player=player.unique_id,
            team=player.team_id,
            action_type='PASS',
            dest_x=target_x,
            dest_y=target_y,
            xThreat_delta=xT_delta
        )
        
        return True
    
    @staticmethod
    def execute_dribble(player, direction: float, distance: float, model) -> bool:
        """
        Execute a dribble action.
        
        Parameters
        ----------
        player : PlayerAgent
            The dribbling player
        direction : float
            Dribble direction in radians
        distance : float
            Dribble distance
        model : FootballModel
            The simulation model
            
        Returns
        -------
        bool
            True if dribble was successful
        """
        # Calculate success probability based on pressure
        nearby_opponents = ActionExecutor._count_nearby_opponents(player, 3.0, model)
        success_prob = (ActionConstraints.BASE_DRIBBLE_SUCCESS - 
                       ActionConstraints.PRESSURE_PENALTY * nearby_opponents)
        success_prob = max(0.1, success_prob)  # Minimum 10% success
        
        # Determine success
        success = np.random.random() < success_prob
        
        if success:
            # Move player and ball
            target_x = player.x + distance * np.cos(direction)
            target_y = player.y + distance * np.sin(direction)
            
            # Ensure within bounds
            target_x = np.clip(target_x, 0, model.pitch_width)
            target_y = np.clip(target_y, 0, model.pitch_length)
            
            player.x = target_x
            player.y = target_y
            model.ball.x = target_x
            model.ball.y = target_y
            
            # Calculate xThreat delta
            current_xT = model.get_xThreat(player.x - distance * np.cos(direction),
                                         player.y - distance * np.sin(direction))
            new_xT = model.get_xThreat(target_x, target_y)
            xT_delta = new_xT - current_xT
            
            model.logger.log_event(
                player=player.unique_id,
                team=player.team_id,
                action_type='DRIBBLE',
                dest_x=target_x,
                dest_y=target_y,
                xThreat_delta=xT_delta
            )
        else:
            # Failed dribble - lose possession
            model.ball.possessing_player = None
            model.logger.log_event(
                player=player.unique_id,
                team=player.team_id,
                action_type='DRIBBLE_FAILED',
                dest_x=player.x,
                dest_y=player.y,
                xThreat_delta=-0.1
            )
        
        return success
    
    @staticmethod
    def execute_shoot(player, model) -> bool:
        """
        Execute a shot action.
        
        Parameters
        ----------
        player : PlayerAgent
            The shooting player
        model : FootballModel
            The simulation model
            
        Returns
        -------
        bool
            True if shot was successful (goal)
        """
        # Simple conversion rate
        goal_scored = np.random.random() < ActionConstraints.BASE_SHOT_CONVERSION
        
        if goal_scored:
            model.logger.log_event(
                player=player.unique_id,
                team=player.team_id,
                action_type='GOAL',
                dest_x=player.x,
                dest_y=player.y,
                xThreat_delta=1.0
            )
            model.end_possession()
            return True
        else:
            model.logger.log_event(
                player=player.unique_id,
                team=player.team_id,
                action_type='SHOT_MISSED',
                dest_x=player.x,
                dest_y=player.y,
                xThreat_delta=-0.2
            )
            # Ball becomes loose
            model.ball.possessing_player = None
            return False
    
    @staticmethod
    def execute_clear(player, model):
        """
        Execute a clearance action.
        
        Parameters
        ----------
        player : PlayerAgent
            The clearing player
        model : FootballModel
            The simulation model
        """
        # Kick towards nearest touchline
        if player.x < model.pitch_width / 2:
            target_x = max(0, player.x - ActionConstraints.CLEAR_DISTANCE)
        else:
            target_x = min(model.pitch_width, player.x + ActionConstraints.CLEAR_DISTANCE)
        
        target_y = player.y  # Same y-coordinate
        
        # Execute clearance
        model.ball.kick(target_x, target_y, 15.0)
        
        model.logger.log_event(
            player=player.unique_id,
            team=player.team_id,
            action_type='CLEAR',
            dest_x=target_x,
            dest_y=target_y,
            xThreat_delta=-0.05
        )
    
    @staticmethod
    def _count_nearby_opponents(player, radius: float, model) -> int:
        """
        Count opponent players within specified radius.
        
        Parameters
        ----------
        player : PlayerAgent
            The reference player
        radius : float
            Search radius in meters
        model : FootballModel
            The simulation model
            
        Returns
        -------
        int
            Number of nearby opponents
        """
        count = 0
        for other_player in model.players:
            if other_player.team_id != player.team_id:
                distance = np.sqrt((other_player.x - player.x)**2 + 
                                 (other_player.y - player.y)**2)
                if distance < radius:
                    count += 1
        return count


class ActionPlanner:
    """
    Plans and suggests actions based on game state analysis.
    
    This class provides utility methods for action planning, including
    evaluation of action outcomes and strategic decision making.
    """
    
    @staticmethod
    def get_available_actions(player, model) -> list:
        """
        Get list of all available actions for a player.
        
        Parameters
        ----------
        player : PlayerAgent
            The player to get actions for
        model : FootballModel
            The simulation model
            
        Returns
        -------
        list
            List of available action dictionaries
        """
        actions = []
        
        # PASS actions
        teammates = [p for p in model.players 
                    if p.team_id == player.team_id and p != player]
        for teammate in teammates:
            if ActionValidator.can_pass(player, teammate, model):
                actions.append({
                    'type': ActionType.PASS,
                    'target': teammate,
                    'target_id': teammate.unique_id
                })
        
        # DRIBBLE actions
        for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
            for distance in [2.0, 4.0, 6.0]:
                if ActionValidator.can_dribble(player, angle, distance, model):
                    actions.append({
                        'type': ActionType.DRIBBLE,
                        'direction': angle,
                        'distance': distance
                    })
        
        # SHOOT action
        if ActionValidator.can_shoot(player, model):
            actions.append({'type': ActionType.SHOOT})
        
        # CLEAR action (always available)
        actions.append({'type': ActionType.CLEAR})
        
        return actions
    
    @staticmethod
    def evaluate_action_xThreat(action: dict, player, model) -> float:
        """
        Evaluate an action's expected xThreat value.
        
        Parameters
        ----------
        action : dict
            Action dictionary to evaluate
        player : PlayerAgent
            The player performing the action
        model : FootballModel
            The simulation model
            
        Returns
        -------
        float
            Expected xThreat delta
        """
        action_type = action['type']
        current_xT = model.get_xThreat(player.x, player.y)
        
        if action_type == ActionType.PASS:
            target = action['target']
            target_xT = model.get_xThreat(target.x, target.y)
            return target_xT - current_xT
            
        elif action_type == ActionType.DRIBBLE:
            direction = action['direction']
            distance = action['distance']
            target_x = player.x + distance * np.cos(direction)
            target_y = player.y + distance * np.sin(direction)
            target_xT = model.get_xThreat(target_x, target_y)
            return target_xT - current_xT
            
        elif action_type == ActionType.SHOOT:
            return 0.3  # Expected value of shot attempt
            
        elif action_type == ActionType.CLEAR:
            return -0.05  # Slightly negative (defensive action)
            
        else:
            return 0.0
    
    @staticmethod
    def select_best_action(player, model, epsilon: float = 0.2) -> dict:
        """
        Select best action using epsilon-greedy strategy.
        
        Parameters
        ----------
        player : PlayerAgent
            The player selecting an action
        model : FootballModel
            The simulation model
        epsilon : float
            Exploration probability
            
        Returns
        -------
        dict
            Selected action dictionary
        """
        available_actions = ActionPlanner.get_available_actions(player, model)
        
        if not available_actions:
            return {'type': ActionType.CLEAR}
        
        # Epsilon-greedy selection
        if np.random.random() < epsilon:
            # Random action
            return np.random.choice(available_actions)
        else:
            # Best action based on weighted xThreat
            best_action = None
            best_value = float('-inf')
            
            for action in available_actions:
                xT_value = ActionPlanner.evaluate_action_xThreat(action, player, model)
                action_weight = player.action_weights.get(action['type'].value, 1.0)
                total_value = xT_value * action_weight
                
                if total_value > best_value:
                    best_value = total_value
                    best_action = action
            
            return best_action if best_action else available_actions[0]
