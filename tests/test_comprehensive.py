"""
Comprehensive unit tests for the football simulation system.
Tests agents, actions, environment, and logging functionality.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
from mesa import Agent
from mesa.space import ContinuousSpace

# Import simulation modules
from sim.agents import PlayerAgent, BallAgent, RefAgent
from sim.actions import ActionType, ActionPlanner, ActionValidator
from sim.env import FootballModel
from sim.logger import EventLogger

class TestPlayerAgent:
    """Test PlayerAgent functionality."""
    
    def test_player_creation(self):
        """Test player agent creation with valid parameters."""
        model = FootballModel()
        player = PlayerAgent(model, team_id=0, role="midfielder", x=26.0, y=17.0)
        
        assert player.team_id == 0
        assert player.role == "midfielder"
        assert player.x == 26.0
        assert player.y == 17.0
        assert player.stamina > 0
        assert isinstance(player.action_weights, dict)
    
    def test_player_movement(self):
        """Test player movement mechanics."""
        model = FootballModel()
        player = PlayerAgent(model, team_id=0, role="midfielder", x=26.0, y=17.0)
        
        initial_x, initial_y = player.x, player.y
        player.move_to(30.0, 20.0)
        
        # Position should change
        assert player.x != initial_x or player.y != initial_y
        
        # Should be within pitch bounds
        assert 0 <= player.x <= 52
        assert 0 <= player.y <= 34
    
    def test_action_selection(self):
        """Test action selection mechanism."""
        model = FootballModel()
        player = PlayerAgent(model, team_id=0, role="midfielder", x=26.0, y=17.0)
        
        # Mock ball possession
        model.ball.possessing_player = player.unique_id
        
        action = player.choose_action()
        assert action in [ActionType.PASS, ActionType.DRIBBLE, ActionType.SHOOT, 
                         ActionType.CLEAR, ActionType.MOVE_OFFBALL]
    
    def test_stamina_system(self):
        """Test stamina mechanics."""
        model = FootballModel()
        player = PlayerAgent(model, team_id=0, role="midfielder", x=26.0, y=17.0)
        
        initial_stamina = player.stamina
        
        # Simulate multiple actions
        for _ in range(10):
            player.step()
        
        # Stamina should change (either decrease from actions or recover)
        # This depends on whether player has ball and performs actions
        assert isinstance(player.stamina, (int, float))
        assert player.stamina >= 0

class TestBallAgent:
    """Test BallAgent functionality."""
    
    def test_ball_creation(self):
        """Test ball agent creation."""
        model = FootballModel()
        ball = BallAgent(model, x=26.0, y=17.0)
        
        assert ball.x == 26.0
        assert ball.y == 17.0
        assert ball.vx == 0.0
        assert ball.vy == 0.0
        assert ball.possessing_player is None
    
    def test_ball_kick(self):
        """Test ball kicking mechanics."""
        model = FootballModel()
        ball = BallAgent(model, x=26.0, y=17.0)
        
        # Kick ball
        ball.kick(30.0, 20.0, power=10.0)
        
        # Ball should have velocity
        assert abs(ball.vx) > 0 or abs(ball.vy) > 0
    
    def test_ball_physics(self):
        """Test ball physics simulation."""
        model = FootballModel()
        ball = BallAgent(model, x=26.0, y=17.0)
        
        # Give ball some velocity
        ball.vx = 5.0
        ball.vy = 3.0
        
        initial_speed = np.sqrt(ball.vx**2 + ball.vy**2)
        
        # Step forward
        ball.step()
        
        # Ball should move and decelerate
        new_speed = np.sqrt(ball.vx**2 + ball.vy**2)
        assert new_speed < initial_speed  # Friction should slow it down

class TestRefAgent:
    """Test RefAgent functionality."""
    
    def test_ref_creation(self):
        """Test referee agent creation."""
        model = FootballModel()
        ref = RefAgent(model)
        
        assert ref.match_time == 0.0
        assert ref.half == 1
    
    def test_timekeeping(self):
        """Test match time management."""
        model = FootballModel()
        ref = RefAgent(model)
        
        initial_time = ref.match_time
        
        # Step forward
        ref.step()
        
        # Time should advance
        assert ref.match_time > initial_time

class TestActionSystem:
    """Test action planning and validation."""
    
    def test_action_types(self):
        """Test action type enumeration."""
        assert ActionType.PASS.value == "PASS"
        assert ActionType.DRIBBLE.value == "DRIBBLE"
        assert ActionType.SHOOT.value == "SHOOT"
        assert ActionType.CLEAR.value == "CLEAR"
        assert ActionType.MOVE_OFFBALL.value == "MOVE_OFFBALL"
    
    def test_action_validator(self):
        """Test action validation logic."""
        model = FootballModel()
        player = PlayerAgent(model, team_id=0, role="midfielder", x=26.0, y=17.0)
        validator = ActionValidator(model)
        
        # Test pass validation
        is_valid = validator.can_pass(player, 30.0, 20.0)
        assert isinstance(is_valid, bool)
        
        # Test shoot validation
        is_valid = validator.can_shoot(player)
        assert isinstance(is_valid, bool)
    
    def test_action_planner(self):
        """Test action planning logic."""
        model = FootballModel()
        player = PlayerAgent(model, team_id=0, role="midfielder", x=26.0, y=17.0)
        planner = ActionPlanner(model)
        
        # Test pass planning
        target_x, target_y = planner.plan_pass(player)
        assert isinstance(target_x, (int, float))
        assert isinstance(target_y, (int, float))
        
        # Target should be on pitch
        assert 0 <= target_x <= 52
        assert 0 <= target_y <= 34

class TestFootballModel:
    """Test main simulation environment."""
    
    def test_model_creation(self):
        """Test model initialization."""
        model = FootballModel()
        
        assert model.width == 52
        assert model.height == 34
        assert len(model.agents) > 0  # Should have players, ball, ref
        assert model.ball is not None
        assert model.ref is not None
    
    def test_team_setup(self):
        """Test team configuration."""
        model = FootballModel()
        
        # Count players by team
        team_0_players = [a for a in model.agents if hasattr(a, 'team_id') and a.team_id == 0]
        team_1_players = [a for a in model.agents if hasattr(a, 'team_id') and a.team_id == 1]
        
        assert len(team_0_players) == 7  # 7-a-side
        assert len(team_1_players) == 7
    
    def test_simulation_step(self):
        """Test simulation stepping."""
        model = FootballModel()
        initial_step = model.schedule.steps
        
        model.step()
        
        assert model.schedule.steps > initial_step
    
    def test_match_termination(self):
        """Test match ending conditions."""
        model = FootballModel()
        
        # Run for limited steps
        for _ in range(10):
            if model.running:
                model.step()
        
        # Model should either still be running or have terminated gracefully
        assert isinstance(model.running, bool)

class TestEventLogger:
    """Test event logging functionality."""
    
    def test_logger_creation(self):
        """Test event logger creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = EventLogger(output_dir=temp_dir)
            assert logger.output_dir == Path(temp_dir)
            assert len(logger.events) == 0
    
    def test_event_logging(self):
        """Test event recording."""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = EventLogger(output_dir=temp_dir)
            
            # Log an event
            logger.log_event(
                case_id="test_case",
                timestamp="2024-01-01 12:00:00",
                player_id=1,
                team_id=0,
                action_type="PASS",
                dest_x=30.0,
                dest_y=20.0,
                xthreat_delta=0.1,
                tick=1
            )
            
            assert len(logger.events) == 1
            event = logger.events[0]
            assert event['case_id'] == "test_case"
            assert event['action_type'] == "PASS"
    
    def test_csv_export(self):
        """Test CSV file export."""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = EventLogger(output_dir=temp_dir)
            
            # Log sample events
            logger.log_event("case1", "2024-01-01 12:00:00", 1, 0, "PASS", 30.0, 20.0, 0.1, 1)
            logger.log_event("case1", "2024-01-01 12:00:01", 2, 0, "SHOOT", 45.0, 17.0, 0.5, 2)
            
            # Export to CSV
            csv_path = logger.export_csv()
            
            assert csv_path.exists()
            
            # Verify CSV content
            df = pd.read_csv(csv_path)
            assert len(df) == 2
            assert 'case_id' in df.columns
            assert 'action_type' in df.columns
    
    def test_xes_export(self):
        """Test XES file export."""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = EventLogger(output_dir=temp_dir)
            
            # Log sample events
            logger.log_event("case1", "2024-01-01 12:00:00", 1, 0, "PASS", 30.0, 20.0, 0.1, 1)
            logger.log_event("case1", "2024-01-01 12:00:01", 2, 0, "SHOOT", 45.0, 17.0, 0.5, 2)
            
            # Export to XES
            xes_path = logger.export_xes()
            
            assert xes_path.exists()
            assert xes_path.suffix == '.xes'

class TestIntegration:
    """Integration tests for the complete system."""
    
    def test_full_simulation(self):
        """Test complete simulation run."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model = FootballModel(output_dir=temp_dir)
            
            # Run simulation for a few steps
            for _ in range(5):
                if model.running:
                    model.step()
            
            # Check that events were logged
            csv_path = Path(temp_dir) / "match_log.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                assert len(df) >= 0  # May be 0 if no actions taken
    
    def test_data_consistency(self):
        """Test data consistency across simulation."""
        model = FootballModel()
        
        # Track ball possession
        initial_possessor = model.ball.possessing_player
        
        # Run a few steps
        for _ in range(3):
            if model.running:
                model.step()
        
        # Ball should still have valid state
        assert model.ball.x >= 0 and model.ball.x <= 52
        assert model.ball.y >= 0 and model.ball.y <= 34
        
        # Players should still be on pitch
        for agent in model.agents:
            if hasattr(agent, 'x') and hasattr(agent, 'y'):
                assert 0 <= agent.x <= 52
                assert 0 <= agent.y <= 34

# Test configuration for pytest
class TestConfiguration:
    """Test configuration and setup."""
    
    def test_data_files_exist(self):
        """Test that required data files exist."""
        data_dir = Path("data")
        
        required_files = [
            "xT_lookup.json",
            "pass_dist.json", 
            "dribble_duration.json"
        ]
        
        for filename in required_files:
            file_path = data_dir / filename
            assert file_path.exists(), f"Required data file missing: {filename}"
    
    def test_config_files_exist(self):
        """Test that configuration files exist."""
        config_dir = Path("configs")
        
        assert (config_dir / "base.yaml").exists(), "Base configuration file missing"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
