"""
Unit tests for configuration functionality.
Tests the config module and parameter validation.
"""

import unittest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_config, set_random_seeds


class TestConfig(unittest.TestCase):
    """Test cases for configuration functionality."""
    
    def test_get_config(self):
        """Test configuration loading."""
        config = get_config()
        
        # Check main configuration sections exist
        expected_sections = [
            'simulation', 'model', 'training', 'evaluation', 'features'
        ]
        
        for section in expected_sections:
            self.assertIn(section, config, f"Missing config section: {section}")
    
    def test_simulation_config(self):
        """Test simulation configuration parameters."""
        config = get_config()
        sim_config = config['simulation']
        
        # Check required simulation parameters
        required_params = [
            'num_skus', 'time_horizon_weeks', 'demand_distribution',
            'seasonality_amplitude', 'holding_cost_rate'
        ]
        
        for param in required_params:
            self.assertIn(param, sim_config, f"Missing simulation parameter: {param}")
        
        # Check parameter types and ranges
        self.assertIsInstance(sim_config['num_skus'], int)
        self.assertGreater(sim_config['num_skus'], 0)
        self.assertIsInstance(sim_config['time_horizon_weeks'], int)
        self.assertGreater(sim_config['time_horizon_weeks'], 0)
        self.assertIsInstance(sim_config['seasonality_amplitude'], (int, float))
        self.assertGreaterEqual(sim_config['seasonality_amplitude'], 0)
        self.assertLessEqual(sim_config['seasonality_amplitude'], 1)
    
    def test_model_config(self):
        """Test model configuration parameters."""
        config = get_config()
        model_config = config['model']
        
        # Check required model parameters
        required_params = [
            'hidden_layers', 'dropout_rate', 'batch_norm', 'activation'
        ]
        
        for param in required_params:
            self.assertIn(param, model_config, f"Missing model parameter: {param}")
        
        # Check parameter types and ranges
        self.assertIsInstance(model_config['hidden_layers'], list)
        self.assertGreater(len(model_config['hidden_layers']), 0)
        self.assertIsInstance(model_config['dropout_rate'], (int, float))
        self.assertGreaterEqual(model_config['dropout_rate'], 0)
        self.assertLessEqual(model_config['dropout_rate'], 1)
        self.assertIsInstance(model_config['batch_norm'], bool)
    
    def test_training_config(self):
        """Test training configuration parameters."""
        config = get_config()
        training_config = config['training']
        
        # Check required training parameters
        required_params = [
            'batch_size', 'learning_rate', 'max_epochs', 'early_stopping_patience'
        ]
        
        for param in required_params:
            self.assertIn(param, training_config, f"Missing training parameter: {param}")
        
        # Check parameter types and ranges
        self.assertIsInstance(training_config['batch_size'], int)
        self.assertGreater(training_config['batch_size'], 0)
        self.assertIsInstance(training_config['learning_rate'], (int, float))
        self.assertGreater(training_config['learning_rate'], 0)
        self.assertIsInstance(training_config['max_epochs'], int)
        self.assertGreater(training_config['max_epochs'], 0)
    
    def test_set_random_seeds(self):
        """Test random seed setting functionality."""
        # Test that function runs without error
        try:
            set_random_seeds(42)
        except Exception as e:
            self.fail(f"set_random_seeds raised an exception: {e}")
        
        # Test with different seed
        try:
            set_random_seeds(123)
        except Exception as e:
            self.fail(f"set_random_seeds raised an exception: {e}")


if __name__ == '__main__':
    unittest.main()
