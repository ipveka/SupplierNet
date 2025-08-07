"""
Basic functional tests for SupplierNet core components.
Tests the main functionality without complex interface dependencies.
"""

import unittest
import pandas as pd
import numpy as np
import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_config, set_random_seeds
from data.simulation import SupplyChainSimulator, SKUMetadata


class TestBasicFunctionality(unittest.TestCase):
    """Test basic functionality of core components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = get_config()
        # Use small values for fast testing
        self.config['simulation']['num_skus'] = 3
        self.config['simulation']['time_horizon_weeks'] = 5
    
    def test_config_loading(self):
        """Test that configuration loads correctly."""
        config = get_config()
        
        # Check main sections exist
        required_sections = ['simulation', 'model', 'training', 'evaluation', 'features']
        for section in required_sections:
            self.assertIn(section, config, f"Missing config section: {section}")
        
        # Check some key parameters
        self.assertGreater(config['simulation']['num_skus'], 0)
        self.assertGreater(config['simulation']['time_horizon_weeks'], 0)
        self.assertIsInstance(config['model']['hidden_layers'], list)
        self.assertGreater(config['training']['batch_size'], 0)
    
    def test_random_seeds(self):
        """Test random seed setting."""
        # Should not raise an exception
        set_random_seeds(42)
        set_random_seeds(123)
    
    def test_sku_creation(self):
        """Test SKUMetadata object creation."""
        sku = SKUMetadata(
            sku_id="TEST_001",
            unit_cost=10.0,
            base_demand=100,
            procurement_lead_time=5,
            manufacturing_lead_time=3,
            ordering_cost=50.0,
            seasonality_phase=0.2
        )
        
        self.assertEqual(sku.sku_id, "TEST_001")
        self.assertEqual(sku.base_demand, 100)
        self.assertGreater(sku.procurement_lead_time, 0)
        self.assertGreater(sku.manufacturing_lead_time, 0)
        self.assertGreater(sku.unit_cost, 0)
    
    def test_simulator_initialization(self):
        """Test simulator initialization."""
        simulator = SupplyChainSimulator(self.config)
        
        self.assertEqual(len(simulator.skus), 3)  # From our test config
        
        # Check SKUs are properly created
        for sku in simulator.skus:
            self.assertIsInstance(sku, SKUMetadata)
            self.assertGreater(sku.base_demand, 0)
    
    def test_dataset_generation(self):
        """Test basic dataset generation."""
        simulator = SupplyChainSimulator(self.config)
        dataset = simulator.generate_dataset()
        
        # Check dataset structure
        self.assertIsInstance(dataset, pd.DataFrame)
        self.assertGreater(len(dataset), 0)
        
        # Check required columns exist
        required_columns = [
            'sku_id', 'week', 'current_inventory', 'actual_demand', 'target_quantity'
        ]
        
        for col in required_columns:
            self.assertIn(col, dataset.columns, f"Missing column: {col}")
        
        # Check data validity
        self.assertTrue(all(dataset['current_inventory'] >= 0))
        self.assertTrue(all(dataset['actual_demand'] >= 0))
        self.assertTrue(all(dataset['target_quantity'] >= 0))
        
        # Check we have expected number of records
        expected_records = 3 * 5  # 3 SKUs * 5 weeks
        self.assertEqual(len(dataset), expected_records)
    
    def test_dataset_summary(self):
        """Test dataset summary generation."""
        simulator = SupplyChainSimulator(self.config)
        dataset = simulator.generate_dataset()
        summary = simulator.get_dataset_summary(dataset)
        
        # Check summary structure
        required_keys = [
            'total_records', 'num_skus', 'time_horizon',
            'avg_demand', 'avg_target_quantity'
        ]
        
        for key in required_keys:
            self.assertIn(key, summary, f"Missing summary key: {key}")
        
        # Check summary values make sense
        self.assertEqual(summary['total_records'], len(dataset))
        self.assertEqual(summary['num_skus'], 3)
        self.assertEqual(summary['time_horizon'], 5)
        self.assertGreater(summary['avg_demand'], 0)
        self.assertGreater(summary['avg_target_quantity'], 0)


class TestTorchIntegration(unittest.TestCase):
    """Test PyTorch integration and basic model functionality."""
    
    def test_torch_availability(self):
        """Test that PyTorch is available and working."""
        # Create a simple tensor
        x = torch.randn(5, 3)
        self.assertEqual(x.shape, (5, 3))
        
        # Test basic operations
        y = torch.sum(x)
        self.assertIsInstance(y.item(), float)
    
    def test_device_detection(self):
        """Test CUDA device detection."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.assertIn(str(device), ['cuda', 'cpu'])
    
    def test_simple_model_creation(self):
        """Test creating a simple PyTorch model."""
        import torch.nn as nn
        
        # Create a simple model
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        )
        
        # Test forward pass
        x = torch.randn(3, 10)
        output = model(x)
        self.assertEqual(output.shape, (3, 1))


class TestDataProcessingBasics(unittest.TestCase):
    """Test basic data processing functionality."""
    
    def test_feature_engineering_basic(self):
        """Test basic feature engineering."""
        # Import here to avoid issues if module has problems
        try:
            from utils.data_processing import DataProcessor
            
            config = get_config()
            processor = DataProcessor(config)
            
            # Create minimal test data
            test_data = pd.DataFrame({
                'current_inventory': [100, 80],
                'pipeline_procurement': [50, 30],
                'pipeline_manufacturing': [25, 15],
                'avg_demand_lookback': [90, 95],
                'demand_trend': [0.1, 0.15],
                'demand_volatility': [10, 12],
                'procurement_lead_time': [5, 5],
                'manufacturing_lead_time': [3, 3],
                'unit_cost': [10.0, 10.0],
                'seasonality_sin': [0.5, 0.7],
                'seasonality_cos': [0.8, 0.6],
                'week': [1, 2],
                'target_quantity': [80, 85]
            })
            
            # Test feature engineering
            engineered = processor.engineer_features(test_data)
            
            # Should have more columns than original
            self.assertGreater(len(engineered.columns), len(test_data.columns))
            
            # Should have same number of rows
            self.assertEqual(len(engineered), len(test_data))
            
        except ImportError:
            self.skipTest("DataProcessor not available")


if __name__ == '__main__':
    unittest.main()
