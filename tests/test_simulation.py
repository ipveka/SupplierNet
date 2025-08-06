"""
Unit tests for data simulation functionality.
Tests the SupplyChainSimulator class and data generation.
"""

import unittest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.simulation import SupplyChainSimulator, SKU
from config import get_config


class TestSupplyChainSimulator(unittest.TestCase):
    """Test cases for SupplyChainSimulator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = get_config()
        self.config['simulation']['num_skus'] = 5  # Small number for testing
        self.config['simulation']['time_horizon_weeks'] = 10
        self.simulator = SupplyChainSimulator(self.config)
    
    def test_simulator_initialization(self):
        """Test that simulator initializes correctly."""
        self.assertEqual(len(self.simulator.skus), 5)
        self.assertEqual(self.simulator.sim_config['time_horizon_weeks'], 10)
        
        # Check that SKUs have required attributes
        for sku in self.simulator.skus:
            self.assertIsInstance(sku, SKU)
            self.assertGreater(sku.base_demand, 0)
            self.assertGreater(sku.procurement_lead_time, 0)
            self.assertGreater(sku.manufacturing_lead_time, 0)
            self.assertGreater(sku.unit_cost, 0)
    
    def test_demand_generation(self):
        """Test demand generation for SKUs."""
        sku = self.simulator.skus[0]
        weeks = 10
        
        demand = self.simulator._generate_demand(sku, weeks)
        
        # Check demand properties
        self.assertEqual(len(demand), weeks)
        self.assertTrue(all(d >= 0 for d in demand))  # Non-negative demand
        self.assertGreater(np.mean(demand), 0)  # Positive average demand
    
    def test_dataset_generation(self):
        """Test complete dataset generation."""
        dataset = self.simulator.generate_dataset()
        
        # Check dataset structure
        self.assertIsInstance(dataset, pd.DataFrame)
        self.assertGreater(len(dataset), 0)
        
        # Check required columns exist
        required_columns = [
            'sku_id', 'week', 'current_inventory', 'demand',
            'target_quantity', 'avg_demand_lookback', 'demand_trend',
            'demand_volatility', 'procurement_lead_time', 'manufacturing_lead_time'
        ]
        
        for col in required_columns:
            self.assertIn(col, dataset.columns, f"Missing column: {col}")
        
        # Check data types and ranges
        self.assertTrue(dataset['sku_id'].dtype == 'object')
        self.assertTrue(dataset['week'].dtype in ['int64', 'int32'])
        self.assertTrue(all(dataset['current_inventory'] >= 0))
        self.assertTrue(all(dataset['demand'] >= 0))
        self.assertTrue(all(dataset['target_quantity'] >= 0))
    
    def test_supply_chain_simulation(self):
        """Test supply chain simulation for a single SKU."""
        sku = self.simulator.skus[0]
        weeks = 10
        
        demand = self.simulator._generate_demand(sku, weeks)
        result = self.simulator._simulate_supply_chain(sku, demand)
        
        # Check simulation results structure
        self.assertIn('inventory', result)
        self.assertIn('orders_procurement', result)
        self.assertIn('orders_manufacturing', result)
        self.assertIn('stockouts', result)
        
        # Check array lengths
        self.assertEqual(len(result['inventory']), weeks)
        self.assertEqual(len(result['orders_procurement']), weeks)
        self.assertEqual(len(result['orders_manufacturing']), weeks)
        self.assertEqual(len(result['stockouts']), weeks)
        
        # Check non-negative values
        self.assertTrue(all(result['inventory'] >= 0))
        self.assertTrue(all(result['stockouts'] >= 0))
    
    def test_dataset_summary(self):
        """Test dataset summary generation."""
        dataset = self.simulator.generate_dataset()
        summary = self.simulator.get_dataset_summary(dataset)
        
        # Check summary structure
        required_keys = [
            'total_records', 'num_skus', 'time_horizon',
            'avg_demand', 'avg_target_quantity', 'stockout_rate'
        ]
        
        for key in required_keys:
            self.assertIn(key, summary, f"Missing summary key: {key}")
        
        # Check summary values
        self.assertEqual(summary['total_records'], len(dataset))
        self.assertEqual(summary['num_skus'], 5)
        self.assertEqual(summary['time_horizon'], 10)
        self.assertGreater(summary['avg_demand'], 0)
        self.assertGreater(summary['avg_target_quantity'], 0)
        self.assertGreaterEqual(summary['stockout_rate'], 0)
        self.assertLessEqual(summary['stockout_rate'], 1)


class TestSKU(unittest.TestCase):
    """Test cases for SKU class."""
    
    def test_sku_creation(self):
        """Test SKU object creation."""
        sku = SKU(
            sku_id="TEST_001",
            base_demand=100,
            seasonality_amplitude=0.2,
            procurement_lead_time=5,
            manufacturing_lead_time=3,
            unit_cost=10.0
        )
        
        self.assertEqual(sku.sku_id, "TEST_001")
        self.assertEqual(sku.base_demand, 100)
        self.assertEqual(sku.seasonality_amplitude, 0.2)
        self.assertEqual(sku.procurement_lead_time, 5)
        self.assertEqual(sku.manufacturing_lead_time, 3)
        self.assertEqual(sku.unit_cost, 10.0)


if __name__ == '__main__':
    unittest.main()
