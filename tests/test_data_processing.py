"""
Unit tests for data processing functionality.
Tests the DataProcessor class and feature engineering.
"""

import unittest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_processing import DataProcessor, SupplyChainDataset
from config import get_config


class TestDataProcessor(unittest.TestCase):
    """Test cases for DataProcessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = get_config()
        self.processor = DataProcessor(self.config)
        
        # Create sample data
        self.sample_data = pd.DataFrame({
            'sku_id': ['SKU_001', 'SKU_001', 'SKU_002', 'SKU_002'],
            'week': [1, 2, 1, 2],
            'current_inventory': [100, 80, 150, 120],
            'pipeline_procurement': [50, 30, 75, 60],
            'pipeline_manufacturing': [25, 15, 35, 30],
            'avg_demand_lookback': [90, 95, 140, 135],
            'demand_trend': [0.1, 0.15, -0.05, -0.02],
            'demand_volatility': [10, 12, 15, 14],
            'procurement_lead_time': [5, 5, 7, 7],
            'manufacturing_lead_time': [3, 3, 4, 4],
            'unit_cost': [10.0, 10.0, 15.0, 15.0],
            'seasonality_sin': [0.5, 0.7, 0.5, 0.7],
            'seasonality_cos': [0.8, 0.6, 0.8, 0.6],
            'target_quantity': [80, 85, 130, 125]
        })
    
    def test_feature_engineering(self):
        """Test feature engineering functionality."""
        engineered_data = self.processor.engineer_features(self.sample_data)
        
        # Check that new features are created
        expected_features = [
            'inventory_to_demand_ratio',
            'pipeline_to_demand_ratio',
            'lead_time_difference',
            'cost_advantage_manufacturing',
            'demand_coefficient_of_variation',
            'week_normalized',
            'inventory_volatility_interaction',
            'leadtime_demand_interaction'
        ]
        
        for feature in expected_features:
            self.assertIn(feature, engineered_data.columns, f"Missing engineered feature: {feature}")
        
        # Check feature calculations
        self.assertTrue(all(engineered_data['inventory_to_demand_ratio'] > 0))
        self.assertTrue(all(engineered_data['pipeline_to_demand_ratio'] >= 0))
        self.assertTrue(all(engineered_data['week_normalized'] >= 0))
        self.assertTrue(all(engineered_data['week_normalized'] <= 1))
    
    def test_feature_selection(self):
        """Test feature selection functionality."""
        engineered_data = self.processor.engineer_features(self.sample_data)
        feature_df, feature_names = self.processor.select_features(engineered_data)
        
        # Check that correct number of features are selected
        expected_feature_count = 19  # As documented in the model specification
        self.assertEqual(len(feature_names), expected_feature_count)
        self.assertEqual(feature_df.shape[1], expected_feature_count)
        
        # Check that core features are included
        core_features = [
            'current_inventory', 'pipeline_procurement', 'pipeline_manufacturing',
            'avg_demand_lookback', 'demand_trend', 'demand_volatility'
        ]
        
        for feature in core_features:
            self.assertIn(feature, feature_names, f"Missing core feature: {feature}")
    
    def test_data_splitting(self):
        """Test train/validation/test splitting."""
        X = np.random.randn(100, 19)  # 100 samples, 19 features
        y = np.random.randn(100)
        
        splits = self.processor.split_data(X, y)
        X_train, X_val, X_test, y_train, y_val, y_test = splits
        
        # Check split sizes
        total_samples = len(X)
        train_size = len(X_train)
        val_size = len(X_val)
        test_size = len(X_test)
        
        self.assertEqual(train_size + val_size + test_size, total_samples)
        
        # Check approximate split ratios (70/15/15)
        self.assertAlmostEqual(train_size / total_samples, 0.7, delta=0.05)
        self.assertAlmostEqual(val_size / total_samples, 0.15, delta=0.05)
        self.assertAlmostEqual(test_size / total_samples, 0.15, delta=0.05)
        
        # Check shapes match
        self.assertEqual(X_train.shape[1], X.shape[1])
        self.assertEqual(len(y_train), len(X_train))
        self.assertEqual(len(y_val), len(X_val))
        self.assertEqual(len(y_test), len(X_test))
    
    def test_normalization(self):
        """Test feature and target normalization."""
        X_train = np.random.randn(100, 19) * 100 + 50  # Mean ~50, std ~100
        y_train = np.random.randn(100) * 200 + 100      # Mean ~100, std ~200
        X_val = np.random.randn(50, 19) * 100 + 50
        
        # Test normalization
        result = self.processor.normalize_features(X_train, X_val, y_train=y_train)
        X_train_norm, X_val_norm, _, y_train_norm = result
        
        # Check that training data is normalized (approximately mean 0, std 1)
        self.assertAlmostEqual(np.mean(X_train_norm), 0, delta=0.1)
        self.assertAlmostEqual(np.std(X_train_norm), 1, delta=0.1)
        
        # Check that validation data uses same scaling
        self.assertEqual(X_val_norm.shape, X_val.shape)
        
        # Check that target is normalized
        if y_train_norm is not None:
            self.assertAlmostEqual(np.mean(y_train_norm), 0, delta=0.1)
            self.assertAlmostEqual(np.std(y_train_norm), 1, delta=0.1)
    
    def test_prepare_data_pipeline(self):
        """Test complete data preparation pipeline."""
        result = self.processor.prepare_data(self.sample_data)
        
        # Check result structure
        self.assertIn('X_train', result)
        self.assertIn('X_val', result)
        self.assertIn('X_test', result)
        self.assertIn('y_train', result)
        self.assertIn('y_val', result)
        self.assertIn('y_test', result)
        self.assertIn('feature_names', result)
        self.assertIn('n_features', result)
        
        # Check shapes
        n_features = result['n_features']
        self.assertEqual(result['X_train'].shape[1], n_features)
        self.assertEqual(result['X_val'].shape[1], n_features)
        self.assertEqual(result['X_test'].shape[1], n_features)
        
        # Check that we have the expected number of features (19)
        self.assertEqual(n_features, 19)


class TestSupplyChainDataset(unittest.TestCase):
    """Test cases for SupplyChainDataset PyTorch dataset."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.X = np.random.randn(50, 19).astype(np.float32)
        self.y = np.random.randn(50).astype(np.float32)
        self.dataset = SupplyChainDataset(self.X, self.y)
    
    def test_dataset_length(self):
        """Test dataset length."""
        self.assertEqual(len(self.dataset), 50)
    
    def test_dataset_getitem(self):
        """Test dataset item access."""
        features, target = self.dataset[0]
        
        # Check types
        self.assertIsInstance(features, np.ndarray)
        self.assertIsInstance(target, (float, np.floating))
        
        # Check shapes
        self.assertEqual(features.shape, (19,))
        self.assertIsInstance(target, (float, np.floating))
    
    def test_dataset_iteration(self):
        """Test dataset iteration."""
        count = 0
        for features, target in self.dataset:
            count += 1
            self.assertEqual(features.shape, (19,))
            self.assertIsInstance(target, (float, np.floating))
        
        self.assertEqual(count, 50)


if __name__ == '__main__':
    unittest.main()
