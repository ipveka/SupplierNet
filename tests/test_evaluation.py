"""
Unit tests for evaluation utilities.
Tests the SupplyChainEvaluator class and metrics calculation.
"""

import unittest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.evaluation import SupplyChainEvaluator
from config import get_config


class TestSupplyChainEvaluator(unittest.TestCase):
    """Test cases for SupplyChainEvaluator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = get_config()
        self.evaluator = SupplyChainEvaluator(self.config)
        
        # Create sample evaluation data
        np.random.seed(42)  # For reproducible tests
        n_samples = 100
        
        self.sample_data = pd.DataFrame({
            'sku_id': [f'SKU_{i//10:03d}' for i in range(n_samples)],
            'week': [i % 52 + 1 for i in range(n_samples)],
            'target_quantity': np.random.uniform(50, 200, n_samples),
            'current_inventory': np.random.uniform(0, 100, n_samples),
            'demand': np.random.uniform(20, 150, n_samples),
            'unit_cost': np.random.uniform(5, 25, n_samples),
            'procurement_lead_time': np.random.uniform(2, 8, n_samples),
            'manufacturing_lead_time': np.random.uniform(1, 5, n_samples)
        })
        
        # Generate predictions (with some noise to simulate model predictions)
        self.predictions = self.sample_data['target_quantity'].values + np.random.normal(0, 10, n_samples)
        self.predictions = np.maximum(self.predictions, 0)  # Ensure non-negative
    
    def test_prediction_metrics(self):
        """Test prediction accuracy metrics calculation."""
        y_true = self.sample_data['target_quantity'].values
        y_pred = self.predictions
        
        metrics = self.evaluator.calculate_prediction_metrics(y_true, y_pred)
        
        # Check that all expected metrics are present
        expected_metrics = ['mae', 'rmse', 'r2', 'mape']
        for metric in expected_metrics:
            self.assertIn(metric, metrics, f"Missing metric: {metric}")
        
        # Check metric properties
        self.assertGreaterEqual(metrics['mae'], 0)  # MAE should be non-negative
        self.assertGreaterEqual(metrics['rmse'], 0)  # RMSE should be non-negative
        self.assertGreaterEqual(metrics['rmse'], metrics['mae'])  # RMSE >= MAE
        self.assertLessEqual(metrics['r2'], 1)  # R² should be <= 1
        self.assertGreaterEqual(metrics['mape'], 0)  # MAPE should be non-negative
    
    def test_cost_metrics(self):
        """Test cost metrics calculation."""
        cost_metrics = self.evaluator.calculate_cost_metrics(
            self.sample_data, self.predictions
        )
        
        # Check that all expected cost components are present
        expected_components = [
            'total_cost', 'holding_cost_total', 'shortage_cost_total',
            'ordering_cost_total', 'production_cost_total'
        ]
        
        for component in expected_components:
            self.assertIn(component, cost_metrics, f"Missing cost component: {component}")
            self.assertGreaterEqual(cost_metrics[component], 0, f"Negative cost: {component}")
        
        # Check cost breakdown
        self.assertIn('cost_breakdown', cost_metrics)
        breakdown = cost_metrics['cost_breakdown']
        
        # Cost breakdown should sum to approximately 1
        total_breakdown = sum(breakdown.values())
        self.assertAlmostEqual(total_breakdown, 1.0, delta=0.01)
    
    def test_service_level_metrics(self):
        """Test service level metrics calculation."""
        service_metrics = self.evaluator.calculate_service_level_metrics(
            self.sample_data, self.predictions
        )
        
        # Check that all expected service metrics are present
        expected_metrics = [
            'avg_service_level', 'min_service_level', 'max_service_level',
            'median_service_level', 'stockout_frequency', 'perfect_service_rate'
        ]
        
        for metric in expected_metrics:
            self.assertIn(metric, service_metrics, f"Missing service metric: {metric}")
        
        # Check metric ranges
        for metric in ['avg_service_level', 'min_service_level', 'max_service_level', 'median_service_level']:
            self.assertGreaterEqual(service_metrics[metric], 0)
            self.assertLessEqual(service_metrics[metric], 1)
        
        self.assertGreaterEqual(service_metrics['stockout_frequency'], 0)
        self.assertLessEqual(service_metrics['stockout_frequency'], 1)
        self.assertGreaterEqual(service_metrics['perfect_service_rate'], 0)
        self.assertLessEqual(service_metrics['perfect_service_rate'], 1)
    
    def test_sku_level_analysis(self):
        """Test SKU-level performance analysis."""
        sku_analysis = self.evaluator.analyze_sku_performance(
            self.sample_data, self.predictions
        )
        
        # Check structure
        self.assertIn('sku_metrics', sku_analysis)
        self.assertIn('summary_stats', sku_analysis)
        
        sku_metrics = sku_analysis['sku_metrics']
        
        # Check that we have metrics for each SKU
        unique_skus = self.sample_data['sku_id'].unique()
        self.assertEqual(len(sku_metrics), len(unique_skus))
        
        # Check that each SKU has required metrics
        for sku_id, metrics in sku_metrics.items():
            self.assertIn('mae', metrics)
            self.assertIn('r2', metrics)
            self.assertIn('n_samples', metrics)
            self.assertGreater(metrics['n_samples'], 0)
    
    def test_comprehensive_evaluation(self):
        """Test comprehensive evaluation pipeline."""
        results = self.evaluator.evaluate_comprehensive(
            self.sample_data, self.predictions
        )
        
        # Check main result structure
        expected_sections = [
            'prediction_metrics', 'cost_metrics', 'service_metrics',
            'sku_analysis', 'evaluation_summary', 'business_impact'
        ]
        
        for section in expected_sections:
            self.assertIn(section, results, f"Missing evaluation section: {section}")
        
        # Check evaluation summary
        summary = results['evaluation_summary']
        self.assertIn('model_accuracy', summary)
        self.assertIn('cost_performance', summary)
        self.assertIn('service_performance', summary)
        
        # Check business impact
        business_impact = results['business_impact']
        self.assertIn('overall_performance_score', business_impact)
        self.assertGreaterEqual(business_impact['overall_performance_score'], 0)
        self.assertLessEqual(business_impact['overall_performance_score'], 1)
    
    def test_perfect_predictions(self):
        """Test evaluation with perfect predictions."""
        y_true = self.sample_data['target_quantity'].values
        y_pred = y_true.copy()  # Perfect predictions
        
        metrics = self.evaluator.calculate_prediction_metrics(y_true, y_pred)
        
        # Perfect predictions should have specific metric values
        self.assertAlmostEqual(metrics['mae'], 0, delta=1e-10)
        self.assertAlmostEqual(metrics['rmse'], 0, delta=1e-10)
        self.assertAlmostEqual(metrics['r2'], 1.0, delta=1e-10)
        self.assertAlmostEqual(metrics['mape'], 0, delta=1e-10)
    
    def test_worst_case_predictions(self):
        """Test evaluation with worst-case predictions."""
        y_true = self.sample_data['target_quantity'].values
        y_pred = np.zeros_like(y_true)  # Always predict zero
        
        metrics = self.evaluator.calculate_prediction_metrics(y_true, y_pred)
        
        # Worst case should have high errors and low R²
        self.assertGreater(metrics['mae'], 0)
        self.assertGreater(metrics['rmse'], 0)
        self.assertLess(metrics['r2'], 0.5)  # Should be quite poor


class TestMetricCalculations(unittest.TestCase):
    """Test individual metric calculation functions."""
    
    def test_mae_calculation(self):
        """Test Mean Absolute Error calculation."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
        
        expected_mae = np.mean(np.abs(y_true - y_pred))
        
        config = get_config()
        evaluator = SupplyChainEvaluator(config)
        metrics = evaluator.calculate_prediction_metrics(y_true, y_pred)
        
        self.assertAlmostEqual(metrics['mae'], expected_mae, places=6)
    
    def test_rmse_calculation(self):
        """Test Root Mean Square Error calculation."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
        
        expected_rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        
        config = get_config()
        evaluator = SupplyChainEvaluator(config)
        metrics = evaluator.calculate_prediction_metrics(y_true, y_pred)
        
        self.assertAlmostEqual(metrics['rmse'], expected_rmse, places=6)
    
    def test_r2_calculation(self):
        """Test R² coefficient calculation."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
        
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        expected_r2 = 1 - (ss_res / ss_tot)
        
        config = get_config()
        evaluator = SupplyChainEvaluator(config)
        metrics = evaluator.calculate_prediction_metrics(y_true, y_pred)
        
        self.assertAlmostEqual(metrics['r2'], expected_r2, places=6)


if __name__ == '__main__':
    unittest.main()
