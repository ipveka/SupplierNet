"""
Evaluation utilities for SupplierNet models.
Provides comprehensive metrics including cost analysis and service level calculations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings

class SupplyChainEvaluator:
    """
    Comprehensive evaluator for supply chain optimization models.
    
    Calculates:
    - Prediction accuracy metrics (MAE, RMSE, R²)
    - Cost-based performance metrics
    - Service level metrics
    - Business impact analysis
    """
    
    def __init__(self, config: Dict):
        """
        Initialize evaluator with configuration.
        
        Args:
            config: Configuration dictionary with cost parameters
        """
        self.config = config
        self.sim_config = config['simulation']
        self.eval_config = config['evaluation']
    
    def calculate_prediction_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Calculate standard prediction accuracy metrics.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            
        Returns:
            Dictionary of prediction metrics
        """
        # Basic regression metrics
        mae = np.mean(np.abs(y_true - y_pred))
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        
        # R-squared
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100
        
        # Symmetric MAPE (handles zero values better)
        smape = np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-6)) * 100
        
        # Prediction intervals
        residuals = y_true - y_pred
        prediction_std = np.std(residuals)
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'smape': smape,
            'prediction_std': prediction_std,
            'mean_true': np.mean(y_true),
            'mean_pred': np.mean(y_pred),
            'std_true': np.std(y_true),
            'std_pred': np.std(y_pred),
            'min_true': np.min(y_true),
            'max_true': np.max(y_true),
            'min_pred': np.min(y_pred),
            'max_pred': np.max(y_pred)
        }
    
    def calculate_cost_metrics(self, df: pd.DataFrame, predictions: np.ndarray) -> Dict:
        """
        Calculate cost-based performance metrics.
        
        Args:
            df: DataFrame with supply chain data
            predictions: Model predictions
            
        Returns:
            Dictionary of cost metrics
        """
        df_eval = df.copy()
        df_eval['predictions'] = predictions
        df_eval['prediction_error'] = predictions - df_eval['target_quantity']
        
        # Cost components
        holding_costs = []
        shortage_costs = []
        ordering_costs = []
        production_costs = []
        
        for _, row in df_eval.iterrows():
            unit_cost = row['unit_cost']
            pred_qty = row['predictions']
            actual_demand = row['actual_demand']
            current_inv = row['current_inventory']
            
            # Simulate inventory after order
            end_inventory = current_inv + pred_qty - actual_demand
            
            # Holding cost (positive inventory)
            holding_cost = max(0, end_inventory) * unit_cost * self.sim_config['holding_cost_rate']
            holding_costs.append(holding_cost)
            
            # Shortage cost (negative inventory)
            shortage_qty = max(0, -end_inventory)
            shortage_cost = shortage_qty * unit_cost * self.sim_config['shortage_penalty_multiplier']
            shortage_costs.append(shortage_cost)
            
            # Ordering/production cost
            if pred_qty > 0:
                # Simplified: assume manufacturing if cost advantage exists
                manuf_cost = unit_cost * self.sim_config['production_cost_multiplier']
                if manuf_cost < unit_cost:
                    production_costs.append(pred_qty * manuf_cost)
                    ordering_costs.append(0)
                else:
                    ordering_costs.append(pred_qty * unit_cost)
                    production_costs.append(0)
            else:
                ordering_costs.append(0)
                production_costs.append(0)
        
        # Convert to arrays
        holding_costs = np.array(holding_costs)
        shortage_costs = np.array(shortage_costs)
        ordering_costs = np.array(ordering_costs)
        production_costs = np.array(production_costs)
        
        total_costs = holding_costs + shortage_costs + ordering_costs + production_costs
        
        return {
            'total_cost': np.sum(total_costs),
            'avg_cost_per_period': np.mean(total_costs),
            'holding_cost_total': np.sum(holding_costs),
            'shortage_cost_total': np.sum(shortage_costs),
            'ordering_cost_total': np.sum(ordering_costs),
            'production_cost_total': np.sum(production_costs),
            'cost_breakdown': {
                'holding': np.sum(holding_costs) / np.sum(total_costs) if np.sum(total_costs) > 0 else 0,
                'shortage': np.sum(shortage_costs) / np.sum(total_costs) if np.sum(total_costs) > 0 else 0,
                'ordering': np.sum(ordering_costs) / np.sum(total_costs) if np.sum(total_costs) > 0 else 0,
                'production': np.sum(production_costs) / np.sum(total_costs) if np.sum(total_costs) > 0 else 0
            }
        }
    
    def calculate_service_level_metrics(self, df: pd.DataFrame, predictions: np.ndarray) -> Dict:
        """
        Calculate service level performance metrics.
        
        Args:
            df: DataFrame with supply chain data
            predictions: Model predictions
            
        Returns:
            Dictionary of service level metrics
        """
        df_eval = df.copy()
        df_eval['predictions'] = predictions
        
        # Simulate service levels
        service_levels = []
        stockout_events = []
        
        for _, row in df_eval.iterrows():
            pred_qty = row['predictions']
            actual_demand = row['actual_demand']
            current_inv = row['current_inventory']
            
            # Available inventory after order
            available_inventory = current_inv + pred_qty
            
            # Demand met
            demand_met = min(actual_demand, available_inventory)
            service_level = demand_met / actual_demand if actual_demand > 0 else 1.0
            
            service_levels.append(service_level)
            stockout_events.append(1 if service_level < 1.0 else 0)
        
        service_levels = np.array(service_levels)
        stockout_events = np.array(stockout_events)
        
        return {
            'avg_service_level': np.mean(service_levels),
            'median_service_level': np.median(service_levels),
            'min_service_level': np.min(service_levels),
            'stockout_frequency': np.mean(stockout_events),
            'perfect_service_rate': np.mean(service_levels == 1.0),
            'service_level_std': np.std(service_levels),
            'service_level_95th_percentile': np.percentile(service_levels, 95),
            'service_level_5th_percentile': np.percentile(service_levels, 5)
        }
    
    def calculate_sku_level_metrics(self, df: pd.DataFrame, predictions: np.ndarray) -> Dict:
        """
        Calculate metrics grouped by SKU.
        
        Args:
            df: DataFrame with supply chain data
            predictions: Model predictions
            
        Returns:
            Dictionary of SKU-level metrics
        """
        df_eval = df.copy()
        df_eval['predictions'] = predictions
        
        sku_metrics = {}
        
        for sku_id in df_eval['sku_id'].unique():
            sku_data = df_eval[df_eval['sku_id'] == sku_id]
            
            y_true = sku_data['target_quantity'].values
            y_pred = sku_data['predictions'].values
            
            # Prediction metrics
            pred_metrics = self.calculate_prediction_metrics(y_true, y_pred)
            
            # Cost metrics
            cost_metrics = self.calculate_cost_metrics(sku_data, y_pred)
            
            # Service level metrics
            service_metrics = self.calculate_service_level_metrics(sku_data, y_pred)
            
            # Additional SKU-specific metrics
            sku_metrics[sku_id] = {
                'n_periods': len(sku_data),
                'avg_demand': sku_data['actual_demand'].mean(),
                'demand_volatility': sku_data['actual_demand'].std(),
                'unit_cost': sku_data['unit_cost'].iloc[0],
                'prediction_metrics': pred_metrics,
                'cost_metrics': cost_metrics,
                'service_metrics': service_metrics
            }
        
        # Summary across SKUs
        summary = {
            'n_skus': len(sku_metrics),
            'avg_mae_across_skus': np.mean([m['prediction_metrics']['mae'] for m in sku_metrics.values()]),
            'avg_r2_across_skus': np.mean([m['prediction_metrics']['r2'] for m in sku_metrics.values()]),
            'avg_service_level_across_skus': np.mean([m['service_metrics']['avg_service_level'] for m in sku_metrics.values()]),
            'total_cost_all_skus': sum([m['cost_metrics']['total_cost'] for m in sku_metrics.values()]),
            'skus_with_good_predictions': sum(1 for m in sku_metrics.values() if m['prediction_metrics']['r2'] > 0.7),
            'skus_with_high_service': sum(1 for m in sku_metrics.values() if m['service_metrics']['avg_service_level'] > 0.95)
        }
        
        return {
            'sku_metrics': sku_metrics,
            'summary': summary
        }
    
    def evaluate_model_performance(self, df: pd.DataFrame, predictions: np.ndarray) -> Dict:
        """
        Comprehensive model evaluation.
        
        Args:
            df: DataFrame with supply chain data
            predictions: Model predictions
            
        Returns:
            Complete evaluation results
        """
        print("Calculating comprehensive model evaluation...")
        
        # Basic prediction metrics
        y_true = df['target_quantity'].values
        prediction_metrics = self.calculate_prediction_metrics(y_true, predictions)
        
        # Cost-based metrics
        cost_metrics = self.calculate_cost_metrics(df, predictions)
        
        # Service level metrics
        service_metrics = self.calculate_service_level_metrics(df, predictions)
        
        # SKU-level analysis
        sku_analysis = self.calculate_sku_level_metrics(df, predictions)
        
        # Business impact summary
        business_impact = self._calculate_business_impact(prediction_metrics, cost_metrics, service_metrics)
        
        return {
            'prediction_metrics': prediction_metrics,
            'cost_metrics': cost_metrics,
            'service_metrics': service_metrics,
            'sku_analysis': sku_analysis,
            'business_impact': business_impact,
            'evaluation_summary': self._create_evaluation_summary(
                prediction_metrics, cost_metrics, service_metrics, sku_analysis
            )
        }
    
    def _calculate_business_impact(self, pred_metrics: Dict, cost_metrics: Dict, service_metrics: Dict) -> Dict:
        """Calculate high-level business impact metrics."""
        # Cost efficiency
        target_service_level = self.eval_config['service_level_threshold']
        service_achievement = service_metrics['avg_service_level'] / target_service_level
        
        # Overall performance score (weighted combination)
        accuracy_score = max(0, pred_metrics['r2'])  # R² as accuracy measure
        cost_efficiency = 1.0 - min(1.0, cost_metrics['shortage_cost_total'] / cost_metrics['total_cost'])
        service_score = min(1.0, service_achievement)
        
        overall_score = 0.3 * accuracy_score + 0.4 * cost_efficiency + 0.3 * service_score
        
        return {
            'overall_performance_score': overall_score,
            'accuracy_score': accuracy_score,
            'cost_efficiency_score': cost_efficiency,
            'service_level_score': service_score,
            'service_level_achievement': service_achievement,
            'cost_per_unit_demand': cost_metrics['total_cost'] / max(1, len(pred_metrics) if isinstance(pred_metrics, list) else 1)
        }
    
    def _create_evaluation_summary(self, pred_metrics: Dict, cost_metrics: Dict, 
                                 service_metrics: Dict, sku_analysis: Dict) -> Dict:
        """Create a summary of key evaluation metrics."""
        return {
            'model_accuracy': {
                'mae': pred_metrics['mae'],
                'rmse': pred_metrics['rmse'],
                'r2': pred_metrics['r2'],
                'mape': pred_metrics['mape']
            },
            'cost_performance': {
                'total_cost': cost_metrics['total_cost'],
                'avg_cost_per_period': cost_metrics['avg_cost_per_period'],
                'cost_breakdown': cost_metrics['cost_breakdown']
            },
            'service_performance': {
                'avg_service_level': service_metrics['avg_service_level'],
                'stockout_frequency': service_metrics['stockout_frequency'],
                'perfect_service_rate': service_metrics['perfect_service_rate']
            },
            'sku_performance': {
                'total_skus': sku_analysis['summary']['n_skus'],
                'skus_with_good_predictions': sku_analysis['summary']['skus_with_good_predictions'],
                'skus_with_high_service': sku_analysis['summary']['skus_with_high_service'],
                'avg_performance_across_skus': {
                    'mae': sku_analysis['summary']['avg_mae_across_skus'],
                    'r2': sku_analysis['summary']['avg_r2_across_skus'],
                    'service_level': sku_analysis['summary']['avg_service_level_across_skus']
                }
            }
        }

def compare_models(evaluations: List[Dict], model_names: List[str]) -> Dict:
    """
    Compare multiple model evaluations.
    
    Args:
        evaluations: List of evaluation dictionaries
        model_names: Names of the models
        
    Returns:
        Comparison results
    """
    if len(evaluations) != len(model_names):
        raise ValueError("Number of evaluations must match number of model names")
    
    comparison = {
        'models': model_names,
        'metrics_comparison': {},
        'rankings': {}
    }
    
    # Extract key metrics for comparison
    metrics_to_compare = [
        ('prediction_metrics', 'mae'),
        ('prediction_metrics', 'rmse'),
        ('prediction_metrics', 'r2'),
        ('cost_metrics', 'total_cost'),
        ('service_metrics', 'avg_service_level'),
        ('service_metrics', 'stockout_frequency')
    ]
    
    for metric_category, metric_name in metrics_to_compare:
        values = []
        for eval_dict in evaluations:
            try:
                value = eval_dict[metric_category][metric_name]
                values.append(value)
            except KeyError:
                values.append(None)
        
        comparison['metrics_comparison'][f"{metric_category}_{metric_name}"] = dict(zip(model_names, values))
        
        # Ranking (lower is better for MAE, RMSE, cost, stockout_frequency)
        if metric_name in ['mae', 'rmse', 'total_cost', 'stockout_frequency']:
            valid_values = [(i, v) for i, v in enumerate(values) if v is not None]
            valid_values.sort(key=lambda x: x[1])  # Sort by value (ascending)
        else:  # Higher is better for R², service level
            valid_values = [(i, v) for i, v in enumerate(values) if v is not None]
            valid_values.sort(key=lambda x: x[1], reverse=True)  # Sort by value (descending)
        
        ranking = {}
        for rank, (model_idx, value) in enumerate(valid_values, 1):
            ranking[model_names[model_idx]] = rank
        
        comparison['rankings'][f"{metric_category}_{metric_name}"] = ranking
    
    return comparison
