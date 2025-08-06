"""
Testing/inference pipeline for SupplierNet models.
Applies trained models to unseen data and generates predictions.

=== TESTING SPECIFICATION ===

PREDICTION TARGET:
- Model predicts 'target_quantity': Optimal supply order quantity
- Continuous values representing cost-optimal procurement/manufacturing decisions
- Denormalized back to original scale (0-500+ units typically)

INPUT FEATURES USED (19 total):
1. INVENTORY STATE (3): current_inventory, pipeline_procurement, pipeline_manufacturing
2. DEMAND PATTERNS (3): avg_demand_lookback, demand_trend, demand_volatility  
3. SUPPLY CHAIN (4): procurement_lead_time, manufacturing_lead_time, unit_cost, seasonality
4. ENGINEERED (9): ratios, interactions, cost advantages, volatility measures

EVALUATION METRICS:
- MAE: Mean Absolute Error (units difference from optimal)
- RMSE: Root Mean Square Error (penalizes large errors)
- R²: Coefficient of determination (variance explained, -∞ to 1)
- MAPE: Mean Absolute Percentage Error (relative accuracy)
- Accuracy ±5%/±10%: Percentage within tolerance bands
- SKU-level analysis: Per-product performance breakdown

BUSINESS INTERPRETATION:
- Good R² > 0.7: Model explains >70% of optimal quantity variance
- Low MAE < 20 units: Predictions within 20 units of optimal
- High accuracy ±10% > 80%: Most predictions within 10% tolerance
"""

import torch
import numpy as np
import pandas as pd
import os
import argparse
import json
from typing import Dict, Tuple, Optional
import warnings

# Local imports
import sys
from pathlib import Path

# Add parent directory to Python path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from config import get_config, set_random_seeds
from data.simulation import SupplyChainSimulator
from utils.data_processing import DataProcessor
from models.neural_net import load_model

def load_test_data(config: Dict, test_data_path: str, 
                   generate_new: bool = False) -> pd.DataFrame:
    """
    Load or generate test dataset.
    
    Args:
        config: Configuration dictionary
        test_data_path: Path to test dataset
        generate_new: Whether to generate new test data
        
    Returns:
        Test dataset DataFrame
    """
    if os.path.exists(test_data_path) and not generate_new:
        print(f"Loading existing test dataset from {test_data_path}")
        df = pd.read_csv(test_data_path)
        print(f"Loaded test dataset with {len(df)} records")
    else:
        print("Generating new test dataset...")
        # Use different random seed for test data to ensure it's truly unseen
        test_config = config.copy()
        test_config['random_seed'] = config['random_seed'] + 1000
        
        simulator = SupplyChainSimulator(test_config)
        df = simulator.generate_dataset()
        
        # Save test dataset
        os.makedirs(os.path.dirname(test_data_path), exist_ok=True)
        simulator.save_dataset(df, test_data_path)
        
        print(f"Generated test dataset with {len(df)} records")
    
    return df

def prepare_test_data(df: pd.DataFrame, processor: DataProcessor) -> Dict:
    """
    Prepare test data using fitted preprocessor.
    
    Args:
        df: Raw test dataset
        processor: Fitted data processor
        
    Returns:
        Dictionary with processed test data
    """
    print("Preparing test data...")
    
    # Feature engineering (same as training)
    df_engineered = processor.engineer_features(df)
    
    # Feature selection
    feature_df, feature_names = processor.select_features(df_engineered)
    
    # Prepare features and targets
    X_test = feature_df.values
    y_test = df_engineered['target_quantity'].values
    
    # Normalize using fitted scalers
    if processor.fitted and processor.feature_scaler is not None:
        X_test = processor.feature_scaler.transform(X_test)
    
    print(f"Test data prepared: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    
    return {
        'X_test': X_test,
        'y_test': y_test,
        'feature_names': feature_names,
        'raw_data': df_engineered
    }

def run_inference(model: torch.nn.Module, X_test: np.ndarray, 
                  device: torch.device, batch_size: int = 1000) -> np.ndarray:
    """
    Run model inference on test data.
    
    Args:
        model: Trained PyTorch model
        X_test: Test features
        device: Inference device
        batch_size: Batch size for inference
        
    Returns:
        Model predictions
    """
    model.eval()
    predictions = []
    
    print(f"Running inference on {len(X_test)} samples...")
    
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            batch_end = min(i + batch_size, len(X_test))
            batch_X = torch.FloatTensor(X_test[i:batch_end]).to(device)
            
            batch_pred = model(batch_X)
            predictions.append(batch_pred.cpu().numpy())
    
    predictions = np.concatenate(predictions, axis=0)
    print(f"Inference completed. Generated {len(predictions)} predictions")
    
    return predictions

def calculate_basic_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """
    Calculate basic regression metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary of metrics
    """
    # Basic metrics
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    
    # R-squared
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100
    
    # Prediction accuracy within tolerance
    tolerance_5pct = np.mean(np.abs(y_true - y_pred) <= 0.05 * np.abs(y_true + 1e-6))
    tolerance_10pct = np.mean(np.abs(y_true - y_pred) <= 0.10 * np.abs(y_true + 1e-6))
    
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'mape': mape,
        'accuracy_5pct': tolerance_5pct,
        'accuracy_10pct': tolerance_10pct,
        'mean_true': np.mean(y_true),
        'mean_pred': np.mean(y_pred),
        'std_true': np.std(y_true),
        'std_pred': np.std(y_pred)
    }

def analyze_predictions_by_sku(df: pd.DataFrame, predictions: np.ndarray) -> Dict:
    """
    Analyze predictions grouped by SKU.
    
    Args:
        df: Raw test data with SKU information
        predictions: Model predictions
        
    Returns:
        Dictionary with per-SKU analysis
    """
    df_analysis = df.copy()
    df_analysis['predictions'] = predictions
    
    # Group by SKU
    sku_metrics = {}
    
    for sku_id in df_analysis['sku_id'].unique():
        sku_data = df_analysis[df_analysis['sku_id'] == sku_id]
        
        y_true = sku_data['target_quantity'].values
        y_pred = sku_data['predictions'].values
        
        metrics = calculate_basic_metrics(y_true, y_pred)
        metrics['n_samples'] = len(sku_data)
        metrics['avg_demand'] = sku_data['actual_demand'].mean()
        metrics['stockout_rate'] = (sku_data['stockout'] > 0).mean()
        
        sku_metrics[sku_id] = metrics
    
    # Summary statistics across SKUs
    summary = {
        'n_skus': len(sku_metrics),
        'avg_mae_across_skus': np.mean([m['mae'] for m in sku_metrics.values()]),
        'avg_r2_across_skus': np.mean([m['r2'] for m in sku_metrics.values()]),
        'best_sku_mae': min([m['mae'] for m in sku_metrics.values()]),
        'worst_sku_mae': max([m['mae'] for m in sku_metrics.values()]),
        'skus_with_good_predictions': sum(1 for m in sku_metrics.values() if m['r2'] > 0.7),
    }
    
    return {
        'sku_metrics': sku_metrics,
        'summary': summary
    }

def test_model(model_path: str, preprocessor_path: str,
               test_data_path: str = '../data/test_dataset.csv',
               results_path: str = '../results/test_results.json',
               generate_new_test_data: bool = False) -> Dict:
    """
    Complete model testing pipeline.
    
    Args:
        model_path: Path to trained model
        preprocessor_path: Path to fitted preprocessor
        test_data_path: Path to test dataset
        results_path: Path to save test results
        generate_new_test_data: Whether to generate new test data
        
    Returns:
        Test results dictionary
    """
    print("=" * 60)
    print("SupplierNet Model Testing Pipeline")
    print("=" * 60)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading model from {model_path}")
    model, config = load_model(model_path, device)
    print("Model loaded successfully")
    
    # Load preprocessor
    print(f"Loading preprocessor from {preprocessor_path}")
    processor = DataProcessor(config)
    processor.load_preprocessor(preprocessor_path)
    
    # Set random seeds
    set_random_seeds(config['random_seed'])
    
    # Load test data
    test_df = load_test_data(config, test_data_path, generate_new_test_data)
    
    # Prepare test data
    test_data = prepare_test_data(test_df, processor)
    
    # Run inference
    predictions = run_inference(model, test_data['X_test'], device)
    
    # Inverse transform predictions if needed
    if processor.target_scaler is not None:
        predictions = processor.inverse_transform_targets(predictions)
        # Also inverse transform true values for fair comparison
        y_test = processor.inverse_transform_targets(test_data['y_test'])
        test_data['y_test'] = y_test
    
    # Calculate metrics
    print("\nCalculating metrics...")
    basic_metrics = calculate_basic_metrics(test_data['y_test'], predictions)
    
    # SKU-level analysis
    print("Performing SKU-level analysis...")
    sku_analysis = analyze_predictions_by_sku(test_data['raw_data'], predictions)
    
    # Compile results
    results = {
        'model_path': model_path,
        'test_data_path': test_data_path,
        'n_test_samples': len(predictions),
        'basic_metrics': basic_metrics,
        'sku_analysis': sku_analysis,
        'config': config,
        'predictions_sample': {
            'true_values': test_data['y_test'][:100].tolist(),
            'predictions': predictions[:100].tolist()
        }
    }
    
    # Save results
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    print(f"Test samples: {len(predictions):,}")
    print(f"MAE: {basic_metrics['mae']:.4f}")
    print(f"RMSE: {basic_metrics['rmse']:.4f}")
    print(f"R²: {basic_metrics['r2']:.4f}")
    print(f"MAPE: {basic_metrics['mape']:.2f}%")
    print(f"Accuracy (±5%): {basic_metrics['accuracy_5pct']:.3f}")
    print(f"Accuracy (±10%): {basic_metrics['accuracy_10pct']:.3f}")
    print(f"\nSKU Analysis:")
    print(f"  SKUs analyzed: {sku_analysis['summary']['n_skus']}")
    print(f"  Average MAE across SKUs: {sku_analysis['summary']['avg_mae_across_skus']:.4f}")
    print(f"  Average R² across SKUs: {sku_analysis['summary']['avg_r2_across_skus']:.4f}")
    print(f"  SKUs with good predictions (R²>0.7): {sku_analysis['summary']['skus_with_good_predictions']}")
    print(f"\nResults saved to: {results_path}")
    
    return results

def main():
    """Main testing script with command line arguments."""
    parser = argparse.ArgumentParser(description='Test SupplierNet model')
    parser.add_argument('--model-path', type=str, default='../models/best_model.pth',
                       help='Path to trained model (default: ../models/best_model.pth)')
    parser.add_argument('--preprocessor-path', type=str,
                       help='Path to fitted preprocessor (auto-detected if not provided)')
    parser.add_argument('--test-data-path', type=str, default='../data/test_dataset.csv',
                       help='Path to test dataset')
    parser.add_argument('--results-path', type=str, default='../results/test_results.json',
                       help='Path to save test results')
    parser.add_argument('--generate-new-test-data', action='store_true',
                       help='Generate new test data')
    
    args = parser.parse_args()
    
    # Auto-detect preprocessor path if not provided
    if args.preprocessor_path is None:
        args.preprocessor_path = args.model_path.replace('.pth', '_preprocessor.pkl')
    
    # Check if files exist
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        return
    
    if not os.path.exists(args.preprocessor_path):
        print(f"Error: Preprocessor file not found: {args.preprocessor_path}")
        return
    
    try:
        results = test_model(
            model_path=args.model_path,
            preprocessor_path=args.preprocessor_path,
            test_data_path=args.test_data_path,
            results_path=args.results_path,
            generate_new_test_data=args.generate_new_test_data
        )
        
        print(f"\nTesting completed successfully!")
        
    except Exception as e:
        print(f"Testing failed with error: {e}")
        raise

if __name__ == "__main__":
    main()
