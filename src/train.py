"""
Training pipeline for SupplierNet models.
Handles data loading, preprocessing, model training, and evaluation.

=== TRAINING SPECIFICATION ===

TARGET PREDICTION:
- Predicts 'target_quantity': Optimal supply order quantity
- Continuous regression problem (normalized to 0-1 range)
- Represents cost-optimal procurement/manufacturing decision

INPUT FEATURES (19 total):
• INVENTORY STATE: current_inventory, pipeline_procurement, pipeline_manufacturing
• DEMAND PATTERNS: avg_demand_lookback, demand_trend, demand_volatility
• SUPPLY PARAMETERS: procurement_lead_time, manufacturing_lead_time, unit_cost
• SEASONALITY: seasonality_sin, seasonality_cos
• ENGINEERED RATIOS: inventory_to_demand_ratio, pipeline_to_demand_ratio
• COST FEATURES: cost_advantage_manufacturing, lead_time_difference
• VOLATILITY MEASURES: demand_coefficient_of_variation
• TIME FEATURES: week_normalized
• INTERACTIONS: inventory_volatility_interaction, leadtime_demand_interaction

TRAINING METRICS:
- Loss Function: MSE (Mean Squared Error) with L2 regularization
- Validation Metrics: MAE, RMSE, R²
- Early Stopping: Based on validation loss (patience=10)
- Learning Rate: 0.001 with Adam optimizer
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import os
import argparse
from typing import Dict, Optional
import json
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
from models.neural_net import create_model, ModelTrainer

def setup_device(config: Dict) -> torch.device:
    """Setup training device (CPU/GPU)."""
    device_name = config['training']['device']
    
    if device_name == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device('cpu')
        print("Using CPU")
        if device_name == 'cuda':
            warnings.warn("CUDA requested but not available, falling back to CPU")
    
    return device

def generate_or_load_data(config: Dict, data_path: str, force_regenerate: bool = False) -> pd.DataFrame:
    """
    Generate new data or load existing dataset.
    
    Args:
        config: Configuration dictionary
        data_path: Path to save/load dataset
        force_regenerate: Force regeneration even if file exists
        
    Returns:
        Dataset DataFrame
    """
    if os.path.exists(data_path) and not force_regenerate:
        print(f"Loading existing dataset from {data_path}")
        df = pd.read_csv(data_path)
        print(f"Loaded dataset with {len(df)} records")
    else:
        print("Generating new dataset...")
        simulator = SupplyChainSimulator(config)
        df = simulator.generate_dataset()
        
        # Save dataset
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        simulator.save_dataset(df, data_path)
        
        # Print dataset summary
        summary = simulator.get_dataset_summary(df)
        print("\nDataset Summary:")
        print(f"  Total records: {summary['total_records']:,}")
        print(f"  Number of SKUs: {summary['num_skus']}")
        print(f"  Time horizon: {summary['time_horizon']} weeks")
        print(f"  Average demand: {summary['avg_demand']:.2f}")
        print(f"  Average target quantity: {summary['avg_target_quantity']:.2f}")
        print(f"  Stockout rate: {summary['stockout_rate']:.3f}")
        print(f"  Manufacturing preference: {summary['manufacturing_preference']:.3f}")
    
    return df

def train_model(config: Dict, model_type: str = 'feedforward', 
                data_path: str = '../data/supply_chain_dataset.csv',
                model_save_path: str = '../models/best_model.pth',
                force_regenerate_data: bool = False) -> Dict:
    """
    Complete model training pipeline.
    
    Args:
        config: Configuration dictionary
        model_type: Type of model to train ('feedforward' or 'lstm')
        data_path: Path to dataset
        model_save_path: Path to save trained model
        force_regenerate_data: Force data regeneration
        
    Returns:
        Training results dictionary
    """
    print("=" * 60)
    print("SupplierNet Model Training Pipeline")
    print("=" * 60)
    
    # Set random seeds for reproducibility
    set_random_seeds(config['random_seed'])
    
    # Setup device
    device = setup_device(config)
    
    # Generate or load data
    df = generate_or_load_data(config, data_path, force_regenerate_data)
    
    # Prepare data
    print("\nPreparing data for training...")
    processor = DataProcessor(config)
    data_dict = processor.prepare_data(df)
    
    # Extract data loaders and info
    train_loader = data_dict['train_loader']
    val_loader = data_dict['val_loader']
    data_info = data_dict['data_info']
    
    print(f"\nData preparation complete:")
    print(f"  Features: {data_info['n_features']}")
    print(f"  Training samples: {data_info['n_train_samples']:,}")
    print(f"  Validation samples: {data_info['n_val_samples']:,}")
    print(f"  Target mean: {data_info['target_mean']:.2f}")
    print(f"  Target std: {data_info['target_std']:.2f}")
    
    # Create model
    print(f"\nCreating {model_type} model...")
    model = create_model(config, data_info['n_features'], model_type)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model created with {total_params:,} total parameters ({trainable_params:,} trainable)")
    
    # Create trainer
    trainer = ModelTrainer(model, config, device)
    
    # Train model
    print(f"\nStarting training...")
    print(f"Max epochs: {config['training']['max_epochs']}")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Learning rate: {config['training']['learning_rate']}")
    print(f"Early stopping patience: {config['training']['early_stopping_patience']}")
    
    # Ensure model save directory exists
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    # Train
    history = trainer.train(train_loader, val_loader, model_save_path)
    
    # Save training results
    results = {
        'model_type': model_type,
        'config': config,
        'data_info': data_info,
        'history': history,
        'best_val_loss': trainer.best_val_loss,
        'total_epochs': len(history['train_loss']),
        'model_path': model_save_path
    }
    
    # Save results
    results_path = model_save_path.replace('.pth', '_results.json')
    with open(results_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = results.copy()
        for key, value in json_results['history'].items():
            if isinstance(value, list) and len(value) > 0 and isinstance(value[0], (np.float32, np.float64)):
                json_results['history'][key] = [float(x) for x in value]
        
        json.dump(json_results, f, indent=2, default=str)
    
    print(f"\nTraining completed!")
    print(f"Best validation loss: {trainer.best_val_loss:.6f}")
    print(f"Model saved to: {model_save_path}")
    print(f"Results saved to: {results_path}")
    
    # Save preprocessor
    preprocessor_path = model_save_path.replace('.pth', '_preprocessor.pkl')
    processor.save_preprocessor(preprocessor_path)
    
    return results

def main():
    """Main training script with command line arguments."""
    parser = argparse.ArgumentParser(description='Train SupplierNet model')
    parser.add_argument('--model-type', type=str, default='feedforward',
                       choices=['feedforward', 'lstm'],
                       help='Type of model to train')
    parser.add_argument('--data-path', type=str, default='../data/supply_chain_dataset.csv',
                       help='Path to dataset file')
    parser.add_argument('--model-path', type=str, default='../models/best_model.pth',
                       help='Path to save trained model')
    parser.add_argument('--config-overrides', type=str, default='{}',
                       help='JSON string with config overrides')
    parser.add_argument('--force-regenerate-data', action='store_true',
                       help='Force regeneration of dataset')
    parser.add_argument('--quick-test', action='store_true',
                       help='Quick test with reduced parameters')
    
    args = parser.parse_args()
    
    # Load base configuration
    config = get_config()
    
    # Apply quick test overrides
    if args.quick_test:
        print("Running in quick test mode...")
        config['simulation']['num_skus'] = 20
        config['simulation']['time_horizon_weeks'] = 52
        config['training']['max_epochs'] = 50
        config['training']['early_stopping_patience'] = 10
    
    # Apply config overrides
    try:
        overrides = json.loads(args.config_overrides)
        for key, value in overrides.items():
            if '.' in key:
                # Handle nested keys like 'training.learning_rate'
                keys = key.split('.')
                current = config
                for k in keys[:-1]:
                    current = current[k]
                current[keys[-1]] = value
            else:
                config[key] = value
        print(f"Applied config overrides: {overrides}")
    except json.JSONDecodeError as e:
        print(f"Error parsing config overrides: {e}")
        return
    
    # Train model
    try:
        results = train_model(
            config=config,
            model_type=args.model_type,
            data_path=args.data_path,
            model_save_path=args.model_path,
            force_regenerate_data=args.force_regenerate_data
        )
        
        print("\n" + "=" * 60)
        print("Training Summary")
        print("=" * 60)
        print(f"Model type: {results['model_type']}")
        print(f"Total epochs: {results['total_epochs']}")
        print(f"Best validation loss: {results['best_val_loss']:.6f}")
        print(f"Final training loss: {results['history']['train_loss'][-1]:.6f}")
        print(f"Model saved to: {results['model_path']}")
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise

if __name__ == "__main__":
    main()
