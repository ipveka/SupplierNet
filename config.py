"""
Configuration parameters for SupplierNet simulation and training.
All parameters are centralized here for easy modification.
"""

from typing import Dict, Any
import numpy as np

# Random seed for reproducibility
RANDOM_SEED = 42

# Simulation Parameters
SIMULATION_CONFIG = {
    # Data generation
    'num_skus': 100,
    'time_horizon_weeks': 104,  # 2 years of weekly data
    'train_test_split': 0.8,
    
    # Demand parameters
    'demand_distribution': 'poisson',  # 'poisson' or 'normal'
    'base_demand_range': (10, 200),  # Min/max base demand per SKU
    'seasonality_amplitude': 0.3,  # Seasonal variation factor
    'demand_noise_std': 0.1,  # Noise as fraction of base demand
    
    # Lead time parameters
    'procurement_lead_time_range': (2, 8),  # weeks
    'manufacturing_lead_time_range': (1, 4),  # weeks
    'lead_time_variability': 0.2,  # CV of lead times
    
    # Cost structure
    'holding_cost_rate': 0.02,  # Per unit per week (2% of unit cost)
    'shortage_penalty_multiplier': 5.0,  # Multiplier of unit cost
    'ordering_cost_range': (50, 200),  # Fixed cost per order
    'production_cost_multiplier': 0.8,  # Manufacturing vs procurement cost ratio
    'unit_cost_range': (5, 100),  # Unit cost range per SKU
    
    # Inventory parameters
    'initial_inventory_weeks': 4,  # Initial inventory as weeks of demand
    'safety_stock_weeks': 2,  # Safety stock target
    'max_order_quantity_weeks': 12,  # Max order as weeks of demand
}

# Neural Network Architecture
MODEL_CONFIG = {
    'input_features': 12,  # Will be calculated based on feature engineering
    'hidden_layers': [128, 64, 32],
    'dropout_rate': 0.2,
    'batch_norm': True,
    'activation': 'relu',
    'output_activation': 'relu',  # Ensure positive quantities
}

# Training Parameters
TRAINING_CONFIG = {
    'batch_size': 64,
    'learning_rate': 0.001,
    'max_epochs': 200,
    'early_stopping_patience': 20,
    'validation_split': 0.2,
    'weight_decay': 1e-5,
    
    # Loss function
    'loss_type': 'mse',  # 'mse' or 'custom'
    'cost_weight': 0.3,  # Weight for cost component in custom loss
    'service_weight': 0.7,  # Weight for service level in custom loss
    
    # Device
    'device': 'cuda',  # Will fallback to 'cpu' if CUDA unavailable
}

# Evaluation Parameters
EVALUATION_CONFIG = {
    'service_level_threshold': 0.95,  # Target service level
    'cost_components': ['holding', 'shortage', 'ordering', 'production'],
    'visualization_skus': 5,  # Number of SKUs to show in detailed plots
}

# Feature Engineering
FEATURE_CONFIG = {
    'lookback_weeks': 8,  # Historical data to include
    'forecast_horizon': 4,  # Weeks to forecast demand
    'include_seasonality': True,
    'include_trend': True,
    'normalize_features': True,
}

def get_config() -> Dict[str, Any]:
    """Return complete configuration dictionary."""
    return {
        'simulation': SIMULATION_CONFIG,
        'model': MODEL_CONFIG,
        'training': TRAINING_CONFIG,
        'evaluation': EVALUATION_CONFIG,
        'features': FEATURE_CONFIG,
        'random_seed': RANDOM_SEED,
    }

def set_random_seeds(seed: int = RANDOM_SEED) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
