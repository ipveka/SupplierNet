"""
Data preprocessing and PyTorch Dataset utilities for SupplierNet.
Handles feature engineering, normalization, and data loading.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Optional, Union
import pickle
import warnings

class SupplyChainDataset(Dataset):
    """PyTorch Dataset for supply chain data."""
    
    def __init__(self, features: np.ndarray, targets: np.ndarray, 
                 feature_names: Optional[List[str]] = None):
        """
        Initialize dataset.
        
        Args:
            features: Feature matrix (n_samples, n_features)
            targets: Target values (n_samples,)
            feature_names: Names of features for interpretability
        """
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.feature_names = feature_names or [f"feature_{i}" for i in range(features.shape[1])]
        
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx]
    
    def get_feature_names(self) -> List[str]:
        return self.feature_names

class DataProcessor:
    """
    Handles all data preprocessing for SupplierNet.
    
    Features:
    - Feature engineering and selection
    - Normalization and scaling
    - Train/validation/test splits
    - PyTorch Dataset creation
    """
    
    def __init__(self, config: Dict):
        """Initialize processor with configuration."""
        self.config = config
        self.sim_config = config['simulation']
        self.feature_config = config['features']
        
        # Scalers for normalization
        self.feature_scaler = None
        self.target_scaler = None
        self.fitted = False
        
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer additional features from raw data.
        
        Args:
            df: Raw dataset from simulator
            
        Returns:
            DataFrame with engineered features
        """
        df = df.copy()
        
        # Inventory ratios
        df['inventory_to_demand_ratio'] = df['current_inventory'] / (df['avg_demand_lookback'] + 1e-6)
        df['pipeline_total'] = df['pipeline_procurement'] + df['pipeline_manufacturing']
        df['pipeline_to_demand_ratio'] = df['pipeline_total'] / (df['avg_demand_lookback'] + 1e-6)
        
        # Lead time features
        df['lead_time_difference'] = df['procurement_lead_time'] - df['manufacturing_lead_time']
        df['min_lead_time'] = np.minimum(df['procurement_lead_time'], df['manufacturing_lead_time'])
        df['max_lead_time'] = np.maximum(df['procurement_lead_time'], df['manufacturing_lead_time'])
        
        # Cost-based features
        manufacturing_cost = df['unit_cost'] * self.sim_config['production_cost_multiplier']
        df['cost_advantage_manufacturing'] = df['unit_cost'] - manufacturing_cost
        df['cost_ratio_manuf_to_proc'] = manufacturing_cost / df['unit_cost']
        
        # Demand features
        df['demand_coefficient_of_variation'] = df['demand_volatility'] / (df['avg_demand_lookback'] + 1e-6)
        
        # Time-based features
        df['week_normalized'] = df['week'] / 52.0  # Normalize to [0, 1] per year
        
        # Interaction features
        df['inventory_volatility_interaction'] = df['current_inventory'] * df['demand_volatility']
        df['leadtime_demand_interaction'] = df['min_lead_time'] * df['avg_demand_lookback']
        
        return df
    
    def select_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Select relevant features for model training.
        
        Args:
            df: DataFrame with engineered features
            
        Returns:
            Tuple of (feature_df, feature_names)
        """
        # Core features
        feature_columns = [
            'current_inventory',
            'pipeline_procurement',
            'pipeline_manufacturing',
            'avg_demand_lookback',
            'demand_trend',
            'demand_volatility',
            'procurement_lead_time',
            'manufacturing_lead_time',
            'unit_cost',
            'seasonality_sin',
            'seasonality_cos',
            
            # Engineered features
            'inventory_to_demand_ratio',
            'pipeline_to_demand_ratio',
            'lead_time_difference',
            'cost_advantage_manufacturing',
            'demand_coefficient_of_variation',
            'week_normalized',
            'inventory_volatility_interaction',
            'leadtime_demand_interaction'
        ]
        
        # Ensure all columns exist
        available_columns = [col for col in feature_columns if col in df.columns]
        if len(available_columns) != len(feature_columns):
            missing = set(feature_columns) - set(available_columns)
            warnings.warn(f"Missing feature columns: {missing}")
        
        feature_df = df[available_columns].copy()
        
        # Handle any remaining NaN values
        feature_df = feature_df.fillna(0)
        
        return feature_df, available_columns
    
    def normalize_features(self, X_train: np.ndarray, X_val: Optional[np.ndarray] = None, 
                          X_test: Optional[np.ndarray] = None, 
                          y_train: Optional[np.ndarray] = None) -> Tuple:
        """
        Normalize features and optionally targets.
        
        Args:
            X_train: Training features
            X_val: Validation features (optional)
            X_test: Test features (optional)
            y_train: Training targets (optional)
            
        Returns:
            Tuple of normalized arrays
        """
        if not self.feature_config['normalize_features']:
            results = [X_train]
            if X_val is not None:
                results.append(X_val)
            if X_test is not None:
                results.append(X_test)
            if y_train is not None:
                results.append(y_train)
            return tuple(results)
        
        # Fit feature scaler on training data
        self.feature_scaler = StandardScaler()
        X_train_scaled = self.feature_scaler.fit_transform(X_train)
        
        results = [X_train_scaled]
        
        # Transform validation and test sets
        if X_val is not None:
            X_val_scaled = self.feature_scaler.transform(X_val)
            results.append(X_val_scaled)
            
        if X_test is not None:
            X_test_scaled = self.feature_scaler.transform(X_test)
            results.append(X_test_scaled)
        
        # Optionally normalize targets (for regression stability)
        if y_train is not None:
            self.target_scaler = MinMaxScaler()
            y_train_scaled = self.target_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
            results.append(y_train_scaled)
        
        self.fitted = True
        return tuple(results)
    
    def inverse_transform_targets(self, y_scaled: np.ndarray) -> np.ndarray:
        """Inverse transform normalized targets back to original scale."""
        if self.target_scaler is None:
            return y_scaled
        return self.target_scaler.inverse_transform(y_scaled.reshape(-1, 1)).ravel()
    
    def prepare_data(self, df: pd.DataFrame, test_size: float = 0.2, 
                    val_size: float = 0.2) -> Dict:
        """
        Complete data preparation pipeline.
        
        Args:
            df: Raw dataset
            test_size: Fraction for test set
            val_size: Fraction of remaining data for validation
            
        Returns:
            Dictionary with datasets and metadata
        """
        print("Starting data preparation pipeline...")
        
        # Feature engineering
        print("Engineering features...")
        df_engineered = self.engineer_features(df)
        
        # Feature selection
        print("Selecting features...")
        feature_df, feature_names = self.select_features(df_engineered)
        
        # Prepare features and targets
        X = feature_df.values
        y = df_engineered['target_quantity'].values
        
        # Split data
        print("Splitting data...")
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.config['random_seed'], 
            stratify=None  # Can't stratify continuous targets
        )
        
        # Second split: train vs val
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size, random_state=self.config['random_seed']
        )
        
        # Normalize data
        print("Normalizing features...")
        if self.feature_config['normalize_features']:
            X_train, X_val, X_test, y_train = self.normalize_features(
                X_train, X_val, X_test, y_train
            )
        
        # Create PyTorch datasets
        print("Creating PyTorch datasets...")
        train_dataset = SupplyChainDataset(X_train, y_train, feature_names)
        val_dataset = SupplyChainDataset(X_val, y_val, feature_names)
        test_dataset = SupplyChainDataset(X_test, y_test, feature_names)
        
        # Create data loaders
        batch_size = self.config['training']['batch_size']
        
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, 
            num_workers=0, pin_memory=torch.cuda.is_available()
        )
        
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=0, pin_memory=torch.cuda.is_available()
        )
        
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=0, pin_memory=torch.cuda.is_available()
        )
        
        # Prepare metadata
        data_info = {
            'n_features': X_train.shape[1],
            'n_train_samples': len(X_train),
            'n_val_samples': len(X_val),
            'n_test_samples': len(X_test),
            'feature_names': feature_names,
            'target_mean': np.mean(y_train),
            'target_std': np.std(y_train),
        }
        
        print(f"Data preparation complete:")
        print(f"  - Training samples: {data_info['n_train_samples']}")
        print(f"  - Validation samples: {data_info['n_val_samples']}")
        print(f"  - Test samples: {data_info['n_test_samples']}")
        print(f"  - Features: {data_info['n_features']}")
        
        return {
            'train_loader': train_loader,
            'val_loader': val_loader,
            'test_loader': test_loader,
            'train_dataset': train_dataset,
            'val_dataset': val_dataset,
            'test_dataset': test_dataset,
            'data_info': data_info,
            'raw_data': {
                'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
                'y_train': y_train, 'y_val': y_val, 'y_test': y_test
            }
        }
    
    def save_preprocessor(self, filepath: str) -> None:
        """Save fitted preprocessor to file."""
        if not self.fitted:
            warnings.warn("Preprocessor not fitted yet. Nothing to save.")
            return
            
        preprocessor_data = {
            'feature_scaler': self.feature_scaler,
            'target_scaler': self.target_scaler,
            'config': self.config,
            'fitted': self.fitted
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(preprocessor_data, f)
        print(f"Preprocessor saved to {filepath}")
    
    def load_preprocessor(self, filepath: str) -> None:
        """Load fitted preprocessor from file."""
        with open(filepath, 'rb') as f:
            preprocessor_data = pickle.load(f)
        
        self.feature_scaler = preprocessor_data['feature_scaler']
        self.target_scaler = preprocessor_data['target_scaler']
        self.config = preprocessor_data['config']
        self.fitted = preprocessor_data['fitted']
        
        print(f"Preprocessor loaded from {filepath}")

def create_data_loaders(config: Dict, df: pd.DataFrame) -> Dict:
    """
    Convenience function to create data loaders from configuration and dataset.
    
    Args:
        config: Configuration dictionary
        df: Dataset DataFrame
        
    Returns:
        Dictionary with data loaders and metadata
    """
    processor = DataProcessor(config)
    return processor.prepare_data(df)

def get_feature_importance_data(data_dict: Dict) -> Dict:
    """
    Extract data needed for feature importance analysis.
    
    Args:
        data_dict: Output from prepare_data()
        
    Returns:
        Dictionary with feature importance data
    """
    return {
        'feature_names': data_dict['data_info']['feature_names'],
        'X_train': data_dict['raw_data']['X_train'],
        'y_train': data_dict['raw_data']['y_train'],
        'X_test': data_dict['raw_data']['X_test'],
        'y_test': data_dict['raw_data']['y_test'],
    }
