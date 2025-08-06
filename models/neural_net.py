"""
PyTorch neural network models for SupplierNet supply chain optimization.

Implements feedforward and LSTM architectures for predicting optimal supply quantities.

=== MODEL SPECIFICATION ===

TARGET VARIABLE:
- target_quantity: Optimal supply quantity to order (continuous, normalized)
  - Represents the ideal quantity to procure or manufacture
  - Calculated by supply chain simulator based on cost optimization
  - Range: 0 to max_demand * safety_factor (typically 0-500 units)

INPUT FEATURES (19 total):
1. INVENTORY STATE (3 features):
   - current_inventory: Current stock level
   - pipeline_procurement: Goods in procurement pipeline
   - pipeline_manufacturing: Goods in manufacturing pipeline

2. DEMAND CHARACTERISTICS (3 features):
   - avg_demand_lookback: Historical average demand
   - demand_trend: Demand growth/decline trend
   - demand_volatility: Demand variability measure

3. SUPPLY CHAIN PARAMETERS (4 features):
   - procurement_lead_time: Days to receive procured goods
   - manufacturing_lead_time: Days to manufacture goods
   - unit_cost: Cost per unit for procurement
   - seasonality_sin/cos: Seasonal demand patterns (2 features)

4. ENGINEERED FEATURES (9 features):
   - inventory_to_demand_ratio: Inventory coverage ratio
   - pipeline_to_demand_ratio: Pipeline coverage ratio
   - lead_time_difference: Procurement vs manufacturing lead time gap
   - cost_advantage_manufacturing: Cost benefit of manufacturing
   - demand_coefficient_of_variation: Demand predictability measure
   - week_normalized: Time position in year (0-1)
   - inventory_volatility_interaction: Inventory * demand volatility
   - leadtime_demand_interaction: Lead time * demand interaction

MODEL ARCHITECTURES:
- Feedforward: Dense layers with dropout and batch normalization
- LSTM: Recurrent layers for time series patterns

EVALUATION METRICS:
- MAE: Mean Absolute Error (primary metric)
- RMSE: Root Mean Square Error
- R²: Coefficient of determination
- MAPE: Mean Absolute Percentage Error
- Business metrics: Cost efficiency, service levels
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

class SupplyChainNet(nn.Module):
    """
    Feedforward neural network for supply chain quantity prediction.
    
    Features:
    - Configurable hidden layers
    - Batch normalization and dropout
    - ReLU activation with positive output constraint
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the neural network.
        
        Args:
            config: Model configuration dictionary
        """
        super(SupplyChainNet, self).__init__()
        
        self.config = config
        model_config = config['model']
        
        # Network architecture
        self.input_size = model_config['input_features']
        self.hidden_sizes = model_config['hidden_layers']
        self.dropout_rate = model_config['dropout_rate']
        self.use_batch_norm = model_config['batch_norm']
        self.activation = model_config['activation']
        self.output_activation = model_config['output_activation']
        
        # Print model specification
        print(f"\n=== FEEDFORWARD MODEL SPECIFICATION ===")
        print(f"INPUT FEATURES: {self.input_size} features")
        print(f"  - 3 Inventory State: current_inventory, pipeline_procurement, pipeline_manufacturing")
        print(f"  - 3 Demand Patterns: avg_demand_lookback, demand_trend, demand_volatility")
        print(f"  - 4 Supply Chain: lead times, costs, seasonality")
        print(f"  - 9 Engineered: ratios, interactions, cost advantages")
        print(f"TARGET: target_quantity (optimal supply order quantity)")
        print(f"ARCHITECTURE: {self.hidden_sizes} -> 1 (regression)")
        print(f"REGULARIZATION: Dropout({self.dropout_rate}), BatchNorm({self.use_batch_norm})")
        print(f"==========================================\n")
        
        # Build layers
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        # Input layer
        prev_size = self.input_size
        
        # Hidden layers
        for hidden_size in self.hidden_sizes:
            # Linear layer
            self.layers.append(nn.Linear(prev_size, hidden_size))
            
            # Batch normalization
            if self.use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_size))
            else:
                self.batch_norms.append(nn.Identity())
            
            # Dropout
            self.dropouts.append(nn.Dropout(self.dropout_rate))
            
            prev_size = hidden_size
        
        # Output layer
        self.output_layer = nn.Linear(prev_size, 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier/He initialization."""
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                if self.activation == 'relu':
                    nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                else:
                    nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias, 0)
        
        # Output layer
        nn.init.xavier_normal_(self.output_layer.weight)
        nn.init.constant_(self.output_layer.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor (batch_size, input_features)
            
        Returns:
            Output tensor (batch_size, 1)
        """
        # Hidden layers
        for i, (layer, batch_norm, dropout) in enumerate(zip(self.layers, self.batch_norms, self.dropouts)):
            x = layer(x)
            x = batch_norm(x)
            
            # Activation function
            if self.activation == 'relu':
                x = F.relu(x)
            elif self.activation == 'tanh':
                x = torch.tanh(x)
            elif self.activation == 'leaky_relu':
                x = F.leaky_relu(x)
            
            x = dropout(x)
        
        # Output layer
        x = self.output_layer(x)
        
        # Output activation (ensure positive quantities)
        if self.output_activation == 'relu':
            x = F.relu(x)
        elif self.output_activation == 'softplus':
            x = F.softplus(x)
        
        return x.squeeze(-1)  # Remove last dimension

class SupplyChainLSTM(nn.Module):
    """
    LSTM-based neural network for sequential supply chain data.
    
    Features:
    - LSTM layers for temporal dependencies
    - Feedforward head for final prediction
    - Configurable architecture
    """
    
    def __init__(self, config: Dict, sequence_length: int = 8):
        """
        Initialize LSTM network.
        
        Args:
            config: Model configuration
            sequence_length: Length of input sequences
        """
        super(SupplyChainLSTM, self).__init__()
        
        self.config = config
        model_config = config['model']
        
        self.input_size = model_config['input_features']
        self.sequence_length = sequence_length
        self.hidden_size = model_config['hidden_layers'][0]  # Use first hidden layer size
        self.num_layers = 2
        self.dropout_rate = model_config['dropout_rate']
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout_rate if self.num_layers > 1 else 0,
            batch_first=True
        )
        
        # Feedforward head
        self.fc_layers = nn.ModuleList()
        prev_size = self.hidden_size
        
        for hidden_size in model_config['hidden_layers'][1:]:
            self.fc_layers.append(nn.Linear(prev_size, hidden_size))
            prev_size = hidden_size
        
        self.output_layer = nn.Linear(prev_size, 1)
        self.dropout = nn.Dropout(self.dropout_rate)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize LSTM and linear layer weights."""
        # LSTM weights
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
        
        # Linear layer weights
        for layer in self.fc_layers:
            nn.init.xavier_normal_(layer.weight)
            nn.init.constant_(layer.bias, 0)
        
        nn.init.xavier_normal_(self.output_layer.weight)
        nn.init.constant_(self.output_layer.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through LSTM network.
        
        Args:
            x: Input tensor (batch_size, sequence_length, input_features)
            
        Returns:
            Output tensor (batch_size,)
        """
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use the last output
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        
        # Feedforward head
        x = last_output
        for layer in self.fc_layers:
            x = F.relu(layer(x))
            x = self.dropout(x)
        
        # Output layer
        x = self.output_layer(x)
        x = F.relu(x)  # Ensure positive quantities
        
        return x.squeeze(-1)

class CustomSupplyChainLoss(nn.Module):
    """
    Custom loss function that combines prediction accuracy with cost-based penalties.
    
    Incorporates:
    - MSE for quantity prediction accuracy
    - Cost penalty for over/under ordering
    - Service level penalty for stockouts
    """
    
    def __init__(self, config: Dict):
        """
        Initialize custom loss function.
        
        Args:
            config: Configuration with loss weights
        """
        super(CustomSupplyChainLoss, self).__init__()
        
        training_config = config['training']
        self.cost_weight = training_config.get('cost_weight', 0.3)
        self.service_weight = training_config.get('service_weight', 0.7)
        
        # Base MSE loss
        self.mse_loss = nn.MSELoss()
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor,
                features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculate custom loss.
        
        Args:
            predictions: Predicted quantities
            targets: Target quantities
            features: Input features (for cost calculation)
            
        Returns:
            Combined loss value
        """
        # Base MSE loss
        mse = self.mse_loss(predictions, targets)
        
        if features is None:
            return mse
        
        # Cost-based penalty (simplified)
        # Penalize over-ordering more than under-ordering
        prediction_error = predictions - targets
        over_order_penalty = torch.relu(prediction_error) ** 2
        under_order_penalty = torch.relu(-prediction_error) ** 2
        
        cost_penalty = torch.mean(2.0 * over_order_penalty + 1.5 * under_order_penalty)
        
        # Service level penalty (penalize predicted quantities that are too low)
        service_penalty = torch.mean(torch.relu(targets - predictions) ** 2)
        
        # Combine losses
        total_loss = mse + self.cost_weight * cost_penalty + self.service_weight * service_penalty
        
        return total_loss

class ModelTrainer:
    """
    Handles model training, validation, and checkpointing.
    """
    
    def __init__(self, model: nn.Module, config: Dict, device: torch.device):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model to train
            config: Training configuration
            device: Training device (CPU/GPU)
        """
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.training_config = config['training']
        
        # Optimizer
        self.optimizer = self._create_optimizer()
        
        # Learning rate scheduler
        self.scheduler = self._create_scheduler()
        
        # Loss function
        self.criterion = self._create_loss_function()
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': []
        }
        
        # Early stopping
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_model_state = None
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer based on configuration."""
        lr = self.training_config['learning_rate']
        weight_decay = self.training_config['weight_decay']
        
        return AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
    
    def _create_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler."""
        return ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
    
    def _create_loss_function(self) -> nn.Module:
        """Create loss function based on configuration."""
        loss_type = self.training_config.get('loss_type', 'mse')
        
        if loss_type == 'custom':
            return CustomSupplyChainLoss(self.config)
        else:
            return nn.MSELoss()
    
    def train_epoch(self, train_loader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_features, batch_targets in train_loader:
            batch_features = batch_features.to(self.device)
            batch_targets = batch_targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(batch_features)
            
            # Calculate loss
            if isinstance(self.criterion, CustomSupplyChainLoss):
                loss = self.criterion(predictions, batch_targets, batch_features)
            else:
                loss = self.criterion(predictions, batch_targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def validate_epoch(self, val_loader) -> float:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_features, batch_targets in val_loader:
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                predictions = self.model(batch_features)
                
                if isinstance(self.criterion, CustomSupplyChainLoss):
                    loss = self.criterion(predictions, batch_targets, batch_features)
                else:
                    loss = self.criterion(predictions, batch_targets)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def train(self, train_loader, val_loader, save_path: Optional[str] = None) -> Dict:
        """
        Complete training loop with early stopping.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            save_path: Path to save best model
            
        Returns:
            Training history dictionary
        """
        max_epochs = self.training_config['max_epochs']
        patience = self.training_config['early_stopping_patience']
        
        print(f"Starting training for up to {max_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(max_epochs):
            # Training
            train_loss = self.train_epoch(train_loader)
            
            # Validation
            val_loss = self.validate_epoch(val_loader)
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            
            # Early stopping check
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.best_model_state = self.model.state_dict().copy()
                
                # Save best model
                if save_path:
                    torch.save({
                        'model_state_dict': self.best_model_state,
                        'config': self.config,
                        'epoch': epoch,
                        'val_loss': val_loss,
                        'history': self.history
                    }, save_path)
            else:
                self.patience_counter += 1
            
            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:3d}/{max_epochs}: "
                      f"Train Loss: {train_loss:.6f}, "
                      f"Val Loss: {val_loss:.6f}, "
                      f"LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Early stopping
            if self.patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        print(f"Training completed. Best validation loss: {self.best_val_loss:.6f}")
        
        return self.history

def create_model(config: Dict, n_features: int, model_type: str = 'feedforward') -> nn.Module:
    """
    Factory function to create models.
    
    Args:
        config: Model configuration
        n_features: Number of input features
        model_type: Type of model ('feedforward' or 'lstm')
        
    Returns:
        Initialized PyTorch model
    """
    # Update config with actual feature count
    config['model']['input_features'] = n_features
    
    if model_type == 'feedforward':
        return SupplyChainNet(config)
    elif model_type == 'lstm':
        return SupplyChainLSTM(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def load_model(checkpoint_path: str, device: torch.device) -> Tuple[nn.Module, Dict]:
    """
    Load a saved model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on
        
    Returns:
        Tuple of (model, config)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    # Create model
    n_features = config['model']['input_features']
    model = create_model(config, n_features)
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    return model, config
