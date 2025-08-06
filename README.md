# SupplierNet: AI-Powered Supply Chain Optimization

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

SupplierNet is a comprehensive supply chain optimization system that uses PyTorch neural networks to predict optimal supply quantities for procurement and manufacturing decisions. The system simulates realistic supply chain scenarios and trains AI models to minimize costs while maintaining high service levels.

## 🎯 Project Overview

SupplierNet addresses the critical challenge of supply chain optimization by:

- **Simulating realistic supply chain data** with configurable parameters for demand patterns, lead times, and cost structures
- **Training PyTorch neural networks** to predict optimal supply quantities (procurement vs. manufacturing)
- **Evaluating model performance** using comprehensive metrics including cost analysis and service level calculations
- **Providing detailed visualizations** for business insights and model interpretability

### Key Features

- 🔧 **Configurable simulation** with seasonal demand patterns, stochastic lead times, and realistic cost structures
- 🧠 **Advanced neural networks** with feedforward and LSTM architectures
- 📊 **Comprehensive evaluation** including prediction accuracy, cost optimization, and service level metrics
- 📈 **Rich visualizations** for model performance analysis and business reporting
- 🎛️ **Modular architecture** with clean separation of concerns and extensible design

## 📁 Project Structure

```
suppliernet/
├── data/
│   └── simulation.py              # Supply chain data simulation
├── models/
│   └── neural_net.py              # PyTorch model architectures
├── src/
│   ├── generate_data.py           # Data generation pipeline
│   ├── train.py                   # Model training pipeline
│   └── test.py                    # Model testing and inference
│   └── evaluate.py                # Comprehensive model evaluation
├── utils/
│   ├── data_processing.py         # Data preprocessing and PyTorch datasets
│   ├── evaluation.py              # Comprehensive evaluation metrics
│   └── visualization.py           # Visualization and reporting tools
├── config.py                      # Configuration parameters
├── requirements.txt               # Project dependencies
└── README.md                      # This file
```

## 🚀 Quick Start

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd SupplierNet
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**:
   ```bash
   python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
   ```

### Basic Usage

#### 1. Generate Data

```bash
# Generate supply chain dataset
cd src
python generate_data.py

# Generate with custom parameters
python generate_data.py --quick-test --force-regenerate

# Generate multiple dataset variations
python generate_data.py --multiple-variations
```

#### 2. Train a Model

```bash
# Train with default configuration
cd src
python train.py

# Train with custom parameters
python train.py --model-type feedforward --quick-test

# Train LSTM model
python train.py --model-type lstm --model-path ../models/lstm_model.pth
```

#### 3. Test the Model

```bash
# Test trained model
cd src
python test.py --model-path ../models/best_model.pth

# Test with new data
python test.py --model-path ../models/best_model.pth --generate-new-test-data
```

#### 4. Comprehensive Evaluation

```bash
# Full evaluation with visualizations (run from root directory)
cd src
python evaluate.py --model-path models/best_model.pth

# Custom output directory
python evaluate.py --model-path models/best_model.pth --output-dir my_evaluation
```

## 🎯 Model Specification: Features, Targets & Metrics

### Target Variable
**What the model predicts:**
- **`target_quantity`**: Optimal supply order quantity (continuous regression)
  - **Definition**: Cost-optimal quantity to procure or manufacture for each SKU per week
  - **Range**: 0 to ~500 units (varies by SKU demand patterns)
  - **Business meaning**: The quantity that minimizes total supply chain costs while maintaining service levels
  - **Calculation**: Determined by supply chain simulator using cost optimization algorithms

### Input Features (19 total)

#### 1. Inventory State Features (3)
- **`current_inventory`**: Current stock level on hand
- **`pipeline_procurement`**: Quantity in procurement pipeline (ordered but not delivered)
- **`pipeline_manufacturing`**: Quantity in manufacturing pipeline

#### 2. Demand Pattern Features (3)
- **`avg_demand_lookback`**: Historical average demand over lookback period
- **`demand_trend`**: Demand growth/decline trend coefficient
- **`demand_volatility`**: Standard deviation of demand (uncertainty measure)

#### 3. Supply Chain Parameter Features (4)
- **`procurement_lead_time`**: Days to receive externally procured goods
- **`manufacturing_lead_time`**: Days to manufacture goods internally
- **`unit_cost`**: Cost per unit for external procurement
- **`seasonality_sin/cos`**: Seasonal demand patterns (2 features)

#### 4. Engineered Features (9)
- **`inventory_to_demand_ratio`**: Current inventory ÷ average demand (coverage ratio)
- **`pipeline_to_demand_ratio`**: Total pipeline ÷ average demand (pipeline coverage)
- **`lead_time_difference`**: Procurement lead time - manufacturing lead time
- **`cost_advantage_manufacturing`**: Cost savings from manufacturing vs procurement
- **`demand_coefficient_of_variation`**: Demand volatility ÷ average demand (predictability)
- **`week_normalized`**: Week position in year (0-1, for seasonal patterns)
- **`inventory_volatility_interaction`**: Current inventory × demand volatility
- **`leadtime_demand_interaction`**: Minimum lead time × average demand

### Evaluation Metrics

#### Prediction Accuracy Metrics
- **MAE (Mean Absolute Error)**: Average absolute difference between predicted and optimal quantities
  - **Good performance**: < 20 units
  - **Business interpretation**: Average prediction error in units

- **RMSE (Root Mean Square Error)**: Square root of average squared errors
  - **Good performance**: < 30 units
  - **Business interpretation**: Emphasizes large prediction errors

- **R² (Coefficient of Determination)**: Proportion of variance in optimal quantities explained by model
  - **Good performance**: > 0.7 (explains >70% of variance)
  - **Range**: -∞ to 1 (1 = perfect predictions)

- **MAPE (Mean Absolute Percentage Error)**: Average percentage error
  - **Good performance**: < 15%
  - **Business interpretation**: Relative prediction accuracy

#### Business Impact Metrics
- **Total Cost Analysis**: Breakdown of holding, shortage, ordering, and production costs
- **Service Level Performance**: Percentage of demand met without stockouts
- **Stockout Frequency**: Percentage of periods with inventory shortages
- **Perfect Service Rate**: Percentage of periods with 100% demand fulfillment
- **Cost Efficiency**: Predicted vs optimal cost comparison

#### SKU-Level Analysis
- **Per-SKU R²**: Model performance for individual products
- **SKU Performance Distribution**: Range of prediction quality across products
- **Good Prediction Rate**: Percentage of SKUs with R² > 0.7

### Model Architectures

#### Feedforward Neural Network
- **Architecture**: Dense layers with dropout and batch normalization
- **Hidden layers**: [128, 64, 32] neurons (configurable)
- **Activation**: ReLU
- **Regularization**: Dropout (0.2) + L2 regularization

#### LSTM Neural Network
- **Architecture**: LSTM layers followed by dense layers
- **Sequence modeling**: Captures temporal dependencies in supply chain data
- **Hidden units**: 64 LSTM units (configurable)

## 🧪 Testing

SupplierNet includes a comprehensive test suite to ensure code quality and reliability.

### Test Coverage

The test suite covers all major components:

- **`test_simulation.py`**: Data simulation and SKU generation
- **`test_data_processing.py`**: Feature engineering and data preprocessing
- **`test_neural_net.py`**: Neural network models and training components
- **`test_evaluation.py`**: Evaluation metrics and performance analysis
- **`test_config.py`**: Configuration validation and parameter checking

### Running Tests

#### Option 1: Using the Test Runner (Recommended)
```bash
# Run all tests with summary report
python run_tests.py
```

#### Option 2: Using unittest directly
```bash
# Run all tests
python -m unittest discover tests/ -v

# Run specific test file
python -m unittest tests.test_simulation -v

# Run specific test class
python -m unittest tests.test_neural_net.TestFeedforwardNet -v

# Run specific test method
python -m unittest tests.test_evaluation.TestSupplyChainEvaluator.test_prediction_metrics -v
```

#### Option 3: Using pytest (if installed)
```bash
# Install pytest (optional)
pip install pytest pytest-cov

# Run tests with coverage
pytest tests/ --cov=. --cov-report=html

# Run tests with verbose output
pytest tests/ -v
```

### Test Categories

#### Unit Tests
- **Model Architecture**: Forward pass, gradient flow, parameter initialization
- **Data Processing**: Feature engineering, normalization, data splitting
- **Simulation**: SKU generation, demand modeling, supply chain dynamics
- **Evaluation**: Metric calculations, business impact analysis
- **Configuration**: Parameter validation, seed setting

#### Integration Tests
- **Training Pipeline**: Complete model training workflow
- **Evaluation Pipeline**: End-to-end model evaluation
- **Data Pipeline**: Simulation → Processing → Model → Evaluation

### Expected Test Results

A successful test run should show:
```
TEST SUMMARY
============================================================
Tests run: 50+
Failures: 0
Errors: 0
Skipped: 0

Success rate: 100.0%

✅ All tests passed!
```

### Test Data

Tests use:
- **Synthetic data**: Generated on-the-fly for reproducible testing
- **Fixed random seeds**: Ensures consistent test results
- **Small datasets**: Fast execution while maintaining coverage
- **Edge cases**: Perfect predictions, worst-case scenarios, boundary conditions

### Adding New Tests

When adding new functionality:

1. **Create test file**: `tests/test_new_module.py`
2. **Follow naming convention**: `TestClassName` for test classes, `test_method_name` for methods
3. **Include docstrings**: Describe what each test validates
4. **Test edge cases**: Normal operation, boundary conditions, error cases
5. **Use assertions**: Validate expected behavior with descriptive messages

Example test structure:
```python
class TestNewFeature(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.config = get_config()
        # Initialize test data
    
    def test_normal_operation(self):
        """Test normal operation of the feature."""
        # Test implementation
        self.assertEqual(expected, actual)
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Test edge cases
        self.assertRaises(ValueError, function, invalid_input)
```

## 📊 Configuration

SupplierNet is highly configurable through `config.py`. Key parameters include:

### Simulation Parameters
```python
SIMULATION_CONFIG = {
    'num_skus': 100,                    # Number of SKUs to simulate
    'time_horizon_weeks': 104,          # Simulation time horizon
    'demand_distribution': 'poisson',   # Demand distribution type
    'seasonality_amplitude': 0.3,       # Seasonal variation factor
    'procurement_lead_time_range': (2, 8),  # Lead time range (weeks)
    'holding_cost_rate': 0.02,          # Holding cost rate
    'shortage_penalty_multiplier': 5.0,  # Shortage penalty
    # ... more parameters
}
```

### Model Architecture
```python
MODEL_CONFIG = {
    'hidden_layers': [128, 64, 32],     # Neural network architecture
    'dropout_rate': 0.2,                # Dropout rate
    'batch_norm': True,                 # Use batch normalization
    'activation': 'relu',               # Activation function
}
```

### Training Parameters
```python
TRAINING_CONFIG = {
    'batch_size': 64,                   # Training batch size
    'learning_rate': 0.001,             # Learning rate
    'max_epochs': 200,                  # Maximum training epochs
    'early_stopping_patience': 20,      # Early stopping patience
}
```

## 🧠 Model Architectures

### Feedforward Neural Network
- **Input**: Tabular features (inventory levels, demand forecasts, lead times, costs)
- **Architecture**: Configurable hidden layers with batch normalization and dropout
- **Output**: Predicted supply quantity (continuous, positive)
- **Loss**: MSE or custom cost-aware loss function

### LSTM Neural Network
- **Input**: Sequential supply chain data
- **Architecture**: LSTM layers followed by feedforward head
- **Output**: Predicted supply quantity
- **Use case**: When temporal dependencies are important

## 📈 Evaluation Metrics

SupplierNet provides comprehensive evaluation across multiple dimensions:

### Prediction Accuracy
- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- R-squared (R²)
- Mean Absolute Percentage Error (MAPE)

### Cost Performance
- Total cost breakdown (holding, shortage, ordering, production)
- Cost efficiency analysis
- Cost per unit demand

### Service Level Metrics
- Average service level
- Stockout frequency
- Perfect service rate
- Service level distribution

### Business Impact
- Overall performance score
- Cost-service level tradeoff analysis
- SKU-level performance comparison

## 📊 Visualizations

The system generates comprehensive visualizations including:

- **Prediction Accuracy Plots**: Scatter plots, residual analysis, Q-Q plots
- **Cost Breakdown Analysis**: Pie charts, bar charts, cost efficiency metrics
- **Service Level Analysis**: Gauge charts, distribution plots, performance summaries
- **SKU Performance Comparison**: Comparative analysis across different SKUs
- **Time Series Analysis**: Demand, inventory, and prediction trends over time

## 🔧 Advanced Usage

### Custom Configuration

```python
# Override specific parameters
python train.py --config-overrides '{"training.learning_rate": 0.01, "simulation.num_skus": 50}'
```

### Programmatic Usage

```python
from config import get_config
from data.simulated_data import SupplyChainSimulator
from utils.data_processing import DataProcessor
from models.neural_net import create_model, ModelTrainer

# Load configuration
config = get_config()

# Generate data
simulator = SupplyChainSimulator(config)
df = simulator.generate_dataset()

# Prepare data
processor = DataProcessor(config)
data_dict = processor.prepare_data(df)

# Train model
model = create_model(config, data_dict['data_info']['n_features'])
trainer = ModelTrainer(model, config, device)
history = trainer.train(data_dict['train_loader'], data_dict['val_loader'])
```

### Hyperparameter Tuning

For advanced users, consider using Optuna for hyperparameter optimization:

```python
import optuna

def objective(trial):
    # Define hyperparameter search space
    lr = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    
    # Update config and train model
    # Return validation loss
    return val_loss

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)
```

## 📋 Requirements

### Core Dependencies
- Python 3.8+
- PyTorch 2.0+
- pandas 1.5+
- numpy 1.21+
- scikit-learn 1.2+
- matplotlib 3.6+
- seaborn 0.12+

### Optional Dependencies
- SHAP 0.41+ (for feature importance analysis)
- Optuna 3.0+ (for hyperparameter tuning)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Install development dependencies (`pip install -r requirements.txt`)
4. Make your changes
5. Add tests if applicable
6. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
7. Push to the branch (`git push origin feature/AmazingFeature`)
8. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- PyTorch team for the excellent deep learning framework
- Scikit-learn for preprocessing utilities
- Matplotlib and Seaborn for visualization capabilities
- The open-source community for inspiration and best practices

## 📞 Support

If you have any questions or run into issues:

1. Check the [Issues](../../issues) page for existing solutions
2. Create a new issue with detailed information about your problem
3. Include your configuration, error messages, and system information

---
