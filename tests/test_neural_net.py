"""
Unit tests for neural network models.
Tests the FeedforwardNet and LSTMNet classes.
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.neural_net import FeedforwardNet, LSTMNet, SupplyChainLoss
from config import get_config


class TestFeedforwardNet(unittest.TestCase):
    """Test cases for FeedforwardNet class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = get_config()
        self.input_size = 19
        self.model = FeedforwardNet(self.config, self.input_size)
    
    def test_model_initialization(self):
        """Test model initialization."""
        self.assertIsInstance(self.model, nn.Module)
        self.assertEqual(self.model.input_size, self.input_size)
        
        # Check that model has layers
        self.assertTrue(hasattr(self.model, 'layers'))
        self.assertTrue(hasattr(self.model, 'output_layer'))
    
    def test_forward_pass(self):
        """Test forward pass through the model."""
        batch_size = 32
        x = torch.randn(batch_size, self.input_size)
        
        # Forward pass
        output = self.model(x)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, 1))
        
        # Check output is finite
        self.assertTrue(torch.isfinite(output).all())
    
    def test_model_parameters(self):
        """Test model parameters."""
        # Check that model has trainable parameters
        params = list(self.model.parameters())
        self.assertGreater(len(params), 0)
        
        # Check parameter shapes are reasonable
        total_params = sum(p.numel() for p in params if p.requires_grad)
        self.assertGreater(total_params, 1000)  # Should have reasonable number of parameters
    
    def test_training_mode(self):
        """Test training vs evaluation mode."""
        # Test training mode
        self.model.train()
        self.assertTrue(self.model.training)
        
        # Test evaluation mode
        self.model.eval()
        self.assertFalse(self.model.training)
    
    def test_gradient_flow(self):
        """Test that gradients flow through the model."""
        x = torch.randn(10, self.input_size, requires_grad=True)
        target = torch.randn(10, 1)
        
        # Forward pass
        output = self.model(x)
        loss = nn.MSELoss()(output, target)
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist
        for param in self.model.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)


class TestLSTMNet(unittest.TestCase):
    """Test cases for LSTMNet class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = get_config()
        self.input_size = 19
        self.sequence_length = 8  # From config lookback_weeks
        self.model = LSTMNet(self.config, self.input_size)
    
    def test_model_initialization(self):
        """Test LSTM model initialization."""
        self.assertIsInstance(self.model, nn.Module)
        self.assertEqual(self.model.input_size, self.input_size)
        
        # Check LSTM-specific components
        self.assertTrue(hasattr(self.model, 'lstm'))
        self.assertTrue(hasattr(self.model, 'fc_layers'))
    
    def test_forward_pass(self):
        """Test forward pass through LSTM model."""
        batch_size = 16
        x = torch.randn(batch_size, self.sequence_length, self.input_size)
        
        # Forward pass
        output = self.model(x)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, 1))
        
        # Check output is finite
        self.assertTrue(torch.isfinite(output).all())
    
    def test_hidden_state_handling(self):
        """Test LSTM hidden state handling."""
        batch_size = 8
        x = torch.randn(batch_size, self.sequence_length, self.input_size)
        
        # Multiple forward passes should work
        output1 = self.model(x)
        output2 = self.model(x)
        
        # Outputs should be the same (deterministic)
        self.assertTrue(torch.allclose(output1, output2, atol=1e-6))


class TestSupplyChainLoss(unittest.TestCase):
    """Test cases for custom loss function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = get_config()
        self.loss_fn = SupplyChainLoss(self.config)
    
    def test_loss_calculation(self):
        """Test loss function calculation."""
        predictions = torch.randn(32, 1)
        targets = torch.randn(32, 1)
        
        loss = self.loss_fn(predictions, targets)
        
        # Check loss properties
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)  # Scalar loss
        self.assertGreaterEqual(loss.item(), 0)  # Non-negative loss
    
    def test_loss_gradient(self):
        """Test loss function gradients."""
        predictions = torch.randn(16, 1, requires_grad=True)
        targets = torch.randn(16, 1)
        
        loss = self.loss_fn(predictions, targets)
        loss.backward()
        
        # Check that gradients exist
        self.assertIsNotNone(predictions.grad)
        self.assertEqual(predictions.grad.shape, predictions.shape)
    
    def test_perfect_predictions(self):
        """Test loss with perfect predictions."""
        targets = torch.randn(10, 1)
        predictions = targets.clone()
        
        loss = self.loss_fn(predictions, targets)
        
        # Loss should be very small for perfect predictions
        self.assertLess(loss.item(), 1e-6)


class TestModelIntegration(unittest.TestCase):
    """Integration tests for model components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = get_config()
        self.input_size = 19
        
    def test_model_training_step(self):
        """Test a complete training step."""
        model = FeedforwardNet(self.config, self.input_size)
        loss_fn = SupplyChainLoss(self.config)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Sample data
        x = torch.randn(16, self.input_size)
        y = torch.randn(16, 1)
        
        # Training step
        model.train()
        optimizer.zero_grad()
        
        predictions = model(x)
        loss = loss_fn(predictions, y)
        loss.backward()
        optimizer.step()
        
        # Check that loss is reasonable
        self.assertIsInstance(loss.item(), float)
        self.assertGreater(loss.item(), 0)
    
    def test_model_evaluation_step(self):
        """Test model evaluation step."""
        model = FeedforwardNet(self.config, self.input_size)
        loss_fn = SupplyChainLoss(self.config)
        
        # Sample data
        x = torch.randn(20, self.input_size)
        y = torch.randn(20, 1)
        
        # Evaluation step
        model.eval()
        with torch.no_grad():
            predictions = model(x)
            loss = loss_fn(predictions, y)
        
        # Check outputs
        self.assertEqual(predictions.shape, (20, 1))
        self.assertIsInstance(loss.item(), float)
    
    def test_model_save_load(self):
        """Test model saving and loading."""
        model = FeedforwardNet(self.config, self.input_size)
        
        # Get initial predictions
        x = torch.randn(5, self.input_size)
        initial_output = model(x)
        
        # Save model state
        state_dict = model.state_dict()
        
        # Create new model and load state
        new_model = FeedforwardNet(self.config, self.input_size)
        new_model.load_state_dict(state_dict)
        
        # Check that outputs are the same
        new_output = new_model(x)
        self.assertTrue(torch.allclose(initial_output, new_output, atol=1e-6))


if __name__ == '__main__':
    # Suppress model initialization prints during testing
    import sys
    from io import StringIO
    
    # Capture stdout to suppress prints
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    try:
        unittest.main()
    finally:
        sys.stdout = old_stdout
