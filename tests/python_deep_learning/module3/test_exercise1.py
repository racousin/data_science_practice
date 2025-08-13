#!/usr/bin/env python3
"""
Test suite for Python Deep Learning Module 3 - Exercise 1: Build Custom MLP
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytest
import math
from typing import Optional, Callable, Tuple, List


class TestExercise1:
    """Test cases for Exercise 1: Build Custom MLP"""
    
    def setup_method(self):
        """Set up test fixtures"""
        torch.manual_seed(42)
        np.random.seed(42)
    
    def test_linear_layer_implementation(self):
        """Test custom linear layer implementation"""
        class CustomLinear(nn.Module):
            def __init__(self, in_features, out_features):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(out_features, in_features))
                self.bias = nn.Parameter(torch.randn(out_features))
            
            def forward(self, x):
                return F.linear(x, self.weight, self.bias)
        
        # Test layer
        layer = CustomLinear(10, 5)
        x = torch.randn(3, 10)
        output = layer(x)
        
        assert output.shape == (3, 5), f"Output shape should be (3, 5), got {output.shape}"
        assert output.requires_grad, "Output should require gradients"
        
        # Test that parameters are learnable
        loss = output.sum()
        loss.backward()
        assert layer.weight.grad is not None, "Weight should have gradients"
        assert layer.bias.grad is not None, "Bias should have gradients"
    
    def test_activation_functions(self):
        """Test various activation functions"""
        x = torch.randn(5, 10)
        
        # Test ReLU
        relu_out = F.relu(x)
        assert torch.all(relu_out >= 0), "ReLU output should be non-negative"
        
        # Test Sigmoid
        sigmoid_out = torch.sigmoid(x)
        assert torch.all((sigmoid_out >= 0) & (sigmoid_out <= 1)), "Sigmoid output should be in [0,1]"
        
        # Test Tanh
        tanh_out = torch.tanh(x)
        assert torch.all((tanh_out >= -1) & (tanh_out <= 1)), "Tanh output should be in [-1,1]"
        
        # Test LeakyReLU
        leaky_relu_out = F.leaky_relu(x, negative_slope=0.01)
        positive_mask = x > 0
        negative_mask = x <= 0
        assert torch.allclose(leaky_relu_out[positive_mask], x[positive_mask]), "LeakyReLU should pass positive values"
        assert torch.allclose(leaky_relu_out[negative_mask], 0.01 * x[negative_mask]), "LeakyReLU should scale negative values"
    
    def test_mlp_forward_pass(self):
        """Test MLP forward pass implementation"""
        class SimpleMLP(nn.Module):
            def __init__(self, input_size, hidden_size, output_size):
                super().__init__()
                self.fc1 = nn.Linear(input_size, hidden_size)
                self.fc2 = nn.Linear(hidden_size, hidden_size)
                self.fc3 = nn.Linear(hidden_size, output_size)
                
            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = self.fc3(x)
                return x
        
        # Test network
        mlp = SimpleMLP(784, 128, 10)
        x = torch.randn(32, 784)
        output = mlp(x)
        
        assert output.shape == (32, 10), f"Output shape should be (32, 10), got {output.shape}"
        assert output.requires_grad, "Output should require gradients"
        
        # Test that all parameters are present
        params = list(mlp.parameters())
        assert len(params) == 6, "Should have 6 parameters (3 weights + 3 biases)"
    
    def test_weight_initialization(self):
        """Test different weight initialization strategies"""
        # Xavier/Glorot initialization
        def xavier_init(layer):
            if hasattr(layer, 'weight'):
                nn.init.xavier_uniform_(layer.weight)
        
        # He initialization
        def he_init(layer):
            if hasattr(layer, 'weight'):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
        
        # Test initialization effects
        layer1 = nn.Linear(100, 50)
        layer2 = nn.Linear(100, 50)
        
        xavier_init(layer1)
        he_init(layer2)
        
        # Check that weights are different after initialization
        assert not torch.allclose(layer1.weight, layer2.weight), "Different initializations should produce different weights"
        
        # Check weight distributions
        xavier_std = layer1.weight.std()
        he_std = layer2.weight.std()
        assert 0.05 < xavier_std < 0.2, f"Xavier weights std should be reasonable, got {xavier_std}"
        assert 0.05 < he_std < 0.3, f"He weights std should be reasonable, got {he_std}"
    
    def test_dropout_implementation(self):
        """Test dropout layer implementation and behavior"""
        x = torch.ones(100, 50)
        dropout = nn.Dropout(p=0.5)
        
        # Test training mode
        dropout.train()
        output_train = dropout(x)
        assert output_train.shape == x.shape, "Dropout should preserve shape"
        
        # Some elements should be zeroed out in training
        zeros_count = (output_train == 0).sum().item()
        assert zeros_count > 0, "Dropout should zero out some elements in training mode"
        
        # Test eval mode
        dropout.eval()
        output_eval = dropout(x)
        assert torch.allclose(output_eval, x), "Dropout should not modify input in eval mode"
    
    def test_batch_normalization(self):
        """Test batch normalization understanding"""
        batch_norm = nn.BatchNorm1d(50)
        x = torch.randn(32, 50)
        
        # Test forward pass
        output = batch_norm(x)
        assert output.shape == x.shape, "BatchNorm should preserve shape"
        
        # In training mode, output should be normalized
        batch_norm.train()
        output_train = batch_norm(x)
        
        # Check normalization properties (approximately)
        mean_out = output_train.mean(dim=0)
        std_out = output_train.std(dim=0)
        
        assert torch.allclose(mean_out, torch.zeros_like(mean_out), atol=1e-6), "BatchNorm output should have zero mean"
        assert torch.allclose(std_out, torch.ones_like(std_out), atol=1e-5), "BatchNorm output should have unit variance"


def test_custom_linear_layer(namespace):
    """Test custom linear layer implementation"""
    import torch
    import torch.nn as nn
    
    # Test CustomLinear class
    if 'CustomLinear' in namespace:
        CustomLinear = namespace['CustomLinear']
        
        # Test instantiation
        layer = CustomLinear(10, 5)
        assert hasattr(layer, 'weight'), "CustomLinear should have weight parameter"
        assert hasattr(layer, 'bias'), "CustomLinear should have bias parameter"
        
        # Test forward pass
        x = torch.randn(3, 10)
        output = layer(x)
        assert output.shape == (3, 5), f"Output shape should be (3, 5), got {output.shape}"
        assert output.requires_grad, "Output should require gradients"
    else:
        raise AssertionError("CustomLinear class not found. Please implement custom linear layer")


def test_mlp_architecture(namespace):
    """Test MLP architecture implementation"""
    import torch
    import torch.nn as nn
    
    # Test MLP class
    if 'MLP' in namespace or 'SimpleMLP' in namespace:
        MLP_class = namespace.get('MLP', namespace.get('SimpleMLP'))
        
        # Test instantiation
        mlp = MLP_class(784, 128, 10)
        assert isinstance(mlp, nn.Module), "MLP should inherit from nn.Module"
        
        # Test forward pass
        x = torch.randn(32, 784)
        output = mlp(x)
        assert output.shape == (32, 10), f"MLP output shape should be (32, 10), got {output.shape}"
        
        # Test that network has multiple layers
        params = list(mlp.parameters())
        assert len(params) >= 4, "MLP should have multiple layers (at least 2 linear layers)"
    else:
        raise AssertionError("MLP class not found. Please implement MLP architecture")


def test_activation_function_usage(namespace):
    """Test usage of different activation functions"""
    import torch
    import torch.nn.functional as F
    
    # Test that different activations were explored
    activations = ['relu_result', 'sigmoid_result', 'tanh_result']
    
    for activation_name in activations:
        if activation_name in namespace:
            result = namespace[activation_name]
            assert torch.is_tensor(result), f"{activation_name} should be a tensor"
            
            # Check activation-specific properties
            if 'relu' in activation_name:
                assert torch.all(result >= 0), "ReLU output should be non-negative"
            elif 'sigmoid' in activation_name:
                assert torch.all((result >= 0) & (result <= 1)), "Sigmoid output should be in [0,1]"
            elif 'tanh' in activation_name:
                assert torch.all((result >= -1) & (result <= 1)), "Tanh output should be in [-1,1]"
        else:
            raise AssertionError(f"{activation_name} not found. Please test {activation_name.split('_')[0]} activation")


def test_weight_initialization_methods(namespace):
    """Test weight initialization implementations"""
    import torch
    import torch.nn as nn
    
    # Test initialization functions
    init_functions = ['xavier_init', 'he_init', 'normal_init']
    
    for init_name in init_functions:
        if init_name in namespace:
            init_func = namespace[init_name]
            assert callable(init_func), f"{init_name} should be a function"
            
            # Test the initialization
            layer = nn.Linear(100, 50)
            original_weight = layer.weight.clone()
            init_func(layer)
            
            # Weight should be modified
            assert not torch.allclose(layer.weight, original_weight), f"{init_name} should modify weights"
        else:
            print(f"Warning: {init_name} not found. Consider implementing different initialization methods")


def test_regularization_techniques(namespace):
    """Test regularization technique implementations"""
    import torch
    import torch.nn as nn
    
    # Test dropout usage
    if 'dropout_layer' in namespace:
        dropout = namespace['dropout_layer']
        assert isinstance(dropout, nn.Dropout), "dropout_layer should be nn.Dropout instance"
        
        # Test dropout behavior
        x = torch.ones(10, 20)
        dropout.train()
        output = dropout(x)
        zeros_count = (output == 0).sum().item()
        assert zeros_count > 0, "Dropout should zero some elements in training mode"
    else:
        raise AssertionError("dropout_layer not found. Please implement dropout regularization")
    
    # Test batch normalization
    if 'batch_norm' in namespace:
        bn = namespace['batch_norm']
        assert isinstance(bn, (nn.BatchNorm1d, nn.BatchNorm2d)), "batch_norm should be BatchNorm layer"
    else:
        print("Warning: batch_norm not found. Consider implementing batch normalization")


def test_mlp_training_step(namespace):
    """Test MLP training step implementation"""
    import torch
    import torch.nn as nn
    
    # Test training components
    if 'model' in namespace and 'optimizer' in namespace and 'criterion' in namespace:
        model = namespace['model']
        optimizer = namespace['optimizer']
        criterion = namespace['criterion']
        
        assert isinstance(model, nn.Module), "model should be nn.Module"
        assert hasattr(optimizer, 'step'), "optimizer should have step method"
        assert callable(criterion), "criterion should be callable"
        
        # Test a training step
        x = torch.randn(32, model.fc1.in_features if hasattr(model, 'fc1') else 784)
        y = torch.randint(0, 10, (32,))
        
        # Forward pass
        output = model(x)
        loss = criterion(output, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        assert loss.item() > 0, "Loss should be positive"
    else:
        raise AssertionError("Training components (model, optimizer, criterion) not found")


def test_network_depth_experiment(namespace):
    """Test experiments with different network depths"""
    import torch
    import torch.nn as nn
    
    # Test that multiple network architectures were compared
    if 'shallow_net' in namespace and 'deep_net' in namespace:
        shallow_net = namespace['shallow_net']
        deep_net = namespace['deep_net']
        
        # Count layers
        shallow_layers = sum(1 for _ in shallow_net.modules() if isinstance(_, nn.Linear))
        deep_layers = sum(1 for _ in deep_net.modules() if isinstance(_, nn.Linear))
        
        assert deep_layers > shallow_layers, "Deep network should have more layers than shallow network"
    else:
        raise AssertionError("Network depth comparison not found. Please compare shallow vs deep networks")


def test_activation_function_comparison(namespace):
    """Test comparison of different activation functions"""
    import torch
    
    # Test that activation function effects were studied
    if 'activation_comparison' in namespace:
        comparison = namespace['activation_comparison']
        assert isinstance(comparison, dict), "activation_comparison should be a dictionary"
        assert len(comparison) >= 2, "Should compare at least 2 activation functions"
    else:
        print("Warning: activation_comparison not found. Consider comparing different activation functions")


def run_tests():
    """Run all tests for Exercise 1"""
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    run_tests()