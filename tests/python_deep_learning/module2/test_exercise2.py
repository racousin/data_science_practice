#!/usr/bin/env python3
"""
Test suite for Python Deep Learning Module 2 - Exercise 2: Gradient Analysis
"""

import sys
import torch
import numpy as np
import pytest
import math
from typing import Optional, Callable, Tuple, List


class TestExercise2:
    """Test cases for Exercise 2: Gradient Analysis"""
    
    def setup_method(self):
        """Set up test fixtures"""
        torch.manual_seed(42)
        np.random.seed(42)
    
    def test_gradient_computation_accuracy(self):
        """Test numerical vs analytical gradient computation"""
        def numerical_gradient(f, x, h=1e-5):
            """Compute numerical gradient using finite differences"""
            grad = torch.zeros_like(x)
            for i in range(x.numel()):
                x_plus = x.clone()
                x_minus = x.clone()
                x_plus.view(-1)[i] += h
                x_minus.view(-1)[i] -= h
                grad.view(-1)[i] = (f(x_plus) - f(x_minus)) / (2 * h)
            return grad
        
        # Test function: f(x) = x^3 + 2*x^2 + x + 1
        def test_function(x):
            return torch.sum(x**3 + 2*x**2 + x + 1)
        
        x = torch.tensor([1.0, 2.0], requires_grad=True)
        
        # Analytical gradient
        loss = test_function(x)
        loss.backward()
        analytical_grad = x.grad.clone()
        
        # Numerical gradient
        x_no_grad = x.detach()
        numerical_grad = numerical_gradient(test_function, x_no_grad)
        
        # Should be close (within tolerance)
        assert torch.allclose(analytical_grad, numerical_grad, atol=1e-4), \
            f"Analytical and numerical gradients should match. Analytical: {analytical_grad}, Numerical: {numerical_grad}"
    
    def test_gradient_flow_through_operations(self):
        """Test gradient flow through various operations"""
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        
        # Test various operations
        y1 = torch.sum(x)  # Sum
        y2 = torch.mean(x)  # Mean
        y3 = torch.max(x)  # Max
        y4 = torch.norm(x)  # L2 norm
        
        # Test gradients for sum
        y1.backward(retain_graph=True)
        expected_grad_sum = torch.ones_like(x)
        assert torch.allclose(x.grad, expected_grad_sum), "Gradient of sum should be all ones"
        
        # Reset gradients and test mean
        x.grad.zero_()
        y2.backward(retain_graph=True)
        expected_grad_mean = torch.ones_like(x) / x.numel()
        assert torch.allclose(x.grad, expected_grad_mean), "Gradient of mean should be 1/n for all elements"
    
    def test_gradient_checkpointing_understanding(self):
        """Test understanding of gradient checkpointing concepts"""
        # Simulate a deep computation that would benefit from checkpointing
        x = torch.tensor(1.0, requires_grad=True)
        
        # Deep computation chain
        y = x
        for i in range(10):
            y = y**2 + 0.1*y  # Repeated squaring with linear term
        
        y.backward()
        
        # Should have computed gradient without error
        assert x.grad is not None, "Gradient should be computed"
        assert torch.isfinite(x.grad), "Gradient should be finite"
    
    def test_gradient_clipping_necessity(self):
        """Test scenarios where gradient clipping is necessary"""
        # Create a scenario prone to exploding gradients
        x = torch.tensor(10.0, requires_grad=True)
        
        # Exponential function leads to large gradients
        y = torch.exp(x)
        y.backward()
        
        original_grad = x.grad.clone()
        
        # Test gradient clipping
        clipped_grad = torch.clamp(original_grad, -1.0, 1.0)
        
        assert torch.abs(clipped_grad) <= 1.0, "Clipped gradient should be within bounds"
        assert torch.abs(original_grad) > 1.0, "Original gradient should be large (demonstrating need for clipping)"


def test_numerical_vs_analytical_gradients(namespace):
    """Test numerical vs analytical gradient comparison"""
    import torch
    
    # Test function definition
    if 'test_func' in namespace:
        test_func = namespace['test_func']
        x_test = torch.tensor([1.0, 2.0], requires_grad=True)
        result = test_func(x_test)
        assert torch.is_tensor(result), "test_func should return a tensor"
    else:
        raise AssertionError("test_func not found. Please define a test function")
    
    # Test analytical gradient computation
    if 'analytical_grad' in namespace:
        analytical_grad = namespace['analytical_grad']
        assert torch.is_tensor(analytical_grad), "analytical_grad should be a tensor"
        assert analytical_grad.shape == torch.Size([2]), "analytical_grad should have shape matching input"
    else:
        raise AssertionError("analytical_grad not found. Please compute analytical gradient using autograd")
    
    # Test numerical gradient computation
    if 'numerical_grad' in namespace:
        numerical_grad = namespace['numerical_grad']
        assert torch.is_tensor(numerical_grad), "numerical_grad should be a tensor"
        assert numerical_grad.shape == torch.Size([2]), "numerical_grad should have shape matching input"
    else:
        raise AssertionError("numerical_grad not found. Please implement numerical gradient computation")
    
    # Test that gradients are close
    if 'gradients_close' in namespace:
        gradients_close = namespace['gradients_close']
        assert gradients_close, "Analytical and numerical gradients should be close"
    else:
        raise AssertionError("gradients_close not found. Please compare the gradients")


def test_gradient_flow_analysis(namespace):
    """Test gradient flow through different operations"""
    import torch
    
    # Test input tensor
    if 'flow_x' in namespace:
        flow_x = namespace['flow_x']
        assert flow_x.requires_grad, "flow_x should require gradients"
        assert flow_x.shape == torch.Size([2, 2]), "flow_x should be 2x2 matrix"
    else:
        raise AssertionError("flow_x not found. Please create a 2x2 tensor with requires_grad=True")
    
    # Test different operations and their gradients
    operations = ['sum_grad', 'mean_grad', 'norm_grad']
    for op_name in operations:
        if op_name in namespace:
            grad = namespace[op_name]
            assert torch.is_tensor(grad), f"{op_name} should be a tensor"
            assert grad.shape == flow_x.shape, f"{op_name} should have same shape as input"
        else:
            raise AssertionError(f"{op_name} not found. Please compute gradients for different operations")


def test_activation_function_gradients(namespace):
    """Test gradients of common activation functions"""
    import torch
    
    # Test activation functions and their gradients
    activations = ['relu_grad', 'sigmoid_grad', 'tanh_grad']
    
    for activation_name in activations:
        if activation_name in namespace:
            grad = namespace[activation_name]
            assert torch.is_tensor(grad), f"{activation_name} should be a tensor"
        else:
            raise AssertionError(f"{activation_name} not found. Please compute gradient of {activation_name.split('_')[0]} activation")
    
    # Test specific properties
    if 'relu_grad' in namespace:
        relu_grad = namespace['relu_grad']
        # ReLU gradient should be 0 or 1
        assert torch.all((relu_grad == 0) | (relu_grad == 1)), "ReLU gradient should be 0 or 1"
    
    if 'sigmoid_grad' in namespace:
        sigmoid_grad = namespace['sigmoid_grad']
        # Sigmoid gradient should be between 0 and 0.25
        assert torch.all((sigmoid_grad >= 0) & (sigmoid_grad <= 0.25)), "Sigmoid gradient should be in [0, 0.25]"


def test_loss_function_gradients(namespace):
    """Test gradients of different loss functions"""
    import torch
    
    # Test MSE loss gradient
    if 'mse_grad' in namespace:
        mse_grad = namespace['mse_grad']
        assert torch.is_tensor(mse_grad), "mse_grad should be a tensor"
    else:
        raise AssertionError("mse_grad not found. Please compute MSE loss gradient")
    
    # Test Cross Entropy loss gradient
    if 'ce_grad' in namespace:
        ce_grad = namespace['ce_grad']
        assert torch.is_tensor(ce_grad), "ce_grad should be a tensor"
    else:
        raise AssertionError("ce_grad not found. Please compute Cross Entropy loss gradient")


def test_gradient_vanishing_exploding(namespace):
    """Test understanding of vanishing and exploding gradients"""
    import torch
    
    # Test deep network gradient magnitudes
    if 'deep_gradients' in namespace:
        deep_gradients = namespace['deep_gradients']
        assert isinstance(deep_gradients, list), "deep_gradients should be a list"
        assert len(deep_gradients) > 0, "deep_gradients should not be empty"
        
        # Check gradient magnitudes
        grad_magnitudes = [torch.norm(grad).item() for grad in deep_gradients if grad is not None]
        assert len(grad_magnitudes) > 0, "Should have computed some gradient magnitudes"
        
    else:
        raise AssertionError("deep_gradients not found. Please analyze gradients in a deep network")
    
    # Test gradient clipping
    if 'clipped_gradients' in namespace:
        clipped_gradients = namespace['clipped_gradients']
        assert isinstance(clipped_gradients, list), "clipped_gradients should be a list"
        
        # Check that clipped gradients are within bounds
        for grad in clipped_gradients:
            if grad is not None:
                assert torch.norm(grad) <= 1.0, "Clipped gradients should have norm <= 1.0"
    else:
        raise AssertionError("clipped_gradients not found. Please implement gradient clipping")


def test_custom_backward_function(namespace):
    """Test implementation of custom backward functions"""
    import torch
    
    # Test custom function implementation
    if 'custom_function_result' in namespace:
        custom_function_result = namespace['custom_function_result']
        assert torch.is_tensor(custom_function_result), "custom_function_result should be a tensor"
        assert custom_function_result.requires_grad, "custom_function_result should require gradients"
    else:
        raise AssertionError("custom_function_result not found. Please implement a custom autograd function")
    
    # Test that custom gradients work
    if 'custom_grad' in namespace:
        custom_grad = namespace['custom_grad']
        assert torch.is_tensor(custom_grad), "custom_grad should be a tensor"
        assert torch.isfinite(custom_grad).all(), "custom_grad should be finite"
    else:
        raise AssertionError("custom_grad not found. Please compute gradients through custom function")


def run_tests():
    """Run all tests for Exercise 2"""
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    run_tests()