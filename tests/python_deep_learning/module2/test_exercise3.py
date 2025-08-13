#!/usr/bin/env python3
"""
Test suite for Python Deep Learning Module 2 - Exercise 3: Optimizer Implementation
"""

import sys
import torch
import numpy as np
import pytest
import math
from typing import Optional, Callable, Tuple, List


class TestExercise3:
    """Test cases for Exercise 3: Optimizer Implementation"""
    
    def setup_method(self):
        """Set up test fixtures"""
        torch.manual_seed(42)
        np.random.seed(42)
    
    def test_sgd_optimizer_implementation(self):
        """Test custom SGD optimizer implementation"""
        # Simple quadratic function: f(x) = (x-3)^2
        x = torch.tensor([0.0], requires_grad=True)
        learning_rate = 0.1
        
        # Custom SGD step
        def sgd_step(param, grad, lr):
            with torch.no_grad():
                param -= lr * grad
        
        # Run optimization steps
        losses = []
        for _ in range(50):
            loss = (x - 3)**2
            losses.append(loss.item())
            loss.backward()
            sgd_step(x, x.grad, learning_rate)
            x.grad.zero_()
        
        # Should converge towards x=3
        assert abs(x.item() - 3.0) < 0.1, f"SGD should converge close to 3.0, got {x.item()}"
        assert losses[-1] < losses[0], "Loss should decrease over time"
    
    def test_momentum_optimizer(self):
        """Test momentum optimizer implementation"""
        x = torch.tensor([0.0], requires_grad=True)
        lr = 0.01
        momentum = 0.9
        velocity = torch.zeros_like(x)
        
        def momentum_step(param, grad, velocity, lr, momentum):
            with torch.no_grad():
                velocity.mul_(momentum).add_(grad)
                param.sub_(lr, velocity)
        
        # Optimize quadratic function with momentum
        for _ in range(100):
            loss = (x - 5)**2
            loss.backward()
            momentum_step(x, x.grad, velocity, lr, momentum)
            x.grad.zero_()
        
        assert abs(x.item() - 5.0) < 0.1, f"Momentum should converge close to 5.0, got {x.item()}"
    
    def test_adam_optimizer_components(self):
        """Test understanding of Adam optimizer components"""
        # Test Adam's bias correction
        beta1, beta2 = 0.9, 0.999
        t = 10  # time step
        
        # Bias correction factors
        bias_correction1 = 1 - beta1**t
        bias_correction2 = 1 - beta2**t
        
        assert bias_correction1 > 0, "Bias correction 1 should be positive"
        assert bias_correction2 > 0, "Bias correction 2 should be positive"
        assert bias_correction1 < 1, "Bias correction 1 should be less than 1"
        assert bias_correction2 < 1, "Bias correction 2 should be less than 1"
    
    def test_learning_rate_scheduling(self):
        """Test learning rate scheduling implementations"""
        initial_lr = 0.1
        
        # Step decay
        def step_decay(epoch, initial_lr, step_size=10, gamma=0.1):
            return initial_lr * (gamma ** (epoch // step_size))
        
        # Exponential decay
        def exponential_decay(epoch, initial_lr, gamma=0.95):
            return initial_lr * (gamma ** epoch)
        
        # Test step decay
        lr_epoch_0 = step_decay(0, initial_lr)
        lr_epoch_10 = step_decay(10, initial_lr)
        assert lr_epoch_0 == initial_lr, "Initial learning rate should be unchanged"
        assert lr_epoch_10 < lr_epoch_0, "Learning rate should decrease after step"
        
        # Test exponential decay
        lr_exp_0 = exponential_decay(0, initial_lr)
        lr_exp_10 = exponential_decay(10, initial_lr)
        assert lr_exp_0 == initial_lr, "Initial learning rate should be unchanged"
        assert lr_exp_10 < lr_exp_0, "Learning rate should decay exponentially"
    
    def test_optimizer_comparison(self):
        """Test comparison between different optimizers"""
        # Define a more complex optimization surface
        def rosenbrock(x):
            return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2
        
        # Test different optimizers on same problem
        optimizers_results = {}
        
        for opt_name in ['SGD', 'Momentum', 'Adam']:
            x = torch.tensor([0.0, 0.0], requires_grad=True)
            
            if opt_name == 'SGD':
                optimizer = torch.optim.SGD([x], lr=0.001)
            elif opt_name == 'Momentum':
                optimizer = torch.optim.SGD([x], lr=0.001, momentum=0.9)
            else:  # Adam
                optimizer = torch.optim.Adam([x], lr=0.01)
            
            # Run optimization
            losses = []
            for _ in range(100):
                optimizer.zero_grad()
                loss = rosenbrock(x)
                losses.append(loss.item())
                loss.backward()
                optimizer.step()
            
            optimizers_results[opt_name] = {
                'final_x': x.detach().clone(),
                'final_loss': losses[-1],
                'losses': losses
            }
        
        # All optimizers should reduce loss
        for opt_name, results in optimizers_results.items():
            assert results['final_loss'] < results['losses'][0], f"{opt_name} should reduce loss"


def test_sgd_implementation(namespace):
    """Test SGD optimizer implementation"""
    import torch
    
    # Test SGD step function
    if 'sgd_step' in namespace:
        sgd_step = namespace['sgd_step']
        assert callable(sgd_step), "sgd_step should be a function"
        
        # Test the function
        x = torch.tensor([1.0], requires_grad=True)
        grad = torch.tensor([2.0])
        lr = 0.1
        original_x = x.clone()
        sgd_step(x, grad, lr)
        expected_x = original_x - lr * grad
        assert torch.allclose(x, expected_x), "SGD step should update parameter correctly"
    else:
        raise AssertionError("sgd_step function not found. Please implement SGD step")
    
    # Test convergence results
    if 'sgd_losses' in namespace:
        sgd_losses = namespace['sgd_losses']
        assert isinstance(sgd_losses, list), "sgd_losses should be a list"
        assert len(sgd_losses) > 10, "Should have multiple loss values"
        assert sgd_losses[-1] < sgd_losses[0], "Loss should decrease"
    else:
        raise AssertionError("sgd_losses not found. Please track losses during optimization")


def test_momentum_implementation(namespace):
    """Test momentum optimizer implementation"""
    import torch
    
    # Test momentum step function
    if 'momentum_step' in namespace:
        momentum_step = namespace['momentum_step']
        assert callable(momentum_step), "momentum_step should be a function"
    else:
        raise AssertionError("momentum_step function not found. Please implement momentum step")
    
    # Test velocity tracking
    if 'velocity' in namespace:
        velocity = namespace['velocity']
        assert torch.is_tensor(velocity), "velocity should be a tensor"
    else:
        raise AssertionError("velocity not found. Please track velocity for momentum")
    
    # Test convergence
    if 'momentum_losses' in namespace:
        momentum_losses = namespace['momentum_losses']
        assert isinstance(momentum_losses, list), "momentum_losses should be a list"
        assert momentum_losses[-1] < momentum_losses[0], "Momentum should improve convergence"
    else:
        raise AssertionError("momentum_losses not found. Please track losses with momentum")


def test_adam_implementation(namespace):
    """Test Adam optimizer components"""
    import torch
    
    # Test Adam step function or usage
    if 'adam_optimizer' in namespace:
        adam_optimizer = namespace['adam_optimizer']
        assert hasattr(adam_optimizer, 'step'), "adam_optimizer should have step method"
    else:
        raise AssertionError("adam_optimizer not found. Please create Adam optimizer")
    
    # Test bias correction understanding
    if 'bias_correction_1' in namespace and 'bias_correction_2' in namespace:
        bc1 = namespace['bias_correction_1']
        bc2 = namespace['bias_correction_2']
        assert bc1 > 0 and bc1 < 1, "bias_correction_1 should be in (0,1)"
        assert bc2 > 0 and bc2 < 1, "bias_correction_2 should be in (0,1)"
    else:
        raise AssertionError("bias_correction factors not found. Please compute bias corrections")
    
    # Test Adam convergence
    if 'adam_losses' in namespace:
        adam_losses = namespace['adam_losses']
        assert isinstance(adam_losses, list), "adam_losses should be a list"
        assert adam_losses[-1] < adam_losses[0], "Adam should reduce loss"
    else:
        raise AssertionError("adam_losses not found. Please track Adam optimization losses")


def test_learning_rate_scheduling(namespace):
    """Test learning rate scheduling implementations"""
    import torch
    
    # Test step decay
    if 'step_decay' in namespace:
        step_decay = namespace['step_decay']
        assert callable(step_decay), "step_decay should be a function"
        
        # Test step decay behavior
        lr_0 = step_decay(0, 0.1, 10, 0.5)
        lr_10 = step_decay(10, 0.1, 10, 0.5)
        assert lr_0 == 0.1, "Initial LR should be unchanged"
        assert lr_10 == 0.05, "LR should be halved after 10 steps"
    else:
        raise AssertionError("step_decay function not found. Please implement step decay")
    
    # Test exponential decay
    if 'exponential_decay' in namespace:
        exponential_decay = namespace['exponential_decay']
        assert callable(exponential_decay), "exponential_decay should be a function"
        
        # Test exponential decay behavior
        lr_0 = exponential_decay(0, 0.1, 0.9)
        lr_1 = exponential_decay(1, 0.1, 0.9)
        assert lr_0 == 0.1, "Initial LR should be unchanged"
        assert abs(lr_1 - 0.09) < 1e-6, "LR should decay exponentially"
    else:
        raise AssertionError("exponential_decay function not found. Please implement exponential decay")


def test_optimizer_comparison(namespace):
    """Test comparison between optimizers"""
    import torch
    
    # Test that different optimizers were compared
    if 'optimizer_results' in namespace:
        results = namespace['optimizer_results']
        assert isinstance(results, dict), "optimizer_results should be a dictionary"
        assert len(results) >= 2, "Should compare at least 2 optimizers"
        
        # Check that each optimizer has results
        for opt_name, opt_results in results.items():
            assert 'losses' in opt_results, f"{opt_name} should have losses tracked"
            assert isinstance(opt_results['losses'], list), f"{opt_name} losses should be a list"
            assert len(opt_results['losses']) > 0, f"{opt_name} should have loss history"
    else:
        raise AssertionError("optimizer_results not found. Please compare different optimizers")


def test_hyperparameter_sensitivity(namespace):
    """Test understanding of hyperparameter sensitivity"""
    import torch
    
    # Test learning rate sensitivity analysis
    if 'lr_sensitivity_results' in namespace:
        lr_results = namespace['lr_sensitivity_results']
        assert isinstance(lr_results, dict), "lr_sensitivity_results should be a dictionary"
        assert len(lr_results) >= 3, "Should test multiple learning rates"
        
        # Check that different learning rates give different results
        loss_values = [results['final_loss'] for results in lr_results.values()]
        assert len(set(loss_values)) > 1, "Different learning rates should give different final losses"
    else:
        raise AssertionError("lr_sensitivity_results not found. Please analyze learning rate sensitivity")


def test_custom_optimizer_implementation(namespace):
    """Test implementation of custom optimizer"""
    import torch
    
    # Test custom optimizer class or function
    if 'CustomOptimizer' in namespace:
        CustomOptimizer = namespace['CustomOptimizer']
        
        # Try to instantiate and use the optimizer
        x = torch.tensor([1.0], requires_grad=True)
        optimizer = CustomOptimizer([x])
        assert hasattr(optimizer, 'step'), "CustomOptimizer should have step method"
        assert hasattr(optimizer, 'zero_grad'), "CustomOptimizer should have zero_grad method"
        
    elif 'custom_optimizer_step' in namespace:
        custom_step = namespace['custom_optimizer_step']
        assert callable(custom_step), "custom_optimizer_step should be a function"
    else:
        raise AssertionError("Custom optimizer implementation not found")


def run_tests():
    """Run all tests for Exercise 3"""
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    run_tests()