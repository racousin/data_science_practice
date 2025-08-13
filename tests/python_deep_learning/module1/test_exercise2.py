#!/usr/bin/env python3
"""
Test suite for Python Deep Learning Module 1 - Exercise 2: Mathematical Implementation
"""

import sys
import torch
import numpy as np
import pytest
import matplotlib.pyplot as plt
from typing import Optional, Callable, Tuple, List
import math

class TestExercise2:
    """Test cases for Exercise 2: Mathematical Implementation"""
    
    def setup_method(self):
        """Set up test fixtures"""
        torch.manual_seed(42)
        np.random.seed(42)
    
    def test_numpy_gradient_descent(self):
        """Test NumPy gradient descent implementation"""
        def quadratic_function(x: np.ndarray) -> float:
            return (x[0] - 3)**2 + (x[1] - 2)**2
        
        def quadratic_gradient(x: np.ndarray) -> np.ndarray:
            return np.array([2*(x[0] - 3), 2*(x[1] - 2)])
        
        # Test basic gradient descent
        def gradient_descent_numpy(func, grad_func, initial_point, learning_rate, num_iterations):
            current_point = initial_point.copy()
            trajectory = []
            losses = []
            
            for i in range(num_iterations):
                current_loss = func(current_point)
                current_gradient = grad_func(current_point)
                
                trajectory.append(current_point.copy())
                losses.append(current_loss)
                
                current_point = current_point - learning_rate * current_gradient
            
            return current_point, trajectory, losses
        
        initial_point = np.array([0.0, 0.0])
        final_point, trajectory, losses = gradient_descent_numpy(
            quadratic_function, quadratic_gradient, initial_point, 0.1, 50
        )
        
        # Check convergence
        assert len(trajectory) == 50, "Should have 50 trajectory points"
        assert len(losses) == 50, "Should have 50 loss values"
        assert losses[-1] < 0.1, "Should converge to low loss"
        
        # Check final point is close to minimum [3, 2]
        assert abs(final_point[0] - 3.0) < 0.1, "x[0] should be close to 3"
        assert abs(final_point[1] - 2.0) < 0.1, "x[1] should be close to 2"
    
    def test_torch_gradient_descent(self):
        """Test PyTorch gradient descent implementation"""
        def quadratic_function_torch(x: torch.Tensor) -> torch.Tensor:
            return (x[0] - 3)**2 + (x[1] - 2)**2
        
        def quadratic_gradient_torch(x: torch.Tensor) -> torch.Tensor:
            return torch.tensor([2*(x[0] - 3), 2*(x[1] - 2)], dtype=x.dtype)
        
        # Test basic gradient descent with PyTorch
        def gradient_descent_torch(func, grad_func, initial_point, learning_rate, num_iterations):
            current_point = initial_point.clone()
            trajectory = []
            losses = []
            
            for i in range(num_iterations):
                current_loss = func(current_point)
                current_gradient = grad_func(current_point)
                
                trajectory.append(current_point.clone())
                losses.append(current_loss.item())
                
                current_point = current_point - learning_rate * current_gradient
            
            return current_point, trajectory, losses
        
        initial_point = torch.tensor([0.0, 0.0], dtype=torch.float32)
        final_point, trajectory, losses = gradient_descent_torch(
            quadratic_function_torch, quadratic_gradient_torch, initial_point, 0.1, 50
        )
        
        # Check convergence
        assert len(trajectory) == 50, "Should have 50 trajectory points"
        assert len(losses) == 50, "Should have 50 loss values"
        assert losses[-1] < 0.1, "Should converge to low loss"
        
        # Check final point is close to minimum [3, 2]
        assert abs(final_point[0].item() - 3.0) < 0.1, "x[0] should be close to 3"
        assert abs(final_point[1].item() - 2.0) < 0.1, "x[1] should be close to 2"
    
    def test_sgd_momentum(self):
        """Test SGD with momentum implementation"""
        def rosenbrock_function(x: np.ndarray) -> float:
            return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2
        
        def rosenbrock_gradient(x: np.ndarray) -> np.ndarray:
            dx0 = -2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2)
            dx1 = 200*(x[1] - x[0]**2)
            return np.array([dx0, dx1])
        
        # Test SGD with momentum
        def sgd_with_momentum(func, grad_func, initial_point, learning_rate, momentum, num_iterations):
            current_point = initial_point.copy()
            velocity = np.zeros_like(current_point)
            trajectory = []
            losses = []
            
            for i in range(num_iterations):
                current_loss = func(current_point)
                current_gradient = grad_func(current_point)
                
                trajectory.append(current_point.copy())
                losses.append(current_loss)
                
                velocity = momentum * velocity + learning_rate * current_gradient
                current_point = current_point - velocity
            
            return current_point, trajectory, losses
        
        initial_point = np.array([-1.0, 1.0])
        final_point, trajectory, losses = sgd_with_momentum(
            rosenbrock_function, rosenbrock_gradient, initial_point, 0.001, 0.9, 100
        )
        
        # Check that momentum helps convergence
        assert len(trajectory) == 100, "Should have 100 trajectory points"
        assert len(losses) == 100, "Should have 100 loss values"
        assert losses[-1] < losses[0], "Loss should decrease"
    
    def test_adaptive_optimizers(self):
        """Test AdaGrad and RMSprop implementations"""
        def quadratic_function(x: np.ndarray) -> float:
            return (x[0] - 3)**2 + (x[1] - 2)**2
        
        def quadratic_gradient(x: np.ndarray) -> np.ndarray:
            return np.array([2*(x[0] - 3), 2*(x[1] - 2)])
        
        # Test AdaGrad
        def adagrad_optimizer(func, grad_func, initial_point, learning_rate, epsilon, num_iterations):
            current_point = initial_point.copy()
            sum_squared_gradients = np.zeros_like(current_point)
            trajectory = []
            losses = []
            
            for i in range(num_iterations):
                current_loss = func(current_point)
                current_gradient = grad_func(current_point)
                
                trajectory.append(current_point.copy())
                losses.append(current_loss)
                
                sum_squared_gradients += current_gradient ** 2
                adaptive_lr = learning_rate / (np.sqrt(sum_squared_gradients) + epsilon)
                current_point = current_point - adaptive_lr * current_gradient
            
            return current_point, trajectory, losses
        
        # Test RMSprop  
        def rmsprop_optimizer(func, grad_func, initial_point, learning_rate, beta, epsilon, num_iterations):
            current_point = initial_point.copy()
            moving_avg_squared_grad = np.zeros_like(current_point)
            trajectory = []
            losses = []
            
            for i in range(num_iterations):
                current_loss = func(current_point)
                current_gradient = grad_func(current_point)
                
                trajectory.append(current_point.copy())
                losses.append(current_loss)
                
                moving_avg_squared_grad = beta * moving_avg_squared_grad + (1 - beta) * current_gradient ** 2
                adaptive_lr = learning_rate / (np.sqrt(moving_avg_squared_grad) + epsilon)
                current_point = current_point - adaptive_lr * current_gradient
            
            return current_point, trajectory, losses
        
        initial_point = np.array([0.0, 0.0])
        
        # Test AdaGrad
        adagrad_final, adagrad_trajectory, adagrad_losses = adagrad_optimizer(
            quadratic_function, quadratic_gradient, initial_point.copy(), 0.1, 1e-8, 50
        )
        
        # Test RMSprop
        rmsprop_final, rmsprop_trajectory, rmsprop_losses = rmsprop_optimizer(
            quadratic_function, quadratic_gradient, initial_point.copy(), 0.01, 0.9, 1e-8, 50
        )
        
        # Check both converge
        assert adagrad_losses[-1] < 0.5, "AdaGrad should converge"
        assert rmsprop_losses[-1] < 0.5, "RMSprop should converge"
    
    def test_learning_rate_schedules(self):
        """Test learning rate scheduling implementations"""
        # Test step schedule
        def step_schedule(initial_lr: float, step_size: int, gamma: float, epoch: int) -> float:
            return initial_lr * (gamma ** (epoch // step_size))
        
        # Test exponential schedule
        def exponential_schedule(initial_lr: float, gamma: float, epoch: int) -> float:
            return initial_lr * (gamma ** epoch)
        
        # Test cosine schedule
        def cosine_schedule(initial_lr: float, max_epochs: int, epoch: int) -> float:
            return initial_lr * 0.5 * (1 + math.cos(math.pi * epoch / max_epochs))
        
        # Test that schedules return expected values
        initial_lr = 0.01
        
        # Step schedule tests
        assert abs(step_schedule(initial_lr, 10, 0.5, 0) - initial_lr) < 1e-6
        assert abs(step_schedule(initial_lr, 10, 0.5, 10) - initial_lr * 0.5) < 1e-6
        assert abs(step_schedule(initial_lr, 10, 0.5, 20) - initial_lr * 0.25) < 1e-6
        
        # Exponential schedule tests
        assert abs(exponential_schedule(initial_lr, 0.9, 0) - initial_lr) < 1e-6
        assert exponential_schedule(initial_lr, 0.9, 10) < initial_lr
        
        # Cosine schedule tests
        assert abs(cosine_schedule(initial_lr, 100, 0) - initial_lr) < 1e-6
        assert cosine_schedule(initial_lr, 100, 50) < initial_lr
        assert cosine_schedule(initial_lr, 100, 100) < initial_lr
    
    def test_visualization(self):
        """Test that visualization function can be called without errors"""
        # Simple test that visualization function exists and can handle basic input
        def quadratic_function(x: np.ndarray) -> float:
            return (x[0] - 3)**2 + (x[1] - 2)**2
        
        # Create a simple trajectory
        trajectory = [np.array([0.0, 0.0]), np.array([1.0, 1.0]), np.array([2.0, 1.5])]
        
        # Test that plot function can handle the data
        def plot_optimization_trajectory(func, trajectory, title="Test"):
            if len(trajectory) == 0:
                return
            
            # Convert trajectory to numpy if needed
            if isinstance(trajectory[0], torch.Tensor):
                trajectory_np = [point.numpy() for point in trajectory]
            else:
                trajectory_np = trajectory
            
            # Create grid
            trajectory_array = np.array(trajectory_np)
            x_min, x_max = trajectory_array[:, 0].min() - 1, trajectory_array[:, 0].max() + 1
            y_min, y_max = trajectory_array[:, 1].min() - 1, trajectory_array[:, 1].max() + 1
            
            x = np.linspace(x_min, x_max, 10)
            y = np.linspace(y_min, y_max, 10)
            X, Y = np.meshgrid(x, y)
            
            # Evaluate function on grid
            Z = np.zeros_like(X)
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    Z[i, j] = func(np.array([X[i, j], Y[i, j]]))
            
            # Should be able to create plot without errors
            assert Z.shape == X.shape, "Function evaluation should match grid shape"
            assert len(trajectory_np) > 0, "Trajectory should not be empty"
        
        # Test the function
        plot_optimization_trajectory(quadratic_function, trajectory)


def test_numpy_gradient_descent(namespace):
    """Test function for numpy gradient descent that can be called directly from notebooks"""
    # Check if gradient_descent_numpy function exists
    if 'gradient_descent_numpy' not in namespace:
        raise AssertionError("gradient_descent_numpy function not found. Please implement the gradient descent function.")
    
    func = namespace['gradient_descent_numpy']
    
    # Test with simple quadratic function
    def quadratic_function(x: np.ndarray) -> float:
        return (x[0] - 3)**2 + (x[1] - 2)**2
    
    def quadratic_gradient(x: np.ndarray) -> np.ndarray:
        return np.array([2*(x[0] - 3), 2*(x[1] - 2)])
    
    initial_point = np.array([0.0, 0.0])
    final_point, trajectory, losses = func(
        quadratic_function, quadratic_gradient, initial_point, 0.1, 50
    )
    
    # Check outputs
    assert isinstance(final_point, np.ndarray), "Final point should be numpy array"
    assert isinstance(trajectory, list), "Trajectory should be a list"
    assert isinstance(losses, list), "Losses should be a list"
    assert len(trajectory) == 50, "Should have 50 trajectory points"
    assert len(losses) == 50, "Should have 50 loss values"
    
    # Check convergence
    assert losses[-1] < losses[0], "Loss should decrease"
    assert abs(final_point[0] - 3.0) < 0.5, "Should converge towards x[0] = 3"
    assert abs(final_point[1] - 2.0) < 0.5, "Should converge towards x[1] = 2"


def test_torch_gradient_descent(namespace):
    """Test function for torch gradient descent that can be called directly from notebooks"""
    if 'gradient_descent_torch' not in namespace:
        raise AssertionError("gradient_descent_torch function not found. Please implement the PyTorch gradient descent function.")
    
    func = namespace['gradient_descent_torch']
    
    # Test with simple quadratic function
    def quadratic_function_torch(x: torch.Tensor) -> torch.Tensor:
        return (x[0] - 3)**2 + (x[1] - 2)**2
    
    def quadratic_gradient_torch(x: torch.Tensor) -> torch.Tensor:
        return torch.tensor([2*(x[0] - 3), 2*(x[1] - 2)], dtype=x.dtype)
    
    initial_point = torch.tensor([0.0, 0.0], dtype=torch.float32)
    final_point, trajectory, losses = func(
        quadratic_function_torch, quadratic_gradient_torch, initial_point, 0.1, 50
    )
    
    # Check outputs
    assert isinstance(final_point, torch.Tensor), "Final point should be torch tensor"
    assert isinstance(trajectory, list), "Trajectory should be a list"
    assert isinstance(losses, list), "Losses should be a list"
    assert len(trajectory) == 50, "Should have 50 trajectory points"
    assert len(losses) == 50, "Should have 50 loss values"
    
    # Check convergence
    assert losses[-1] < losses[0], "Loss should decrease"
    assert abs(final_point[0].item() - 3.0) < 0.5, "Should converge towards x[0] = 3"
    assert abs(final_point[1].item() - 2.0) < 0.5, "Should converge towards x[1] = 2"


def test_visualization(namespace):
    """Test function for visualization that can be called directly from notebooks"""
    if 'plot_optimization_trajectory' not in namespace:
        raise AssertionError("plot_optimization_trajectory function not found. Please implement the visualization function.")
    
    func = namespace['plot_optimization_trajectory']
    
    # Test with simple data
    def simple_function(x):
        return x[0]**2 + x[1]**2
    
    trajectory = [np.array([0.0, 0.0]), np.array([1.0, 1.0])]
    
    # Should not raise an error
    try:
        func(simple_function, trajectory, "Test Plot")
    except Exception as e:
        # Allow matplotlib-related errors in testing environment
        if "display" not in str(e).lower() and "tkinter" not in str(e).lower():
            raise AssertionError(f"Visualization function failed: {e}")


def test_sgd_momentum(namespace):
    """Test function for SGD with momentum that can be called directly from notebooks"""
    if 'sgd_with_momentum' not in namespace:
        raise AssertionError("sgd_with_momentum function not found. Please implement the SGD with momentum function.")
    
    func = namespace['sgd_with_momentum']
    
    # Test with simple quadratic function
    def quadratic_function(x: np.ndarray) -> float:
        return (x[0] - 3)**2 + (x[1] - 2)**2
    
    def quadratic_gradient(x: np.ndarray) -> np.ndarray:
        return np.array([2*(x[0] - 3), 2*(x[1] - 2)])
    
    initial_point = np.array([0.0, 0.0])
    final_point, trajectory, losses = func(
        quadratic_function, quadratic_gradient, initial_point, 0.1, 0.9, 50
    )
    
    # Check outputs
    assert isinstance(final_point, np.ndarray), "Final point should be numpy array"
    assert isinstance(trajectory, list), "Trajectory should be a list"
    assert isinstance(losses, list), "Losses should be a list"
    assert len(trajectory) == 50, "Should have 50 trajectory points"
    assert len(losses) == 50, "Should have 50 loss values"
    
    # Check convergence
    assert losses[-1] < losses[0], "Loss should decrease"


def test_adaptive_optimizers(namespace):
    """Test function for adaptive optimizers that can be called directly from notebooks"""
    if 'adagrad_optimizer' not in namespace:
        raise AssertionError("adagrad_optimizer function not found. Please implement the AdaGrad optimizer.")
    
    if 'rmsprop_optimizer' not in namespace:
        raise AssertionError("rmsprop_optimizer function not found. Please implement the RMSprop optimizer.")
    
    adagrad_func = namespace['adagrad_optimizer']
    rmsprop_func = namespace['rmsprop_optimizer']
    
    # Test with simple quadratic function
    def quadratic_function(x: np.ndarray) -> float:
        return (x[0] - 3)**2 + (x[1] - 2)**2
    
    def quadratic_gradient(x: np.ndarray) -> np.ndarray:
        return np.array([2*(x[0] - 3), 2*(x[1] - 2)])
    
    initial_point = np.array([0.0, 0.0])
    
    # Test AdaGrad
    adagrad_final, adagrad_trajectory, adagrad_losses = adagrad_func(
        quadratic_function, quadratic_gradient, initial_point.copy(), 0.1, 1e-8, 50
    )
    
    # Test RMSprop
    rmsprop_final, rmsprop_trajectory, rmsprop_losses = rmsprop_func(
        quadratic_function, quadratic_gradient, initial_point.copy(), 0.01, 0.9, 1e-8, 50
    )
    
    # Check AdaGrad outputs
    assert isinstance(adagrad_final, np.ndarray), "AdaGrad final point should be numpy array"
    assert len(adagrad_trajectory) == 50, "AdaGrad should have 50 trajectory points"
    assert len(adagrad_losses) == 50, "AdaGrad should have 50 loss values"
    assert adagrad_losses[-1] < adagrad_losses[0], "AdaGrad loss should decrease"
    
    # Check RMSprop outputs
    assert isinstance(rmsprop_final, np.ndarray), "RMSprop final point should be numpy array"
    assert len(rmsprop_trajectory) == 50, "RMSprop should have 50 trajectory points"
    assert len(rmsprop_losses) == 50, "RMSprop should have 50 loss values"
    assert rmsprop_losses[-1] < rmsprop_losses[0], "RMSprop loss should decrease"


def test_learning_rate_schedules(namespace):
    """Test function for learning rate schedules that can be called directly from notebooks"""
    required_functions = ['step_schedule', 'exponential_schedule', 'cosine_schedule', 'sgd_with_schedule']
    
    for func_name in required_functions:
        if func_name not in namespace:
            raise AssertionError(f"{func_name} function not found. Please implement the learning rate schedule function.")
    
    step_func = namespace['step_schedule']
    exp_func = namespace['exponential_schedule']
    cos_func = namespace['cosine_schedule']
    sgd_sched_func = namespace['sgd_with_schedule']
    
    # Test schedule functions
    initial_lr = 0.01
    
    # Test step schedule
    lr_step_0 = step_func(initial_lr, 10, 0.5, 0)
    lr_step_10 = step_func(initial_lr, 10, 0.5, 10)
    
    assert abs(lr_step_0 - initial_lr) < 1e-6, "Step schedule should return initial_lr at epoch 0"
    assert lr_step_10 < lr_step_0, "Step schedule should decay learning rate"
    
    # Test exponential schedule
    lr_exp_0 = exp_func(initial_lr, 0.9, 0)
    lr_exp_10 = exp_func(initial_lr, 0.9, 10)
    
    assert abs(lr_exp_0 - initial_lr) < 1e-6, "Exponential schedule should return initial_lr at epoch 0"
    assert lr_exp_10 < lr_exp_0, "Exponential schedule should decay learning rate"
    
    # Test cosine schedule
    lr_cos_0 = cos_func(initial_lr, 100, 0)
    lr_cos_50 = cos_func(initial_lr, 100, 50)
    
    assert abs(lr_cos_0 - initial_lr) < 1e-6, "Cosine schedule should return initial_lr at epoch 0"
    assert lr_cos_50 < lr_cos_0, "Cosine schedule should decay learning rate"
    
    # Test SGD with schedule
    def quadratic_function(x: np.ndarray) -> float:
        return (x[0] - 3)**2 + (x[1] - 2)**2
    
    def quadratic_gradient(x: np.ndarray) -> np.ndarray:
        return np.array([2*(x[0] - 3), 2*(x[1] - 2)])
    
    schedule_func = lambda epoch: step_func(initial_lr, 10, 0.5, epoch)
    initial_point = np.array([0.0, 0.0])
    
    final_point, trajectory, losses, learning_rates = sgd_sched_func(
        quadratic_function, quadratic_gradient, initial_point, initial_lr, schedule_func, 25
    )
    
    # Check outputs
    assert isinstance(final_point, np.ndarray), "Final point should be numpy array"
    assert len(trajectory) == 25, "Should have 25 trajectory points"
    assert len(losses) == 25, "Should have 25 loss values"
    assert len(learning_rates) == 25, "Should have 25 learning rates"
    
    # Check that learning rates change according to schedule
    assert learning_rates[0] == initial_lr, "First learning rate should be initial_lr"
    assert learning_rates[10] < learning_rates[0], "Learning rate should decay according to schedule"


def run_tests():
    """Run all tests for Exercise 2"""
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    run_tests()