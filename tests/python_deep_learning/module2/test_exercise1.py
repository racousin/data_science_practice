#!/usr/bin/env python3
"""
Test suite for Python Deep Learning Module 2 - Exercise 1: Autograd Exploration
"""

import sys
import torch
import numpy as np
import pytest
import math
from typing import Optional, Callable, Tuple, List


class TestExercise1:
    """Test cases for Exercise 1: Autograd Exploration"""
    
    def setup_method(self):
        """Set up test fixtures"""
        torch.manual_seed(42)
        np.random.seed(42)
    
    def test_basic_autograd(self):
        """Test basic autograd functionality"""
        # Test simple gradient computation
        x = torch.tensor(2.0, requires_grad=True)
        y = x**2 + 3*x + 1
        y.backward()
        
        # Expected gradient: dy/dx = 2x + 3 = 2*2 + 3 = 7
        expected_grad = 2*2 + 3
        assert torch.isclose(x.grad, torch.tensor(expected_grad)), f"Expected gradient {expected_grad}, got {x.grad}"
    
    def test_multivariable_gradients(self):
        """Test gradients with multiple variables"""
        x = torch.tensor(1.0, requires_grad=True)
        y = torch.tensor(2.0, requires_grad=True)
        z = x**2 + y**3 + x*y
        z.backward()
        
        # Expected gradients: dz/dx = 2x + y = 2*1 + 2 = 4, dz/dy = 3y^2 + x = 3*4 + 1 = 13
        assert torch.isclose(x.grad, torch.tensor(4.0)), f"Expected x gradient 4.0, got {x.grad}"
        assert torch.isclose(y.grad, torch.tensor(13.0)), f"Expected y gradient 13.0, got {y.grad}"
    
    def test_vector_gradients(self):
        """Test gradients with vector operations"""
        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = torch.sum(x**2)  # Sum of squares
        y.backward()
        
        # Expected gradient: dy/dx = 2x = [2, 4, 6]
        expected_grad = torch.tensor([2.0, 4.0, 6.0])
        assert torch.allclose(x.grad, expected_grad), f"Expected gradient {expected_grad}, got {x.grad}"
    
    def test_matrix_gradients(self):
        """Test gradients with matrix operations"""
        A = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        loss = torch.sum(A**2)
        loss.backward()
        
        # Expected gradient: dloss/dA = 2*A
        expected_grad = 2 * torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        assert torch.allclose(A.grad, expected_grad), f"Expected gradient {expected_grad}, got {A.grad}"
    
    def test_computational_graph(self):
        """Test understanding of computational graph"""
        x = torch.tensor(2.0, requires_grad=True)
        y = x**2
        z = 3*y + 1
        w = z**2
        w.backward()
        
        # Chain rule: dw/dx = dw/dz * dz/dy * dy/dx = 2z * 3 * 2x = 6z * 2x = 12zx
        # z = 3*y + 1 = 3*4 + 1 = 13, so dw/dx = 12*13*2 = 312
        expected_grad = 12 * 13 * 2
        assert torch.isclose(x.grad, torch.tensor(float(expected_grad))), f"Expected gradient {expected_grad}, got {x.grad}"


def test_basic_autograd_functions(namespace):
    """Test basic autograd functionality that can be called directly from notebooks"""
    import torch
    
    # Test requires_grad tensor creation
    if 'x' in namespace:
        x = namespace['x']
        assert x.requires_grad, "x should have requires_grad=True"
        assert x.item() == 2.0, "x should have value 2.0"
    else:
        raise AssertionError("x not found. Please create x = torch.tensor(2.0, requires_grad=True)")
    
    # Test function and gradient
    if 'y' in namespace:
        y = namespace['y']
        expected_y = 2.0**2 + 3*2.0 + 1  # 4 + 6 + 1 = 11
        assert torch.isclose(y, torch.tensor(expected_y)), f"y should equal {expected_y}, got {y}"
    else:
        raise AssertionError("y not found. Please compute y = x**2 + 3*x + 1")
    
    # Test gradient computation
    if hasattr(x, 'grad') and x.grad is not None:
        expected_grad = 2*2.0 + 3  # 7
        assert torch.isclose(x.grad, torch.tensor(expected_grad)), f"x.grad should be {expected_grad}, got {x.grad}"
    else:
        raise AssertionError("x.grad not found or is None. Please call y.backward() to compute gradients")


def test_multivariable_gradients_functions(namespace):
    """Test multivariable gradients that can be called directly from notebooks"""
    import torch
    
    # Test variable creation
    variables = ['x1', 'x2']
    for var_name in variables:
        if var_name in namespace:
            var = namespace[var_name]
            assert var.requires_grad, f"{var_name} should have requires_grad=True"
        else:
            raise AssertionError(f"{var_name} not found. Please create {var_name} with requires_grad=True")
    
    # Test function computation
    if 'z' in namespace:
        z = namespace['z']
        x1, x2 = namespace['x1'], namespace['x2']
        expected_z = x1**2 + x2**3 + x1*x2
        assert torch.isclose(z, expected_z), f"z should equal x1**2 + x2**3 + x1*x2"
    else:
        raise AssertionError("z not found. Please compute z = x1**2 + x2**3 + x1*x2")
    
    # Test gradients
    x1, x2 = namespace['x1'], namespace['x2']
    if x1.grad is not None and x2.grad is not None:
        # Expected gradients based on x1=1, x2=2: dx1 = 2*1 + 2 = 4, dx2 = 3*4 + 1 = 13
        expected_x1_grad = 2*x1.item() + x2.item()
        expected_x2_grad = 3*(x2.item()**2) + x1.item()
        
        assert torch.isclose(x1.grad, torch.tensor(expected_x1_grad)), f"x1.grad should be {expected_x1_grad}"
        assert torch.isclose(x2.grad, torch.tensor(expected_x2_grad)), f"x2.grad should be {expected_x2_grad}"
    else:
        raise AssertionError("Gradients not computed. Please call z.backward()")


def test_vector_gradients_functions(namespace):
    """Test vector gradients that can be called directly from notebooks"""
    import torch
    
    # Test vector creation
    if 'vec_x' in namespace:
        vec_x = namespace['vec_x']
        assert vec_x.requires_grad, "vec_x should have requires_grad=True"
        assert vec_x.shape == torch.Size([3]), "vec_x should have shape (3,)"
    else:
        raise AssertionError("vec_x not found. Please create a 3-element vector with requires_grad=True")
    
    # Test function
    if 'vec_loss' in namespace:
        vec_loss = namespace['vec_loss']
        expected_loss = torch.sum(vec_x**2)
        assert torch.isclose(vec_loss, expected_loss), "vec_loss should be sum of squares of vec_x"
    else:
        raise AssertionError("vec_loss not found. Please compute vec_loss = torch.sum(vec_x**2)")
    
    # Test gradient
    if vec_x.grad is not None:
        expected_grad = 2 * vec_x.detach()
        assert torch.allclose(vec_x.grad, expected_grad), f"vec_x.grad should be 2*vec_x = {expected_grad}"
    else:
        raise AssertionError("vec_x.grad not computed. Please call vec_loss.backward()")


def test_matrix_gradients_functions(namespace):
    """Test matrix gradients that can be called directly from notebooks"""
    import torch
    
    # Test matrix creation
    if 'mat_A' in namespace:
        mat_A = namespace['mat_A']
        assert mat_A.requires_grad, "mat_A should have requires_grad=True"
        assert mat_A.shape == torch.Size([2, 2]), "mat_A should have shape (2, 2)"
    else:
        raise AssertionError("mat_A not found. Please create a 2x2 matrix with requires_grad=True")
    
    # Test function
    if 'mat_loss' in namespace:
        mat_loss = namespace['mat_loss']
        expected_loss = torch.sum(mat_A**2)
        assert torch.isclose(mat_loss, expected_loss), "mat_loss should be sum of squares of mat_A"
    else:
        raise AssertionError("mat_loss not found. Please compute mat_loss = torch.sum(mat_A**2)")
    
    # Test gradient
    if mat_A.grad is not None:
        expected_grad = 2 * mat_A.detach()
        assert torch.allclose(mat_A.grad, expected_grad), f"mat_A.grad should be 2*mat_A"
    else:
        raise AssertionError("mat_A.grad not computed. Please call mat_loss.backward()")


def test_computational_graph_functions(namespace):
    """Test computational graph understanding that can be called directly from notebooks"""
    import torch
    
    # Test variables
    if 'graph_x' in namespace:
        graph_x = namespace['graph_x']
        assert graph_x.requires_grad, "graph_x should have requires_grad=True"
    else:
        raise AssertionError("graph_x not found. Please create graph_x with requires_grad=True")
    
    # Test intermediate computations
    vars_to_check = ['graph_y', 'graph_z', 'graph_w']
    for var_name in vars_to_check:
        if var_name not in namespace:
            raise AssertionError(f"{var_name} not found. Please compute the computational graph step by step")
    
    # Test final gradient
    if graph_x.grad is not None:
        # For x=2: y=4, z=13, w=169, dw/dx = 12*z*x = 12*13*2 = 312
        expected_grad = 12 * 13 * 2
        assert torch.isclose(graph_x.grad, torch.tensor(float(expected_grad))), f"graph_x.grad should be {expected_grad}"
    else:
        raise AssertionError("graph_x.grad not computed. Please call graph_w.backward()")


def test_grad_context_functions(namespace):
    """Test gradient context management that can be called directly from notebooks"""
    import torch
    
    # Test no_grad context
    if 'no_grad_result' in namespace:
        no_grad_result = namespace['no_grad_result']
        assert not no_grad_result.requires_grad, "no_grad_result should not require gradients"
    else:
        raise AssertionError("no_grad_result not found. Please compute a result within torch.no_grad() context")
    
    # Test detach
    if 'detached_result' in namespace:
        detached_result = namespace['detached_result']
        assert not detached_result.requires_grad, "detached_result should not require gradients after detach"
    else:
        raise AssertionError("detached_result not found. Please detach a tensor from the computational graph")


def test_higher_order_gradients_functions(namespace):
    """Test higher-order gradients that can be called directly from notebooks"""
    import torch
    
    # Test second derivative computation
    if 'second_derivative' in namespace:
        second_derivative = namespace['second_derivative']
        # For f(x) = x^4, f''(x) = 12x^2. At x=2, f''(2) = 12*4 = 48
        expected_second_deriv = 48.0
        assert torch.isclose(second_derivative, torch.tensor(expected_second_deriv)), f"Second derivative should be {expected_second_deriv}"
    else:
        raise AssertionError("second_derivative not found. Please compute the second derivative")


def run_tests():
    """Run all tests for Exercise 1"""
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    run_tests()