#!/usr/bin/env python3
"""
Test suite for Python Deep Learning Module 1 - Exercise 1: Tensor Basics
"""

import sys
import torch
import numpy as np
import pytest
from typing import Optional

class TestExercise1:
    """Test cases for Exercise 1: Tensor Basics"""
    
    def setup_method(self):
        """Set up test fixtures"""
        torch.manual_seed(42)
        np.random.seed(42)
    
    def test_part1_tensor_creation(self):
        """Test Part 1: Tensor Creation"""
        # Test tensor_zeros
        tensor_zeros = torch.zeros(3, 3)
        assert tensor_zeros.shape == (3, 3), "tensor_zeros should have shape (3, 3)"
        assert torch.all(tensor_zeros == 0), "tensor_zeros should contain all zeros"
        
        # Test tensor_ones
        tensor_ones = torch.ones(2, 4)
        assert tensor_ones.shape == (2, 4), "tensor_ones should have shape (2, 4)"
        assert torch.all(tensor_ones == 1), "tensor_ones should contain all ones"
        
        # Test tensor_identity
        tensor_identity = torch.eye(3)
        assert tensor_identity.shape == (3, 3), "tensor_identity should have shape (3, 3)"
        assert torch.allclose(tensor_identity, torch.eye(3)), "tensor_identity should be an identity matrix"
        
        # Test tensor_random
        tensor_random = torch.rand(2, 3, 4)
        assert tensor_random.shape == (2, 3, 4), "tensor_random should have shape (2, 3, 4)"
        assert torch.all((tensor_random >= 0) & (tensor_random <= 1)), "tensor_random values should be between 0 and 1"
        
        # Test tensor_from_list
        tensor_from_list = torch.tensor([[1, 2, 3], [4, 5, 6]])
        assert tensor_from_list.shape == (2, 3), "tensor_from_list should have shape (2, 3)"
        expected = torch.tensor([[1, 2, 3], [4, 5, 6]])
        assert torch.equal(tensor_from_list, expected), "tensor_from_list should match expected values"
        
        # Test tensor_range
        tensor_range = torch.arange(10)
        assert tensor_range.shape == (10,), "tensor_range should have shape (10,)"
        assert torch.equal(tensor_range, torch.arange(10)), "tensor_range should contain values from 0 to 9"
    
    def test_part2_tensor_attributes(self):
        """Test Part 2: Tensor Attributes"""
        sample_tensor = torch.randn(3, 4, 5)
        
        tensor_shape = sample_tensor.shape
        assert tensor_shape == torch.Size([3, 4, 5]), "tensor_shape should be (3, 4, 5)"
        
        tensor_dtype = sample_tensor.dtype
        assert tensor_dtype == torch.float32, "tensor_dtype should be torch.float32"
        
        tensor_device = sample_tensor.device
        assert tensor_device.type == 'cpu', "tensor_device should be CPU"
        
        tensor_ndim = sample_tensor.ndim
        assert tensor_ndim == 3, "tensor_ndim should be 3"
        
        tensor_numel = sample_tensor.numel()
        assert tensor_numel == 60, "tensor_numel should be 60 (3*4*5)"
    
    def test_part3_indexing_slicing(self):
        """Test Part 3: Tensor Indexing and Slicing"""
        tensor = torch.arange(24).reshape(4, 6)
        
        # Test element access
        element = tensor[1, 3]
        assert element == 9, "Element at (1, 3) should be 9"
        
        # Test row access
        second_row = tensor[1]
        expected_row = torch.tensor([6, 7, 8, 9, 10, 11])
        assert torch.equal(second_row, expected_row), "Second row should match expected values"
        
        # Test column access
        last_column = tensor[:, -1]
        expected_column = torch.tensor([5, 11, 17, 23])
        assert torch.equal(last_column, expected_column), "Last column should match expected values"
        
        # Test submatrix
        submatrix = tensor[:2, :2]
        expected_submatrix = torch.tensor([[0, 1], [6, 7]])
        assert torch.equal(submatrix, expected_submatrix), "Submatrix should match expected values"
        
        # Test alternating elements
        alternating_elements = tensor[0, ::2]
        expected_alternating = torch.tensor([0, 2, 4])
        assert torch.equal(alternating_elements, expected_alternating), "Alternating elements should match expected values"
    
    def test_part4_tensor_reshaping(self):
        """Test Part 4: Tensor Reshaping"""
        original = torch.arange(12)
        
        # Test reshape to 3x4
        reshaped_3x4 = original.reshape(3, 4)
        assert reshaped_3x4.shape == (3, 4), "reshaped_3x4 should have shape (3, 4)"
        
        # Test reshape to 2x2x3
        reshaped_2x2x3 = original.reshape(2, 2, 3)
        assert reshaped_2x2x3.shape == (2, 2, 3), "reshaped_2x2x3 should have shape (2, 2, 3)"
        
        # Test flatten
        flattened = reshaped_2x2x3.flatten()
        assert flattened.shape == (12,), "flattened should have shape (12,)"
        assert torch.equal(flattened, original), "flattened should match original"
        
        # Test unsqueeze
        unsqueezed = original.unsqueeze(0)
        assert unsqueezed.shape == (1, 12), "unsqueezed should have shape (1, 12)"
        
        # Test squeeze
        tensor_with_singles = torch.randn(1, 3, 1, 4)
        squeezed = tensor_with_singles.squeeze()
        assert squeezed.shape == (3, 4), "squeezed should have shape (3, 4)"
    
    def test_part5_data_types(self):
        """Test Part 5: Tensor Data Types"""
        # Test float32 tensor
        float32_tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        assert float32_tensor.dtype == torch.float32, "float32_tensor should have dtype float32"
        
        # Test float64 conversion
        float64_tensor = float32_tensor.double()
        assert float64_tensor.dtype == torch.float64, "float64_tensor should have dtype float64"
        
        # Test integer to float conversion
        int_tensor = torch.tensor([1, 2, 3])
        int_to_float = int_tensor.float()
        assert int_to_float.dtype == torch.float32, "int_to_float should have dtype float32"
        
        # Test boolean tensor
        comparison_tensor = torch.tensor([1, 2, 3, 4, 5])
        bool_tensor = comparison_tensor > 3
        expected_bool = torch.tensor([False, False, False, True, True])
        assert torch.equal(bool_tensor, expected_bool), "bool_tensor should match expected values"
    
    def test_part6_numpy_interop(self):
        """Test Part 6: NumPy Interoperability"""
        # Test NumPy to tensor
        numpy_array = np.array([[1, 2, 3], [4, 5, 6]])
        tensor_from_numpy = torch.from_numpy(numpy_array)
        assert tensor_from_numpy.shape == (2, 3), "tensor_from_numpy should have shape (2, 3)"
        
        # Test tensor to NumPy
        pytorch_tensor = torch.randn(2, 3)
        numpy_from_tensor = pytorch_tensor.numpy()
        assert numpy_from_tensor.shape == (2, 3), "numpy_from_tensor should have shape (2, 3)"
        
        # Test shared memory
        shared_numpy = np.ones((2, 2))
        shared_tensor = torch.from_numpy(shared_numpy)
        shared_numpy[0, 0] = 999
        assert shared_tensor[0, 0] == 999, "Shared tensor should reflect changes in numpy array"


def run_tests():
    """Run all tests for Exercise 1"""
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    run_tests()