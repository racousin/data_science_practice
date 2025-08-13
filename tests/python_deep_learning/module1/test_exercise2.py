#!/usr/bin/env python3
"""
Test suite for Python Deep Learning Module 1 - Exercise 2: PyTorch Operations
"""

import sys
import torch
import numpy as np
import pytest
from typing import Optional

class TestExercise2:
    """Test cases for Exercise 2: PyTorch Operations"""
    
    def setup_method(self):
        """Set up test fixtures"""
        torch.manual_seed(42)
        np.random.seed(42)
    
    def test_part1_elementwise_operations(self):
        """Test Part 1: Element-wise Operations"""
        tensor_a = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
        tensor_b = torch.tensor([[7, 8, 9], [10, 11, 12]], dtype=torch.float32)
        
        # Test addition
        tensor_sum = tensor_a + tensor_b
        expected_sum = torch.tensor([[8, 10, 12], [14, 16, 18]], dtype=torch.float32)
        assert torch.allclose(tensor_sum, expected_sum), "tensor_sum should match expected values"
        
        # Test subtraction
        tensor_diff = tensor_a - tensor_b
        expected_diff = torch.tensor([[-6, -6, -6], [-6, -6, -6]], dtype=torch.float32)
        assert torch.allclose(tensor_diff, expected_diff), "tensor_diff should match expected values"
        
        # Test multiplication
        tensor_mul = tensor_a * tensor_b
        expected_mul = torch.tensor([[7, 16, 27], [40, 55, 72]], dtype=torch.float32)
        assert torch.allclose(tensor_mul, expected_mul), "tensor_mul should match expected values"
        
        # Test division
        tensor_div = tensor_a / tensor_b
        assert tensor_div.shape == tensor_a.shape, "tensor_div should have same shape as tensor_a"
        
        # Test power
        tensor_pow = tensor_a ** 2
        expected_pow = torch.tensor([[1, 4, 9], [16, 25, 36]], dtype=torch.float32)
        assert torch.allclose(tensor_pow, expected_pow), "tensor_pow should match expected values"
        
        # Test square root
        tensor_sqrt = torch.sqrt(tensor_a)
        assert tensor_sqrt.shape == tensor_a.shape, "tensor_sqrt should have same shape as tensor_a"
    
    def test_part2_reduction_operations(self):
        """Test Part 2: Reduction Operations"""
        tensor = torch.randn(3, 4, 5)
        
        # Test sum
        total_sum = tensor.sum()
        assert total_sum.ndim == 0, "total_sum should be a scalar"
        
        # Test mean along dimension
        mean_dim1 = tensor.mean(dim=1)
        assert mean_dim1.shape == (3, 5), "mean_dim1 should have shape (3, 5)"
        
        # Test max values and indices
        max_values, max_indices = tensor.max(dim=2)
        assert max_values.shape == (3, 4), "max_values should have shape (3, 4)"
        assert max_indices.shape == (3, 4), "max_indices should have shape (3, 4)"
        
        # Test standard deviation
        std_dim0 = tensor.std(dim=0)
        assert std_dim0.shape == (4, 5), "std_dim0 should have shape (4, 5)"
        
        # Test product
        small_tensor = torch.tensor([1., 2., 3.])
        total_product = small_tensor.prod()
        assert torch.allclose(total_product, torch.tensor(6.)), "total_product should be 6"
    
    def test_part3_matrix_operations(self):
        """Test Part 3: Matrix Operations"""
        matrix_a = torch.randn(3, 4)
        matrix_b = torch.randn(4, 5)
        square_matrix = torch.randn(3, 3)
        
        # Test matrix multiplication
        matmul_result = torch.matmul(matrix_a, matrix_b)
        assert matmul_result.shape == (3, 5), "matmul_result should have shape (3, 5)"
        
        # Test transpose
        transpose_a = matrix_a.T
        assert transpose_a.shape == (4, 3), "transpose_a should have shape (4, 3)"
        
        # Test inverse (make sure matrix is invertible)
        identity = torch.eye(3)
        inverse_matrix = torch.inverse(identity)
        assert torch.allclose(inverse_matrix, identity), "inverse of identity should be identity"
        
        # Test determinant
        determinant = torch.det(torch.eye(3))
        assert torch.allclose(determinant, torch.tensor(1.)), "determinant of identity should be 1"
        
        # Test eigenvalues and eigenvectors
        eigenvalues, eigenvectors = torch.linalg.eig(square_matrix)
        assert eigenvalues.shape == (3,), "eigenvalues should have shape (3,)"
        assert eigenvectors.shape == (3, 3), "eigenvectors should have shape (3, 3)"
    
    def test_part4_broadcasting(self):
        """Test Part 4: Broadcasting"""
        tensor_3x4 = torch.randn(3, 4)
        tensor_1x4 = torch.randn(1, 4)
        tensor_3x1 = torch.randn(3, 1)
        scalar = torch.tensor(2.0)
        
        # Test broadcasting along dim 0
        broadcast_add_dim0 = tensor_3x4 + tensor_1x4
        assert broadcast_add_dim0.shape == (3, 4), "broadcast_add_dim0 should have shape (3, 4)"
        
        # Test broadcasting along dim 1
        broadcast_mul_dim1 = tensor_3x1 * tensor_3x4
        assert broadcast_mul_dim1.shape == (3, 4), "broadcast_mul_dim1 should have shape (3, 4)"
        
        # Test scalar broadcasting
        scalar_broadcast = tensor_3x4 + scalar
        assert scalar_broadcast.shape == (3, 4), "scalar_broadcast should have shape (3, 4)"
        
        # Test custom broadcasting
        tensor_custom1 = torch.randn(2, 1, 3)
        tensor_custom2 = torch.randn(1, 3, 1)
        broadcast_custom = tensor_custom1 + tensor_custom2
        assert broadcast_custom.shape == (2, 3, 3), "broadcast_custom should have shape (2, 3, 3)"
    
    def test_part5_advanced_indexing(self):
        """Test Part 5: Advanced Indexing"""
        tensor = torch.randn(4, 5)
        
        # Test boolean mask
        mask = tensor > 0
        assert mask.dtype == torch.bool, "mask should be boolean tensor"
        
        # Test masked selection
        masked_elements = tensor[mask]
        assert masked_elements.ndim == 1, "masked_elements should be 1D tensor"
        
        # Test ReLU using boolean indexing
        tensor_relu = tensor.clone()
        tensor_relu[tensor_relu < 0] = 0
        assert torch.all(tensor_relu >= 0), "tensor_relu should have no negative values"
        
        # Test gather
        indices = torch.tensor([[1], [2], [0], [4]])
        gathered = torch.gather(tensor, 1, indices)
        assert gathered.shape == (4, 1), "gathered should have shape (4, 1)"
        
        # Test scatter
        scattered = torch.zeros(3, 5)
        src = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32)
        index = torch.tensor([[0, 1, 2], [2, 0, 1], [1, 2, 0]])
        scattered.scatter_(1, index, src)
        assert scattered.shape == (3, 5), "scattered should have shape (3, 5)"
    
    def test_part6_concatenation_stacking(self):
        """Test Part 6: Tensor Concatenation and Stacking"""
        tensor1 = torch.randn(2, 3)
        tensor2 = torch.randn(2, 3)
        tensor3 = torch.randn(2, 3)
        
        # Test concatenation along dim 0
        concat_dim0 = torch.cat([tensor1, tensor2, tensor3], dim=0)
        assert concat_dim0.shape == (6, 3), "concat_dim0 should have shape (6, 3)"
        
        # Test concatenation along dim 1
        concat_dim1 = torch.cat([tensor1, tensor2, tensor3], dim=1)
        assert concat_dim1.shape == (2, 9), "concat_dim1 should have shape (2, 9)"
        
        # Test stacking along dim 0
        stacked_dim0 = torch.stack([tensor1, tensor2, tensor3], dim=0)
        assert stacked_dim0.shape == (3, 2, 3), "stacked_dim0 should have shape (3, 2, 3)"
        
        # Test stacking along dim 2
        stacked_dim2 = torch.stack([tensor1, tensor2, tensor3], dim=2)
        assert stacked_dim2.shape == (2, 3, 3), "stacked_dim2 should have shape (2, 3, 3)"
        
        # Test splitting
        large_tensor = torch.randn(10, 4)
        chunks = torch.chunk(large_tensor, 5, dim=0)
        assert len(chunks) == 5, "Should have 5 chunks"
        assert all(chunk.shape == (2, 4) for chunk in chunks), "Each chunk should have shape (2, 4)"
    
    def test_part7_inplace_operations(self):
        """Test Part 7: In-place Operations"""
        tensor = torch.randn(3, 3)
        original_id = id(tensor)
        
        # Test in-place addition
        tensor.add_(1)
        assert id(tensor) == original_id, "In-place addition should not change tensor ID"
        
        # Test in-place multiplication
        tensor.mul_(2)
        assert id(tensor) == original_id, "In-place multiplication should not change tensor ID"
        
        # Test in-place ReLU
        tensor.relu_()
        assert id(tensor) == original_id, "In-place ReLU should not change tensor ID"
        assert torch.all(tensor >= 0), "After ReLU, all values should be non-negative"
        
        # Test difference between in-place and regular operations
        tensor_copy = torch.randn(2, 2)
        original_copy_id = id(tensor_copy)
        
        # Regular operation (creates new tensor)
        regular_result = tensor_copy + 1
        assert id(regular_result) != original_copy_id, "Regular operation should create new tensor"
        
        # In-place operation (modifies existing tensor)
        inplace_result = tensor_copy.add_(1)
        assert id(inplace_result) == original_copy_id, "In-place operation should not create new tensor"


def run_tests():
    """Run all tests for Exercise 2"""
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    run_tests()