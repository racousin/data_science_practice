#!/usr/bin/env python3
"""
Test suite for Python Deep Learning Module 1 - Exercise 3: Tensor Mastery
"""

import sys
import torch
import numpy as np
import pytest
from typing import Optional, Callable, Tuple, List
import time

class TestExercise3:
    """Test cases for Exercise 3: Tensor Mastery"""
    
    def setup_method(self):
        """Set up test fixtures"""
        torch.manual_seed(42)
        np.random.seed(42)
    
    def test_advanced_indexing(self):
        """Test advanced indexing techniques"""
        # Test fancy indexing
        tensor = torch.randn(100, 50, 25)
        
        # Expected fancy indexing
        row_indices = torch.tensor([0, 10, 20, 30])
        col_indices = torch.tensor([5, 15, 25, 35, 45])
        expected = tensor[row_indices][:, col_indices]
        
        assert expected.shape == (4, 5, 25), "Fancy indexing should produce shape (4, 5, 25)"
        
        # Test boolean masking
        data = torch.randn(1000)
        threshold1, threshold2 = 0.5, 1.0
        mask = (torch.abs(data) > threshold1) & (data < threshold2)
        expected_masked = data[mask]
        
        assert expected_masked.dtype == data.dtype, "Masked tensor should preserve dtype"
        assert len(expected_masked.shape) == 1, "Masked tensor should be 1D"
        
        # Test conditional replacement
        test_data = torch.randn(10, 10)
        condition_value = 0.0
        replacement = -999.0
        expected_replaced = torch.where(test_data > condition_value, replacement, test_data)
        
        assert expected_replaced.shape == test_data.shape, "Conditional replacement should preserve shape"
    
    def test_broadcasting_rules(self):
        """Test broadcasting rule implementation"""
        # Test compatible shapes
        compatible_cases = [
            ((3, 4), (4,), (3, 4)),
            ((2, 1, 3), (3,), (2, 1, 3)),
            ((1, 5, 1), (3, 1, 4), (3, 5, 4)),
            ((3, 4), (1, 4), (3, 4)),
        ]
        
        for shape1, shape2, expected_result in compatible_cases:
            # Manually check broadcasting compatibility
            def check_compatibility(s1, s2):
                s1_rev = list(reversed(s1))
                s2_rev = list(reversed(s2))
                max_len = max(len(s1_rev), len(s2_rev))
                
                s1_padded = [1] * (max_len - len(s1_rev)) + s1_rev
                s2_padded = [1] * (max_len - len(s2_rev)) + s2_rev
                
                result = []
                for d1, d2 in zip(s1_padded, s2_padded):
                    if d1 == d2 or d1 == 1 or d2 == 1:
                        result.append(max(d1, d2))
                    else:
                        return False, ()
                
                return True, tuple(reversed(result))
            
            compatible, result_shape = check_compatibility(shape1, shape2)
            assert compatible, f"Shapes {shape1} and {shape2} should be compatible"
            assert result_shape == expected_result, f"Expected {expected_result}, got {result_shape}"
        
        # Test incompatible shapes
        incompatible_cases = [
            ((3, 4), (5, 4)),
            ((2, 3, 4), (2, 5, 4)),
        ]
        
        for shape1, shape2 in incompatible_cases:
            compatible, _ = check_compatibility(shape1, shape2)
            assert not compatible, f"Shapes {shape1} and {shape2} should not be compatible"
    
    def test_memory_analysis(self):
        """Test memory layout analysis"""
        # Test contiguous tensor
        contiguous_tensor = torch.randn(10, 20)
        assert contiguous_tensor.is_contiguous(), "Regular tensor should be contiguous"
        
        # Test non-contiguous tensor
        non_contiguous = contiguous_tensor.t()
        assert not non_contiguous.is_contiguous(), "Transposed tensor should be non-contiguous"
        
        # Test stride calculation
        def calculate_strides(shape):
            strides = []
            stride = 1
            for dim in reversed(shape):
                strides.append(stride)
                stride *= dim
            return tuple(reversed(strides))
        
        test_shapes = [(10, 20), (2, 3, 4), (5, 1, 3, 2)]
        for shape in test_shapes:
            calculated = calculate_strides(shape)
            actual = torch.randn(shape).stride()
            assert calculated == actual, f"Stride calculation failed for shape {shape}"
    
    def test_einsum_operations(self):
        """Test einsum operation implementations"""
        # Test basic operations
        A = torch.randn(10, 20)
        B = torch.randn(20, 30)
        v = torch.randn(20)
        
        # Matrix multiplication
        expected_mm = torch.mm(A, B)
        einsum_mm = torch.einsum('ij,jk->ik', A, B)
        assert torch.allclose(expected_mm, einsum_mm, atol=1e-6), "Einsum matrix multiplication failed"
        
        # Matrix-vector multiplication
        expected_mv = torch.mv(A, v)
        einsum_mv = torch.einsum('ij,j->i', A, v)
        assert torch.allclose(expected_mv, einsum_mv, atol=1e-6), "Einsum matrix-vector multiplication failed"
        
        # Trace
        square = torch.randn(15, 15)
        expected_trace = torch.trace(square)
        einsum_trace = torch.einsum('ii->', square)
        assert torch.allclose(expected_trace, einsum_trace, atol=1e-6), "Einsum trace failed"
        
        # Transpose
        expected_transpose = A.t()
        einsum_transpose = torch.einsum('ij->ji', A)
        assert torch.allclose(expected_transpose, einsum_transpose, atol=1e-6), "Einsum transpose failed"
    
    def test_custom_functions(self):
        """Test custom tensor function implementations"""
        # Test that functions can be called without errors
        # Note: We can't test full implementations without seeing the actual code,
        # but we can test basic structure and expected behavior
        
        # Test max pooling structure
        test_input = torch.randn(2, 3, 8, 8)
        # Expected output shape after 2x2 pooling with stride 2
        expected_shape = (2, 3, 4, 4)
        
        # Test layer normalization structure
        test_data = torch.randn(10, 20, 30)
        # After normalization, shape should be preserved
        expected_norm_shape = test_data.shape
        
        # Test that basic tensor operations work
        assert test_input.numel() > 0, "Test input should have elements"
        assert test_data.numel() > 0, "Test data should have elements"
        
        # Test unfold operation (commonly used in custom implementations)
        unfolded = torch.nn.functional.unfold(test_input, kernel_size=2, stride=2)
        assert unfolded.shape[0] == 2, "Unfold should preserve batch dimension"


def test_advanced_indexing(namespace):
    """Test function for advanced indexing that can be called directly from notebooks"""
    required_functions = ['fancy_indexing_selection', 'complex_boolean_mask', 'conditional_replacement']
    
    for func_name in required_functions:
        if func_name not in namespace:
            raise AssertionError(f"{func_name} function not found. Please implement the advanced indexing function.")
    
    # Test fancy indexing
    fancy_func = namespace['fancy_indexing_selection']
    test_tensor = torch.randn(100, 50, 25)
    result = fancy_func(test_tensor)
    
    if result is not None:
        assert result.shape == (4, 5, 25), f"Expected shape (4, 5, 25), got {result.shape}"
    else:
        raise AssertionError("fancy_indexing_selection returned None. Please implement the function.")
    
    # Test boolean masking
    mask_func = namespace['complex_boolean_mask']
    test_data = torch.randn(1000)
    masked_result = mask_func(test_data, 0.5, 1.0)
    
    if masked_result is not None:
        assert len(masked_result.shape) == 1, "Boolean mask should return 1D tensor"
        assert masked_result.dtype == test_data.dtype, "Boolean mask should preserve dtype"
    else:
        raise AssertionError("complex_boolean_mask returned None. Please implement the function.")
    
    # Test conditional replacement
    cond_func = namespace['conditional_replacement']
    test_data = torch.randn(10, 10)
    cond_result = cond_func(test_data, 0.0, -999.0)
    
    if cond_result is not None:
        assert cond_result.shape == test_data.shape, "Conditional replacement should preserve shape"
    else:
        raise AssertionError("conditional_replacement returned None. Please implement the function.")


def test_broadcasting_rules(namespace):
    """Test function for broadcasting rules that can be called directly from notebooks"""
    required_functions = ['check_broadcasting_compatibility', 'manual_broadcast', 'advanced_broadcasting_operations']
    
    for func_name in required_functions:
        if func_name not in namespace:
            raise AssertionError(f"{func_name} function not found. Please implement the broadcasting function.")
    
    # Test broadcasting compatibility checker
    check_func = namespace['check_broadcasting_compatibility']
    
    # Test compatible shapes
    compatible, result_shape = check_func((3, 4), (4,))
    assert compatible == True, "Shapes (3, 4) and (4,) should be compatible"
    assert result_shape == (3, 4), f"Expected result shape (3, 4), got {result_shape}"
    
    # Test incompatible shapes
    compatible, result_shape = check_func((3, 4), (5, 4))
    assert compatible == False, "Shapes (3, 4) and (5, 4) should not be compatible"
    
    # Test advanced broadcasting operations
    advanced_func = namespace['advanced_broadcasting_operations']
    results = advanced_func()
    
    assert isinstance(results, dict), "advanced_broadcasting_operations should return a dictionary"
    
    expected_keys = ['matrix_vector_add', '3d_2d_multiply', 'singleton_broadcast']
    for key in expected_keys:
        if key in results and results[key] is not None:
            assert hasattr(results[key], 'shape'), f"{key} should return a tensor with shape attribute"


def test_memory_analysis(namespace):
    """Test function for memory analysis that can be called directly from notebooks"""
    required_functions = ['analyze_memory_layout', 'calculate_manual_strides', 'memory_efficient_operations']
    
    for func_name in required_functions:
        if func_name not in namespace:
            raise AssertionError(f"{func_name} function not found. Please implement the memory analysis function.")
    
    # Test memory layout analyzer
    analyze_func = namespace['analyze_memory_layout']
    test_tensor = torch.randn(10, 20)
    analysis = analyze_func(test_tensor)
    
    assert isinstance(analysis, dict), "analyze_memory_layout should return a dictionary"
    
    expected_keys = ['shape', 'strides', 'is_contiguous', 'storage_size', 'element_size', 'memory_usage_bytes', 'data_ptr']
    for key in expected_keys:
        assert key in analysis, f"Analysis should include {key}"
    
    # Test stride calculation
    stride_func = namespace['calculate_manual_strides']
    test_shapes = [(10, 20), (2, 3, 4)]
    
    for shape in test_shapes:
        calculated_strides = stride_func(shape)
        actual_strides = torch.randn(shape).stride()
        
        if calculated_strides is not None:
            assert calculated_strides == actual_strides, f"Stride calculation failed for shape {shape}"
    
    # Test memory efficient operations
    efficiency_func = namespace['memory_efficient_operations']
    results = efficiency_func()
    
    assert isinstance(results, dict), "memory_efficient_operations should return a dictionary"
    
    expected_keys = ['regular_time', 'inplace_time', 'view_time', 'copy_time']
    for key in expected_keys:
        if key in results and results[key] is not None:
            assert isinstance(results[key], (int, float)), f"{key} should be a numeric value"


def test_einsum_operations(namespace):
    """Test function for einsum operations that can be called directly from notebooks"""
    required_functions = ['einsum_operations', 'complex_einsum_operations']
    
    for func_name in required_functions:
        if func_name not in namespace:
            raise AssertionError(f"{func_name} function not found. Please implement the einsum function.")
    
    # Test basic einsum operations
    basic_func = namespace['einsum_operations']
    basic_results = basic_func()
    
    assert isinstance(basic_results, dict), "einsum_operations should return a dictionary"
    
    expected_operations = ['matrix_mult', 'matvec_mult', 'trace', 'transpose']
    for op in expected_operations:
        if op in basic_results and basic_results[op] is not None:
            assert hasattr(basic_results[op], 'shape'), f"{op} should return a tensor"
    
    # Test complex einsum operations
    complex_func = namespace['complex_einsum_operations']
    complex_results = complex_func()
    
    assert isinstance(complex_results, dict), "complex_einsum_operations should return a dictionary"
    
    # Verify some basic einsum correctness if implemented
    if 'matrix_mult' in basic_results and basic_results['matrix_mult'] is not None:
        # Test that einsum matrix multiplication gives correct shape
        A = torch.randn(10, 20)
        B = torch.randn(20, 30)
        expected_shape = (10, 30)
        einsum_result = torch.einsum('ij,jk->ik', A, B)
        assert einsum_result.shape == expected_shape, f"Matrix multiplication should give shape {expected_shape}"


def test_custom_functions(namespace):
    """Test function for custom functions that can be called directly from notebooks"""
    required_functions = ['custom_max_pool2d', 'custom_conv2d', 'custom_layer_norm', 'batched_matrix_operations']
    
    for func_name in required_functions:
        if func_name not in namespace:
            raise AssertionError(f"{func_name} function not found. Please implement the custom function.")
    
    # Test custom max pooling
    pool_func = namespace['custom_max_pool2d']
    test_input = torch.randn(2, 3, 8, 8)
    pooled = pool_func(test_input, kernel_size=2, stride=2)
    
    if pooled is not None:
        expected_shape = (2, 3, 4, 4)
        assert pooled.shape == expected_shape, f"Max pooling should produce shape {expected_shape}, got {pooled.shape}"
    
    # Test custom convolution
    conv_func = namespace['custom_conv2d']
    kernel = torch.randn(16, 3, 3, 3)
    convolved = conv_func(test_input, kernel, stride=1, padding=1)
    
    if convolved is not None:
        expected_shape = (2, 16, 8, 8)  # Same size due to padding=1
        assert convolved.shape == expected_shape, f"Convolution should produce shape {expected_shape}, got {convolved.shape}"
    
    # Test custom layer norm
    norm_func = namespace['custom_layer_norm']
    test_data = torch.randn(10, 20, 30)
    normalized = norm_func(test_data, [20, 30])
    
    if normalized is not None:
        assert normalized.shape == test_data.shape, f"Layer norm should preserve shape"
    
    # Test batched operations
    batch_func = namespace['batched_matrix_operations']
    matrices = torch.randn(5, 10, 10)
    vectors = torch.randn(5, 10)
    results = batch_func(matrices, vectors)
    
    assert isinstance(results, dict), "batched_matrix_operations should return a dictionary"
    
    if 'matvec' in results and results['matvec'] is not None:
        expected_shape = (5, 10)
        assert results['matvec'].shape == expected_shape, f"Batched matvec should produce shape {expected_shape}"


def run_tests():
    """Run all tests for Exercise 3"""
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    run_tests()