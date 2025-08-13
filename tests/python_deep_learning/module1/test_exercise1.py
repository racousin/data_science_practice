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


def test_tensor_creation(namespace):
    """Test function that can be called directly from notebooks"""
    import torch
    
    # Test tensor_zeros
    if 'tensor_zeros' in namespace:
        tensor_zeros = namespace['tensor_zeros']
        assert tensor_zeros.shape == (3, 3), "tensor_zeros should have shape (3, 3)"
        assert torch.all(tensor_zeros == 0), "tensor_zeros should contain all zeros"
    else:
        raise AssertionError("tensor_zeros not found. Please create a 3x3 tensor of zeros named 'tensor_zeros'")
    
    # Test tensor_ones
    if 'tensor_ones' in namespace:
        tensor_ones = namespace['tensor_ones']
        assert tensor_ones.shape == (2, 4), "tensor_ones should have shape (2, 4)"
        assert torch.all(tensor_ones == 1), "tensor_ones should contain all ones"
    else:
        raise AssertionError("tensor_ones not found. Please create a 2x4 tensor of ones named 'tensor_ones'")
    
    # Test tensor_identity
    if 'tensor_identity' in namespace:
        tensor_identity = namespace['tensor_identity']
        assert tensor_identity.shape == (3, 3), "tensor_identity should have shape (3, 3)"
        assert torch.allclose(tensor_identity, torch.eye(3)), "tensor_identity should be an identity matrix"
    else:
        raise AssertionError("tensor_identity not found. Please create a 3x3 identity tensor named 'tensor_identity'")
    
    # Test tensor_random
    if 'tensor_random' in namespace:
        tensor_random = namespace['tensor_random']
        assert tensor_random.shape == (2, 3, 4), "tensor_random should have shape (2, 3, 4)"
        assert torch.all((tensor_random >= 0) & (tensor_random <= 1)), "tensor_random values should be between 0 and 1"
    else:
        raise AssertionError("tensor_random not found. Please create a 2x3x4 tensor of random values named 'tensor_random'")
    
    # Test tensor_from_list
    if 'tensor_from_list' in namespace:
        tensor_from_list = namespace['tensor_from_list']
        assert tensor_from_list.shape == (2, 3), "tensor_from_list should have shape (2, 3)"
        expected = torch.tensor([[1, 2, 3], [4, 5, 6]])
        assert torch.equal(tensor_from_list, expected), "tensor_from_list should match expected values [[1, 2, 3], [4, 5, 6]]"
    else:
        raise AssertionError("tensor_from_list not found. Please create a tensor from the list [[1, 2, 3], [4, 5, 6]] named 'tensor_from_list'")
    
    # Test tensor_range
    if 'tensor_range' in namespace:
        tensor_range = namespace['tensor_range']
        assert tensor_range.shape == (10,), "tensor_range should have shape (10,)"
        assert torch.equal(tensor_range, torch.arange(10)), "tensor_range should contain values from 0 to 9"
    else:
        raise AssertionError("tensor_range not found. Please create a tensor with values 0-9 named 'tensor_range'")


def test_tensor_attributes(namespace):
    """Test function for tensor attributes that can be called directly from notebooks"""
    import torch
    
    # Create the sample tensor if it doesn't exist
    if 'sample_tensor' not in namespace:
        namespace['sample_tensor'] = torch.randn(3, 4, 5)
    
    sample_tensor = namespace['sample_tensor']
    
    # Test tensor_shape
    if 'tensor_shape' in namespace:
        tensor_shape = namespace['tensor_shape']
        assert tensor_shape == sample_tensor.shape, f"tensor_shape should be {sample_tensor.shape}, got {tensor_shape}"
    else:
        raise AssertionError("tensor_shape not found. Please assign sample_tensor.shape to tensor_shape")
    
    # Test tensor_dtype
    if 'tensor_dtype' in namespace:
        tensor_dtype = namespace['tensor_dtype']
        assert tensor_dtype == sample_tensor.dtype, f"tensor_dtype should be {sample_tensor.dtype}, got {tensor_dtype}"
    else:
        raise AssertionError("tensor_dtype not found. Please assign sample_tensor.dtype to tensor_dtype")
    
    # Test tensor_device
    if 'tensor_device' in namespace:
        tensor_device = namespace['tensor_device']
        assert tensor_device == sample_tensor.device, f"tensor_device should be {sample_tensor.device}, got {tensor_device}"
    else:
        raise AssertionError("tensor_device not found. Please assign sample_tensor.device to tensor_device")
    
    # Test tensor_ndim
    if 'tensor_ndim' in namespace:
        tensor_ndim = namespace['tensor_ndim']
        assert tensor_ndim == sample_tensor.ndim, f"tensor_ndim should be {sample_tensor.ndim}, got {tensor_ndim}"
    else:
        raise AssertionError("tensor_ndim not found. Please assign sample_tensor.ndim to tensor_ndim")
    
    # Test tensor_numel
    if 'tensor_numel' in namespace:
        tensor_numel = namespace['tensor_numel']
        assert tensor_numel == sample_tensor.numel(), f"tensor_numel should be {sample_tensor.numel()}, got {tensor_numel}"
    else:
        raise AssertionError("tensor_numel not found. Please assign sample_tensor.numel() to tensor_numel")


def test_tensor_indexing(namespace):
    """Test function for tensor indexing that can be called directly from notebooks"""
    import torch
    
    # Create the tensor if it doesn't exist
    if 'tensor' not in namespace:
        namespace['tensor'] = torch.arange(24).reshape(4, 6)
    
    tensor = namespace['tensor']
    
    # Test element access
    if 'element' in namespace:
        element = namespace['element']
        expected = tensor[1, 3]
        assert element == expected, f"element should be {expected}, got {element}"
    else:
        raise AssertionError("element not found. Please assign tensor[1, 3] to element")
    
    # Test second_row
    if 'second_row' in namespace:
        second_row = namespace['second_row']
        expected = tensor[1]
        assert torch.equal(second_row, expected), f"second_row should equal tensor[1]"
    else:
        raise AssertionError("second_row not found. Please assign tensor[1] to second_row")
    
    # Test last_column
    if 'last_column' in namespace:
        last_column = namespace['last_column']
        expected = tensor[:, -1]
        assert torch.equal(last_column, expected), f"last_column should equal tensor[:, -1]"
    else:
        raise AssertionError("last_column not found. Please assign tensor[:, -1] to last_column")
    
    # Test submatrix
    if 'submatrix' in namespace:
        submatrix = namespace['submatrix']
        expected = tensor[:2, :2]
        assert torch.equal(submatrix, expected), f"submatrix should equal tensor[:2, :2]"
    else:
        raise AssertionError("submatrix not found. Please assign tensor[:2, :2] to submatrix")
    
    # Test alternating_elements
    if 'alternating_elements' in namespace:
        alternating_elements = namespace['alternating_elements']
        expected = tensor[0, ::2]
        assert torch.equal(alternating_elements, expected), f"alternating_elements should equal tensor[0, ::2]"
    else:
        raise AssertionError("alternating_elements not found. Please assign tensor[0, ::2] to alternating_elements")


def test_tensor_reshaping(namespace):
    """Test function for tensor reshaping that can be called directly from notebooks"""
    import torch
    
    # Create the original tensor if it doesn't exist
    if 'original' not in namespace:
        namespace['original'] = torch.arange(12)
    
    original = namespace['original']
    
    # Test reshaped_3x4
    if 'reshaped_3x4' in namespace:
        reshaped_3x4 = namespace['reshaped_3x4']
        assert reshaped_3x4.shape == (3, 4), f"reshaped_3x4 should have shape (3, 4), got {reshaped_3x4.shape}"
    else:
        raise AssertionError("reshaped_3x4 not found. Please assign original.reshape(3, 4) to reshaped_3x4")
    
    # Test reshaped_2x2x3
    if 'reshaped_2x2x3' in namespace:
        reshaped_2x2x3 = namespace['reshaped_2x2x3']
        assert reshaped_2x2x3.shape == (2, 2, 3), f"reshaped_2x2x3 should have shape (2, 2, 3), got {reshaped_2x2x3.shape}"
    else:
        raise AssertionError("reshaped_2x2x3 not found. Please assign original.reshape(2, 2, 3) to reshaped_2x2x3")
    
    # Test flattened
    if 'flattened' in namespace:
        flattened = namespace['flattened']
        assert flattened.shape == (12,), f"flattened should have shape (12,), got {flattened.shape}"
        if 'reshaped_2x2x3' in namespace:
            assert torch.equal(flattened, original), "flattened should equal original tensor"
    else:
        raise AssertionError("flattened not found. Please assign reshaped_2x2x3.flatten() to flattened")
    
    # Test unsqueezed
    if 'unsqueezed' in namespace:
        unsqueezed = namespace['unsqueezed']
        assert unsqueezed.shape == (1, 12), f"unsqueezed should have shape (1, 12), got {unsqueezed.shape}"
    else:
        raise AssertionError("unsqueezed not found. Please assign original.unsqueeze(0) to unsqueezed")
    
    # Test squeezed
    if 'squeezed' in namespace:
        squeezed = namespace['squeezed']
        assert squeezed.shape == (3, 4), f"squeezed should have shape (3, 4), got {squeezed.shape}"
    else:
        raise AssertionError("squeezed not found. Please assign tensor_with_singles.squeeze() to squeezed")


def test_tensor_dtypes(namespace):
    """Test function for tensor data types that can be called directly from notebooks"""
    import torch
    
    # Test float32_tensor
    if 'float32_tensor' in namespace:
        float32_tensor = namespace['float32_tensor']
        assert float32_tensor.dtype == torch.float32, f"float32_tensor should have dtype float32, got {float32_tensor.dtype}"
    else:
        raise AssertionError("float32_tensor not found. Please create a float32 tensor")
    
    # Test float64_tensor
    if 'float64_tensor' in namespace:
        float64_tensor = namespace['float64_tensor']
        assert float64_tensor.dtype == torch.float64, f"float64_tensor should have dtype float64, got {float64_tensor.dtype}"
    else:
        raise AssertionError("float64_tensor not found. Please convert float32_tensor to float64")
    
    # Test int_tensor and int_to_float
    if 'int_tensor' in namespace:
        int_tensor = namespace['int_tensor']
        assert int_tensor.dtype in [torch.int32, torch.int64], f"int_tensor should have integer dtype, got {int_tensor.dtype}"
    else:
        raise AssertionError("int_tensor not found. Please create an integer tensor")
    
    if 'int_to_float' in namespace:
        int_to_float = namespace['int_to_float']
        assert int_to_float.dtype == torch.float32, f"int_to_float should have dtype float32, got {int_to_float.dtype}"
    else:
        raise AssertionError("int_to_float not found. Please convert int_tensor to float")
    
    # Test bool_tensor
    if 'bool_tensor' in namespace:
        bool_tensor = namespace['bool_tensor']
        assert bool_tensor.dtype == torch.bool, f"bool_tensor should have dtype bool, got {bool_tensor.dtype}"
        expected = torch.tensor([False, False, False, True, True])
        assert torch.equal(bool_tensor, expected), "bool_tensor should show elements > 3"
    else:
        raise AssertionError("bool_tensor not found. Please create a boolean tensor showing comparison_tensor > 3")


def test_numpy_interop(namespace):
    """Test function for numpy interoperability that can be called directly from notebooks"""
    import torch
    import numpy as np
    
    # Test tensor_from_numpy
    if 'tensor_from_numpy' in namespace:
        tensor_from_numpy = namespace['tensor_from_numpy']
        assert tensor_from_numpy.shape == (2, 3), f"tensor_from_numpy should have shape (2, 3), got {tensor_from_numpy.shape}"
        if 'numpy_array' in namespace:
            expected_values = torch.from_numpy(namespace['numpy_array'])
            assert torch.equal(tensor_from_numpy, expected_values), "tensor_from_numpy should match numpy_array values"
    else:
        raise AssertionError("tensor_from_numpy not found. Please convert numpy_array to tensor using torch.from_numpy()")
    
    # Test numpy_from_tensor
    if 'numpy_from_tensor' in namespace:
        numpy_from_tensor = namespace['numpy_from_tensor']
        assert isinstance(numpy_from_tensor, np.ndarray), "numpy_from_tensor should be a numpy array"
        assert numpy_from_tensor.shape == (2, 3), f"numpy_from_tensor should have shape (2, 3), got {numpy_from_tensor.shape}"
    else:
        raise AssertionError("numpy_from_tensor not found. Please convert pytorch_tensor to numpy using .numpy()")
    
    # Test shared memory
    if 'shared_tensor' in namespace:
        shared_tensor = namespace['shared_tensor']
        if 'shared_numpy' in namespace:
            shared_numpy = namespace['shared_numpy']
            # Modify numpy array and check if tensor reflects the change
            original_value = shared_numpy[0, 0]
            shared_numpy[0, 0] = 999
            assert shared_tensor[0, 0] == 999, "shared_tensor should reflect changes in shared_numpy (memory sharing)"
            # Restore original value
            shared_numpy[0, 0] = original_value
    else:
        raise AssertionError("shared_tensor not found. Please create a tensor from shared_numpy using torch.from_numpy()")


def run_tests():
    """Run all tests for Exercise 1"""
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    run_tests()