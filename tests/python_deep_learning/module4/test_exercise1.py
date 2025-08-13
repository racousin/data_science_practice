#!/usr/bin/env python3
"""
Test suite for Python Deep Learning Module 4 - Exercise 1: Performance Profiling
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytest
import time
import math
from typing import Optional, Callable, Tuple, List


class TestExercise1:
    """Test cases for Exercise 1: Performance Profiling"""
    
    def setup_method(self):
        """Set up test fixtures"""
        torch.manual_seed(42)
        np.random.seed(42)
    
    def test_timing_measurements(self):
        """Test basic timing measurement functionality"""
        # Test CPU timing
        start_time = time.time()
        
        # Simulate some computation
        x = torch.randn(1000, 1000)
        y = torch.matmul(x, x.T)
        
        end_time = time.time()
        cpu_time = end_time - start_time
        
        assert cpu_time > 0, "CPU timing should be positive"
        assert cpu_time < 10.0, "Simple computation should be fast"
        
        # Test CUDA timing if available
        if torch.cuda.is_available():
            device = torch.device('cuda')
            x_gpu = x.to(device)
            
            # CUDA events for accurate GPU timing
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            y_gpu = torch.matmul(x_gpu, x_gpu.T)
            end_event.record()
            
            torch.cuda.synchronize()
            gpu_time = start_event.elapsed_time(end_event)  # in milliseconds
            
            assert gpu_time > 0, "GPU timing should be positive"
            assert gpu_time < 10000, "GPU computation should be reasonable"
    
    def test_memory_profiling(self):
        """Test memory usage profiling"""
        # Test memory tracking
        if torch.cuda.is_available():
            device = torch.device('cuda')
            
            # Clear cache and get initial memory
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()
            
            # Allocate tensors
            tensors = []
            for i in range(10):
                tensor = torch.randn(100, 100, device=device)
                tensors.append(tensor)
            
            # Check memory increase
            after_allocation = torch.cuda.memory_allocated()
            memory_used = after_allocation - initial_memory
            
            assert memory_used > 0, "Memory usage should increase after allocation"
            
            # Clean up
            del tensors
            torch.cuda.empty_cache()
            
            final_memory = torch.cuda.memory_allocated()
            assert final_memory <= initial_memory + 1024, "Memory should be cleaned up"
        else:
            # CPU memory tracking (approximate)
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss
            
            # Allocate large tensor
            large_tensor = torch.randn(5000, 5000)
            
            after_memory = process.memory_info().rss
            memory_increase = after_memory - initial_memory
            
            assert memory_increase > 0, "Memory should increase after allocation"
            
            del large_tensor
    
    def test_computation_complexity_analysis(self):
        """Test computational complexity understanding"""
        sizes = [100, 200, 400]
        times = []
        
        for size in sizes:
            x = torch.randn(size, size)
            
            start_time = time.time()
            # Matrix multiplication is O(n^3)
            y = torch.matmul(x, x)
            end_time = time.time()
            
            times.append(end_time - start_time)
        
        # Check that time increases with size (roughly cubic for matrix multiplication)
        assert times[1] > times[0], "Time should increase with input size"
        assert times[2] > times[1], "Time should continue to increase"
        
        # Rough check for cubic scaling (allowing for variance)
        ratio1 = times[1] / times[0] if times[0] > 0 else float('inf')
        ratio2 = times[2] / times[1] if times[1] > 0 else float('inf')
        
        # For 2x size increase, we expect roughly 8x time increase for O(n^3)
        # But allow for significant variance due to system factors
        assert ratio1 > 1.5, "Should see significant time increase for larger inputs"
    
    def test_batch_size_performance(self):
        """Test performance impact of different batch sizes"""
        model = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        
        batch_sizes = [32, 64, 128]
        times_per_sample = []
        
        for batch_size in batch_sizes:
            x = torch.randn(batch_size, 784)
            
            # Warm up
            _ = model(x)
            
            # Time the forward pass
            start_time = time.time()
            for _ in range(10):  # Multiple runs for better measurement
                _ = model(x)
            end_time = time.time()
            
            time_per_sample = (end_time - start_time) / (10 * batch_size)
            times_per_sample.append(time_per_sample)
        
        # Generally, larger batch sizes should be more efficient per sample
        # (though this can vary based on hardware)
        assert all(t > 0 for t in times_per_sample), "All times should be positive"
        assert max(times_per_sample) / min(times_per_sample) < 10, "Times shouldn't vary too dramatically"
    
    def test_cpu_vs_gpu_performance(self):
        """Test CPU vs GPU performance comparison"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for GPU testing")
        
        # Test data
        x = torch.randn(1000, 1000)
        y = torch.randn(1000, 1000)
        
        # CPU timing
        start_time = time.time()
        result_cpu = torch.matmul(x, y)
        cpu_time = time.time() - start_time
        
        # GPU timing
        x_gpu = x.cuda()
        y_gpu = y.cuda()
        torch.cuda.synchronize()  # Ensure GPU is ready
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        result_gpu = torch.matmul(x_gpu, y_gpu)
        end_event.record()
        torch.cuda.synchronize()
        
        gpu_time = start_event.elapsed_time(end_event) / 1000.0  # Convert to seconds
        
        # Results should be similar (allowing for numerical differences)
        assert torch.allclose(result_cpu, result_gpu.cpu(), rtol=1e-5), "CPU and GPU results should match"
        
        # Both should complete in reasonable time
        assert cpu_time < 10.0, "CPU computation should complete reasonably fast"
        assert gpu_time < 10.0, "GPU computation should complete reasonably fast"
        
        print(f"CPU time: {cpu_time:.4f}s, GPU time: {gpu_time:.4f}s")
    
    def test_model_parameter_count(self):
        """Test model parameter counting and memory estimation"""
        model = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Expected: (784*512 + 512) + (512*256 + 256) + (256*10 + 10)
        expected_params = (784 * 512 + 512) + (512 * 256 + 256) + (256 * 10 + 10)
        
        assert total_params == expected_params, f"Parameter count should be {expected_params}, got {total_params}"
        assert trainable_params == total_params, "All parameters should be trainable by default"
        
        # Estimate memory usage (approximate)
        model_memory_bytes = total_params * 4  # 4 bytes per float32 parameter
        model_memory_mb = model_memory_bytes / (1024 * 1024)
        
        assert model_memory_mb > 0, "Model should use some memory"
        assert model_memory_mb < 100, "Model shouldn't be too large for this simple architecture"
        
        print(f"Model parameters: {total_params:,}, Memory: {model_memory_mb:.2f} MB")
    
    def test_profiler_usage(self):
        """Test PyTorch profiler usage"""
        model = nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10)
        )
        
        x = torch.randn(32, 100)
        
        # Test profiler context
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            record_shapes=True,
            profile_memory=True
        ) as prof:
            # Forward pass
            output = model(x)
            loss = output.sum()
            
            # Backward pass
            loss.backward()
        
        # Check that profiler captured some events
        events = prof.events()
        assert len(events) > 0, "Profiler should capture some events"
        
        # Check for key operations
        event_names = [event.name for event in events]
        assert any('linear' in name.lower() or 'addmm' in name.lower() for name in event_names), \
            "Should capture linear layer operations"


def test_timing_measurement_implementation(namespace):
    """Test timing measurement implementation"""
    import torch
    import time
    
    # Test basic timing function
    if 'measure_time' in namespace:
        measure_time = namespace['measure_time']
        assert callable(measure_time), "measure_time should be a function"
        
        # Test the timing function
        def simple_op():
            return torch.randn(100, 100).sum()
        
        elapsed_time = measure_time(simple_op)
        assert isinstance(elapsed_time, (int, float)), "measure_time should return a number"
        assert elapsed_time > 0, "Measured time should be positive"
    else:
        raise AssertionError("measure_time function not found. Please implement timing measurement")


def test_cuda_timing_implementation(namespace):
    """Test CUDA timing implementation"""
    import torch
    
    if torch.cuda.is_available():
        # Test CUDA timing function
        if 'measure_gpu_time' in namespace:
            measure_gpu_time = namespace['measure_gpu_time']
            assert callable(measure_gpu_time), "measure_gpu_time should be a function"
            
            # Test GPU timing
            def gpu_op():
                x = torch.randn(500, 500, device='cuda')
                return torch.matmul(x, x)
            
            gpu_time = measure_gpu_time(gpu_op)
            assert isinstance(gpu_time, (int, float)), "GPU timing should return a number"
            assert gpu_time > 0, "GPU time should be positive"
        else:
            print("Warning: measure_gpu_time not found. Consider implementing GPU timing")
    else:
        print("CUDA not available - skipping GPU timing tests")


def test_memory_profiling_implementation(namespace):
    """Test memory profiling implementation"""
    import torch
    
    # Test memory measurement function
    if 'measure_memory_usage' in namespace:
        measure_memory = namespace['measure_memory_usage']
        assert callable(measure_memory), "measure_memory_usage should be a function"
        
        # Test memory measurement
        def memory_intensive_op():
            return [torch.randn(200, 200) for _ in range(10)]
        
        memory_used = measure_memory(memory_intensive_op)
        assert isinstance(memory_used, (int, float)), "Memory measurement should return a number"
        assert memory_used >= 0, "Memory usage should be non-negative"
    else:
        print("Warning: measure_memory_usage not found. Consider implementing memory profiling")


def test_performance_comparison_results(namespace):
    """Test performance comparison results"""
    import torch
    
    # Test CPU vs GPU comparison
    if 'cpu_gpu_comparison' in namespace:
        comparison = namespace['cpu_gpu_comparison']
        assert isinstance(comparison, dict), "cpu_gpu_comparison should be a dictionary"
        
        expected_keys = ['cpu_time', 'gpu_time']
        for key in expected_keys:
            assert key in comparison, f"Comparison should include {key}"
            assert isinstance(comparison[key], (int, float)), f"{key} should be a number"
            assert comparison[key] > 0, f"{key} should be positive"
    else:
        print("Warning: cpu_gpu_comparison not found")


def test_batch_size_analysis_results(namespace):
    """Test batch size analysis results"""
    import torch
    
    # Test batch size performance analysis
    if 'batch_size_performance' in namespace:
        perf_results = namespace['batch_size_performance']
        assert isinstance(perf_results, dict), "batch_size_performance should be a dictionary"
        assert len(perf_results) >= 2, "Should test at least 2 batch sizes"
        
        # Check that all results are positive numbers
        for batch_size, time_result in perf_results.items():
            assert isinstance(batch_size, int), "Batch size should be integer"
            assert batch_size > 0, "Batch size should be positive"
            assert isinstance(time_result, (int, float)), "Time result should be a number"
            assert time_result > 0, "Time should be positive"
    else:
        print("Warning: batch_size_performance not found")


def test_model_complexity_analysis(namespace):
    """Test model complexity analysis"""
    import torch
    import torch.nn as nn
    
    # Test parameter counting
    if 'count_parameters' in namespace:
        count_params = namespace['count_parameters']
        assert callable(count_params), "count_parameters should be a function"
        
        # Test with a simple model
        test_model = nn.Sequential(nn.Linear(10, 5), nn.Linear(5, 1))
        param_count = count_params(test_model)
        
        expected_count = (10 * 5 + 5) + (5 * 1 + 1)  # weights + biases
        assert param_count == expected_count, f"Parameter count should be {expected_count}, got {param_count}"
    else:
        raise AssertionError("count_parameters function not found")
    
    # Test model size estimation
    if 'estimate_model_size' in namespace:
        estimate_size = namespace['estimate_model_size']
        assert callable(estimate_size), "estimate_model_size should be a function"
        
        test_model = nn.Linear(100, 10)
        size_mb = estimate_size(test_model)
        assert isinstance(size_mb, (int, float)), "Model size should be a number"
        assert size_mb > 0, "Model size should be positive"
    else:
        print("Warning: estimate_model_size not found")


def test_profiler_usage_implementation(namespace):
    """Test PyTorch profiler usage"""
    import torch
    
    # Test profiling function
    if 'profile_model' in namespace:
        profile_func = namespace['profile_model']
        assert callable(profile_func), "profile_model should be a function"
    elif 'profiler_results' in namespace:
        results = namespace['profiler_results']
        assert isinstance(results, (dict, list)), "profiler_results should contain profiling data"
    else:
        print("Warning: Profiler usage not implemented")


def test_optimization_recommendations(namespace):
    """Test optimization recommendations based on profiling"""
    import torch
    
    # Test optimization suggestions
    if 'optimization_recommendations' in namespace:
        recommendations = namespace['optimization_recommendations']
        assert isinstance(recommendations, (list, dict, str)), "Should provide optimization recommendations"
        
        if isinstance(recommendations, list):
            assert len(recommendations) > 0, "Should provide at least one recommendation"
        elif isinstance(recommendations, dict):
            assert len(recommendations) > 0, "Should provide some recommendations"
    else:
        print("Warning: optimization_recommendations not found")


def test_bottleneck_identification(namespace):
    """Test bottleneck identification"""
    import torch
    
    # Test bottleneck analysis
    if 'bottleneck_analysis' in namespace:
        analysis = namespace['bottleneck_analysis']
        assert isinstance(analysis, (dict, list, str)), "Should provide bottleneck analysis"
    else:
        print("Warning: bottleneck_analysis not found")


def test_performance_visualization(namespace):
    """Test performance visualization"""
    import torch
    
    # Test visualization function or data
    if 'plot_performance' in namespace:
        plot_func = namespace['plot_performance']
        assert callable(plot_func), "plot_performance should be a function"
    elif 'performance_data' in namespace:
        perf_data = namespace['performance_data']
        assert isinstance(perf_data, (dict, list)), "performance_data should be structured data"
    else:
        print("Warning: Performance visualization not implemented")


def test_hardware_utilization_analysis(namespace):
    """Test hardware utilization analysis"""
    import torch
    
    # Test utilization metrics
    if 'hardware_utilization' in namespace:
        utilization = namespace['hardware_utilization']
        assert isinstance(utilization, dict), "hardware_utilization should be a dictionary"
        
        # Check for common metrics
        possible_keys = ['cpu_usage', 'memory_usage', 'gpu_usage', 'gpu_memory']
        found_keys = [key for key in possible_keys if key in utilization]
        assert len(found_keys) > 0, "Should track at least one hardware metric"
    else:
        print("Warning: hardware_utilization not found")


def run_tests():
    """Run all tests for Exercise 1"""
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    run_tests()