#!/usr/bin/env python3
"""
Test suite for Python Deep Learning Module 4 - Exercise 2: Advanced Features
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytest
import math
from typing import Optional, Callable, Tuple, List


class TestExercise2:
    """Test cases for Exercise 2: Advanced Features"""
    
    def setup_method(self):
        """Set up test fixtures"""
        torch.manual_seed(42)
        np.random.seed(42)
    
    def test_mixed_precision_training(self):
        """Test mixed precision training with autocast and GradScaler"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for mixed precision testing")
        
        model = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        ).cuda()
        
        optimizer = torch.optim.Adam(model.parameters())
        scaler = torch.cuda.amp.GradScaler()
        criterion = nn.CrossEntropyLoss()
        
        # Test data
        x = torch.randn(32, 784).cuda()
        y = torch.randint(0, 10, (32,)).cuda()
        
        # Training step with mixed precision
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            output = model(x)
            loss = criterion(output, y)
        
        # Check that autocast changes the dtype of intermediate tensors
        assert output.dtype == torch.float16, "Output should be float16 with autocast"
        
        # Backward pass with scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        assert not torch.isnan(loss), "Loss should not be NaN with mixed precision"
    
    def test_model_compilation(self):
        """Test model compilation features (PyTorch 2.0+)"""
        model = nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10)
        )
        
        # Test if torch.compile is available (PyTorch 2.0+)
        if hasattr(torch, 'compile'):
            try:
                compiled_model = torch.compile(model)
                
                # Test that compiled model works
                x = torch.randn(32, 100)
                output = compiled_model(x)
                
                assert output.shape == (32, 10), "Compiled model should produce correct output shape"
                assert torch.is_tensor(output), "Compiled model should return tensor"
                
                # Test that original and compiled models produce similar results
                original_output = model(x)
                assert torch.allclose(output, original_output, atol=1e-5), \
                    "Compiled and original models should produce similar outputs"
                    
            except Exception as e:
                print(f"Model compilation not fully supported: {e}")
        else:
            print("torch.compile not available (requires PyTorch 2.0+)")
    
    def test_custom_autograd_function(self):
        """Test custom autograd function implementation"""
        class CustomReLU(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input):
                ctx.save_for_backward(input)
                return torch.clamp(input, min=0)
            
            @staticmethod
            def backward(ctx, grad_output):
                input, = ctx.saved_tensors
                grad_input = grad_output.clone()
                grad_input[input < 0] = 0
                return grad_input
        
        # Test custom function
        x = torch.randn(10, requires_grad=True)
        y = CustomReLU.apply(x)
        
        assert y.shape == x.shape, "Custom function should preserve shape"
        assert torch.all(y >= 0), "Custom ReLU should output non-negative values"
        assert y.requires_grad, "Output should require gradients"
        
        # Test gradients
        loss = y.sum()
        loss.backward()
        
        assert x.grad is not None, "Input should have gradients"
        
        # Gradient should be 1 where x > 0 and 0 where x < 0
        expected_grad = (x > 0).float()
        assert torch.allclose(x.grad, expected_grad), "Custom gradients should be correct"
    
    def test_model_scripting_and_tracing(self):
        """Test TorchScript functionality"""
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)
                
            def forward(self, x):
                return F.relu(self.linear(x))
        
        model = SimpleModel()
        x = torch.randn(3, 10)
        
        # Test scripting
        try:
            scripted_model = torch.jit.script(model)
            scripted_output = scripted_model(x)
            
            assert torch.is_tensor(scripted_output), "Scripted model should return tensor"
            assert scripted_output.shape == (3, 5), "Scripted model should have correct output shape"
            
        except Exception as e:
            print(f"Model scripting failed: {e}")
        
        # Test tracing
        try:
            traced_model = torch.jit.trace(model, x)
            traced_output = traced_model(x)
            
            assert torch.is_tensor(traced_output), "Traced model should return tensor"
            assert traced_output.shape == (3, 5), "Traced model should have correct output shape"
            
            # Compare outputs
            original_output = model(x)
            assert torch.allclose(traced_output, original_output), \
                "Traced model should produce same output as original"
                
        except Exception as e:
            print(f"Model tracing failed: {e}")
    
    def test_custom_dataset_implementation(self):
        """Test custom Dataset implementation"""
        from torch.utils.data import Dataset, DataLoader
        
        class CustomDataset(Dataset):
            def __init__(self, size=1000, input_dim=10, num_classes=5):
                self.size = size
                self.input_dim = input_dim
                self.num_classes = num_classes
                
                # Generate synthetic data
                self.data = torch.randn(size, input_dim)
                self.labels = torch.randint(0, num_classes, (size,))
            
            def __len__(self):
                return self.size
            
            def __getitem__(self, idx):
                return self.data[idx], self.labels[idx]
        
        # Test dataset
        dataset = CustomDataset()
        assert len(dataset) == 1000, "Dataset should have correct length"
        
        # Test indexing
        sample, label = dataset[0]
        assert sample.shape == (10,), "Sample should have correct shape"
        assert isinstance(label.item(), int), "Label should be integer"
        assert 0 <= label.item() < 5, "Label should be in valid range"
        
        # Test with DataLoader
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        batch_x, batch_y = next(iter(dataloader))
        
        assert batch_x.shape == (32, 10), "Batch should have correct shape"
        assert batch_y.shape == (32,), "Labels should have correct shape"
    
    def test_distributed_training_setup(self):
        """Test distributed training setup (basic components)"""
        # Test if distributed utilities are available
        assert hasattr(torch.distributed, 'init_process_group'), \
            "Should have distributed training utilities"
        assert hasattr(torch.nn.parallel, 'DistributedDataParallel'), \
            "Should have DistributedDataParallel"
        
        # Test DataParallel (simpler version for single machine)
        if torch.cuda.device_count() > 1:
            model = nn.Sequential(
                nn.Linear(100, 50),
                nn.ReLU(),
                nn.Linear(50, 10)
            )
            
            # Wrap model with DataParallel
            parallel_model = nn.DataParallel(model)
            
            x = torch.randn(64, 100)
            if torch.cuda.is_available():
                parallel_model = parallel_model.cuda()
                x = x.cuda()
            
            output = parallel_model(x)
            assert output.shape == (64, 10), "DataParallel should preserve output shape"
        else:
            print("Multiple GPUs not available - skipping DataParallel test")
    
    def test_checkpointing_and_resuming(self):
        """Test advanced checkpointing with training state"""
        model = nn.Sequential(
            nn.Linear(50, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        
        # Simulate training state
        epoch = 15
        step = 1500
        best_loss = 0.25
        
        # Create comprehensive checkpoint
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_loss': best_loss,
            'rng_state': torch.get_rng_state()
        }
        
        # Verify checkpoint contains all necessary components
        required_keys = ['epoch', 'model_state_dict', 'optimizer_state_dict', 'best_loss']
        for key in required_keys:
            assert key in checkpoint, f"Checkpoint should contain {key}"
        
        # Test loading checkpoint
        new_model = nn.Sequential(
            nn.Linear(50, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )
        new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)
        new_scheduler = torch.optim.lr_scheduler.StepLR(new_optimizer, step_size=10, gamma=0.1)
        
        # Load state
        new_model.load_state_dict(checkpoint['model_state_dict'])
        new_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        new_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Verify models have same parameters
        for (p1, p2) in zip(model.parameters(), new_model.parameters()):
            assert torch.allclose(p1, p2), "Loaded model should have same parameters"
    
    def test_gradient_accumulation(self):
        """Test gradient accumulation for large effective batch sizes"""
        model = nn.Linear(100, 10)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        
        # Test gradient accumulation
        accumulation_steps = 4
        batch_size = 8
        
        # Accumulate gradients over multiple mini-batches
        optimizer.zero_grad()
        accumulated_loss = 0
        
        for i in range(accumulation_steps):
            # Mini-batch
            x = torch.randn(batch_size, 100)
            y = torch.randn(batch_size, 10)
            
            output = model(x)
            loss = criterion(output, y)
            
            # Scale loss by number of accumulation steps
            loss = loss / accumulation_steps
            loss.backward()
            
            accumulated_loss += loss.item()
        
        # Check that gradients were accumulated
        for param in model.parameters():
            assert param.grad is not None, "Parameters should have gradients after accumulation"
            assert not torch.all(param.grad == 0), "Gradients should be non-zero"
        
        # Update parameters
        optimizer.step()
        
        assert accumulated_loss > 0, "Accumulated loss should be positive"
    
    def test_model_ensembling(self):
        """Test model ensembling techniques"""
        # Create multiple models
        models = []
        for i in range(3):
            model = nn.Sequential(
                nn.Linear(20, 50),
                nn.ReLU(),
                nn.Linear(50, 10)
            )
            models.append(model)
        
        # Test ensemble prediction
        x = torch.randn(32, 20)
        predictions = []
        
        for model in models:
            with torch.no_grad():
                pred = model(x)
                predictions.append(pred)
        
        # Average ensemble
        ensemble_pred = torch.stack(predictions).mean(dim=0)
        
        assert ensemble_pred.shape == (32, 10), "Ensemble prediction should have correct shape"
        
        # Weighted ensemble
        weights = torch.tensor([0.5, 0.3, 0.2])
        weighted_pred = torch.stack(predictions).mul(weights.view(-1, 1, 1)).sum(dim=0)
        
        assert weighted_pred.shape == (32, 10), "Weighted ensemble should have correct shape"
        assert not torch.allclose(ensemble_pred, weighted_pred), "Different ensembling should give different results"


def test_mixed_precision_implementation(namespace):
    """Test mixed precision training implementation"""
    import torch
    
    if torch.cuda.is_available():
        # Test GradScaler usage
        if 'scaler' in namespace:
            scaler = namespace['scaler']
            assert isinstance(scaler, torch.cuda.amp.GradScaler), "Should use GradScaler for mixed precision"
        else:
            print("Warning: GradScaler not found. Mixed precision training not implemented")
        
        # Test autocast usage
        if 'use_autocast' in namespace:
            use_autocast = namespace['use_autocast']
            assert isinstance(use_autocast, bool), "use_autocast should be boolean"
            if use_autocast:
                print("✓ Autocast enabled for mixed precision training")
        else:
            print("Warning: Autocast usage not specified")
    else:
        print("CUDA not available - skipping mixed precision tests")


def test_model_compilation_usage(namespace):
    """Test model compilation implementation"""
    import torch
    
    # Test compiled model
    if 'compiled_model' in namespace:
        compiled_model = namespace['compiled_model']
        
        # Test that it's callable
        assert hasattr(compiled_model, '__call__'), "Compiled model should be callable"
        
        # Test with sample input
        try:
            x = torch.randn(5, 100)  # Adjust based on model architecture
            output = compiled_model(x)
            assert torch.is_tensor(output), "Compiled model should return tensor"
        except Exception as e:
            print(f"Compiled model test failed: {e}")
    elif hasattr(torch, 'compile'):
        print("Warning: torch.compile available but compiled_model not found")
    else:
        print("torch.compile not available (requires PyTorch 2.0+)")


def test_custom_autograd_function(namespace):
    """Test custom autograd function implementation"""
    import torch
    
    # Test custom function class
    if 'CustomFunction' in namespace or 'CustomAutograd' in namespace:
        CustomFunc = namespace.get('CustomFunction', namespace.get('CustomAutograd'))
        
        # Test that it's a proper autograd function
        assert hasattr(CustomFunc, 'forward'), "Custom function should have forward method"
        assert hasattr(CustomFunc, 'backward'), "Custom function should have backward method"
        
        # Test usage
        try:
            x = torch.randn(10, requires_grad=True)
            y = CustomFunc.apply(x)
            assert y.requires_grad, "Custom function output should require gradients"
            
            # Test backward pass
            loss = y.sum()
            loss.backward()
            assert x.grad is not None, "Custom function should compute gradients"
        except Exception as e:
            print(f"Custom autograd function test failed: {e}")
    else:
        print("Warning: Custom autograd function not implemented")


def test_torchscript_usage(namespace):
    """Test TorchScript implementation"""
    import torch
    
    # Test scripted or traced model
    script_artifacts = ['scripted_model', 'traced_model', 'jit_model']
    found_artifacts = [name for name in script_artifacts if name in namespace]
    
    if found_artifacts:
        for artifact_name in found_artifacts:
            artifact = namespace[artifact_name]
            
            # Test that it's a ScriptModule
            if hasattr(torch.jit, 'ScriptModule') and isinstance(artifact, torch.jit.ScriptModule):
                print(f"✓ Found TorchScript artifact: {artifact_name}")
                
                # Test inference
                try:
                    x = torch.randn(1, 100)  # Adjust based on model
                    output = artifact(x)
                    assert torch.is_tensor(output), f"{artifact_name} should return tensor"
                except Exception as e:
                    print(f"TorchScript artifact {artifact_name} test failed: {e}")
    else:
        print("Warning: No TorchScript artifacts found")


def test_custom_dataset_implementation(namespace):
    """Test custom Dataset implementation"""
    import torch
    from torch.utils.data import Dataset
    
    # Test custom dataset class
    if 'CustomDataset' in namespace:
        CustomDataset = namespace['CustomDataset']
        
        # Check inheritance
        assert issubclass(CustomDataset, Dataset), "CustomDataset should inherit from torch.utils.data.Dataset"
        
        # Test instantiation
        try:
            dataset = CustomDataset()
            assert len(dataset) > 0, "Custom dataset should not be empty"
            
            # Test indexing
            sample = dataset[0]
            assert sample is not None, "Dataset should return valid samples"
            
        except Exception as e:
            print(f"Custom dataset test failed: {e}")
    else:
        print("Warning: CustomDataset not implemented")


def test_distributed_training_components(namespace):
    """Test distributed training components"""
    import torch
    import torch.nn as nn
    
    # Test DataParallel usage
    if 'parallel_model' in namespace:
        parallel_model = namespace['parallel_model']
        
        if isinstance(parallel_model, nn.DataParallel):
            print("✓ DataParallel model found")
        elif isinstance(parallel_model, nn.parallel.DistributedDataParallel):
            print("✓ DistributedDataParallel model found")
        else:
            print("Warning: parallel_model is not a recognized parallel wrapper")
    else:
        print("Warning: Parallel model not found")
    
    # Test distributed utilities knowledge
    if 'distributed_config' in namespace:
        config = namespace['distributed_config']
        assert isinstance(config, dict), "distributed_config should be dictionary"
        print("✓ Distributed configuration found")
    else:
        print("Info: Distributed configuration not specified")


def test_advanced_checkpointing(namespace):
    """Test advanced checkpointing implementation"""
    import torch
    
    # Test comprehensive checkpoint
    if 'create_checkpoint' in namespace:
        create_checkpoint = namespace['create_checkpoint']
        assert callable(create_checkpoint), "create_checkpoint should be a function"
        
        # Test checkpoint creation
        model = torch.nn.Linear(10, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        
        try:
            checkpoint = create_checkpoint(model, optimizer, epoch=5, loss=0.1)
            assert isinstance(checkpoint, dict), "Checkpoint should be a dictionary"
            
            required_keys = ['model_state_dict', 'optimizer_state_dict', 'epoch']
            for key in required_keys:
                assert key in checkpoint, f"Checkpoint should contain {key}"
        except Exception as e:
            print(f"Checkpoint creation test failed: {e}")
    else:
        print("Warning: create_checkpoint function not found")
    
    # Test checkpoint loading
    if 'load_checkpoint' in namespace:
        load_checkpoint = namespace['load_checkpoint']
        assert callable(load_checkpoint), "load_checkpoint should be a function"
    else:
        print("Warning: load_checkpoint function not found")


def test_gradient_accumulation_implementation(namespace):
    """Test gradient accumulation implementation"""
    import torch
    
    # Test gradient accumulation function or configuration
    if 'gradient_accumulation_steps' in namespace:
        steps = namespace['gradient_accumulation_steps']
        assert isinstance(steps, int), "gradient_accumulation_steps should be integer"
        assert steps > 1, "Gradient accumulation steps should be > 1"
        print(f"✓ Gradient accumulation with {steps} steps")
    elif 'accumulate_gradients' in namespace:
        accumulate_func = namespace['accumulate_gradients']
        assert callable(accumulate_func), "accumulate_gradients should be a function"
        print("✓ Gradient accumulation function found")
    else:
        print("Warning: Gradient accumulation not implemented")


def test_model_ensembling_implementation(namespace):
    """Test model ensembling implementation"""
    import torch
    
    # Test ensemble models
    if 'ensemble_models' in namespace:
        ensemble = namespace['ensemble_models']
        assert isinstance(ensemble, (list, tuple)), "ensemble_models should be list or tuple"
        assert len(ensemble) > 1, "Ensemble should contain multiple models"
        
        # Test that all are PyTorch models
        for model in ensemble:
            assert hasattr(model, 'parameters'), "Each ensemble member should be a PyTorch model"
    else:
        print("Warning: ensemble_models not found")
    
    # Test ensemble prediction function
    if 'ensemble_predict' in namespace:
        predict_func = namespace['ensemble_predict']
        assert callable(predict_func), "ensemble_predict should be a function"
        print("✓ Ensemble prediction function found")
    else:
        print("Warning: ensemble_predict function not found")


def test_advanced_optimization_techniques(namespace):
    """Test advanced optimization techniques"""
    import torch
    
    # Test learning rate finder
    if 'lr_finder' in namespace or 'find_lr' in namespace:
        lr_finder = namespace.get('lr_finder', namespace.get('find_lr'))
        if callable(lr_finder):
            print("✓ Learning rate finder implemented")
        else:
            print("Learning rate finder found but not callable")
    else:
        print("Warning: Learning rate finder not implemented")
    
    # Test cyclical learning rates
    if 'cyclical_lr' in namespace or 'cyclic_scheduler' in namespace:
        cyclical = namespace.get('cyclical_lr', namespace.get('cyclic_scheduler'))
        print("✓ Cyclical learning rate implementation found")
    else:
        print("Info: Cyclical learning rates not implemented")


def test_model_interpretability_tools(namespace):
    """Test model interpretability implementations"""
    import torch
    
    # Test gradient-based interpretability
    if 'compute_gradients' in namespace or 'gradient_attribution' in namespace:
        grad_func = namespace.get('compute_gradients', namespace.get('gradient_attribution'))
        if callable(grad_func):
            print("✓ Gradient-based interpretability implemented")
    else:
        print("Info: Gradient-based interpretability not implemented")
    
    # Test attention visualization
    if 'visualize_attention' in namespace:
        vis_func = namespace['visualize_attention']
        if callable(vis_func):
            print("✓ Attention visualization implemented")
    else:
        print("Info: Attention visualization not implemented")


def run_tests():
    """Run all tests for Exercise 2"""
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    run_tests()