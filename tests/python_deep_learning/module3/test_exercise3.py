#!/usr/bin/env python3
"""
Test suite for Python Deep Learning Module 3 - Exercise 3: Data & Optimization
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import torchvision.transforms as transforms
import numpy as np
import pytest
import math
from typing import Optional, Callable, Tuple, List


class TestExercise3:
    """Test cases for Exercise 3: Data & Optimization"""
    
    def setup_method(self):
        """Set up test fixtures"""
        torch.manual_seed(42)
        np.random.seed(42)
    
    def test_data_augmentation(self):
        """Test data augmentation techniques"""
        # Create dummy image data
        dummy_image = torch.randn(3, 32, 32)
        
        # Test various transformations
        transform_list = [
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomResizedCrop(32, scale=(0.8, 1.0))
        ]
        
        for transform in transform_list:
            augmented = transform(dummy_image)
            assert augmented.shape == dummy_image.shape, f"Transform should preserve shape"
            assert torch.is_tensor(augmented), "Transform should return tensor"
    
    def test_normalization_transforms(self):
        """Test data normalization"""
        # Test standard normalization
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225])
        
        # Create dummy RGB image
        image = torch.rand(3, 32, 32)
        normalized = normalize(image)
        
        assert normalized.shape == image.shape, "Normalization should preserve shape"
        assert not torch.allclose(normalized, image), "Normalization should change values"
        
        # Test that normalization is approximately correct for uniform distribution
        uniform_image = torch.ones(3, 32, 32) * 0.5
        normalized_uniform = normalize(uniform_image)
        
        # Should be approximately centered around expected values
        expected_means = torch.tensor([(0.5 - 0.485) / 0.229, 
                                     (0.5 - 0.456) / 0.224, 
                                     (0.5 - 0.406) / 0.225])
        
        actual_means = normalized_uniform.mean(dim=(1, 2))
        assert torch.allclose(actual_means, expected_means, atol=1e-6), "Normalization should be correct"
    
    def test_train_val_split(self):
        """Test train/validation split"""
        # Create dataset
        X = torch.randn(1000, 10)
        y = torch.randint(0, 5, (1000,))
        dataset = TensorDataset(X, y)
        
        # Test split
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        assert len(train_dataset) == train_size, f"Train size should be {train_size}"
        assert len(val_dataset) == val_size, f"Validation size should be {val_size}"
        assert len(train_dataset) + len(val_dataset) == len(dataset), "Split should preserve total size"
        
        # Test that splits are different
        train_indices = train_dataset.indices
        val_indices = val_dataset.indices
        assert len(set(train_indices) & set(val_indices)) == 0, "Train and val should not overlap"
    
    def test_learning_rate_scheduling(self):
        """Test learning rate scheduling"""
        model = nn.Linear(10, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        
        # Test StepLR scheduler
        step_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        
        initial_lr = optimizer.param_groups[0]['lr']
        assert initial_lr == 0.1, "Initial learning rate should be 0.1"
        
        # After 9 steps, LR should be unchanged
        for _ in range(9):
            step_scheduler.step()
        assert optimizer.param_groups[0]['lr'] == 0.1, "LR should be unchanged before step_size"
        
        # After step 10, LR should be reduced
        step_scheduler.step()
        assert optimizer.param_groups[0]['lr'] == 0.01, "LR should be reduced after step_size"
        
        # Test ExponentialLR scheduler
        optimizer2 = torch.optim.SGD(model.parameters(), lr=0.1)
        exp_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer2, gamma=0.9)
        
        initial_lr = optimizer2.param_groups[0]['lr']
        exp_scheduler.step()
        new_lr = optimizer2.param_groups[0]['lr']
        
        assert abs(new_lr - initial_lr * 0.9) < 1e-6, "Exponential scheduler should multiply by gamma"
    
    def test_advanced_optimizers(self):
        """Test advanced optimizer configurations"""
        model = nn.Sequential(
            nn.Linear(20, 50),
            nn.ReLU(),
            nn.Linear(50, 10)
        )
        
        # Test Adam with different parameters
        adam_optimizer = torch.optim.Adam(model.parameters(), 
                                        lr=0.001, 
                                        betas=(0.9, 0.999), 
                                        eps=1e-8, 
                                        weight_decay=1e-4)
        
        assert adam_optimizer.param_groups[0]['lr'] == 0.001, "Learning rate should be set correctly"
        assert adam_optimizer.param_groups[0]['betas'] == (0.9, 0.999), "Betas should be set correctly"
        assert adam_optimizer.param_groups[0]['weight_decay'] == 1e-4, "Weight decay should be set correctly"
        
        # Test AdamW (Adam with decoupled weight decay)
        adamw_optimizer = torch.optim.AdamW(model.parameters(), 
                                          lr=0.001, 
                                          weight_decay=0.01)
        
        assert adamw_optimizer.param_groups[0]['weight_decay'] == 0.01, "AdamW weight decay should be set"
        
        # Test RMSprop
        rmsprop_optimizer = torch.optim.RMSprop(model.parameters(), 
                                              lr=0.01, 
                                              alpha=0.99)
        
        assert rmsprop_optimizer.param_groups[0]['lr'] == 0.01, "RMSprop lr should be set correctly"
        assert rmsprop_optimizer.param_groups[0]['alpha'] == 0.99, "RMSprop alpha should be set correctly"
    
    def test_gradient_clipping(self):
        """Test gradient clipping implementation"""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 1)
        )
        
        # Create data that might cause large gradients
        x = torch.randn(32, 10) * 10  # Large input values
        y = torch.randn(32, 1) * 100  # Large target values
        
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        criterion = nn.MSELoss()
        
        # Forward pass
        output = model(x)
        loss = criterion(output, y)
        
        # Backward pass
        loss.backward()
        
        # Compute gradient norms before clipping
        total_norm_before = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm_before += param_norm.item() ** 2
        total_norm_before = total_norm_before ** 0.5
        
        # Apply gradient clipping
        max_norm = 1.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        
        # Compute gradient norms after clipping
        total_norm_after = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm_after += param_norm.item() ** 2
        total_norm_after = total_norm_after ** 0.5
        
        assert total_norm_after <= max_norm + 1e-6, f"Gradient norm should be clipped to {max_norm}"
        
        if total_norm_before > max_norm:
            assert total_norm_after < total_norm_before, "Gradient norm should be reduced by clipping"
    
    def test_regularization_techniques(self):
        """Test regularization techniques"""
        # Test L2 regularization via weight decay
        model = nn.Linear(10, 1)
        
        # Optimizer with weight decay
        optimizer_l2 = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.01)
        
        # Create simple training step
        x = torch.randn(32, 10)
        y = torch.randn(32, 1)
        criterion = nn.MSELoss()
        
        # Training step with L2 regularization
        optimizer_l2.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer_l2.step()
        
        # Test dropout regularization
        model_with_dropout = nn.Sequential(
            nn.Linear(20, 50),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(50, 10),
            nn.Dropout(0.3)
        )
        
        # Test dropout behavior in training vs eval mode
        x_test = torch.randn(32, 20)
        
        model_with_dropout.train()
        output_train = model_with_dropout(x_test)
        
        model_with_dropout.eval()
        output_eval = model_with_dropout(x_test)
        
        assert output_train.shape == output_eval.shape, "Outputs should have same shape"
        # In most cases, outputs will be different due to dropout
        # assert not torch.allclose(output_train, output_eval), "Dropout should cause different outputs in train vs eval"
    
    def test_batch_size_effects(self):
        """Test understanding of batch size effects"""
        model = nn.Sequential(nn.Linear(10, 1))
        criterion = nn.MSELoss()
        
        X = torch.randn(1000, 10)
        y = torch.randn(1000, 1)
        
        # Test different batch sizes
        batch_sizes = [32, 64, 128]
        gradient_norms = {}
        
        for batch_size in batch_sizes:
            model_copy = nn.Sequential(nn.Linear(10, 1))
            model_copy.load_state_dict(model.state_dict())
            
            dataset = TensorDataset(X, y)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            
            # Take one batch and compute gradient
            batch_x, batch_y = next(iter(dataloader))
            output = model_copy(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            
            # Compute gradient norm
            total_norm = 0
            for p in model_copy.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            
            gradient_norms[batch_size] = total_norm
        
        # Gradient norms should be different for different batch sizes
        norm_values = list(gradient_norms.values())
        assert len(set(norm_values)) > 1, "Different batch sizes should produce different gradient norms"


def test_data_augmentation_implementation(namespace):
    """Test data augmentation implementation"""
    import torch
    import torchvision.transforms as transforms
    
    # Test data augmentation transforms
    if 'augmentation_transforms' in namespace:
        aug_transforms = namespace['augmentation_transforms']
        assert isinstance(aug_transforms, (transforms.Compose, list)), "Should use torchvision transforms"
        
        # Test that augmentation works
        dummy_image = torch.randn(3, 32, 32)
        if isinstance(aug_transforms, transforms.Compose):
            augmented = aug_transforms(dummy_image)
            assert torch.is_tensor(augmented), "Augmentation should return tensor"
    else:
        raise AssertionError("augmentation_transforms not found. Please implement data augmentation")


def test_normalization_implementation(namespace):
    """Test data normalization implementation"""
    import torch
    import torchvision.transforms as transforms
    
    # Test normalization
    if 'normalize_transform' in namespace:
        normalize = namespace['normalize_transform']
        
        # Test normalization
        image = torch.rand(3, 32, 32)
        normalized = normalize(image)
        assert normalized.shape == image.shape, "Normalization should preserve shape"
        assert not torch.allclose(normalized, image), "Normalization should change values"
    else:
        print("Warning: normalize_transform not found. Consider implementing normalization")


def test_dataset_splitting(namespace):
    """Test train/validation/test split implementation"""
    import torch
    from torch.utils.data import random_split
    
    # Test dataset splits
    datasets = ['train_dataset', 'val_dataset']
    for dataset_name in datasets:
        if dataset_name in namespace:
            dataset = namespace[dataset_name]
            assert len(dataset) > 0, f"{dataset_name} should not be empty"
        else:
            raise AssertionError(f"{dataset_name} not found. Please implement dataset splitting")
    
    # Test split ratios
    if 'train_dataset' in namespace and 'val_dataset' in namespace:
        train_size = len(namespace['train_dataset'])
        val_size = len(namespace['val_dataset'])
        total_size = train_size + val_size
        
        train_ratio = train_size / total_size
        assert 0.6 <= train_ratio <= 0.9, f"Train ratio should be reasonable, got {train_ratio:.2f}"


def test_learning_rate_scheduler_usage(namespace):
    """Test learning rate scheduler implementation"""
    import torch
    
    # Test scheduler
    if 'scheduler' in namespace or 'lr_scheduler' in namespace:
        scheduler = namespace.get('scheduler', namespace.get('lr_scheduler'))
        assert hasattr(scheduler, 'step'), "Scheduler should have step method"
        
        # Test that scheduler affects learning rate
        if hasattr(scheduler, 'get_last_lr'):
            initial_lr = scheduler.get_last_lr()
            scheduler.step()
            new_lr = scheduler.get_last_lr()
            # Learning rate might change depending on scheduler type
    else:
        raise AssertionError("Learning rate scheduler not found. Please implement LR scheduling")


def test_advanced_optimizer_usage(namespace):
    """Test usage of advanced optimizers"""
    import torch
    
    # Test that advanced optimizers were used
    optimizers = ['adam_optimizer', 'adamw_optimizer', 'rmsprop_optimizer']
    found_optimizers = []
    
    for opt_name in optimizers:
        if opt_name in namespace:
            optimizer = namespace[opt_name]
            assert hasattr(optimizer, 'step'), f"{opt_name} should have step method"
            found_optimizers.append(opt_name)
    
    assert len(found_optimizers) >= 1, "Should use at least one advanced optimizer"


def test_gradient_clipping_implementation(namespace):
    """Test gradient clipping implementation"""
    import torch
    
    # Test gradient clipping function
    if 'apply_gradient_clipping' in namespace:
        clip_func = namespace['apply_gradient_clipping']
        assert callable(clip_func), "apply_gradient_clipping should be a function"
    elif 'max_grad_norm' in namespace:
        max_norm = namespace['max_grad_norm']
        assert isinstance(max_norm, (int, float)), "max_grad_norm should be a number"
        assert max_norm > 0, "max_grad_norm should be positive"
    else:
        print("Warning: Gradient clipping not implemented")


def test_regularization_implementation(namespace):
    """Test regularization techniques implementation"""
    import torch
    import torch.nn as nn
    
    # Test L2 regularization (weight decay)
    if 'optimizer_with_l2' in namespace:
        optimizer = namespace['optimizer_with_l2']
        weight_decay = optimizer.param_groups[0].get('weight_decay', 0)
        assert weight_decay > 0, "Should use weight decay for L2 regularization"
    else:
        print("Warning: L2 regularization (weight decay) not found")
    
    # Test dropout usage
    if 'model_with_dropout' in namespace:
        model = namespace['model_with_dropout']
        has_dropout = any(isinstance(module, nn.Dropout) for module in model.modules())
        assert has_dropout, "Model should contain dropout layers"
    else:
        print("Warning: Dropout regularization not found")
    
    # Test batch normalization
    if 'model_with_batchnorm' in namespace:
        model = namespace['model_with_batchnorm']
        has_batchnorm = any(isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)) 
                           for module in model.modules())
        assert has_batchnorm, "Model should contain batch normalization layers"
    else:
        print("Warning: Batch normalization not found")


def test_hyperparameter_tuning(namespace):
    """Test hyperparameter tuning implementation"""
    import torch
    
    # Test hyperparameter grid search or results
    if 'hyperparameter_grid' in namespace:
        grid = namespace['hyperparameter_grid']
        assert isinstance(grid, (dict, list)), "hyperparameter_grid should be dict or list"
        assert len(grid) > 1, "Should test multiple hyperparameter combinations"
    elif 'hp_results' in namespace or 'hyperparameter_results' in namespace:
        results = namespace.get('hp_results', namespace.get('hyperparameter_results'))
        assert isinstance(results, (dict, list)), "Hyperparameter results should be dict or list"
        assert len(results) > 1, "Should have results for multiple configurations"
    else:
        print("Warning: Hyperparameter tuning not implemented")


def test_batch_size_analysis(namespace):
    """Test batch size analysis"""
    import torch
    
    # Test batch size experiments
    if 'batch_size_results' in namespace:
        results = namespace['batch_size_results']
        assert isinstance(results, dict), "batch_size_results should be a dictionary"
        assert len(results) >= 2, "Should test at least 2 different batch sizes"
        
        # Check that different batch sizes were tested
        batch_sizes = list(results.keys())
        assert len(set(batch_sizes)) > 1, "Should test different batch sizes"
    else:
        print("Warning: Batch size analysis not implemented")


def test_data_pipeline_efficiency(namespace):
    """Test data loading efficiency optimizations"""
    import torch
    from torch.utils.data import DataLoader
    
    # Test dataloader configurations
    if 'optimized_dataloader' in namespace:
        dataloader = namespace['optimized_dataloader']
        assert isinstance(dataloader, DataLoader), "Should use DataLoader"
        
        # Check for efficiency settings
        assert dataloader.num_workers >= 0, "Should consider num_workers setting"
        # pin_memory is beneficial for GPU training
        if torch.cuda.is_available():
            assert hasattr(dataloader, 'pin_memory'), "Should consider pin_memory for GPU training"
    else:
        print("Warning: Optimized dataloader not found")


def test_training_optimization_techniques(namespace):
    """Test various training optimization techniques"""
    import torch
    
    # Test mixed precision training setup
    if 'use_mixed_precision' in namespace:
        use_mixed_precision = namespace['use_mixed_precision']
        assert isinstance(use_mixed_precision, bool), "use_mixed_precision should be boolean"
        
        if use_mixed_precision:
            # Check for scaler
            if 'scaler' in namespace:
                scaler = namespace['scaler']
                assert hasattr(scaler, 'scale'), "Should use GradScaler for mixed precision"
    else:
        print("Warning: Mixed precision training not implemented")
    
    # Test model compilation (PyTorch 2.0+)
    if 'compiled_model' in namespace:
        compiled_model = namespace['compiled_model']
        assert torch.jit.is_scripting() or hasattr(compiled_model, '__call__'), "Should use model compilation"
    else:
        print("Warning: Model compilation not used")


def run_tests():
    """Run all tests for Exercise 3"""
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    run_tests()