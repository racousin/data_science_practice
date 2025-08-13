#!/usr/bin/env python3
"""
Test suite for Python Deep Learning Module 3 - Exercise 2: Complete Training Pipeline
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pytest
import math
from typing import Optional, Callable, Tuple, List


class TestExercise2:
    """Test cases for Exercise 2: Complete Training Pipeline"""
    
    def setup_method(self):
        """Set up test fixtures"""
        torch.manual_seed(42)
        np.random.seed(42)
    
    def test_dataset_creation(self):
        """Test dataset creation and data loading"""
        # Create sample dataset
        X = torch.randn(1000, 20)
        y = torch.randint(0, 5, (1000,))
        
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Test dataset properties
        assert len(dataset) == 1000, "Dataset should have 1000 samples"
        
        # Test dataloader
        batch_x, batch_y = next(iter(dataloader))
        assert batch_x.shape == (32, 20), f"Batch X shape should be (32, 20), got {batch_x.shape}"
        assert batch_y.shape == (32,), f"Batch y shape should be (32,), got {batch_y.shape}"
        assert batch_y.dtype == torch.long, "Labels should be long type for classification"
    
    def test_loss_functions(self):
        """Test different loss functions"""
        # Test Cross Entropy Loss
        logits = torch.randn(10, 5)
        targets = torch.randint(0, 5, (10,))
        
        ce_loss = F.cross_entropy(logits, targets)
        assert ce_loss.item() > 0, "Cross entropy loss should be positive"
        
        # Test MSE Loss for regression
        predictions = torch.randn(10, 1)
        targets_reg = torch.randn(10, 1)
        
        mse_loss = F.mse_loss(predictions, targets_reg)
        assert mse_loss.item() >= 0, "MSE loss should be non-negative"
        
        # Test that losses have gradients
        ce_loss.backward(retain_graph=True)
        mse_loss.backward()
    
    def test_training_loop_components(self):
        """Test components of training loop"""
        # Create simple model and data
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        )
        
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        X = torch.randn(100, 10)
        y = torch.randint(0, 5, (100,))
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=20)
        
        # Test one training step
        model.train()
        initial_loss = float('inf')
        
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            if initial_loss == float('inf'):
                initial_loss = loss.item()
            
            break  # Just test one batch
        
        assert loss.item() > 0, "Loss should be positive"
        assert not torch.isnan(loss), "Loss should not be NaN"
        
        # Test that gradients were computed
        for param in model.parameters():
            assert param.grad is not None, "All parameters should have gradients"
    
    def test_validation_loop(self):
        """Test validation loop implementation"""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        )
        
        criterion = nn.CrossEntropyLoss()
        
        # Create validation data
        X_val = torch.randn(50, 10)
        y_val = torch.randint(0, 5, (50,))
        val_dataset = TensorDataset(X_val, y_val)
        val_dataloader = DataLoader(val_dataset, batch_size=10)
        
        # Test validation loop
        model.eval()
        val_losses = []
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_dataloader:
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_losses.append(loss.item())
                
                _, predicted = torch.max(outputs.data, 1)
                total_predictions += batch_y.size(0)
                correct_predictions += (predicted == batch_y).sum().item()
        
        avg_val_loss = sum(val_losses) / len(val_losses)
        accuracy = correct_predictions / total_predictions
        
        assert avg_val_loss > 0, "Validation loss should be positive"
        assert 0 <= accuracy <= 1, f"Accuracy should be between 0 and 1, got {accuracy}"
    
    def test_metrics_computation(self):
        """Test computation of training metrics"""
        # Simulate predictions and targets
        predictions = torch.tensor([[0.1, 0.9, 0.0], [0.8, 0.1, 0.1], [0.2, 0.3, 0.5]])
        targets = torch.tensor([1, 0, 2])
        
        # Test accuracy
        _, predicted_classes = torch.max(predictions, 1)
        accuracy = (predicted_classes == targets).float().mean()
        
        assert accuracy.item() == 1.0, "All predictions should be correct in this test case"
        
        # Test top-k accuracy (for larger datasets)
        predictions_large = torch.randn(100, 10)
        targets_large = torch.randint(0, 10, (100,))
        
        _, top3_pred = torch.topk(predictions_large, 3, dim=1)
        top3_accuracy = (top3_pred == targets_large.unsqueeze(1)).any(dim=1).float().mean()
        
        assert 0 <= top3_accuracy <= 1, "Top-3 accuracy should be between 0 and 1"
    
    def test_early_stopping(self):
        """Test early stopping implementation"""
        class EarlyStopping:
            def __init__(self, patience=5, min_delta=0.001):
                self.patience = patience
                self.min_delta = min_delta
                self.best_loss = float('inf')
                self.wait = 0
                
            def __call__(self, val_loss):
                if val_loss < self.best_loss - self.min_delta:
                    self.best_loss = val_loss
                    self.wait = 0
                    return False
                else:
                    self.wait += 1
                    return self.wait >= self.patience
        
        # Test early stopping
        early_stopping = EarlyStopping(patience=3)
        
        # Simulate improving losses
        assert not early_stopping(1.0), "Should not stop with improving loss"
        assert not early_stopping(0.8), "Should not stop with improving loss"
        
        # Simulate stagnating losses
        assert not early_stopping(0.81), "Should not stop yet"
        assert not early_stopping(0.82), "Should not stop yet"
        assert early_stopping(0.83), "Should stop after patience exceeded"
    
    def test_model_checkpointing(self):
        """Test model checkpointing functionality"""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        )
        
        optimizer = torch.optim.Adam(model.parameters())
        
        # Create checkpoint
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': 10,
            'loss': 0.5
        }
        
        # Test that checkpoint contains required keys
        required_keys = ['model_state_dict', 'optimizer_state_dict', 'epoch', 'loss']
        for key in required_keys:
            assert key in checkpoint, f"Checkpoint should contain {key}"
        
        # Test loading checkpoint
        new_model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        )
        new_optimizer = torch.optim.Adam(new_model.parameters())
        
        new_model.load_state_dict(checkpoint['model_state_dict'])
        new_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Models should have same parameters
        for (p1, p2) in zip(model.parameters(), new_model.parameters()):
            assert torch.allclose(p1, p2), "Loaded model should have same parameters"


def test_dataset_and_dataloader(namespace):
    """Test dataset and dataloader implementation"""
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    
    # Test dataset creation
    if 'train_dataset' in namespace and 'train_dataloader' in namespace:
        train_dataset = namespace['train_dataset']
        train_dataloader = namespace['train_dataloader']
        
        assert len(train_dataset) > 0, "Dataset should not be empty"
        assert isinstance(train_dataloader, DataLoader), "Should use DataLoader"
        
        # Test batch loading
        batch_x, batch_y = next(iter(train_dataloader))
        assert torch.is_tensor(batch_x), "Batch X should be tensor"
        assert torch.is_tensor(batch_y), "Batch y should be tensor"
    else:
        raise AssertionError("train_dataset and train_dataloader not found")
    
    # Test validation dataset
    if 'val_dataset' in namespace and 'val_dataloader' in namespace:
        val_dataset = namespace['val_dataset']
        val_dataloader = namespace['val_dataloader']
        
        assert len(val_dataset) > 0, "Validation dataset should not be empty"
        assert isinstance(val_dataloader, DataLoader), "Should use DataLoader for validation"
    else:
        print("Warning: Validation dataset not found. Consider implementing train/val split")


def test_training_loop_implementation(namespace):
    """Test training loop implementation"""
    import torch
    import torch.nn as nn
    
    # Test training function or loop
    if 'train_model' in namespace or 'training_loop' in namespace:
        train_func = namespace.get('train_model', namespace.get('training_loop'))
        assert callable(train_func), "Training function should be callable"
    else:
        print("Warning: Training function not found")
    
    # Test that training was executed
    if 'training_losses' in namespace:
        losses = namespace['training_losses']
        assert isinstance(losses, list), "training_losses should be a list"
        assert len(losses) > 0, "Should have recorded some losses"
        assert all(loss >= 0 for loss in losses), "All losses should be non-negative"
    else:
        raise AssertionError("training_losses not found. Please track training losses")


def test_validation_implementation(namespace):
    """Test validation implementation"""
    import torch
    
    # Test validation function
    if 'validate_model' in namespace or 'validation_loop' in namespace:
        val_func = namespace.get('validate_model', namespace.get('validation_loop'))
        assert callable(val_func), "Validation function should be callable"
    else:
        print("Warning: Validation function not found")
    
    # Test validation results
    if 'val_losses' in namespace:
        val_losses = namespace['val_losses']
        assert isinstance(val_losses, list), "val_losses should be a list"
        assert len(val_losses) > 0, "Should have validation losses"
    else:
        raise AssertionError("val_losses not found. Please implement validation")
    
    # Test accuracy tracking
    if 'val_accuracies' in namespace:
        val_accuracies = namespace['val_accuracies']
        assert isinstance(val_accuracies, list), "val_accuracies should be a list"
        assert all(0 <= acc <= 1 for acc in val_accuracies), "Accuracies should be between 0 and 1"
    else:
        print("Warning: val_accuracies not found. Consider tracking validation accuracy")


def test_loss_function_usage(namespace):
    """Test loss function implementation and usage"""
    import torch
    import torch.nn as nn
    
    # Test loss function
    if 'criterion' in namespace or 'loss_function' in namespace:
        criterion = namespace.get('criterion', namespace.get('loss_function'))
        
        # Test that it's a proper loss function
        if hasattr(criterion, '__call__'):
            # Test with dummy data
            logits = torch.randn(10, 5)
            targets = torch.randint(0, 5, (10,))
            loss = criterion(logits, targets)
            assert torch.is_tensor(loss), "Loss should be a tensor"
            assert loss.item() >= 0, "Loss should be non-negative"
    else:
        raise AssertionError("criterion or loss_function not found")


def test_optimizer_configuration(namespace):
    """Test optimizer configuration"""
    import torch
    
    # Test optimizer
    if 'optimizer' in namespace:
        optimizer = namespace['optimizer']
        assert hasattr(optimizer, 'step'), "Optimizer should have step method"
        assert hasattr(optimizer, 'zero_grad'), "Optimizer should have zero_grad method"
        
        # Check learning rate
        lr = optimizer.param_groups[0]['lr']
        assert 0 < lr < 1, f"Learning rate should be reasonable, got {lr}"
    else:
        raise AssertionError("optimizer not found. Please configure optimizer")


def test_metrics_tracking(namespace):
    """Test metrics tracking implementation"""
    import torch
    
    # Test accuracy computation
    if 'compute_accuracy' in namespace:
        compute_accuracy = namespace['compute_accuracy']
        assert callable(compute_accuracy), "compute_accuracy should be a function"
        
        # Test the function
        predictions = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
        targets = torch.tensor([1, 0])
        accuracy = compute_accuracy(predictions, targets)
        assert accuracy == 1.0, "Should compute 100% accuracy for this test case"
    else:
        print("Warning: compute_accuracy function not found")
    
    # Test that metrics were tracked
    if 'train_accuracies' in namespace:
        train_accuracies = namespace['train_accuracies']
        assert isinstance(train_accuracies, list), "train_accuracies should be a list"
        assert all(0 <= acc <= 1 for acc in train_accuracies), "Accuracies should be in [0,1]"
    else:
        print("Warning: train_accuracies not found. Consider tracking training accuracy")


def test_early_stopping_implementation(namespace):
    """Test early stopping implementation"""
    import torch
    
    # Test early stopping class or function
    if 'EarlyStopping' in namespace:
        EarlyStopping = namespace['EarlyStopping']
        
        # Test instantiation
        early_stopping = EarlyStopping(patience=3)
        
        # Test early stopping logic
        assert not early_stopping(1.0), "Should not stop initially"
        assert not early_stopping(0.9), "Should not stop with improvement"
        assert not early_stopping(0.91), "Should not stop yet"
        assert not early_stopping(0.92), "Should not stop yet"
    elif 'early_stopping' in namespace:
        early_stopping = namespace['early_stopping']
        assert callable(early_stopping), "early_stopping should be callable"
    else:
        print("Warning: Early stopping not implemented")


def test_model_checkpointing(namespace):
    """Test model checkpointing implementation"""
    import torch
    
    # Test checkpoint saving
    if 'save_checkpoint' in namespace:
        save_checkpoint = namespace['save_checkpoint']
        assert callable(save_checkpoint), "save_checkpoint should be a function"
    else:
        print("Warning: save_checkpoint function not found")
    
    # Test checkpoint loading
    if 'load_checkpoint' in namespace:
        load_checkpoint = namespace['load_checkpoint']
        assert callable(load_checkpoint), "load_checkpoint should be a function"
    else:
        print("Warning: load_checkpoint function not found")
    
    # Test that checkpoints were created
    if 'best_checkpoint' in namespace:
        checkpoint = namespace['best_checkpoint']
        assert isinstance(checkpoint, dict), "Checkpoint should be a dictionary"
        
        required_keys = ['model_state_dict', 'optimizer_state_dict', 'epoch', 'loss']
        for key in required_keys:
            assert key in checkpoint, f"Checkpoint should contain {key}"
    else:
        print("Warning: best_checkpoint not found. Consider implementing checkpointing")


def test_learning_curves(namespace):
    """Test learning curves plotting and analysis"""
    import torch
    
    # Test that learning curves data exists
    if 'plot_learning_curves' in namespace:
        plot_func = namespace['plot_learning_curves']
        assert callable(plot_func), "plot_learning_curves should be a function"
    else:
        print("Warning: plot_learning_curves not found")
    
    # Test training history
    if 'training_history' in namespace:
        history = namespace['training_history']
        assert isinstance(history, dict), "training_history should be a dictionary"
        
        expected_keys = ['train_loss', 'val_loss']
        for key in expected_keys:
            assert key in history, f"History should track {key}"
    else:
        print("Warning: training_history not found. Consider tracking training history")


def test_hyperparameter_experiment(namespace):
    """Test hyperparameter experimentation"""
    import torch
    
    # Test hyperparameter grid or results
    if 'hyperparameter_results' in namespace:
        results = namespace['hyperparameter_results']
        assert isinstance(results, (dict, list)), "hyperparameter_results should be dict or list"
        assert len(results) > 1, "Should test multiple hyperparameter combinations"
    else:
        print("Warning: hyperparameter_results not found. Consider experimenting with hyperparameters")


def run_tests():
    """Run all tests for Exercise 2"""
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    run_tests()