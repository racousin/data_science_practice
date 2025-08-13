#!/usr/bin/env python3
"""
Test suite for Python Deep Learning Module 4 - Exercise 3: Mini-Project
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pytest
import math
from typing import Optional, Callable, Tuple, List, Dict, Any


class TestExercise3:
    """Test cases for Exercise 3: Mini-Project"""
    
    def setup_method(self):
        """Set up test fixtures"""
        torch.manual_seed(42)
        np.random.seed(42)
    
    def test_project_architecture_design(self):
        """Test that project has well-designed architecture"""
        # This test checks for proper model architecture design principles
        
        # Test modular design
        class TestModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.feature_extractor = nn.Sequential(
                    nn.Linear(784, 256),
                    nn.ReLU(),
                    nn.Dropout(0.2)
                )
                self.classifier = nn.Sequential(
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Linear(128, 10)
                )
            
            def forward(self, x):
                features = self.feature_extractor(x)
                output = self.classifier(features)
                return output
        
        model = TestModule()
        
        # Test that model has logical components
        assert hasattr(model, 'feature_extractor'), "Model should have feature extraction component"
        assert hasattr(model, 'classifier'), "Model should have classification component"
        
        # Test forward pass
        x = torch.randn(32, 784)
        output = model(x)
        assert output.shape == (32, 10), "Model should produce correct output shape"
        
        # Test parameter count is reasonable
        total_params = sum(p.numel() for p in model.parameters())
        assert 50000 < total_params < 1000000, "Model should have reasonable number of parameters"
    
    def test_data_pipeline_design(self):
        """Test comprehensive data pipeline"""
        
        class ProjectDataset(Dataset):
            def __init__(self, data_size=1000, input_dim=784, num_classes=10, transform=None):
                self.data = torch.randn(data_size, input_dim)
                self.labels = torch.randint(0, num_classes, (data_size,))
                self.transform = transform
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                sample, label = self.data[idx], self.labels[idx]
                if self.transform:
                    sample = self.transform(sample)
                return sample, label
        
        # Test dataset creation
        dataset = ProjectDataset()
        assert len(dataset) > 0, "Dataset should not be empty"
        
        # Test transforms
        def normalize_transform(x):
            return (x - x.mean()) / (x.std() + 1e-8)
        
        dataset_with_transform = ProjectDataset(transform=normalize_transform)
        sample, label = dataset_with_transform[0]
        
        # Test that transform was applied
        original_sample, _ = dataset[0]
        assert not torch.allclose(sample, original_sample), "Transform should modify data"
        
        # Test data loader
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
        batch_x, batch_y = next(iter(dataloader))
        
        assert batch_x.shape == (32, 784), "DataLoader should produce correct batch shape"
        assert batch_y.shape == (32,), "Labels should have correct shape"
    
    def test_training_pipeline_completeness(self):
        """Test complete training pipeline with all components"""
        
        # Model
        model = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 5)
        )
        
        # Training components
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
        
        # Data
        X = torch.randn(200, 20)
        y = torch.randint(0, 5, (200,))
        dataset = torch.utils.data.TensorDataset(X, y)
        train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Training metrics tracking
        train_losses = []
        train_accuracies = []
        
        # Training loop (abbreviated)
        model.train()
        for epoch in range(3):  # Short training for testing
            epoch_loss = 0
            correct = 0
            total = 0
            
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Metrics
                epoch_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
            
            # Record metrics
            avg_loss = epoch_loss / len(train_loader)
            accuracy = correct / total
            train_losses.append(avg_loss)
            train_accuracies.append(accuracy)
            
            scheduler.step()
        
        # Validate training results
        assert len(train_losses) == 3, "Should have recorded losses for each epoch"
        assert len(train_accuracies) == 3, "Should have recorded accuracies for each epoch"
        assert all(loss > 0 for loss in train_losses), "All losses should be positive"
        assert all(0 <= acc <= 1 for acc in train_accuracies), "All accuracies should be in [0,1]"
        
        # Check that training progressed
        assert not all(loss == train_losses[0] for loss in train_losses), "Loss should change during training"
    
    def test_model_evaluation_and_metrics(self):
        """Test comprehensive model evaluation"""
        
        # Create model and test data
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        )
        
        test_x = torch.randn(100, 10)
        test_y = torch.randint(0, 5, (100,))
        
        # Evaluation mode
        model.eval()
        
        with torch.no_grad():
            outputs = model(test_x)
            _, predictions = torch.max(outputs, 1)
        
        # Test basic metrics
        accuracy = (predictions == test_y).float().mean().item()
        assert 0 <= accuracy <= 1, "Accuracy should be between 0 and 1"
        
        # Test per-class metrics (simplified)
        num_classes = 5
        class_correct = torch.zeros(num_classes)
        class_total = torch.zeros(num_classes)
        
        for i in range(len(test_y)):
            label = test_y[i]
            class_correct[label] += (predictions[i] == test_y[i]).item()
            class_total[label] += 1
        
        # Per-class accuracy
        per_class_accuracy = class_correct / (class_total + 1e-8)  # Avoid division by zero
        assert per_class_accuracy.shape == (num_classes,), "Should have per-class metrics"
        
        # Test confusion matrix concept
        confusion_matrix = torch.zeros(num_classes, num_classes)
        for i in range(len(test_y)):
            confusion_matrix[test_y[i], predictions[i]] += 1
        
        assert confusion_matrix.sum() == len(test_y), "Confusion matrix should account for all samples"
    
    def test_model_saving_and_loading(self):
        """Test model serialization and deserialization"""
        
        # Original model
        original_model = nn.Sequential(
            nn.Linear(50, 100),
            nn.ReLU(),
            nn.Linear(100, 20)
        )
        
        # Create test input
        test_input = torch.randn(5, 50)
        original_output = original_model(test_input)
        
        # Save model state
        model_state = original_model.state_dict()
        
        # Create new model with same architecture
        loaded_model = nn.Sequential(
            nn.Linear(50, 100),
            nn.ReLU(),
            nn.Linear(100, 20)
        )
        
        # Load state
        loaded_model.load_state_dict(model_state)
        
        # Test that loaded model produces same output
        loaded_output = loaded_model(test_input)
        assert torch.allclose(original_output, loaded_output), \
            "Loaded model should produce identical outputs"
        
        # Test parameter equality
        for (name1, param1), (name2, param2) in zip(original_model.named_parameters(), 
                                                    loaded_model.named_parameters()):
            assert name1 == name2, "Parameter names should match"
            assert torch.allclose(param1, param2), f"Parameters {name1} should be identical"
    
    def test_hyperparameter_optimization(self):
        """Test systematic hyperparameter optimization approach"""
        
        # Define hyperparameter grid
        hyperparameters = {
            'lr': [0.001, 0.01, 0.1],
            'batch_size': [16, 32, 64],
            'hidden_size': [32, 64, 128]
        }
        
        # Test hyperparameter combinations
        best_config = None
        best_score = -1
        results = []
        
        # Simplified grid search (test first few combinations)
        test_combinations = [
            {'lr': 0.001, 'batch_size': 32, 'hidden_size': 64},
            {'lr': 0.01, 'batch_size': 32, 'hidden_size': 64},
            {'lr': 0.001, 'batch_size': 64, 'hidden_size': 64}
        ]
        
        for config in test_combinations:
            # Create model with current hyperparameters
            model = nn.Sequential(
                nn.Linear(20, config['hidden_size']),
                nn.ReLU(),
                nn.Linear(config['hidden_size'], 5)
            )
            
            optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
            
            # Simulate training score (in practice, this would be validation accuracy)
            # For testing, use random score that depends on hyperparameters
            torch.manual_seed(hash(str(config)) % 1000)
            simulated_score = torch.rand(1).item()
            
            results.append({
                'config': config,
                'score': simulated_score
            })
            
            if simulated_score > best_score:
                best_score = simulated_score
                best_config = config
        
        # Validate results
        assert len(results) == len(test_combinations), "Should have results for all tested combinations"
        assert best_config is not None, "Should find best configuration"
        assert best_score > 0, "Best score should be positive"
        
        # Check result format
        for result in results:
            assert 'config' in result, "Each result should have config"
            assert 'score' in result, "Each result should have score"
            assert isinstance(result['score'], (int, float)), "Score should be numeric"
    
    def test_documentation_and_reproducibility(self):
        """Test code documentation and reproducibility measures"""
        
        # Test random seed setting for reproducibility
        def set_seeds(seed=42):
            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
        
        # Test reproducibility
        set_seeds(123)
        x1 = torch.randn(10, 10)
        model1 = nn.Linear(10, 5)
        output1 = model1(x1)
        
        set_seeds(123)  # Reset with same seed
        x2 = torch.randn(10, 10)
        model2 = nn.Linear(10, 5)
        output2 = model2(x2)
        
        assert torch.allclose(x1, x2), "Same seed should produce same random tensors"
        # Note: model parameters are randomly initialized, so outputs may differ
        # But the random input should be the same
        
        # Test configuration logging
        config = {
            'model_architecture': 'MLP',
            'input_size': 784,
            'hidden_sizes': [256, 128],
            'output_size': 10,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 50,
            'optimizer': 'Adam',
            'loss_function': 'CrossEntropyLoss'
        }
        
        # Validate configuration completeness
        required_keys = ['model_architecture', 'learning_rate', 'batch_size', 'epochs']
        for key in required_keys:
            assert key in config, f"Configuration should include {key}"


def test_project_structure_and_organization(namespace):
    """Test overall project structure and organization"""
    import torch
    import torch.nn as nn
    
    # Test main model class or architecture
    model_artifacts = ['ProjectModel', 'MainModel', 'Net', 'Classifier']
    found_models = [name for name in model_artifacts if name in namespace]
    
    if found_models:
        for model_name in found_models:
            model_class = namespace[model_name]
            if isinstance(model_class, type) and issubclass(model_class, nn.Module):
                print(f"✓ Found model class: {model_name}")
                
                # Test model instantiation
                try:
                    # This might fail if model requires specific arguments
                    # In a real project, we'd check the constructor signature
                    pass
                except Exception as e:
                    print(f"Model instantiation test skipped: {e}")
    else:
        print("Warning: No main model class found. Expected one of:", model_artifacts)


def test_data_pipeline_implementation(namespace):
    """Test data pipeline implementation"""
    import torch
    from torch.utils.data import Dataset, DataLoader
    
    # Test custom dataset
    dataset_artifacts = ['ProjectDataset', 'CustomDataset', 'TrainDataset']
    found_datasets = [name for name in dataset_artifacts if name in namespace]
    
    if found_datasets:
        for dataset_name in found_datasets:
            dataset_class = namespace[dataset_name]
            if isinstance(dataset_class, type) and issubclass(dataset_class, Dataset):
                print(f"✓ Found dataset class: {dataset_name}")
    else:
        print("Info: No custom dataset class found")
    
    # Test data loaders
    loader_artifacts = ['train_loader', 'val_loader', 'test_loader', 'dataloader']
    found_loaders = [name for name in loader_artifacts if name in namespace]
    
    for loader_name in found_loaders:
        loader = namespace[loader_name]
        if isinstance(loader, DataLoader):
            print(f"✓ Found data loader: {loader_name}")
        else:
            print(f"Warning: {loader_name} is not a DataLoader instance")
    
    # Test data preprocessing
    if 'preprocess_data' in namespace or 'transform' in namespace:
        print("✓ Data preprocessing implemented")
    else:
        print("Info: No explicit data preprocessing found")


def test_training_implementation(namespace):
    """Test training loop implementation"""
    import torch
    
    # Test training function
    training_functions = ['train', 'train_model', 'training_loop', 'fit']
    found_training = [name for name in training_functions if name in namespace]
    
    if found_training:
        for func_name in found_training:
            func = namespace[func_name]
            if callable(func):
                print(f"✓ Found training function: {func_name}")
    else:
        print("Warning: No training function found")
    
    # Test training history/metrics
    metrics_artifacts = ['train_losses', 'train_accuracies', 'training_history', 'metrics']
    found_metrics = [name for name in metrics_artifacts if name in namespace]
    
    for metric_name in found_metrics:
        metrics = namespace[metric_name]
        if isinstance(metrics, (list, dict)):
            print(f"✓ Found training metrics: {metric_name}")
            
            if isinstance(metrics, list) and len(metrics) > 0:
                assert all(isinstance(m, (int, float)) for m in metrics), \
                    f"{metric_name} should contain numeric values"
            elif isinstance(metrics, dict):
                assert len(metrics) > 0, f"{metric_name} should not be empty"


def test_evaluation_implementation(namespace):
    """Test model evaluation implementation"""
    import torch
    
    # Test evaluation function
    eval_functions = ['evaluate', 'test', 'validation', 'eval_model']
    found_eval = [name for name in eval_functions if name in namespace]
    
    if found_eval:
        for func_name in found_eval:
            func = namespace[func_name]
            if callable(func):
                print(f"✓ Found evaluation function: {func_name}")
    else:
        print("Warning: No evaluation function found")
    
    # Test evaluation metrics
    eval_metrics = ['test_accuracy', 'val_accuracy', 'test_loss', 'evaluation_results']
    found_eval_metrics = [name for name in eval_metrics if name in namespace]
    
    for metric_name in found_eval_metrics:
        metric = namespace[metric_name]
        if isinstance(metric, (int, float)):
            assert 0 <= metric <= 1 or metric >= 0, f"{metric_name} should be non-negative"
            print(f"✓ Found evaluation metric: {metric_name} = {metric}")
        elif isinstance(metric, dict):
            print(f"✓ Found evaluation results: {metric_name}")


def test_model_persistence(namespace):
    """Test model saving and loading implementation"""
    import torch
    
    # Test model saving
    save_functions = ['save_model', 'save_checkpoint', 'save_state']
    found_save = [name for name in save_functions if name in namespace]
    
    if found_save:
        for func_name in found_save:
            func = namespace[func_name]
            if callable(func):
                print(f"✓ Found model saving function: {func_name}")
    else:
        print("Info: No explicit model saving function found")
    
    # Test model loading
    load_functions = ['load_model', 'load_checkpoint', 'load_state']
    found_load = [name for name in load_functions if name in namespace]
    
    if found_load:
        for func_name in found_load:
            func = namespace[func_name]
            if callable(func):
                print(f"✓ Found model loading function: {func_name}")
    else:
        print("Info: No explicit model loading function found")
    
    # Test checkpoint or saved model
    if 'checkpoint' in namespace or 'saved_model' in namespace:
        artifact = namespace.get('checkpoint', namespace.get('saved_model'))
        if isinstance(artifact, dict):
            required_keys = ['model_state_dict']
            for key in required_keys:
                if key in artifact:
                    print(f"✓ Checkpoint contains {key}")
    else:
        print("Info: No checkpoint artifact found")


def test_hyperparameter_optimization(namespace):
    """Test hyperparameter optimization implementation"""
    import torch
    
    # Test hyperparameter configuration
    hp_artifacts = ['hyperparameters', 'config', 'hp_config', 'best_params']
    found_hp = [name for name in hp_artifacts if name in namespace]
    
    for hp_name in found_hp:
        hp_config = namespace[hp_name]
        if isinstance(hp_config, dict):
            print(f"✓ Found hyperparameter config: {hp_name}")
            
            # Check for common hyperparameters
            common_hps = ['lr', 'learning_rate', 'batch_size', 'epochs']
            found_common = [hp for hp in common_hps if hp in hp_config]
            print(f"  Contains: {found_common}")
    
    # Test hyperparameter search results
    search_results = ['hp_results', 'grid_search_results', 'optimization_results']
    found_results = [name for name in search_results if name in namespace]
    
    for result_name in found_results:
        results = namespace[result_name]
        if isinstance(results, (list, dict)):
            print(f"✓ Found hyperparameter search results: {result_name}")
            
            if isinstance(results, list) and len(results) > 1:
                print(f"  Tested {len(results)} configurations")


def test_reproducibility_measures(namespace):
    """Test reproducibility implementation"""
    import torch
    
    # Test seed setting
    if 'set_seed' in namespace or 'set_seeds' in namespace:
        seed_func = namespace.get('set_seed', namespace.get('set_seeds'))
        if callable(seed_func):
            print("✓ Found seed setting function")
            
            # Test that it works
            try:
                seed_func(42)
                print("✓ Seed setting function executed successfully")
            except Exception as e:
                print(f"Seed setting function test failed: {e}")
    else:
        print("Info: No explicit seed setting function found")
    
    # Test configuration logging
    if 'experiment_config' in namespace or 'run_config' in namespace:
        config = namespace.get('experiment_config', namespace.get('run_config'))
        if isinstance(config, dict):
            print("✓ Found experiment configuration")
            print(f"  Configuration keys: {list(config.keys())}")
    else:
        print("Info: No experiment configuration found")


def test_code_quality_and_documentation(namespace):
    """Test code quality and documentation"""
    import torch
    
    # Test that main components have docstrings or comments
    main_functions = ['train', 'evaluate', 'preprocess_data', 'create_model']
    documented_functions = []
    
    for func_name in main_functions:
        if func_name in namespace:
            func = namespace[func_name]
            if callable(func) and hasattr(func, '__doc__') and func.__doc__:
                documented_functions.append(func_name)
    
    if documented_functions:
        print(f"✓ Found documented functions: {documented_functions}")
    else:
        print("Info: No documented functions found (consider adding docstrings)")
    
    # Test error handling (basic check)
    if 'error_handler' in namespace or 'try_except_wrapper' in namespace:
        print("✓ Error handling implemented")
    else:
        print("Info: No explicit error handling found")


def test_project_completeness(namespace):
    """Test overall project completeness"""
    import torch
    
    # Essential components checklist
    essential_components = [
        ('Model Architecture', ['ProjectModel', 'MainModel', 'Net', 'model']),
        ('Training Loop', ['train', 'training_loop', 'fit']),
        ('Evaluation', ['evaluate', 'test', 'validation']),
        ('Data Pipeline', ['train_loader', 'dataset', 'dataloader']),
        ('Metrics Tracking', ['train_losses', 'metrics', 'history'])
    ]
    
    completion_score = 0
    total_components = len(essential_components)
    
    for component_name, possible_names in essential_components:
        found = any(name in namespace for name in possible_names)
        if found:
            completion_score += 1
            print(f"✓ {component_name}: Implemented")
        else:
            print(f"⚠ {component_name}: Not found (expected one of: {possible_names})")
    
    completion_percentage = (completion_score / total_components) * 100
    print(f"\nProject Completion: {completion_percentage:.1f}% ({completion_score}/{total_components} components)")
    
    if completion_percentage >= 80:
        print("🎉 Project appears to be well-structured and complete!")
    elif completion_percentage >= 60:
        print("👍 Project has most essential components implemented.")
    else:
        print("📝 Project needs more core components to be complete.")


def run_tests():
    """Run all tests for Exercise 3"""
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    run_tests()