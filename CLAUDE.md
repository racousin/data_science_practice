# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## PRIMARY FOCUS: Python Deep Learning Exercise Development

We are actively developing exercises for the Python Deep Learning course. Each exercise follows a standardized pattern that ensures consistency and quality.

### Exercise Structure Pattern

Each exercise notebook (`/website/public/modules/python-deep-learning/module{N}/exercises/exercise{N}.ipynb`) must include:

1. **Title and Learning Objectives**
   ```markdown
   # Module N - Exercise N: Topic Name
   
   ## Learning Objectives
   - Clear, specific learning goals
   - One objective per line
   - Actionable and measurable
   ```

2. **Test Framework Setup**
   ```python
   # Clone the test repository
   !git clone https://github.com/racousin/data_science_practice.git /tmp/tests 2>/dev/null || true
   
   # Import required modules
   import sys
   sys.path.append('/tmp/tests/tests/python_deep_learning')
   
   # Import the improved test utilities
   from test_utils import NotebookTestRunner, create_inline_test
   from moduleN.test_exerciseN import ExerciseNValidator, EXERCISEN_SECTIONS
   
   # Create test runner and validator
   test_runner = NotebookTestRunner("moduleN", N)
   validator = ExerciseNValidator()
   ```

3. **Environment Setup**
   - Import necessary libraries (torch, numpy, matplotlib, etc.)
   - Set random seeds for reproducibility
   - Verify CUDA availability if relevant

4. **Section Pattern**
   Each section follows this structure:
   ```markdown
   ## Section N: Topic Name
   
   Brief description of what students will learn in this section.
   Theorical and mathematical questions
   ```
   
   ```python
   # TODO comments for each task the student needs to complete
   # Students replace None with their implementation
   variable_name = None
   
   # Display/verification code to help students check their work
   print(f"Your result: {variable_name}")
   ```
   
   ```python
   # Test Section N: Topic Name
   section_tests = [(getattr(validator, name), desc) for name, desc in EXERCISEN_SECTIONS["Section N: Topic Name"]]
   test_runner.test_section("Section N: Topic Name", validator, section_tests, locals())
   ```

5. **Final Validation**
   ```python
   # Display final summary of all tests
   test_runner.final_summary()
   ```

### Test File Structure Pattern

Each test file (`/tests/python_deep_learning/module{N}/test_exercise{N}.py`) must:

1. **Import Dependencies**
   ```python
   import sys
   import torch
   import numpy as np
   import pytest
   from typing import Dict, Any
   
   sys.path.append('..')
   from test_utils import TestValidator, NotebookTestRunner
   ```

2. **Define Validator Class**
   ```python
   class ExerciseNValidator(TestValidator):
       """Validator for Module N Exercise N: Topic"""
   ```

3. **Implement Test Methods**
   - One test method per student task
   - Use descriptive test names: `test_variable_name()`
   - Use helper methods from TestValidator:
     - `check_variable()` - Verify variable exists
     - `check_tensor_shape()` - Validate tensor dimensions
     - `check_tensor_dtype()` - Check data types
     - `check_tensor_values()` - Compare expected values

4. **Define Section Dictionary**
   ```python
   EXERCISEN_SECTIONS = {
       "Section 1: Topic Name": [
           ("test_method_name", "Description of what's being tested"),
           ...
       ],
       ...
   }
   ```
#### Module 1: Foundations of Deep Learning
- Exercise 1: Environment & Basics
- Exercise 2: Gradient Descent
- Exercise 3: First Step with MLP

#### Module 2: Automatic Differentiation
- Exercise 1: Autograd Exploration
- Exercise 2: Optimization with PyTorch Autograd
- Exercise 3: Gradient Flow

#### Module 3: Neural Network Training & Monitoring
- Exercise 1: Data Pipeline & Training Loop
- Exercise 2: Essential Layers
- Exercise 3: Monitoring & Visualization with TensorBoard

#### Module 4: Performance Optimization & Scale
- Exercise 1: Model Resource Profiling
- Exercise 2: Fine Tunning
- Exercise 3: Performance Optimization Techniques
### Development Guidelines

1. **Consistency**: All exercises must follow the exact patternß
2. **Progressive Difficulty**: Start simple, gradually increase complexityß
3. **Clear TODOs**: Every student task starts with `# TODO:` comment
4. **Immediate Feedback**: Tests run after each section for validation
5. **Self-Contained**: No external dependencies beyond PyTorch and standard libraries
6. **Colab-Friendly**: Everything runs in Google Colab without additional setup