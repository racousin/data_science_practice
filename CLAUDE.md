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


#### Module 1: Foundations of Deep Learning
- Exercise 1: Environment & Basics
- Exercise 2: Gradient Descent
- Exercise 3: First Step with MLP

#### Module 2: Automatic Differentiation
- Exercise 1: Autograd Exploration
- Exercise 2: Optimization with PyTorch Autograd
- Exercise 3: Gradient Flow

#### Module 3: Neural Network Training & Monitoring
- Exercise 0: Training Basic
- Exercise 1: Data Pipeline & Training Loop
- Exercise 2: Essential Layers
- Exercise 3: Monitoring & Visualization with TensorBoard

#### Module 4: Performance Optimization & Scale
- Exercise 1: Resource Profiling
- Exercise 2: Performance Optimization Techniques
- Exercise 2: Fine Tunning

### Development Guidelines

1. **Consistency**: All exercises must follow the exact patternß
2. **Progressive Difficulty**: Start simple, gradually increase complexityß
3. **Clear TODOs**: Every student task starts with `# TODO:` comment
4. **Immediate Feedback**: Tests run after each section for validation
5. **Self-Contained**: No external dependencies beyond PyTorch and standard libraries
6. **Colab-Friendly**: Everything runs in Google Colab without additional setup