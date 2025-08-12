import React from 'react';
import { Stack, Title, Text, Table, Alert, Grid, Box } from '@mantine/core';
import { InlineMath, BlockMath } from 'react-katex';
import CodeBlock from 'components/CodeBlock';

const HyperparameterOptimization = () => {
  return (
    <Stack spacing="md">

      <Title order={4}>1. Grid Search</Title>
      <Text>
        Systematically explores every combination of specified hyperparameter values.
        Best for small search spaces with few parameters.
      </Text>

      <CodeBlock
        language="python"
        code={`
import itertools

def grid_search_cv(model_class, param_grid, train_loader, val_loader, device):
    best_val_loss = float('inf')
    best_params = None
    
    # Generate all combinations of parameters
    param_combinations = [dict(zip(param_grid.keys(), v)) 
                         for v in itertools.product(*param_grid.values())]
    
    for params in param_combinations:
        # Initialize model with current parameters
        model = model_class(
            input_size=params['input_size'],
            hidden_size=params['hidden_size'],
            output_size=params['output_size']
        ).to(device)
        
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=params['learning_rate']
        )
        
        # Train model
        model.train()
        for epoch in range(params['epochs']):
            train_loss = train_epoch(model, train_loader, 
                                   criterion, optimizer, device)
            val_loss = validate(model, val_loader, criterion, device)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_params = params
                
    return best_params, best_val_loss

# Example usage
param_grid = {
    'input_size': [1],
    'hidden_size': [32, 64, 128],
    'output_size': [1],
    'learning_rate': [0.001, 0.0001],
    'epochs': [50]
}

best_params, best_loss = grid_search_cv(
    RegressionNet, param_grid, train_loader, val_loader, device
)`}
      />

      <Title order={4}>2. Random Search</Title>
      <Text>
        More efficient than grid search for high-dimensional spaces, randomly samples 
        from parameter distributions.
      </Text>

      <CodeBlock
        language="python"
        code={`
import numpy as np
from functools import partial

def random_search_cv(model_class, param_distributions, n_iter, 
                    train_loader, val_loader, device):
    best_val_loss = float('inf')
    best_params = None
    
    for _ in range(n_iter):
        # Sample parameters
        params = {
            k: np.random.choice(v) if isinstance(v, list) 
               else v for k, v in param_distributions.items()
        }
        
        # Initialize and train model with sampled parameters
        model = model_class(**params).to(device)
        optimizer = torch.optim.Adam(model.parameters(), 
                                   lr=params['learning_rate'])
        
        # Training loop
        for epoch in range(params['epochs']):
            train_loss = train_epoch(model, train_loader, 
                                   criterion, optimizer, device)
            val_loss = validate(model, val_loader, criterion, device)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_params = params
    
    return best_params, best_val_loss

# Example usage
param_distributions = {
    'input_size': [1],
    'hidden_size': [32, 64, 128, 256],
    'output_size': [1],
    'learning_rate': [1e-4, 3e-4, 1e-3, 3e-3],
    'epochs': [50]
}`}
      />

      <Title order={4}>3. Bayesian Optimization</Title>
      <Text>
        Uses probabilistic models to guide the search, particularly effective for 
        expensive-to-evaluate objectives.
      </Text>

      <CodeBlock
        language="python"
        code={`
from bayes_opt import BayesianOptimization

def objective_function(hidden_size, learning_rate, model_class, 
                      train_loader, val_loader, device):
    # Convert continuous parameters to discrete if needed
    hidden_size = int(hidden_size)
    
    # Initialize model
    model = model_class(
        input_size=1,
        hidden_size=hidden_size,
        output_size=1
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train and evaluate
    best_val_loss = float('inf')
    for epoch in range(50):
        train_loss = train_epoch(model, train_loader, criterion, 
                               optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        best_val_loss = min(best_val_loss, val_loss)
    
    return -best_val_loss  # Negative because BayesianOptimization maximizes

# Setup Bayesian Optimization
pbounds = {
    'hidden_size': (32, 256),
    'learning_rate': (1e-4, 1e-2)
}

optimizer = BayesianOptimization(
    f=partial(objective_function, 
             model_class=RegressionNet,
             train_loader=train_loader,
             val_loader=val_loader,
             device=device),
    pbounds=pbounds,
    random_state=1
)

optimizer.maximize(init_points=5, n_iter=20)`}
      />

    </Stack>
  );
};

export default HyperparameterOptimization;