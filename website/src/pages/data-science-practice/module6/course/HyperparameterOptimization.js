import React from 'react';
import { Container, Title, Text, Stack, List, Group, Alert } from '@mantine/core';
import { IconAdjustments, IconArrowsShuffle, IconBrain, IconAlertCircle, IconChartBar, IconFold} from '@tabler/icons-react';
import CodeBlock from 'components/CodeBlock';
import { InlineMath, BlockMath } from 'react-katex';
import 'katex/dist/katex.min.css';

const HyperparameterOptimization = () => {
  return (
    <Container fluid>
      <Title order={1} id="hyperparameter-optimization" mt="xl" mb="md">Hyperparameter Optimization</Title>
      
      <Text>
        Hyperparameter optimization is the process of finding the best set of hyperparameters for a machine learning model. Hyperparameters are parameters that are set before the learning process begins, as opposed to parameters that are learned during training.
      </Text>

      <Stack spacing="xl" mt="xl">
        <Title order={3} id="grid-search">Grid Search</Title>
          <Text>
            Grid search is an exhaustive search through a manually specified subset of the hyperparameter space. It tries all possible combinations of the specified hyperparameter values.
          </Text>

          <CodeBlock
            language="python"
            code={`
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.datasets import load_iris

# Load the iris dataset
X, y = load_iris(return_X_y=True)

# Define the parameter grid
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['rbf', 'linear'],
    'gamma': [0.1, 1, 'scale']
}

# Create an SVC classifier
svc = SVC()

# Perform grid search
grid_search = GridSearchCV(svc, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X, y)

print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)
# Best parameters: {'C': 1, 'gamma': 0.1, 'kernel': 'rbf'}
# Best cross-validation score: 0.9800000000000001
            `}
          />

          <Text mt="md">
            Grid search is comprehensive but can be computationally expensive, especially with a large number of hyperparameters or a wide range of values.
          </Text>
        

        <Title order={3} id="random-search">Random Search</Title>
          <Text>
            Random search samples random combinations of hyperparameters from specified distributions. It can be more efficient than grid search, especially when not all hyperparameters are equally important.
          </Text>

          <CodeBlock
            language="python"
            code={`
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from scipy.stats import randint, uniform

# Load the iris dataset
X, y = load_iris(return_X_y=True)

# Define the parameter distributions
param_dist = {
    'n_estimators': randint(10, 200),
    'max_depth': randint(1, 20),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': uniform(0, 1)
}

# Create a random forest classifier
rf = RandomForestClassifier()

# Perform random search
random_search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=100, cv=5, scoring='accuracy')
random_search.fit(X, y)

print("Best parameters:", random_search.best_params_)
print("Best cross-validation score:", random_search.best_score_)
# Best parameters: {'max_depth': 3, 'max_features': 0.7360065387687408, 'min_samples_leaf': 3, 'min_samples_split': 7, 'n_estimators': 170}
# Best cross-validation score: 0.9666666666666668
            `}
          />

          <Text mt="md">
            Random search can often find good hyperparameters in fewer iterations than grid search, making it a popular choice for many applications.
          </Text>
        

        <Title order={3} id="bayesian-optimization">Bayesian Optimization</Title>
  <Text>
    Bayesian optimization uses probabilistic models to guide the search for optimal hyperparameters. It tries to balance exploration of unknown regions and exploitation of known good regions.
  </Text>

<List>
  <List.Item>
    <><span style={{ fontWeight: 700 }}>Prior:</span> We start with a prior belief <InlineMath math="P(\theta)" /> about good hyperparameter values <InlineMath math="\theta" />. For example, we might believe the learning rate is uniformly distributed between 0.0001 and 0.1.</>
  </List.Item>
  <List.Item>
    <><span style={{ fontWeight: 700 }}>Likelihood:</span> We try some hyperparameter values and observe the model's performance. This gives us the likelihood <InlineMath math="P(D|\theta)" />, where <InlineMath math="D" /> is our observed data.</>
  </List.Item>
  <List.Item>
    <><span style={{ fontWeight: 700 }}>Posterior:</span> We update our belief using Bayes' theorem:</>
    <BlockMath math="P(\theta|D) = \frac{P(D|\theta)P(\theta)}{P(D)}" />
    <Text>This posterior <InlineMath math="P(\theta|D)" /> represents our updated belief about good hyperparameter values.</Text>
  </List.Item>
</List>

  <Title order={3} mt="md" mb="sm">Common Priors in Bayesian Optimization</Title>

  <List>
    <List.Item>
      <><span style={{ fontWeight: 700 }}>Uniform Prior:</span> Assumes all values in a range are equally likely. Used when you have no prior knowledge about the parameter.</>
    </List.Item>
    <List.Item>
      <><span style={{ fontWeight: 700 }}>Log-Uniform Prior:</span> Useful for parameters that span several orders of magnitude, like learning rates or regularization strengths.</>
    </List.Item>
    <List.Item>
      <><span style={{ fontWeight: 700 }}>Normal (Gaussian) Prior:</span> Assumes values are more likely near a central value and less likely far from it. Used when you have an idea of a good starting point.</>
    </List.Item>
    <List.Item>
      <><span style={{ fontWeight: 700 }}>Beta Prior:</span> Useful for parameters bounded between 0 and 1, like dropout rates.</>
    </List.Item>
    <List.Item>
      <><span style={{ fontWeight: 700 }}>Categorical Prior:</span> Used for non-numeric hyperparameters, like choice of activation function or kernel type.</>
    </List.Item>
  </List>

  <CodeBlock
    language="python"
    code={`
from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from skopt.callbacks import DeltaYStopper

# Load the Boston Housing dataset
X, y = load_boston(return_X_y=True)

# Define the search space with diverse priors
search_spaces = {
    'n_estimators': Integer(10, 200),  # Uniform prior
    'max_depth': Integer(1, 20),  # Uniform prior
    'min_samples_split': Real(0.1, 1.0, prior='uniform'),  # Uniform prior
    'min_samples_leaf': Real(0.1, 0.5, prior='log-uniform'),  # Log-uniform prior
    'max_features': Categorical(['auto', 'sqrt', 'log2']),  # Categorical prior
    'bootstrap': Categorical([True, False])  # Categorical prior
}

# Create a random forest regressor
rf = RandomForestRegressor(random_state=42)


# Define the DeltaYStopper callback
# This stops the optimization early if the change in the objective function (score) 
# is less than 0.001 for the best 3 consecutive iterations.
delta_stopper = DeltaYStopper(delta=0.001, n_best=3)


# Perform Bayesian optimization
opt = BayesSearchCV(
    rf,
    search_spaces,
    n_iter=50, # The algorithm will try up to 50 different hyperparameter combinations.
    cv=5,
    n_jobs=-1,
    random_state=42,
    scoring='accuracy'
)

opt.fit(X, y, callback=[delta_stopper])

print("Best parameters:", opt.best_params_)
print("Best cross-validation score:", opt.best_score_)
`}
  />

  <Text mt="md">
    Bayesian optimization can be more efficient than both grid and random search, especially for expensive-to-evaluate objective functions. The choice of prior can significantly impact the efficiency of the optimization process.
  </Text>



<Title order={3} id="scoring-methods">Scoring Methods</Title>
  <Text>
    Scoring methods define how we evaluate model performance during hyperparameter optimization. scikit-learn provides built-in scoring metrics, and you can also create custom scoring functions.
  </Text>

  <CodeBlock
    language="python"
    code={`
from sklearn.metrics import make_scorer, mean_squared_error, accuracy_score
from sklearn.model_selection import cross_val_score
import numpy as np

# Example 1: Using built-in scoring metrics
param_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring={
        'accuracy': 'accuracy',
        'precision_macro': 'precision_macro',
        'recall_macro': 'recall_macro'
    },
    refit='accuracy'  # Use accuracy to select the best model
)

# Example 2: Custom scoring function
def custom_metric(y_true, y_pred):
    # Example: Weighted combination of precision and recall
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return 0.7 * precision + 0.3 * recall

# Create a scorer object
custom_scorer = make_scorer(custom_metric)

# Use custom scorer in GridSearchCV
param_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring=custom_scorer
)

`}
  />


<Title order={3} id="cv-strategies">Cross-validation Strategies</Title>
  <Text>
    Cross-validation strategies determine how the data is split during hyperparameter optimization. Different problems may require different CV strategies.
  </Text>

  <CodeBlock
    language="python"
    code={`
from sklearn.model_selection import (
    KFold, StratifiedKFold, TimeSeriesSplit,
    GroupKFold, LeaveOneOut
)

# Example 1: Basic K-Fold CV
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
param_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=kfold
)

# Example 2: Stratified K-Fold for imbalanced classification
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
param_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=skf
)

# Example 3: Time Series CV
# Maintains temporal order and prevents future data leakage
tscv = TimeSeriesSplit(n_splits=5)
param_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=tscv
)
`}
  />

        <Title order={3} id="best-practices">Best Practices</Title>
          <List>
            <List.Item>
              <Text><span style={{ fontWeight: 700 }}>Start with a broad search:</span> Begin with a wide range of hyperparameters and gradually narrow down.</Text>
            </List.Item>
            <List.Item>
              <Text><span style={{ fontWeight: 700 }}>Consider computational resources:</span> Choose the optimization method based on the available computational budget/time.</Text>
            </List.Item>
            <List.Item>
              <Text><span style={{ fontWeight: 700 }}>Be cautious of data leakage</span> ensure that the test set remains completely unseen during the entire model selection process.</Text>
            </List.Item>
          </List>
        
      </Stack>
    </Container>
  );
};



export default HyperparameterOptimization;