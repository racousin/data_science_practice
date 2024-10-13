import React from 'react';
import { Container, Title, Text, Stack, List, Group, Image } from '@mantine/core';
import { IconAdjustments, IconArrowsShuffle, IconBrain } from '@tabler/icons-react';
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
        <Section
          icon={<IconAdjustments size={28} />}
          title="Grid Search"
          id="grid-search"
        >
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
            `}
          />

          <Text mt="md">
            Grid search is comprehensive but can be computationally expensive, especially with a large number of hyperparameters or a wide range of values.
          </Text>
        </Section>

        <Section
          icon={<IconArrowsShuffle size={28} />}
          title="Random Search"
          id="random-search"
        >
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
            `}
          />

          <Text mt="md">
            Random search can often find good hyperparameters in fewer iterations than grid search, making it a popular choice for many applications.
          </Text>
        </Section>

        <Section
          icon={<IconBrain size={28} />}
          title="Bayesian Optimization"
          id="bayesian-optimization"
        >
          <Text>
            Bayesian optimization uses probabilistic models to guide the search for optimal hyperparameters. It tries to balance exploration of unknown regions and exploitation of known good regions.
          </Text>

          <CodeBlock
            language="python"
            code={`
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

# Load the iris dataset
X, y = load_iris(return_X_y=True)

# Define the search space
search_spaces = {
    'C': Real(1e-6, 1e+6, prior='log-uniform'),
    'gamma': Real(1e-6, 1e+1, prior='log-uniform'),
    'degree': Integer(1, 8),
    'kernel': Categorical(['linear', 'poly', 'rbf'])
}

# Create an SVC classifier
svc = SVC()

# Perform Bayesian optimization
opt = BayesSearchCV(
    svc,
    search_spaces,
    n_iter=50,
    cv=5,
    n_jobs=-1,
    verbose=0
)

opt.fit(X, y)

print("Best parameters:", opt.best_params_)
print("Best cross-validation score:", opt.best_score_)
            `}
          />

          <Text mt="md">
            Bayesian optimization can be more efficient than both grid and random search, especially for expensive-to-evaluate objective functions.
          </Text>
        </Section>

        <Section
          title="Best Practices"
          id="best-practices"
        >
          <List>
            <List.Item>
              <Text><span style={{ fontWeight: 700 }}>Start with a broad search:</span> Begin with a wide range of hyperparameters and gradually narrow down.</Text>
            </List.Item>
            <List.Item>
              <Text><span style={{ fontWeight: 700 }}>Use domain knowledge:</span> Incorporate prior knowledge about good hyperparameter ranges when available.</Text>
            </List.Item>
            <List.Item>
              <Text><span style={{ fontWeight: 700 }}>Consider computational resources:</span> Choose the optimization method based on the available computational budget.</Text>
            </List.Item>
            <List.Item>
              <Text><span style={{ fontWeight: 700 }}>Avoid overfitting:</span> Use cross-validation to ensure the optimized hyperparameters generalize well.</Text>
            </List.Item>
            <List.Item>
              <Text><span style={{ fontWeight: 700 }}>Log-scale for certain parameters:</span> Use log-scale for parameters like learning rates or regularization strengths that can span several orders of magnitude.</Text>
            </List.Item>
          </List>
        </Section>
      </Stack>
    </Container>
  );
};

const Section = ({ icon, title, id, children }) => (
  <Stack spacing="sm">
    <Group spacing="xs">
      {icon}
      <Title order={2} id={id}>{title}</Title>
    </Group>
    {children}
  </Stack>
);

export default HyperparameterOptimization;