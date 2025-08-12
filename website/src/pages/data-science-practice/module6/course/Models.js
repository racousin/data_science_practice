import React from 'react';
import { Container, Title, Text, Stack, List, Group, Table } from '@mantine/core';
import { IconCheck, IconX } from '@tabler/icons-react';
import CodeBlock from 'components/CodeBlock';
import { InlineMath, BlockMath } from 'react-katex';
import 'katex/dist/katex.min.css';

const Models = () => {
  return (
    <Container fluid>
      <Title order={1} mt="xl" mb="md">Machine Learning Models</Title>

      <Stack spacing="xl">
        <ModelSection
          title="Linear Models"
          id="linear-models"
          math={<BlockMath math="y = \beta_0 + \beta_1x_1 + \beta_2x_2 + ... + \beta_nx_n + \epsilon" />}
          description="Linear models assume a linear relationship between the input features and the target variable. They are simple yet powerful for many tasks."
          hyperparameters={[
            { name: 'fit_intercept', description: 'Whether to calculate the intercept (β₀)' },
            { name: 'normalize', description: 'Whether to normalize the input features' },
            { name: 'C', description: 'Inverse of regularization strength (for regularized models)' },
            { name: 'penalty', description: 'Type of regularization (L1, L2, ElasticNet)' },
          ]}
          checks={{
            unscaled: false,
            missing: false,
            categorical: false,
            regression: true,
            classification: true
          }}
          bayesianOptimization={`
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from sklearn.linear_model import LogisticRegression, LinearRegression

param_space = {
    'C': Real(1e-6, 1e+6, prior='log-uniform'),
    'penalty': Categorical(['l1', 'l2', 'elasticnet']),
    'l1_ratio': Real(0, 1),
    'solver': Categorical(['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'])
}

# For classification
opt_clf = BayesSearchCV(LogisticRegression(), param_space, n_iter=50, cv=5, n_jobs=-1, verbose=0)
opt_clf.fit(X_train, y_train)

# For regression
opt_reg = BayesSearchCV(LinearRegression(), {}, n_iter=50, cv=5, n_jobs=-1, verbose=0)
opt_reg.fit(X_train, y_train)

print("Best parameters (classification):", opt_clf.best_params_)
print("Best cross-validation score (classification):", opt_clf.best_score_)
print("Best parameters (regression):", opt_reg.best_params_)
print("Best cross-validation score (regression):", opt_reg.best_score_)
          `}
        />

        <ModelSection
          title="K-Nearest Neighbors (KNN)"
          id="knn"
          math={<BlockMath math="f(x) = \frac{1}{k} \sum_{i \in N_k(x, D)} y_i" />}
          description="KNN is a non-parametric method used for classification and regression. It uses the k nearest neighbors of a query point to make predictions."
          hyperparameters={[
            { name: 'n_neighbors', description: 'Number of neighbors to use' },
            { name: 'weights', description: 'Weight function used in prediction' },
            { name: 'metric', description: 'Distance metric to use' },
            { name: 'p', description: 'Power parameter for Minkowski metric' },
          ]}
          checks={{
            unscaled: false,
            missing: false,
            categorical: false,
            regression: true,
            classification: true
          }}
          bayesianOptimization={`
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

param_space = {
    'n_neighbors': Integer(1, 50),
    'weights': Categorical(['uniform', 'distance']),
    'metric': Categorical(['euclidean', 'manhattan', 'minkowski']),
    'p': Integer(1, 5)
}

# For classification
opt_clf = BayesSearchCV(KNeighborsClassifier(), param_space, n_iter=50, cv=5, n_jobs=-1, verbose=0)
opt_clf.fit(X_train, y_train)

# For regression
opt_reg = BayesSearchCV(KNeighborsRegressor(), param_space, n_iter=50, cv=5, n_jobs=-1, verbose=0)
opt_reg.fit(X_train, y_train)

print("Best parameters (classification):", opt_clf.best_params_)
print("Best cross-validation score (classification):", opt_clf.best_score_)
print("Best parameters (regression):", opt_reg.best_params_)
print("Best cross-validation score (regression):", opt_reg.best_score_)
          `}
        />

        <ModelSection
          title="Support Vector Machines (SVM)"
          id="svm"
          math={<BlockMath math="f(x) = \text{sign}(\mathbf{w}^T\mathbf{x} + b)" />}
          description="SVM finds the hyperplane that best separates classes in the feature space. It can be used for both classification and regression."
          hyperparameters={[
            { name: 'C', description: 'Regularization parameter' },
            { name: 'kernel', description: 'Kernel type to be used' },
            { name: 'gamma', description: 'Kernel coefficient for RBF, poly and sigmoid kernels' },
            { name: 'degree', description: 'Degree of the polynomial kernel function' },
          ]}
          checks={{
            unscaled: false,
            missing: false,
            categorical: false,
            regression: true,
            classification: true
          }}
          bayesianOptimization={`
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from sklearn.svm import SVC, SVR

param_space = {
    'C': Real(1e-6, 1e+6, prior='log-uniform'),
    'gamma': Real(1e-6, 1e+1, prior='log-uniform'),
    'kernel': Categorical(['rbf', 'linear', 'poly']),
    'degree': Integer(1, 5)
}

# For classification
opt_clf = BayesSearchCV(SVC(), param_space, n_iter=50, cv=5, n_jobs=-1, verbose=0)
opt_clf.fit(X_train, y_train)

# For regression
opt_reg = BayesSearchCV(SVR(), param_space, n_iter=50, cv=5, n_jobs=-1, verbose=0)
opt_reg.fit(X_train, y_train)

print("Best parameters (classification):", opt_clf.best_params_)
print("Best cross-validation score (classification):", opt_clf.best_score_)
print("Best parameters (regression):", opt_reg.best_params_)
print("Best cross-validation score (regression):", opt_reg.best_score_)
          `}
        />

        <ModelSection
          title="Decision Trees"
          id="decision-trees"
          math={<BlockMath math="\text{Information Gain} = H(S) - \sum_{i=1}^m \frac{|S_i|}{|S|} H(S_i)" />}
          description="Decision Trees are non-parametric supervised learning methods used for classification and regression. The model is represented as a tree structure."
          hyperparameters={[
            { name: 'max_depth', description: 'Maximum depth of the tree' },
            { name: 'min_samples_split', description: 'Minimum number of samples required to split an internal node' },
            { name: 'min_samples_leaf', description: 'Minimum number of samples required to be at a leaf node' },
            { name: 'max_features', description: 'Number of features to consider when looking for the best split' },
          ]}
          checks={{
            unscaled: true,
            missing: false,
            categorical: true,
            regression: true,
            classification: true
          }}
          bayesianOptimization={`
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

param_space = {
    'max_depth': Integer(1, 20),
    'min_samples_split': Integer(2, 20),
    'min_samples_leaf': Integer(1, 20),
    'max_features': Real(0.1, 1.0)
}

# For classification
opt_clf = BayesSearchCV(DecisionTreeClassifier(), param_space, n_iter=50, cv=5, n_jobs=-1, verbose=0)
opt_clf.fit(X_train, y_train)

# For regression
opt_reg = BayesSearchCV(DecisionTreeRegressor(), param_space, n_iter=50, cv=5, n_jobs=-1, verbose=0)
opt_reg.fit(X_train, y_train)

print("Best parameters (classification):", opt_clf.best_params_)
print("Best cross-validation score (classification):", opt_clf.best_score_)
print("Best parameters (regression):", opt_reg.best_params_)
print("Best cross-validation score (regression):", opt_reg.best_score_)
          `}
        />
      </Stack>
    </Container>
  );
};

const ModelSection = ({ title, id, math, description, hyperparameters, checks, bayesianOptimization }) => (
  <Stack spacing="md">
    <Title order={2} id={id}>{title}</Title>
    {math}
    <Text>{description}</Text>
    <Title order={3}>Key Hyperparameters</Title>
    <Table>
      <thead>
        <tr>
          <th>Parameter</th>
          <th>Description</th>
        </tr>
      </thead>
      <tbody>
        {hyperparameters.map((param, index) => (
          <tr key={index}>
            <td><code>{param.name}</code></td>
            <td>{param.description}</td>
          </tr>
        ))}
      </tbody>
    </Table>
    <Title order={3}>Model Characteristics</Title>
    <Table>
      <thead>
        <tr>
          <th>Characteristic</th>
          <th>Status</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>Robust to unscaled data</td>
          <td>{checks.unscaled ? <IconCheck color="green" /> : <IconX color="red" />}</td>
        </tr>
        <tr>
          <td>Handles missing values</td>
          <td>{checks.missing ? <IconCheck color="green" /> : <IconX color="red" />}</td>
        </tr>
        {/* <tr>
          <td>Handles categorical data</td>
          <td>{checks.categorical ? <IconCheck color="green" /> : <IconX color="red" />}</td>
        </tr> */}
        <tr>
          <td>Supports regression</td>
          <td>{checks.regression ? <IconCheck color="green" /> : <IconX color="red" />}</td>
        </tr>
        <tr>
          <td>Supports classification</td>
          <td>{checks.classification ? <IconCheck color="green" /> : <IconX color="red" />}</td>
        </tr>
      </tbody>
    </Table>
    <Title order={3}>Bayesian Optimization Cheat Sheet</Title>
    <CodeBlock language="python" code={bayesianOptimization} />
  </Stack>
);

export default Models;