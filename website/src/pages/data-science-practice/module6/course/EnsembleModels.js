import React from 'react';
import { Container, Title, Text, Stack, List, Group, Table, Badge } from '@mantine/core';
import { IconCheck, IconX } from '@tabler/icons-react';
import CodeBlock from 'components/CodeBlock';
import { InlineMath, BlockMath } from 'react-katex';
import 'katex/dist/katex.min.css';

const EnsembleModels = () => {
  return (
    <Container fluid>
      <Title order={1} mt="xl" mb="md">Ensemble Models</Title>

      <Stack spacing="xl">
        <ModelSection
          title="Boosting"
          id="boosting"
          math={<BlockMath math="F_m(x) = F_{m-1}(x) + \alpha_m h_m(x)" />}
          description="Boosting builds models sequentially, where each new model tries to correct the errors of the previous ones. It converts weak learners into strong learners."
          submodels={[
            {
              name: "AdaBoost",
              description: "Adjusts weights of misclassified instances after each iteration.",
              hyperparameters: [
                { name: 'n_estimators', description: 'Number of weak learners' },
                { name: 'learning_rate', description: 'Weight applied to each classifier at each boosting iteration' },
              ],
              checks: {
                unscaled: false,
                missing: false,
                categorical: false,
                regression: true,
                classification: true
              },
              bayesianOptimization: `
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor

param_space = {
    'n_estimators': Integer(50, 500),
    'learning_rate': Real(0.01, 1.0, prior='log-uniform'),
}

# For classification
opt_clf = BayesSearchCV(AdaBoostClassifier(), param_space, n_iter=50, cv=5, n_jobs=-1, verbose=0)
opt_clf.fit(X_train, y_train)

# For regression
opt_reg = BayesSearchCV(AdaBoostRegressor(), param_space, n_iter=50, cv=5, n_jobs=-1, verbose=0)
opt_reg.fit(X_train, y_train)
              `
            },
            {
              name: "Gradient Boosting",
              description: "Minimizes the loss function of the entire ensemble using gradient descent.",
              hyperparameters: [
                { name: 'n_estimators', description: 'Number of boosting stages to perform' },
                { name: 'learning_rate', description: 'Shrinks the contribution of each tree' },
                { name: 'max_depth', description: 'Maximum depth of the individual regression estimators' },
              ],
              checks: {
                unscaled: true,
                missing: false,
                categorical: false,
                regression: true,
                classification: true
              },
              bayesianOptimization: `
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

param_space = {
    'n_estimators': Integer(50, 500),
    'learning_rate': Real(0.01, 1.0, prior='log-uniform'),
    'max_depth': Integer(3, 10),
    'min_samples_split': Integer(2, 20),
    'min_samples_leaf': Integer(1, 20)
}

# For classification
opt_clf = BayesSearchCV(GradientBoostingClassifier(), param_space, n_iter=50, cv=5, n_jobs=-1, verbose=0)
opt_clf.fit(X_train, y_train)

# For regression
opt_reg = BayesSearchCV(GradientBoostingRegressor(), param_space, n_iter=50, cv=5, n_jobs=-1, verbose=0)
opt_reg.fit(X_train, y_train)
              `
            }
          ]}
        />

        <ModelSection
          title="Random Forests"
          id="random-forests"
          math={<BlockMath math="f(x) = \frac{1}{B} \sum_{b=1}^B f_b(x)" />}
          description="Random Forests are an ensemble of decision trees, where each tree is trained on a random subset of the data and features. The final prediction is the average (for regression) or majority vote (for classification) of all trees."
          hyperparameters={[
            { name: 'n_estimators', description: 'Number of trees in the forest' },
            { name: 'max_depth', description: 'Maximum depth of the trees' },
            { name: 'min_samples_split', description: 'Minimum number of samples required to split an internal node' },
            { name: 'min_samples_leaf', description: 'Minimum number of samples required to be at a leaf node' },
            { name: 'max_features', description: 'Number of features to consider when looking for the best split' },
          ]}
          checks={{
            unscaled: true,
            missing: true,
            categorical: false,
            regression: true,
            classification: true
          }}
          bayesianOptimization={`
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

param_space = {
    'n_estimators': Integer(10, 500),
    'max_depth': Integer(1, 20),
    'min_samples_split': Integer(2, 20),
    'min_samples_leaf': Integer(1, 20),
    'max_features': Real(0.1, 1.0)
}

# For classification
opt_clf = BayesSearchCV(RandomForestClassifier(), param_space, n_iter=50, cv=5, n_jobs=-1, verbose=0)
opt_clf.fit(X_train, y_train)

# For regression
opt_reg = BayesSearchCV(RandomForestRegressor(), param_space, n_iter=50, cv=5, n_jobs=-1, verbose=0)
opt_reg.fit(X_train, y_train)
          `}
        />

        <ModelSection
          title="Advanced Gradient Boosting"
          id="advanced-gradient-boosting"
          math={<BlockMath math="L(\phi) = \sum_i l(y_i, \hat{y}_i) + \sum_k \Omega(f_k)" />}
          description="Advanced Gradient Boosting implementations like XGBoost, LightGBM, and CatBoost offer improved performance and efficiency over traditional Gradient Boosting."
          submodels={[
            {
              name: "XGBoost",
              description: "Optimized distributed gradient boosting library.",
              hyperparameters: [
                { name: 'n_estimators', description: 'Number of gradient boosted trees' },
                { name: 'learning_rate', description: 'Boosting learning rate' },
                { name: 'max_depth', description: 'Maximum tree depth for base learners' },
                { name: 'subsample', description: 'Subsample ratio of the training instance' },
                { name: 'colsample_bytree', description: 'Subsample ratio of columns when constructing each tree' },
              ],
              checks: {
                unscaled: true,
                missing: true,
                categorical: false,
                regression: true,
                classification: true
              },
              bayesianOptimization: `
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from xgboost import XGBClassifier, XGBRegressor

param_space = {
    'n_estimators': Integer(50, 1000),
    'learning_rate': Real(0.01, 1.0, prior='log-uniform'),
    'max_depth': Integer(3, 10),
    'subsample': Real(0.5, 1.0),
    'colsample_bytree': Real(0.5, 1.0)
}

# For classification
opt_clf = BayesSearchCV(XGBClassifier(), param_space, n_iter=50, cv=5, n_jobs=-1, verbose=0)
opt_clf.fit(X_train, y_train)

# For regression
opt_reg = BayesSearchCV(XGBRegressor(), param_space, n_iter=50, cv=5, n_jobs=-1, verbose=0)
opt_reg.fit(X_train, y_train)
              `
            },
            {
              name: "LightGBM",
              description: "Light Gradient Boosting Machine, uses histogram-based algorithms.",
              hyperparameters: [
                { name: 'n_estimators', description: 'Number of boosting iterations' },
                { name: 'learning_rate', description: 'Boosting learning rate' },
                { name: 'num_leaves', description: 'Maximum tree leaves for base learners' },
                { name: 'feature_fraction', description: 'Fraction of features to be used in each iteration' },
                { name: 'bagging_fraction', description: 'Fraction of data to be used for each iteration' },
              ],
              checks: {
                unscaled: true,
                missing: true,
                categorical: true,
                regression: true,
                classification: true
              },
              bayesianOptimization: `
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from lightgbm import LGBMClassifier, LGBMRegressor

param_space = {
    'n_estimators': Integer(50, 1000),
    'learning_rate': Real(0.01, 1.0, prior='log-uniform'),
    'num_leaves': Integer(20, 100),
    'feature_fraction': Real(0.5, 1.0),
    'bagging_fraction': Real(0.5, 1.0)
}

# For classification
opt_clf = BayesSearchCV(LGBMClassifier(), param_space, n_iter=50, cv=5, n_jobs=-1, verbose=0)
opt_clf.fit(X_train, y_train)

# For regression
opt_reg = BayesSearchCV(LGBMRegressor(), param_space, n_iter=50, cv=5, n_jobs=-1, verbose=0)
opt_reg.fit(X_train, y_train)
              `
            },
            {
              name: "CatBoost",
              description: "Implements ordered boosting and an innovative algorithm for processing categorical features.",
              hyperparameters: [
                { name: 'iterations', description: 'Number of boosting iterations' },
                { name: 'learning_rate', description: 'Boosting learning rate' },
                { name: 'depth', description: 'Depth of the tree' },
                { name: 'l2_leaf_reg', description: 'L2 regularization coefficient' },
                { name: 'border_count', description: 'Number of splits for numerical features' },
              ],
              checks: {
                unscaled: true,
                missing: true,
                categorical: true,
                regression: true,
                classification: true
              },
              bayesianOptimization: `
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from catboost import CatBoostClassifier, CatBoostRegressor

param_space = {
    'iterations': Integer(50, 1000),
    'learning_rate': Real(0.01, 1.0, prior='log-uniform'),
    'depth': Integer(3, 10),
    'l2_leaf_reg': Real(1e-3, 10, prior='log-uniform'),
    'border_count': Integer(32, 255)
}

# For classification
opt_clf = BayesSearchCV(CatBoostClassifier(verbose=False), param_space, n_iter=50, cv=5, n_jobs=-1, verbose=0)
opt_clf.fit(X_train, y_train)

# For regression
opt_reg = BayesSearchCV(CatBoostRegressor(verbose=False), param_space, n_iter=50, cv=5, n_jobs=-1, verbose=0)
opt_reg.fit(X_train, y_train)
              `
            }
          ]}
        />
      </Stack>
    </Container>
  );
};

const ModelSection = ({ title, id, math, description, hyperparameters, submodels, checks, bayesianOptimization }) => (
  <Stack spacing="md">
    <Title order={2} id={id}>{title}</Title>
    {math}
    <Text>{description}</Text>
    {submodels ? (
      submodels.map((submodel, index) => (
        <Stack key={index} spacing="sm">
          <Title order={3}>{submodel.name}</Title>
          <Text>{submodel.description}</Text>
          <HyperparameterTable hyperparameters={submodel.hyperparameters} />
          <ChecksTable checks={submodel.checks} />
          <Title order={4}>Bayesian Optimization Cheat Sheet</Title>
          <CodeBlock language="python" code={submodel.bayesianOptimization} />
        </Stack>
      ))
    ) : (
      <>
        <HyperparameterTable hyperparameters={hyperparameters} />
        <ChecksTable checks={checks} />
        <Title order={3}>Bayesian Optimization Cheat Sheet</Title>
        <CodeBlock language="python" code={bayesianOptimization} />
      </>
    )}
  </Stack>
);

const HyperparameterTable = ({ hyperparameters }) => (
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
);

const ChecksTable = ({ checks }) => (
  <Table>
    <thead>
      <tr>
        <th>Check</th>
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
);

export default EnsembleModels;