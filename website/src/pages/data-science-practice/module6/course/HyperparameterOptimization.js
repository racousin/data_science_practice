import React from 'react';
import { Container, Title, Text, List, Flex, Image } from '@mantine/core';
import CodeBlock from 'components/CodeBlock';
import { InlineMath, BlockMath } from 'react-katex';
import 'katex/dist/katex.min.css';

const HyperparameterOptimization = () => {
  return (
    <Container fluid>
      {/* Slide 1: Introduction */}
      <div data-slide>
        <Title order={1} id="hyperparameter-optimization" mt="xl" mb="md">Hyperparameter Optimization</Title>

        <Text mb="md">
          Hyperparameter optimization is the process of finding the best set of hyperparameters for a machine learning model.
        </Text>

        <Text mb="md">
          <strong>Key distinction:</strong> Hyperparameters are set before training begins, unlike model parameters that are learned during training.
        </Text>

        <Flex direction="column" align="center">
          <Image
            src="/assets/data-science-practice/module6/tunning.jpg"
            style={{ maxWidth: 'min(800px, 90vw)', height: 'auto' }}
            fluid
          />
        </Flex>
      </div>

      {/* Slide 2: Grid Search Introduction */}
      <div data-slide>
        <Title order={2} id="grid-search">Grid Search</Title>

        <Text mb="md">
          Grid search exhaustively tries all possible combinations of specified hyperparameter values. It's comprehensive but can be computationally expensive.
        </Text>

        <Text mb="md">
          <strong>Step 1: Define the parameter grid</strong>
        </Text>

        <CodeBlock
          language="python"
          code={`from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# Define parameter combinations to test
param_grid = {
    'C': [0.1, 1, 10],           # 3 values
    'kernel': ['rbf', 'linear'],  # 2 values
    'gamma': [0.1, 1, 'scale']    # 3 values
}
# Total combinations: 3 × 2 × 3 = 18`}
        />

        <Text mt="md" mb="md">
          <strong>Step 2: Setup and run grid search</strong>
        </Text>

        <CodeBlock
          language="python"
          code={`# Create model and grid search
svc = SVC()
grid_search = GridSearchCV(
    svc, param_grid,
    cv=5,                # 5-fold cross-validation
    scoring='accuracy'   # Metric to optimize
)`}
        />

        <Text mt="md" mb="md">
          <strong>Step 3: Fit and get results</strong>
        </Text>

        <CodeBlock
          language="python"
          code={`# Fit on your data
grid_search.fit(X, y)

# Best parameters found
print("Best params:", grid_search.best_params_)
# {'C': 1, 'gamma': 0.1, 'kernel': 'rbf'}

print("Best score:", grid_search.best_score_)
# 0.98`}
        />
      </div>

      {/* Slide 3: Understanding Grid Search */}
      <div data-slide>
        <Title order={2}>Grid Search: Pros and Cons</Title>

        <Text mb="md">
          <strong>Advantages:</strong>
        </Text>

        <List mb="md">
          <List.Item>Guarantees finding the best combination within the specified grid</List.Item>
          <List.Item>Simple to understand and implement</List.Item>
          <List.Item>Results are reproducible</List.Item>
        </List>

        <Text mb="md">
          <strong>Disadvantages:</strong>
        </Text>

        <List mb="md">
          <List.Item>Computationally expensive: O(n_combinations × n_folds)</List.Item>
          <List.Item>Doesn't scale well with many hyperparameters</List.Item>
          <List.Item>May miss optimal values between grid points</List.Item>
        </List>

        <Text mb="md">
          <strong>Accessing grid search results:</strong>
        </Text>

        <CodeBlock
          language="python"
          code={`# Access all results
results = grid_search.cv_results_

# Get mean test scores for each combination
mean_scores = results['mean_test_score']
params = results['params']

# Find top 3 parameter combinations
top_3_idx = np.argsort(mean_scores)[-3:]
for idx in top_3_idx:
    print(f"Score: {mean_scores[idx]:.3f}, Params: {params[idx]}")`}
        />
      </div>

      {/* Slide 4: Random Search */}
      <div data-slide>
        <Title order={2} id="random-search">Random Search</Title>

        <Text mb="md">
          Random search samples random combinations from specified distributions. Often more efficient than grid search, especially when not all hyperparameters are equally important.
        </Text>

        <Text mb="md">
          <strong>Step 1: Define parameter distributions</strong>
        </Text>

        <CodeBlock
          language="python"
          code={`from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# Define distributions to sample from
param_dist = {
    'n_estimators': randint(10, 200),      # Integer from 10-200
    'max_depth': randint(1, 20),           # Integer from 1-20
    'min_samples_split': randint(2, 20),   # Integer from 2-20
    'max_features': uniform(0, 1)          # Float from 0-1
}`}
        />

        <Text mt="md" mb="md">
          <strong>Step 2: Configure random search</strong>
        </Text>

        <CodeBlock
          language="python"
          code={`from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
random_search = RandomizedSearchCV(
    rf,
    param_distributions=param_dist,
    n_iter=100,      # Try 100 random combinations
    cv=5,            # 5-fold CV
    random_state=42  # For reproducibility
)`}
        />

        <Text mt="md" mb="md">
          <strong>Why random can beat grid search:</strong>
        </Text>

        <Text>
          If only some hyperparameters matter, random search explores more values of important parameters while grid search wastes time on unimportant combinations.
        </Text>
      </div>

      {/* Slide 5: Random vs Grid Comparison */}
      <div data-slide>
        <Title order={2}>Random Search vs Grid Search</Title>

        <Text mb="md">
          <strong>Efficiency comparison:</strong>
        </Text>

        <CodeBlock
          language="python"
          code={`import time

# Grid Search: 3×3×3×3 = 81 combinations
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Random Search: 50 random combinations
param_dist = {
    'n_estimators': randint(50, 150),
    'max_depth': randint(5, 15),
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 4)
}`}
        />

        <Text mt="md" mb="md">
          <strong>Performance comparison:</strong>
        </Text>

        <CodeBlock
          language="python"
          code={`# Compare execution times
start = time.time()
grid_search.fit(X_train, y_train)
grid_time = time.time() - start

start = time.time()
random_search.fit(X_train, y_train)
random_time = time.time() - start

print(f"Grid: {grid_time:.1f}s, Score: {grid_search.best_score_:.3f}")
print(f"Random: {random_time:.1f}s, Score: {random_search.best_score_:.3f}")`}
        />

        <Text mt="md">
          <strong>Rule of thumb:</strong> Use random search when you have many hyperparameters or limited computational budget.
        </Text>
      </div>

      {/* Slide 6: Bayesian Optimization Introduction */}
      <div data-slide>
        <Title order={2} id="bayesian-optimization">Bayesian Optimization</Title>

        <Text mb="md">
          Bayesian optimization uses probabilistic models to guide the search. It balances exploration of unknown regions with exploitation of promising areas.
        </Text>

        <Text mb="md">
          <strong>The Bayesian approach:</strong>
        </Text>

        <List mb="md">
          <List.Item>
            <><strong>Prior:</strong> Initial belief <InlineMath math="P(\theta)" /> about good hyperparameter values</>
          </List.Item>
          <List.Item>
            <><strong>Likelihood:</strong> Observed performance <InlineMath math="P(D|\theta)" /> for tried values</>
          </List.Item>
          <List.Item>
            <><strong>Posterior:</strong> Updated belief after seeing results</>
          </List.Item>
        </List>

        <BlockMath math="P(\theta|D) = \frac{P(D|\theta)P(\theta)}{P(D)}" />

        <Text mt="md">
          The posterior represents our updated knowledge about which hyperparameters are likely to perform well.
        </Text>
      </div>

      {/* Slide 7: Bayesian Optimization Priors */}
      <div data-slide>
        <Title order={2}>Common Priors in Bayesian Optimization</Title>

        <List spacing="md">
          <List.Item>
            <><strong>Uniform Prior:</strong> All values equally likely</>
            <CodeBlock
              language="python"
              code={`Real(0.001, 0.1, prior='uniform')  # Learning rate`}
            />
          </List.Item>

          <List.Item>
            <><strong>Log-Uniform Prior:</strong> For parameters spanning orders of magnitude</>
            <CodeBlock
              language="python"
              code={`Real(1e-4, 1e-1, prior='log-uniform')  # Better for learning rates`}
            />
          </List.Item>

          <List.Item>
            <><strong>Integer Prior:</strong> For discrete values</>
            <CodeBlock
              language="python"
              code={`Integer(10, 200)  # Number of trees`}
            />
          </List.Item>

          <List.Item>
            <><strong>Categorical Prior:</strong> For non-numeric choices</>
            <CodeBlock
              language="python"
              code={`Categorical(['relu', 'tanh', 'sigmoid'])  # Activation functions`}
            />
          </List.Item>
        </List>

        <Text mt="md">
          <strong>Tip:</strong> Use log-uniform for learning rates, regularization strengths, and other parameters that vary across scales.
        </Text>
      </div>

      {/* Slide 8: Bayesian Optimization Implementation */}
      <div data-slide>
        <Title order={2}>Implementing Bayesian Optimization</Title>

        <Text mb="md">
          <strong>Step 1: Define search space with appropriate priors</strong>
        </Text>

        <CodeBlock
          language="python"
          code={`from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

search_spaces = {
    'n_estimators': Integer(10, 200),
    'max_depth': Integer(1, 20),
    'min_samples_leaf': Real(0.1, 0.5, prior='log-uniform'),
    'max_features': Categorical(['auto', 'sqrt', 'log2'])
}`}
        />

        <Text mt="md" mb="md">
          <strong>Step 2: Setup Bayesian search</strong>
        </Text>

        <CodeBlock
          language="python"
          code={`from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(random_state=42)
bayes_search = BayesSearchCV(
    rf,
    search_spaces,
    n_iter=50,      # Maximum iterations
    cv=5,           # Cross-validation folds
    n_jobs=-1,      # Parallel processing
    random_state=42
)`}
        />

        <Text mt="md" mb="md">
          <strong>Step 3: Add early stopping</strong>
        </Text>

        <CodeBlock
          language="python"
          code={`from skopt.callbacks import DeltaYStopper

# Stop if improvement < 0.001 for 3 iterations
stopper = DeltaYStopper(delta=0.001, n_best=3)

# Fit with early stopping
bayes_search.fit(X, y, callback=[stopper])`}
        />
      </div>

      {/* Slide 9: Scoring Methods */}
      <div data-slide>
        <Title order={2} id="scoring-methods">Scoring Methods</Title>

        <Text mb="md">
          Scoring methods define how we evaluate model performance during optimization.
        </Text>

        <Text mb="md">
          <strong>Using built-in metrics:</strong>
        </Text>

        <CodeBlock
          language="python"
          code={`# Single metric
grid_search = GridSearchCV(
    model, param_grid,
    scoring='accuracy'  # or 'f1', 'roc_auc', 'r2', etc.
)

# Multiple metrics (refit on best 'accuracy')
grid_search = GridSearchCV(
    model, param_grid,
    scoring=['accuracy', 'precision', 'recall'],
    refit='accuracy'
)`}
        />

        <Text mt="md" mb="md">
          <strong>Creating custom scorers:</strong>
        </Text>

        <CodeBlock
          language="python"
          code={`from sklearn.metrics import make_scorer

def custom_metric(y_true, y_pred):
    # Example: Weighted F1 for imbalanced classes
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return 2.0 * precision * recall / (1.5*precision + recall)`}
        />

        <Text mt="md" mb="md">
          <strong>Using custom scorer:</strong>
        </Text>

        <CodeBlock
          language="python"
          code={`# Create scorer (greater_is_better=True by default)
custom_scorer = make_scorer(custom_metric)

# Use in optimization
grid_search = GridSearchCV(
    model, param_grid,
    scoring=custom_scorer
)`}
        />
      </div>

      {/* Slide 11: Cross-Validation Strategies */}
      <div data-slide>
        <Title order={2} id="cv-strategies">Cross-Validation Strategies</Title>

        <Text mb="md">
          Different problems require different CV strategies during hyperparameter optimization.
        </Text>

        <Text mb="md">
          <strong>Standard K-Fold:</strong>
        </Text>

        <CodeBlock
          language="python"
          code={`from sklearn.model_selection import KFold

# For general regression/classification
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(model, param_grid, cv=kfold)`}
        />

        <Text mt="md" mb="md">
          <strong>Stratified K-Fold for imbalanced data:</strong>
        </Text>

        <CodeBlock
          language="python"
          code={`from sklearn.model_selection import StratifiedKFold

# Maintains class distribution in each fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(model, param_grid, cv=skf)`}
        />

        <Text mt="md" mb="md">
          <strong>Time Series Split:</strong>
        </Text>

        <CodeBlock
          language="python"
          code={`from sklearn.model_selection import TimeSeriesSplit

# Respects temporal order
tscv = TimeSeriesSplit(n_splits=5)
grid_search = GridSearchCV(model, param_grid, cv=tscv)`}
        />

        <Text mt="md" mb="md">
          <strong>Group K-Fold for grouped data:</strong>
        </Text>

        <CodeBlock
          language="python"
          code={`from sklearn.model_selection import GroupKFold

# Ensures groups don't split across folds
gkf = GroupKFold(n_splits=5)
grid_search = GridSearchCV(model, param_grid, cv=gkf)
grid_search.fit(X, y, groups=group_labels)`}
        />
      </div>

      {/* Slide 12: Nested Cross-Validation */}
      <div data-slide>
        <Title order={2}>Nested Cross-Validation</Title>
        <Flex direction="column" align="center">
          <Image
            src="/assets/data-science-practice/module6/d-nested-cv.png"
            style={{ maxWidth: 'min(800px, 90vw)', height: 'auto' }}
            fluid
          />
        </Flex>
        <Text mb="md">
          Nested CV provides unbiased performance estimation when doing hyperparameter optimization.
        </Text>

        <Text mb="md">
          <strong>The problem with single CV:</strong> Using the same data for hyperparameter selection and performance evaluation leads to optimistic estimates.
        </Text>

        <Text mb="md">
          <strong>Solution: Two nested loops</strong>
        </Text>

        <CodeBlock
          language="python"
          code={`from sklearn.model_selection import cross_val_score

# Inner loop: hyperparameter optimization
param_grid = {'C': [0.1, 1, 10], 'gamma': [0.01, 0.1, 1]}
inner_cv = KFold(n_splits=3)

# Setup grid search for inner loop
clf = GridSearchCV(SVC(), param_grid, cv=inner_cv)

# Outer loop: unbiased performance estimation
outer_cv = KFold(n_splits=5)
scores = cross_val_score(clf, X, y, cv=outer_cv)`}
        />

        <Text mt="md" mb="md">
          <strong>Interpretation:</strong>
        </Text>

        <CodeBlock
          language="python"
          code={`print(f"Unbiased score: {scores.mean():.3f} (+/- {scores.std():.3f})")
# This score is a reliable estimate of real-world performance`}
        />
      </div>

      {/* Slide 13: Practical Tips */}
      <div data-slide>
        <Title order={2}>Practical Tips for Hyperparameter Optimization</Title>

        <Text mb="md">
          <strong>1. Start with defaults, then optimize:</strong>
        </Text>

        <CodeBlock
          language="python"
          code={`# First: baseline with defaults
baseline_model = RandomForestClassifier()
baseline_score = cross_val_score(baseline_model, X, y).mean()

# Then: optimize if baseline is promising`}
        />

        <Text mt="md" mb="md">
          <strong>2. Use coarse-to-fine search:</strong>
        </Text>

        <CodeBlock
          language="python"
          code={`# Round 1: Coarse grid
param_grid_coarse = {'C': [0.001, 0.1, 10, 1000]}

# Round 2: Fine grid around best value
# If best C=10, search:
param_grid_fine = {'C': [5, 7, 10, 15, 20]}`}
        />

        <Text mt="md" mb="md">
          <strong>3. Monitor search progress:</strong>
        </Text>

        <CodeBlock
          language="python"
          code={`# Use verbose to track progress
grid_search = GridSearchCV(
    model, param_grid,
    cv=5, verbose=2,  # Shows progress
    n_jobs=-1        # Use all cores
)`}
        />
      </div>

    </Container>
  );
};

export default HyperparameterOptimization;