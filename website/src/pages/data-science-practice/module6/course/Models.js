import React from 'react';
import { Container, Title, Text, Stack, List, Flex, Image } from '@mantine/core';
import CodeBlock from 'components/CodeBlock';
import { InlineMath, BlockMath } from 'react-katex';
import 'katex/dist/katex.min.css';

const Models = () => {
  return (
    <Container fluid>
      <Title order={1} mt="xl" mb="md">Machine Learning Models</Title>

      <Stack spacing="xl">
        <div data-slide>
          <Title order={2}>Linear Models</Title>
          <BlockMath math="y = \beta_0 + \beta_1x_1 + \beta_2x_2 + ... + \beta_nx_n + \epsilon" />

          <Title order={3} mt="lg">Learning Principle</Title>
          <Text mb="md">
            Linear models minimize the residual sum of squares <InlineMath math="\sum_{i=1}^n (y_i - \hat{y}_i)^2" /> through closed-form solution <InlineMath math="\beta = (X^TX)^{-1}X^Ty" /> or gradient descent for large datasets.
          </Text>

          <Title order={3}>Key Hyperparameters</Title>
          <List spacing="sm">
            <List.Item>
              <strong>C</strong> (regularization): Controls overfitting. Small C = strong regularization, large C = weak regularization. Typical range: <InlineMath math="10^{-3}" /> to <InlineMath math="10^{3}" />
            </List.Item>
            <List.Item>
              <strong>penalty</strong>: L1 produces sparse models (feature selection), L2 handles collinearity, ElasticNet combines both with mixing parameter <InlineMath math="\alpha" />
            </List.Item>
            <List.Item>
              <strong>solver</strong>: 'liblinear' for small datasets, 'sag'/'saga' for large datasets, 'lbfgs' for multiclass problems
            </List.Item>
          </List>

          <CodeBlock language="python" code={`# Optimization example
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(C=1.0, penalty='l2', solver='lbfgs')
model.fit(X_train, y_train)`} />

          <Title order={3} mt="lg">Bayesian Optimization</Title>
          <CodeBlock language="python" code={`from skopt import BayesSearchCV
from skopt.space import Real, Categorical
from sklearn.linear_model import LogisticRegression, Ridge

# Classification
param_space = {
    'C': Real(1e-3, 1e3, prior='log-uniform'),
    'penalty': Categorical(['l1', 'l2']),
    'solver': Categorical(['liblinear', 'saga'])
}
opt = BayesSearchCV(LogisticRegression(), param_space, n_iter=32, cv=5)

# Regression
param_space_reg = {'alpha': Real(1e-3, 1e3, prior='log-uniform')}
opt_reg = BayesSearchCV(Ridge(), param_space_reg, n_iter=32, cv=5)`} />
        </div>

        <div data-slide>
          <Title order={2}>K-Nearest Neighbors (KNN)</Title>
          <BlockMath math="f(x) = \frac{1}{k} \sum_{i \in N_k(x, D)} y_i" />

          <Title order={3} mt="lg">Learning Principle</Title>
          <Text mb="md">
            KNN is a lazy learner - no explicit training phase. Predictions are made by finding k nearest points using distance metric <InlineMath math="d(x_i, x_j)" /> and aggregating their labels (majority vote for classification, mean for regression).
          </Text>

          <Title order={3}>Key Hyperparameters</Title>
          <List spacing="sm">
            <List.Item>
              <strong>n_neighbors (k)</strong>: Small k = high variance (overfitting), large k = high bias (underfitting). Odd values prevent ties. Typical: 3-20
            </List.Item>
            <List.Item>
              <strong>weights</strong>: 'uniform' treats all neighbors equally, 'distance' weights by <InlineMath math="1/d" /> giving closer points more influence
            </List.Item>
            <List.Item>
              <strong>metric & p</strong>: Euclidean (p=2), Manhattan (p=1), or Minkowski (general). Choice depends on feature scale and domain
            </List.Item>
          </List>

          <CodeBlock language="python" code={`# KNN with distance weighting
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=5, weights='distance')
model.fit(X_train, y_train)`} />

          <Title order={3} mt="lg">Bayesian Optimization</Title>
          <CodeBlock language="python" code={`from skopt import BayesSearchCV
from skopt.space import Integer, Categorical
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

param_space = {
    'n_neighbors': Integer(3, 30),
    'weights': Categorical(['uniform', 'distance']),
    'metric': Categorical(['euclidean', 'manhattan'])
}

# Classification & Regression use same hyperparameters
opt_clf = BayesSearchCV(KNeighborsClassifier(), param_space, n_iter=32, cv=5)
opt_reg = BayesSearchCV(KNeighborsRegressor(), param_space, n_iter=32, cv=5)`} />

          <Flex direction="column" align="center" mt="md">
            <Image
              src="/assets/data-science-practice/module6/knn.png"
              style={{ maxWidth: 'min(600px, 80vw)', height: 'auto' }}
              fluid
            />
          </Flex>
        </div>

        <div data-slide>
          <Title order={2}>Support Vector Machines (SVM)</Title>
          <BlockMath math="\min_{\mathbf{w},b} \frac{1}{2}||\mathbf{w}||^2 + C\sum_{i=1}^n \xi_i" />

          <Title order={3} mt="lg">Learning Principle</Title>
          <Text mb="md">
            SVM maximizes margin between classes by solving quadratic optimization. Maps data to higher dimensions via kernel trick: <InlineMath math="K(x_i, x_j) = \phi(x_i)^T\phi(x_j)" /> without explicit transformation.
          </Text>

          <Title order={3}>Key Hyperparameters</Title>
          <List spacing="sm">
            <List.Item>
              <strong>C</strong>: Trade-off between margin maximization and misclassification. Small C = soft margin, large C = hard margin. Range: <InlineMath math="10^{-2}" /> to <InlineMath math="10^{4}" />
            </List.Item>
            <List.Item>
              <strong>kernel</strong>: 'linear' for linearly separable, 'rbf' (Gaussian) for non-linear, 'poly' for polynomial boundaries
            </List.Item>
            <List.Item>
              <strong>gamma</strong>: Kernel coefficient controlling influence radius. Small = far reach, large = close reach. Auto = <InlineMath math="1/(n\_features \cdot var(X))" />
            </List.Item>
          </List>

          <CodeBlock language="python" code={`# SVM with RBF kernel
from sklearn.svm import SVC
model = SVC(kernel='rbf', C=1.0, gamma='scale')
model.fit(X_train, y_train)`} />

          <Title order={3} mt="lg">Bayesian Optimization</Title>
          <CodeBlock language="python" code={`from skopt import BayesSearchCV
from skopt.space import Real, Categorical
from sklearn.svm import SVC, SVR

param_space = {
    'C': Real(1e-2, 1e4, prior='log-uniform'),
    'gamma': Real(1e-4, 1e1, prior='log-uniform'),
    'kernel': Categorical(['rbf', 'linear'])
}

opt_clf = BayesSearchCV(SVC(), param_space, n_iter=32, cv=5)
opt_reg = BayesSearchCV(SVR(), param_space, n_iter=32, cv=5)`} />

          <Flex direction="column" align="center" mt="md">
            <Image
              src="/assets/data-science-practice/module6/svm.png"
              style={{ maxWidth: 'min(600px, 80vw)', height: 'auto' }}
              fluid
            />
          </Flex>
        </div>

        <div data-slide>
          <Title order={2}>Decision Trees</Title>
          <BlockMath math="\text{Gain}(S, A) = H(S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|} H(S_v)" />

          <Title order={3} mt="lg">Learning Principle</Title>
          <Text mb="md">
            Trees recursively partition feature space using greedy splits that maximize information gain (entropy reduction) or minimize Gini impurity: <InlineMath math="Gini = 1 - \sum_{i=1}^C p_i^2" />. Each node tests one feature, creating axis-aligned decision boundaries.
          </Text>

          <Title order={3}>Key Hyperparameters</Title>
          <List spacing="sm">
            <List.Item>
              <strong>max_depth</strong>: Controls tree complexity. Deep trees overfit, shallow trees underfit. Start with 3-10 for interpretability
            </List.Item>
            <List.Item>
              <strong>min_samples_split</strong>: Minimum samples to split node. Higher values prevent overfitting. Typical: 2-20 or 1-5% of data
            </List.Item>
            <List.Item>
              <strong>max_features</strong>: Features considered per split. 'sqrt' for classification, 'auto' for regression, reduces variance
            </List.Item>
            <List.Item>
              <strong>criterion</strong>: 'gini' for Gini impurity (faster), 'entropy' for information gain (slightly better for imbalanced data)
            </List.Item>
          </List>

          <CodeBlock language="python" code={`# Decision tree with pruning
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth=5, min_samples_split=20)
model.fit(X_train, y_train)`} />

          <Title order={3} mt="lg">Bayesian Optimization</Title>
          <CodeBlock language="python" code={`from skopt import BayesSearchCV
from skopt.space import Integer, Real, Categorical
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

param_space = {
    'max_depth': Integer(2, 15),
    'min_samples_split': Integer(2, 50),
    'min_samples_leaf': Integer(1, 20),
    'max_features': Categorical(['sqrt', 'log2', None])
}

opt_clf = BayesSearchCV(DecisionTreeClassifier(), param_space, n_iter=32, cv=5)
opt_reg = BayesSearchCV(DecisionTreeRegressor(), param_space, n_iter=32, cv=5)`} />

          <Flex direction="column" align="center" mt="md">
            <Image
              src="/assets/data-science-practice/module6/tree.png"
              style={{ maxWidth: 'min(600px, 80vw)', height: 'auto' }}
              fluid
            />
          </Flex>
        </div>


      </Stack>
    </Container>
  );
};

export default Models;