import React from 'react';
import { Container, Title, Text, Stack, List, Flex, Image } from '@mantine/core';
import CodeBlock from 'components/CodeBlock';
import { InlineMath, BlockMath } from 'react-katex';
import 'katex/dist/katex.min.css';

const EnsembleModels = () => {
  return (
    <Container fluid>
      <Title order={1} mt="xl" mb="md">Ensemble Models</Title>

      <Stack spacing="xl">
        <div data-slide>
          <Title order={2}>Random Forests</Title>
          <BlockMath math="f(x) = \frac{1}{B} \sum_{b=1}^B f_b(x)" />

          <Title order={3} mt="lg">Learning Principle</Title>
          <Text mb="md">
            Random Forests combine bagging with random feature selection. Each tree is trained on bootstrap sample with <InlineMath math="\sqrt{p}" /> features randomly selected at each split. Reduces variance through averaging uncorrelated trees.
          </Text>

          <Title order={3}>Key Hyperparameters</Title>
          <List spacing="sm">
            <List.Item>
              <strong>n_estimators</strong>: Number of trees. More trees = better performance but diminishing returns. Typical: 100-500
            </List.Item>
            <List.Item>
              <strong>max_features</strong>: Features per split. 'sqrt' for classification, 'auto' (all) for regression, controls diversity-accuracy trade-off
            </List.Item>
            <List.Item>
              <strong>max_depth</strong>: Tree depth. None = fully grown trees (high variance), shallow = underfitting. Often left unlimited
            </List.Item>
            <List.Item>
              <strong>min_samples_split</strong>: Minimum samples to split. Higher values prevent overfitting. Typical: 2-20
            </List.Item>
          </List>

          <CodeBlock language="python" code={`# Random Forest with typical settings
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=200, max_features='sqrt')
model.fit(X_train, y_train)`} />

          <Title order={3} mt="lg">Bayesian Optimization</Title>
          <CodeBlock language="python" code={`from skopt import BayesSearchCV
from skopt.space import Integer, Categorical
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

param_space = {
    'n_estimators': Integer(50, 300),
    'max_depth': Integer(5, 30),
    'min_samples_split': Integer(2, 20),
    'max_features': Categorical(['sqrt', 'log2', 0.5])
}

opt_clf = BayesSearchCV(RandomForestClassifier(), param_space, n_iter=32, cv=5)
opt_reg = BayesSearchCV(RandomForestRegressor(), param_space, n_iter=32, cv=5)`} />

          <Flex direction="column" align="center" mt="md">
            <Image
              src="/assets/data-science-practice/module6/rf.webp"
              style={{ maxWidth: 'min(600px, 80vw)', height: 'auto' }}
              fluid
            />
          </Flex>
        </div>

        <div data-slide>
          <Title order={2}>AdaBoost</Title>
          <BlockMath math="\alpha_t = \frac{1}{2}\ln\left(\frac{1-\epsilon_t}{\epsilon_t}\right), \quad w_i^{(t+1)} = w_i^{(t)} e^{-\alpha_t y_i h_t(x_i)}" />

          <Title order={3} mt="lg">Learning Principle</Title>
          <Text mb="md">
            AdaBoost sequentially trains weak learners on weighted data. Misclassified samples receive higher weights <InlineMath math="w_i" />, forcing subsequent learners to focus on hard cases. Final prediction combines all learners weighted by their accuracy.
          </Text>

          <Title order={3}>Key Hyperparameters</Title>
          <List spacing="sm">
            <List.Item>
              <strong>n_estimators</strong>: Number of weak learners. More = better fit but risk overfitting. Typical: 50-200
            </List.Item>
            <List.Item>
              <strong>learning_rate</strong>: Shrinks contribution of each classifier. Small = needs more estimators. Range: 0.01-1.0
            </List.Item>
            <List.Item>
              <strong>base_estimator</strong>: Weak learner type. Default: DecisionTreeClassifier(max_depth=1) - decision stumps
            </List.Item>
          </List>

          <CodeBlock language="python" code={`# AdaBoost with decision stumps
from sklearn.ensemble import AdaBoostClassifier
model = AdaBoostClassifier(n_estimators=100, learning_rate=1.0)
model.fit(X_train, y_train)`} />

          <Title order={3} mt="lg">Bayesian Optimization</Title>
          <CodeBlock language="python" code={`from skopt import BayesSearchCV
from skopt.space import Integer, Real
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor

param_space = {
    'n_estimators': Integer(50, 200),
    'learning_rate': Real(0.01, 2.0, prior='log-uniform')
}

opt_clf = BayesSearchCV(AdaBoostClassifier(), param_space, n_iter=32, cv=5)
opt_reg = BayesSearchCV(AdaBoostRegressor(), param_space, n_iter=32, cv=5)`} />
        </div>

        <div data-slide>
          <Title order={2}>Gradient Boosting</Title>
          <BlockMath math="F_m(x) = F_{m-1}(x) + \gamma_m h_m(x), \quad h_m = \arg\min_h \sum_{i=1}^n L(y_i, F_{m-1}(x_i) + h(x_i))" />

          <Title order={3} mt="lg">Learning Principle</Title>
          <Text mb="md">
            Gradient boosting fits new predictors to residual errors made by previous predictors. Each tree corrects its predecessor by minimizing loss gradient. Learning rate <InlineMath math="\gamma" /> controls contribution of each tree.
          </Text>

          <Title order={3}>Key Hyperparameters</Title>
          <List spacing="sm">
            <List.Item>
              <strong>n_estimators</strong>: Number of boosting stages. More stages = better fit but slower and risk overfitting. Typical: 100-1000
            </List.Item>
            <List.Item>
              <strong>learning_rate</strong>: Shrinkage coefficient. Small values need more trees but generalize better. Range: 0.01-0.3
            </List.Item>
            <List.Item>
              <strong>max_depth</strong>: Tree complexity. Shallow trees (3-8) work well, acting as weak learners
            </List.Item>
            <List.Item>
              <strong>subsample</strong>: Fraction of samples for fitting. Values {'<'} 1.0 lead to stochastic gradient boosting, reducing variance
            </List.Item>
          </List>

          <CodeBlock language="python" code={`# Gradient Boosting with typical settings
from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
model.fit(X_train, y_train)`} />

          <Title order={3} mt="lg">Bayesian Optimization</Title>
          <CodeBlock language="python" code={`from skopt import BayesSearchCV
from skopt.space import Integer, Real
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

param_space = {
    'n_estimators': Integer(50, 300),
    'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
    'max_depth': Integer(3, 8),
    'subsample': Real(0.7, 1.0)
}

opt_clf = BayesSearchCV(GradientBoostingClassifier(), param_space, n_iter=32, cv=5)
opt_reg = BayesSearchCV(GradientBoostingRegressor(), param_space, n_iter=32, cv=5)`} />

          <Flex direction="column" align="center" mt="md">
            <Image
              src="/assets/data-science-practice/module6/adaboost.jpg"
              style={{ maxWidth: 'min(600px, 80vw)', height: 'auto' }}
              fluid
            />
          </Flex>
        </div>

        <div data-slide>
          <Title order={2}>XGBoost</Title>
          <BlockMath math="\mathcal{L}(\phi) = \sum_i l(y_i, \hat{y}_i) + \sum_k \Omega(f_k), \quad \Omega(f) = \gamma T + \frac{1}{2}\lambda ||w||^2" />

          <Title order={3} mt="lg">Learning Principle</Title>
          <Text mb="md">
            XGBoost optimizes a regularized objective combining loss and complexity penalty. Uses second-order Taylor expansion for optimization and implements column subsampling. Handles sparse data efficiently through default direction learning.
            {' '}<a href="https://arxiv.org/pdf/1603.02754.pdf" target="_blank" rel="noopener noreferrer">[Chen & Guestrin, 2016]</a>
          </Text>

          <Title order={3}>Key Hyperparameters</Title>
          <List spacing="sm">
            <List.Item>
              <strong>n_estimators</strong>: Boosting rounds. Use early stopping to find optimal value. Typical: 100-1000
            </List.Item>
            <List.Item>
              <strong>learning_rate (eta)</strong>: Step size shrinkage. Lower values prevent overfitting. Range: 0.01-0.3
            </List.Item>
            <List.Item>
              <strong>max_depth</strong>: Tree depth. Deeper trees capture more complex patterns. Range: 3-10
            </List.Item>
            <List.Item>
              <strong>subsample</strong>: Row sampling ratio. Prevents overfitting. Range: 0.5-1.0
            </List.Item>
            <List.Item>
              <strong>colsample_bytree</strong>: Column sampling ratio. Adds regularization. Range: 0.3-1.0
            </List.Item>
            <List.Item>
              <strong>reg_lambda (λ)</strong>: L2 regularization on weights. Higher values = more conservative model
            </List.Item>
          </List>

          <CodeBlock language="python" code={`# XGBoost with regularization
from xgboost import XGBClassifier
model = XGBClassifier(n_estimators=200, learning_rate=0.1,
                      max_depth=6, subsample=0.8)
model.fit(X_train, y_train)`} />

          <Title order={3} mt="lg">Bayesian Optimization</Title>
          <CodeBlock language="python" code={`from skopt import BayesSearchCV
from skopt.space import Integer, Real
from xgboost import XGBClassifier, XGBRegressor

param_space = {
    'n_estimators': Integer(100, 500),
    'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
    'max_depth': Integer(3, 10),
    'subsample': Real(0.6, 1.0),
    'colsample_bytree': Real(0.6, 1.0)
}

opt_clf = BayesSearchCV(XGBClassifier(use_label_encoder=False), param_space, n_iter=32, cv=5)
opt_reg = BayesSearchCV(XGBRegressor(), param_space, n_iter=32, cv=5)`} />
        </div>

        <div data-slide>
          <Title order={2}>LightGBM</Title>
          <BlockMath math="\text{Gain} = \frac{1}{2} \left[ \frac{(\sum_{i \in L} g_i)^2}{n_L} + \frac{(\sum_{i \in R} g_i)^2}{n_R} - \frac{(\sum_{i} g_i)^2}{n} \right]" />

          <Title order={3} mt="lg">Learning Principle</Title>
          <Text mb="md">
            LightGBM uses histogram-based algorithms for faster training. Implements Gradient-based One-Side Sampling (GOSS) to keep instances with large gradients and Exclusive Feature Bundling (EFB) to reduce features. Grows trees leaf-wise rather than level-wise.
            {' '}<a href="https://papers.nips.cc/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf" target="_blank" rel="noopener noreferrer">[Ke et al., NeurIPS 2017]</a>
          </Text>

          <Title order={3}>Key Hyperparameters</Title>
          <List spacing="sm">
            <List.Item>
              <strong>num_leaves</strong>: Maximum leaves in one tree. Should be {'<'} <InlineMath math="2^{\text{max\_depth}}" />. Typical: 31-127
            </List.Item>
            <List.Item>
              <strong>learning_rate</strong>: Boosting learning rate. Lower = more rounds needed. Range: 0.01-0.3
            </List.Item>
            <List.Item>
              <strong>feature_fraction</strong>: Randomly select features for each iteration. Prevents overfitting. Range: 0.5-1.0
            </List.Item>
            <List.Item>
              <strong>bagging_fraction</strong>: Randomly select data for each iteration. Speeds up training. Range: 0.5-1.0
            </List.Item>
            <List.Item>
              <strong>lambda_l1/l2</strong>: L1/L2 regularization. Controls model complexity
            </List.Item>
          </List>

          <CodeBlock language="python" code={`# LightGBM with leaf-wise growth
from lightgbm import LGBMClassifier
model = LGBMClassifier(num_leaves=31, learning_rate=0.1,
                       feature_fraction=0.9)
model.fit(X_train, y_train)`} />

          <Title order={3} mt="lg">Bayesian Optimization</Title>
          <CodeBlock language="python" code={`from skopt import BayesSearchCV
from skopt.space import Integer, Real
from lightgbm import LGBMClassifier, LGBMRegressor

param_space = {
    'num_leaves': Integer(20, 100),
    'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
    'feature_fraction': Real(0.6, 1.0),
    'bagging_fraction': Real(0.6, 1.0),
    'min_child_samples': Integer(5, 30)
}

opt_clf = BayesSearchCV(LGBMClassifier(verbosity=-1), param_space, n_iter=32, cv=5)
opt_reg = BayesSearchCV(LGBMRegressor(verbosity=-1), param_space, n_iter=32, cv=5)`} />
        </div>

        <div data-slide>
          <Title order={2}>CatBoost</Title>
          <BlockMath math="\text{Ordered TS} = \frac{\sum_{j=1}^{p-1} [x_{j,k} = x_{i,k}] \cdot y_j + a \cdot P}{\sum_{j=1}^{p-1} [x_{j,k} = x_{i,k}] + a}" />

          <Title order={3} mt="lg">Learning Principle</Title>
          <Text mb="md">
            CatBoost implements ordered boosting to prevent prediction shift caused by target leakage. Uses symmetric trees as base predictors and novel algorithm for processing categorical features through target statistics with random permutations.
            {' '}<a href="https://arxiv.org/pdf/1706.09516.pdf" target="_blank" rel="noopener noreferrer">[Prokhorenkova et al., NeurIPS 2018]</a>
          </Text>

          <Title order={3}>Key Hyperparameters</Title>
          <List spacing="sm">
            <List.Item>
              <strong>iterations</strong>: Number of trees. CatBoost prevents overfitting well, can use more. Typical: 500-2000
            </List.Item>
            <List.Item>
              <strong>learning_rate</strong>: Step size. CatBoost auto-adjusts if not set. Range: 0.01-0.3
            </List.Item>
            <List.Item>
              <strong>depth</strong>: Tree depth. Symmetric trees, so depth has larger impact. Range: 4-10
            </List.Item>
            <List.Item>
              <strong>l2_leaf_reg</strong>: L2 regularization coefficient. Higher = more conservative. Range: 1-10
            </List.Item>
            <List.Item>
              <strong>cat_features</strong>: List of categorical feature indices. Handles categoricals natively without encoding
            </List.Item>
          </List>

          <CodeBlock language="python" code={`# CatBoost with categorical features
from catboost import CatBoostClassifier
model = CatBoostClassifier(iterations=1000, depth=6, learning_rate=0.1,
                          cat_features=categorical_indices, verbose=False)
model.fit(X_train, y_train)`} />

          <Title order={3} mt="lg">Bayesian Optimization</Title>
          <CodeBlock language="python" code={`from skopt import BayesSearchCV
from skopt.space import Integer, Real
from catboost import CatBoostClassifier, CatBoostRegressor

param_space = {
    'iterations': Integer(200, 1000),
    'depth': Integer(4, 10),
    'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
    'l2_leaf_reg': Real(1, 10, prior='log-uniform')
}

opt_clf = BayesSearchCV(CatBoostClassifier(verbose=False), param_space, n_iter=32, cv=5)
opt_reg = BayesSearchCV(CatBoostRegressor(verbose=False), param_space, n_iter=32, cv=5)`} />
        </div>

      </Stack>
    </Container>
  );
};

export default EnsembleModels;