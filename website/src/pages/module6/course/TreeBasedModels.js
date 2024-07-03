import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";
import { InlineMath, BlockMath } from "react-katex";
import "katex/dist/katex.min.css";

const TreeBasedModels = () => {
  return (
    <Container fluid>
      <h1 className="my-4">Tree-based Models</h1>

      <section>
        <h2 id="decision-trees">Decision Trees</h2>
        <h3>Theory and Decision Boundaries</h3>
        <p>
          Decision trees are non-parametric supervised learning methods used for
          classification and regression. They work by creating a model that
          predicts the target variable by learning simple decision rules
          inferred from the data features.
        </p>
        <p>The tree structure consists of:</p>
        <ul>
          <li>Root Node: The topmost node in the tree</li>
          <li>Internal Nodes: Nodes that test an attribute and branch</li>
          <li>
            Leaf Nodes: Terminal nodes that provide the final decision or
            prediction
          </li>
        </ul>

        <h3>Implementation with scikit-learn</h3>
        <CodeBlock
          language="python"
          code={`
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

# Classification
X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(f"Classification Accuracy: {accuracy_score(y_test, y_pred)}")

# Regression
X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

reg = DecisionTreeRegressor(random_state=42)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
print(f"Regression MSE: {mean_squared_error(y_test, y_pred)}")
          `}
        />

        <h3>Key Hyperparameters</h3>
        <ul>
          <li>
            <code>max_depth</code>: Maximum depth of the tree
          </li>
          <li>
            <code>min_samples_split</code>: Minimum number of samples required
            to split an internal node
          </li>
          <li>
            <code>min_samples_leaf</code>: Minimum number of samples required to
            be at a leaf node
          </li>
          <li>
            <code>max_features</code>: Number of features to consider when
            looking for the best split
          </li>
          <li>
            <code>criterion</code>: Function to measure the quality of a split
          </li>
        </ul>
      </section>

      <section>
        <h2 id="random-forests">Random Forests</h2>
        <h3>Ensemble Learning Concept</h3>
        <p>
          Random Forests are an ensemble learning method that operate by
          constructing multiple decision trees during training and outputting
          the class that is the mode of the classes (classification) or mean
          prediction (regression) of the individual trees.
        </p>
        <p>Key concepts:</p>
        <ul>
          <li>
            Bagging (Bootstrap Aggregating): Training each tree on a random
            subset of the data
          </li>
          <li>
            Feature Randomness: Considering only a random subset of features for
            splitting at each node
          </li>
        </ul>

        <h3>Implementation and Key Hyperparameters</h3>
        <CodeBlock
          language="python"
          code={`
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

# Classification
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)
y_pred = rf_clf.predict(X_test)
print(f"Random Forest Classification Accuracy: {accuracy_score(y_test, y_pred)}")

# Regression
X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train, y_train)
y_pred = rf_reg.predict(X_test)
print(f"Random Forest Regression MSE: {mean_squared_error(y_test, y_pred)}")
          `}
        />

        <h3>Feature Importance</h3>
        <p>
          Random Forests provide a measure of feature importance, which can be
          useful for feature selection and interpretation.
        </p>
        <CodeBlock
          language="python"
          code={`
import pandas as pd
import matplotlib.pyplot as plt

feature_importance = pd.DataFrame({
    'feature': range(X.shape[1]),
    'importance': rf_clf.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.bar(feature_importance['feature'], feature_importance['importance'])
plt.xlabel('Feature Index')
plt.ylabel('Importance')
plt.title('Random Forest Feature Importance')
plt.show()
          `}
        />
      </section>

      <section>
        <h2 id="gradient-boosting">Gradient Boosting Machines</h2>
        <p>
          Gradient Boosting is a machine learning technique that produces a
          prediction model in the form of an ensemble of weak prediction models,
          typically decision trees.
        </p>

        <h3>XGBoost, LightGBM, and CatBoost</h3>
        <p>These are popular implementations of gradient boosting:</p>
        <ul>
          <li>
            <strong>XGBoost</strong>: Optimized distributed gradient boosting
            library
          </li>
          <li>
            <strong>LightGBM</strong>: Light Gradient Boosting Machine, uses
            histogram-based algorithms
          </li>
          <li>
            <strong>CatBoost</strong>: Implements ordered boosting and an
            innovative algorithm for processing categorical features
          </li>
        </ul>

        <h3>Implementation and Key Hyperparameters</h3>
        <CodeBlock
          language="python"
          code={`
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost
xgb = XGBClassifier(n_estimators=100, random_state=42)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
print(f"XGBoost Accuracy: {accuracy_score(y_test, y_pred_xgb)}")

# LightGBM
lgbm = LGBMClassifier(n_estimators=100, random_state=42)
lgbm.fit(X_train, y_train)
y_pred_lgbm = lgbm.predict(X_test)
print(f"LightGBM Accuracy: {accuracy_score(y_test, y_pred_lgbm)}")

# CatBoost
cb = CatBoostClassifier(n_estimators=100, random_state=42, verbose=0)
cb.fit(X_train, y_train)
y_pred_cb = cb.predict(X_test)
print(f"CatBoost Accuracy: {accuracy_score(y_test, y_pred_cb)}")
          `}
        />

        <h3>Handling Categorical Variables</h3>
        <p>
          Gradient boosting models can handle categorical variables differently:
        </p>
        <ul>
          <li>
            XGBoost and LightGBM require encoding of categorical variables
            (e.g., one-hot encoding)
          </li>
          <li>
            CatBoost can handle categorical variables natively using its ordered
            boosting algorithm
          </li>
        </ul>
      </section>

      <section>
        <h2>Conclusion</h2>
        <p>
          Tree-based models are powerful and flexible algorithms that can
          capture complex non-linear relationships in data. They are widely used
          in various applications due to their good performance and
          interpretability. Random Forests and Gradient Boosting methods further
          improve upon single decision trees by creating ensembles, often
          leading to state-of-the-art performance on many machine learning
          tasks.
        </p>
      </section>
    </Container>
  );
};

export default TreeBasedModels;
