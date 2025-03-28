import React from 'react';
import { Container, Title, Text, Stack, List, Group, Table } from '@mantine/core';
import CodeBlock from 'components/CodeBlock';
import { InlineMath, BlockMath } from 'react-katex';
import 'katex/dist/katex.min.css';

// TODO add boosting

const BaggingSVG = () => (
  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 400">
    <rect x="10" y="10" width="120" height="80" fill="#f0f0f0" stroke="#000000" strokeWidth="2"/>
    <text x="70" y="55" textAnchor="middle" fontFamily="Arial, sans-serif" fontSize="14">Original Dataset</text>

    <rect x="200" y="10" width="100" height="60" fill="#e6f3ff" stroke="#000000" strokeWidth="2"/>
    <text x="250" y="45" textAnchor="middle" fontFamily="Arial, sans-serif" fontSize="12">Bootstrap Sample 1</text>

    <rect x="200" y="80" width="100" height="60" fill="#e6f3ff" stroke="#000000" strokeWidth="2"/>
    <text x="250" y="115" textAnchor="middle" fontFamily="Arial, sans-serif" fontSize="12">Bootstrap Sample 2</text>

    <rect x="200" y="150" width="100" height="60" fill="#e6f3ff" stroke="#000000" strokeWidth="2"/>
    <text x="250" y="185" textAnchor="middle" fontFamily="Arial, sans-serif" fontSize="12">Bootstrap Sample 3</text>

    <rect x="400" y="10" width="80" height="60" fill="#ffe6e6" stroke="#000000" strokeWidth="2"/>
    <text x="440" y="45" textAnchor="middle" fontFamily="Arial, sans-serif" fontSize="12">Model 1</text>

    <rect x="400" y="80" width="80" height="60" fill="#ffe6e6" stroke="#000000" strokeWidth="2"/>
    <text x="440" y="115" textAnchor="middle" fontFamily="Arial, sans-serif" fontSize="12">Model 2</text>

    <rect x="400" y="150" width="80" height="60" fill="#ffe6e6" stroke="#000000" strokeWidth="2"/>
    <text x="440" y="185" textAnchor="middle" fontFamily="Arial, sans-serif" fontSize="12">Model 3</text>

    <rect x="560" y="10" width="80" height="60" fill="#e6ffe6" stroke="#000000" strokeWidth="2"/>
    <text x="600" y="45" textAnchor="middle" fontFamily="Arial, sans-serif" fontSize="12">Prediction 1</text>

    <rect x="560" y="80" width="80" height="60" fill="#e6ffe6" stroke="#000000" strokeWidth="2"/>
    <text x="600" y="115" textAnchor="middle" fontFamily="Arial, sans-serif" fontSize="12">Prediction 2</text>

    <rect x="560" y="150" width="80" height="60" fill="#e6ffe6" stroke="#000000" strokeWidth="2"/>
    <text x="600" y="185" textAnchor="middle" fontFamily="Arial, sans-serif" fontSize="12">Prediction 3</text>

    <rect x="700" y="80" width="90" height="60" fill="#ffffe6" stroke="#000000" strokeWidth="2"/>
    <text x="745" y="115" textAnchor="middle" fontFamily="Arial, sans-serif" fontSize="12">Final Prediction</text>

    <defs>
      <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="0" refY="3.5" orient="auto">
        <polygon points="0 0, 10 3.5, 0 7" />
      </marker>
    </defs>

    <line x1="130" y1="50" x2="190" y2="40" stroke="#000" strokeWidth="2" markerEnd="url(#arrowhead)"/>
    <line x1="130" y1="50" x2="190" y2="110" stroke="#000" strokeWidth="2" markerEnd="url(#arrowhead)"/>
    <line x1="130" y1="50" x2="190" y2="180" stroke="#000" strokeWidth="2" markerEnd="url(#arrowhead)"/>

    <line x1="300" y1="40" x2="390" y2="40" stroke="#000" strokeWidth="2" markerEnd="url(#arrowhead)"/>
    <line x1="300" y1="110" x2="390" y2="110" stroke="#000" strokeWidth="2" markerEnd="url(#arrowhead)"/>
    <line x1="300" y1="180" x2="390" y2="180" stroke="#000" strokeWidth="2" markerEnd="url(#arrowhead)"/>

    <line x1="480" y1="40" x2="550" y2="40" stroke="#000" strokeWidth="2" markerEnd="url(#arrowhead)"/>
    <line x1="480" y1="110" x2="550" y2="110" stroke="#000" strokeWidth="2" markerEnd="url(#arrowhead)"/>
    <line x1="480" y1="180" x2="550" y2="180" stroke="#000" strokeWidth="2" markerEnd="url(#arrowhead)"/>

    <line x1="640" y1="40" x2="690" y2="100" stroke="#000" strokeWidth="2" markerEnd="url(#arrowhead)"/>
    <line x1="640" y1="110" x2="690" y2="110" stroke="#000" strokeWidth="2" markerEnd="url(#arrowhead)"/>
    <line x1="640" y1="180" x2="690" y2="120" stroke="#000" strokeWidth="2" markerEnd="url(#arrowhead)"/>

    <text x="160" y="30" textAnchor="middle" fontFamily="Arial, sans-serif" fontSize="12">Bootstrap</text>
    <text x="350" y="30" textAnchor="middle" fontFamily="Arial, sans-serif" fontSize="12">Train</text>
    <text x="520" y="30" textAnchor="middle" fontFamily="Arial, sans-serif" fontSize="12">Predict</text>
    <text x="670" y="70" textAnchor="middle" fontFamily="Arial, sans-serif" fontSize="12">Aggregate</text>
  </svg>
);

const StackingSVG = () => (
  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 400">
    {/* Original Dataset */}
    <rect x="10" y="10" width="120" height="80" fill="#f0f0f0" stroke="#000000" strokeWidth="2"/>
    <text x="70" y="55" textAnchor="middle" fontFamily="Arial, sans-serif" fontSize="14">Original Dataset</text>

    {/* Base Models */}
    <rect x="200" y="10" width="100" height="60" fill="#e6f3ff" stroke="#000000" strokeWidth="2"/>
    <text x="250" y="45" textAnchor="middle" fontFamily="Arial, sans-serif" fontSize="12">Base Model 1</text>

    <rect x="200" y="80" width="100" height="60" fill="#e6f3ff" stroke="#000000" strokeWidth="2"/>
    <text x="250" y="115" textAnchor="middle" fontFamily="Arial, sans-serif" fontSize="12">Base Model 2</text>

    <rect x="200" y="150" width="100" height="60" fill="#e6f3ff" stroke="#000000" strokeWidth="2"/>
    <text x="250" y="185" textAnchor="middle" fontFamily="Arial, sans-serif" fontSize="12">Base Model 3</text>

    {/* Predictions / Meta-features */}
    <rect x="400" y="10" width="100" height="60" fill="#ffe6e6" stroke="#000000" strokeWidth="2"/>
    <text x="450" y="45" textAnchor="middle" fontFamily="Arial, sans-serif" fontSize="12">Predictions 1</text>

    <rect x="400" y="80" width="100" height="60" fill="#ffe6e6" stroke="#000000" strokeWidth="2"/>
    <text x="450" y="115" textAnchor="middle" fontFamily="Arial, sans-serif" fontSize="12">Predictions 2</text>

    <rect x="400" y="150" width="100" height="60" fill="#ffe6e6" stroke="#000000" strokeWidth="2"/>
    <text x="450" y="185" textAnchor="middle" fontFamily="Arial, sans-serif" fontSize="12">Predictions 3</text>

    {/* Meta-features */}
    <rect x="560" y="80" width="100" height="60" fill="#e6ffe6" stroke="#000000" strokeWidth="2"/>
    <text x="610" y="115" textAnchor="middle" fontFamily="Arial, sans-serif" fontSize="12">Meta-features</text>

    {/* Meta-model */}
    <rect x="700" y="80" width="90" height="60" fill="#ffffe6" stroke="#000000" strokeWidth="2"/>
    <text x="745" y="115" textAnchor="middle" fontFamily="Arial, sans-serif" fontSize="12">Meta-model</text>

    {/* Arrows */}
    <defs>
      <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="0" refY="3.5" orient="auto">
        <polygon points="0 0, 10 3.5, 0 7" />
      </marker>
    </defs>

    {/* Original Dataset to Base Models */}
    <line x1="130" y1="50" x2="190" y2="40" stroke="#000" strokeWidth="2" markerEnd="url(#arrowhead)"/>
    <line x1="130" y1="50" x2="190" y2="110" stroke="#000" strokeWidth="2" markerEnd="url(#arrowhead)"/>
    <line x1="130" y1="50" x2="190" y2="180" stroke="#000" strokeWidth="2" markerEnd="url(#arrowhead)"/>

    {/* Base Models to Predictions */}
    <line x1="300" y1="40" x2="390" y2="40" stroke="#000" strokeWidth="2" markerEnd="url(#arrowhead)"/>
    <line x1="300" y1="110" x2="390" y2="110" stroke="#000" strokeWidth="2" markerEnd="url(#arrowhead)"/>
    <line x1="300" y1="180" x2="390" y2="180" stroke="#000" strokeWidth="2" markerEnd="url(#arrowhead)"/>

    {/* Predictions to Meta-features */}
    <line x1="500" y1="40" x2="550" y2="100" stroke="#000" strokeWidth="2" markerEnd="url(#arrowhead)"/>
    <line x1="500" y1="110" x2="550" y2="110" stroke="#000" strokeWidth="2" markerEnd="url(#arrowhead)"/>
    <line x1="500" y1="180" x2="550" y2="120" stroke="#000" strokeWidth="2" markerEnd="url(#arrowhead)"/>

    {/* Meta-features to Meta-model */}
    <line x1="660" y1="110" x2="690" y2="110" stroke="#000" strokeWidth="2" markerEnd="url(#arrowhead)"/>

    {/* Labels */}
    <text x="160" y="30" textAnchor="middle" fontFamily="Arial, sans-serif" fontSize="12">Train</text>
    <text x="350" y="30" textAnchor="middle" fontFamily="Arial, sans-serif" fontSize="12">Predict</text>
    <text x="530" y="70" textAnchor="middle" fontFamily="Arial, sans-serif" fontSize="12">Combine</text>
    <text x="680" y="70" textAnchor="middle" fontFamily="Arial, sans-serif" fontSize="12">Train</text>
  </svg>
);


const EnsembleTechniques = () => {
  return (
    <Container fluid>
      <Title order={1} mt="xl" mb="md">Ensemble Techniques</Title>

      <Stack spacing="xl">
        <Section
          title="Bagging (Bootstrap Aggregating)"
          id="bagging"
          description="Bagging is an ensemble technique that combines multiple models trained on different subsets of the data to reduce overfitting and improve generalization."
          concept={`
Bagging works as follows:
1. Create multiple subsets of the original dataset by sampling with replacement (bootstrap samples).
2. Train a separate model on each subset.
3. Combine the predictions of all models by voting (for classification) or averaging (for regression).

Key benefits:
- Reduces overfitting
- Decreases variance
- Improves stability and accuracy
          `}
          svgType={'bagging'}
          simpleImplementation={`
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification

# Create a synthetic dataset
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# Simple Bagging implementation
class SimpleBagging:
    def __init__(self, n_estimators=10):
        self.n_estimators = n_estimators
        self.estimators = []

    def fit(self, X, y):
        n_samples = X.shape[0]
        for _ in range(self.n_estimators):
            # Bootstrap sampling
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_bootstrap, y_bootstrap = X[indices], y[indices]
            
            # Train a decision tree on the bootstrap sample
            estimator = DecisionTreeClassifier()
            estimator.fit(X_bootstrap, y_bootstrap)
            self.estimators.append(estimator)

    def predict(self, X):
        predictions = np.array([estimator.predict(X) for estimator in self.estimators])
        return np.mean(predictions, axis=0).round().astype(int)

# Use the simple bagging implementation
bagging = SimpleBagging(n_estimators=10)
bagging.fit(X, y)
y_pred = bagging.predict(X)
accuracy = np.mean(y_pred == y)
print(f"Accuracy: {accuracy:.4f}")
          `}
          sklearnImplementation={`
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Create a synthetic dataset
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the BaggingClassifier
bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=10,
    max_samples=0.8,
    max_features=0.8,
    bootstrap=True,
    bootstrap_features=True,
    n_jobs=-1,
    random_state=42
)
bagging.fit(X_train, y_train)

# Make predictions and evaluate
y_pred = bagging.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Best practices:
# 1. Use a sufficient number of estimators (e.g., 10-100)
# 2. Experiment with different base estimators
# 3. Adjust max_samples and max_features to control diversity
# 4. Use cross-validation for more robust performance estimation
# 5. Consider using out-of-bag (OOB) samples for model evaluation
          `}
        />

        <Section
          title="Stacking"
          id="stacking"
          description="Stacking is an ensemble technique that combines multiple models by training a meta-model on their predictions to improve overall performance."
          concept={`
Stacking works as follows:
1. Train multiple base models (level-0 models) on the original dataset.
2. Use the predictions of these base models as features to train a meta-model (level-1 model).
3. The meta-model learns how to best combine the predictions of the base models.

Key benefits:
- Can capture complex patterns that single models might miss
- Often produces higher accuracy than individual models
- Allows combining diverse types of models
          `}
          svgType={'stacking'}
          simpleImplementation={`
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Create a synthetic dataset
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Correct Stacking implementation with CV
class StackingCV:
    def __init__(self, base_models, meta_model, n_splits=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_splits = n_splits
        self.base_models_ = base_models

    def fit(self, X, y):
        # Prepare cross-validation
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        meta_features = np.zeros((X.shape[0], len(self.base_models)))
        
        for i, (train_idx, valid_idx) in enumerate(kf.split(X)):
            X_train_fold, X_valid_fold = X[train_idx], X[valid_idx]
            y_train_fold, y_valid_fold = y[train_idx], y[valid_idx]
            
            # Train each base model on the training fold and predict on the validation fold
            for j, model in enumerate(self.base_models_):
                model[i].fit(X_train_fold, y_train_fold)
                meta_features[valid_idx, j] = model[i].predict(X_valid_fold)
        
        # Train the meta-model on out-of-fold predictions (meta-features)
        self.meta_model.fit(meta_features, y)

        # Train base models on the entire dataset for final predictions
        for model in self.base_models:
            model.fit(X, y)

    def predict(self, X):
        # Generate meta-features for test data
        meta_features = np.column_stack([model.predict(X) for model in self.base_models])
        return self.meta_model.predict(meta_features)

# Use the StackingCV implementation

base_models = [
    LogisticRegression(),
    DecisionTreeClassifier(),
    SVC(probability=True)
]
meta_model = LogisticRegression()

stacking = StackingCV(base_models, meta_model)
stacking.fit(X_train, y_train)
y_pred = stacking.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
          `}
          sklearnImplementation={`
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Create a synthetic dataset
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the StackingClassifier
estimators = [
    ('lr', LogisticRegression()),
    ('dt', DecisionTreeClassifier()),
    ('svc', SVC(probability=True))
]
stacking = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(),
    cv=5,
    stack_method='predict_proba'
)
stacking.fit(X_train, y_train)

# Make predictions and evaluate
y_pred = stacking.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Best practices:
# 1. Use diverse base models to capture different aspects of the data
# 2. Experiment with different meta-models
# 3. Use cross-validation to prevent overfitting
# 4. Consider using feature selection or dimensionality reduction
# 5. Balance model complexity with computational resources
# 6. Evaluate the stacked model against individual base models
          `}
        />
      </Stack>
    </Container>
  );
};

const Section = ({ title, id, description, concept, svgType, simpleImplementation, sklearnImplementation }) => (
  <Stack spacing="md">
    <Title order={2} id={id}>{title}</Title>
    <Text>{description}</Text>
    <div style={{ width: '100%', maxWidth: '800px', margin: '0 auto' }}>
      {svgType === 'bagging' ? <BaggingSVG /> : <StackingSVG />}
    </div>
    <Title order={3}>Concept</Title>
    <Text style={{ whiteSpace: 'pre-line' }}>{concept}</Text>
    <Title order={3}>Simple Implementation</Title>
    <CodeBlock language="python" code={simpleImplementation} />
    <Title order={3}>Sklearn Implementation and Best Practices</Title>
    <CodeBlock language="python" code={sklearnImplementation} />
  </Stack>
);
export default EnsembleTechniques;