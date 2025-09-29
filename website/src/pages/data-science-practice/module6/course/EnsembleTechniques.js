import React from 'react';
import { Container, Title, Text, Stack, List, Flex, Image } from '@mantine/core';
import CodeBlock from 'components/CodeBlock';
import { InlineMath, BlockMath } from 'react-katex';
import 'katex/dist/katex.min.css';



const EnsembleTechniques = () => {
  return (
    <Container fluid>
      <Title order={1} mt="xl" mb="md">Ensemble Techniques</Title>
          <Flex direction="column" align="center" mt="md">
            <Image
              src="/assets/data-science-practice/module6/votbag.png"
              style={{ maxWidth: 'min(600px, 80vw)', height: 'auto' }}
              fluid
            />
          </Flex>
      <Stack spacing="xl">
        <div data-slide>
          <Title order={2}>Voting Ensemble</Title>
          <BlockMath math="\hat{y} = \text{mode}(h_1(x), h_2(x), ..., h_n(x))" />

          <Title order={3} mt="lg">Training Process</Title>
          <Text mb="md">
            Train multiple <strong>different algorithms</strong> independently on the <strong>same full dataset</strong>. Each model learns different patterns due to algorithmic differences and inductive biases.
          </Text>

          <CodeBlock language="python" code={`# Training: different algorithms on same data
from sklearn.ensemble import VotingClassifier
voting = VotingClassifier([('lr', LogisticRegression()),
                           ('rf', RandomForestClassifier()),
                           ('svm', SVC(probability=True))])
voting.fit(X_train, y_train)  # All models see same X_train`} />

          <Title order={3} mt="lg">Key Characteristics</Title>
          <List spacing="sm">
            <List.Item><strong>Diversity source</strong>: Different algorithms (SVM vs Trees vs Linear)</List.Item>
            <List.Item><strong>Data usage</strong>: All models see 100% of training data</List.Item>
            <List.Item><strong>Model types</strong>: Heterogeneous (different families)</List.Item>
            <List.Item><strong>Purpose</strong>: Leverage different algorithmic strengths</List.Item>
          </List>

          <Title order={3} mt="lg">Prediction Process</Title>
          <Text mb="md">
            Hard voting: majority vote among predictions. Soft voting: average predicted probabilities, then select class with highest average.
          </Text>

          <CodeBlock language="python" code={`# Hard voting: majority wins
predictions = voting.predict(X_test)  # [1, 0, 1] → 1

# Soft voting: average probabilities
voting_soft = VotingClassifier(estimators, voting='soft')
prob_avg = voting_soft.predict_proba(X_test)`} />
        </div>

        <div data-slide>
          <Title order={2}>Bagging (Bootstrap Aggregating)</Title>

          <Title order={3} mt="lg">Training Process</Title>
          <Text mb="md">
            Train the <strong>same algorithm</strong> multiple times on <strong>different bootstrap samples</strong>. Each sample created by sampling with replacement has ~63.2% unique instances. Each model sees different subset of data.
          </Text>

          <CodeBlock language="python" code={`# Training: same algorithm on different data samples
from sklearn.ensemble import BaggingClassifier
bagging = BaggingClassifier(base_estimator=DecisionTreeClassifier(),
                            n_estimators=100, max_samples=1.0)
bagging.fit(X_train, y_train)  # Creates 100 bootstrap samples`} />

          <Title order={3} mt="lg">Key Characteristics</Title>
          <List spacing="sm">
            <List.Item><strong>Diversity source</strong>: Different data samples (bootstrap)</List.Item>
            <List.Item><strong>Data usage</strong>: Each model sees ~63.2% unique instances</List.Item>
            <List.Item><strong>Model types</strong>: Homogeneous (same algorithm)</List.Item>
            <List.Item><strong>Purpose</strong>: Reduce variance, prevent overfitting</List.Item>
          </List>

          <Title order={3} mt="lg">Random Forest: Advanced Bagging</Title>
          <Text mb="md">
            <strong>Random Forest is a bagging algorithm</strong> that adds extra randomness: besides bootstrap sampling, it also randomly selects features at each split (<InlineMath math="\sqrt{p}" /> for classification, <InlineMath math="p/3" /> for regression).
          </Text>

          <CodeBlock language="python" code={`# Random Forest = Bagging + Feature Randomness
rf = RandomForestClassifier(n_estimators=100)  # Bagging built-in
# Each tree: different bootstrap sample + random features at splits`} />
        </div>

        <div data-slide>
          <Title order={2}>Stacking (Stacked Generalization)</Title>
          <Flex direction="column" align="center" mt="md">
            <Image
              src="/assets/data-science-practice/module6/stacking.jpg"
              style={{ maxWidth: 'min(600px, 80vw)', height: 'auto' }}
              fluid
            />
          </Flex>
          <Title order={3} mt="lg">Training Process - Two Levels</Title>
          <Text mb="md">
            Level 1: Train diverse base models on original data. Level 2: Use base model predictions as features to train meta-learner. Uses k-fold CV to avoid overfitting.
          </Text>

          <CodeBlock language="python" code={`# Level 1: Train base models with cross-validation
from sklearn.ensemble import StackingClassifier
base_models = [('rf', RandomForestClassifier()),
               ('gb', GradientBoostingClassifier())]
stacking = StackingClassifier(estimators=base_models,
                             final_estimator=LogisticRegression(),
                             cv=5)  # 5-fold for meta-features`} />

          <Title order={3} mt="lg">Meta-Feature Generation</Title>
          <Text mb="md">
            For each fold, train base models on other folds and predict current fold. This creates out-of-fold predictions preventing overfitting.
          </Text>

          <CodeBlock language="python" code={`# Generate meta-features via cross-validation
# For fold i: train on folds ≠ i, predict fold i
meta_features = cross_val_predict(base_model, X, y, cv=5)
# Shape: (n_samples, n_base_models)`} />

          <Title order={3} mt="lg">Prediction Process</Title>
          <Text mb="md">
            Base models predict on new data → predictions become meta-features → meta-learner makes final prediction using these features.
          </Text>

          <CodeBlock language="python" code={`# Two-stage prediction
base_predictions = np.column_stack([m.predict(X_test) for m in base_models])
final_prediction = meta_model.predict(base_predictions)`} />
        </div>

        <div data-slide>
          <Title order={2}>Blending</Title>
          <Flex direction="column" align="center" mt="md">
            <Image
              src="/assets/data-science-practice/module6/blending.jpg"
              style={{ maxWidth: 'min(600px, 80vw)', height: 'auto' }}
              fluid
            />
          </Flex>
          <Title order={3} mt="lg">Training Process</Title>
          <Text mb="md">
            Similar to stacking but simpler: split training data into blend set and training set. Train base models on training set, generate predictions on blend set, train meta-model on blend predictions.
          </Text>

          <CodeBlock language="python" code={`# Split data for blending
X_blend, X_train_blend = train_test_split(X_train, test_size=0.2)
# Train base models on X_train_blend
# Predict on X_blend to create meta-features
blend_features = np.column_stack([m.predict(X_blend) for m in models])`} />

          <Title order={3} mt="lg">Key Difference from Stacking</Title>
          <Text mb="md">
            Blending uses holdout validation (single split) while stacking uses k-fold CV. Blending is faster but uses less data for meta-model training.
          </Text>
        </div>


        <div data-slide>
          <Title order={2}>Ensemble Diversity Strategies</Title>

          <Title order={3} mt="lg">Data Diversity</Title>
          <List spacing="sm">
            <List.Item>Bagging: Bootstrap sampling creates different training sets</List.Item>
            <List.Item>Feature bagging: Random subsets of features (Random Forest)</List.Item>
            <List.Item>Boosting: Weighted sampling focuses on hard examples</List.Item>
          </List>

          <CodeBlock language="python" code={`# Combine multiple diversity strategies
ensemble = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(max_features='sqrt'),
    max_samples=0.8, max_features=0.8)  # Sample both rows and columns`} />

          <Title order={3} mt="lg">Algorithm Diversity</Title>
          <Text mb="md">
            Combine models with different inductive biases: linear models, trees, neural networks, instance-based methods.
          </Text>

          <CodeBlock language="python" code={`# Diverse algorithms in voting ensemble
diverse_ensemble = VotingClassifier([
    ('linear', LogisticRegression()),
    ('tree', RandomForestClassifier()),
    ('neighbor', KNeighborsClassifier())])`} />

          <Title order={3} mt="lg">Hyperparameter Diversity</Title>
          <Text mb="md">
            Same algorithm with different hyperparameters creates diverse models.
          </Text>

          <CodeBlock language="python" code={`# Multiple configurations of same algorithm
models = [RandomForestClassifier(max_depth=d, min_samples_split=s)
          for d in [5, 10, None] for s in [2, 10, 20]]`} />
        </div>
      </Stack>
    </Container>
  );
};

export default EnsembleTechniques;