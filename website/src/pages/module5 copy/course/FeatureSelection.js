import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";

const FeatureSelection = () => {
  return (
    <Container fluid>
      <h1 className="my-4">Feature Selection Techniques</h1>
      <p>
        In this section, you will learn about various techniques for selecting
        the most important features for your model.
      </p>
      <Row>
        <Col>
          <h2>Filter Methods</h2>
          <p>
            Filter methods use statistical measures to evaluate the relevance of
            features. Examples include correlation and Chi-squared test.
          </p>
          <CodeBlock
            code={`# Example of correlation-based feature selection
import numpy as np
import pandas as pd

corr = df.corr()
corr_target = abs(corr["target"])
relevant_features = corr_target[corr_target>0.5]`}
          />
          <h2>Wrapper Methods</h2>
          <p>
            Wrapper methods use a machine learning algorithm to evaluate the
            relevance of features. Examples include recursive feature
            elimination.
          </p>
          <CodeBlock
            code={`# Example of recursive feature elimination
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
rfe = RFE(model, 3)
fit = rfe.fit(X, y)
print("Num Features: %d" % fit.n_features_)
print("Selected Features: %s" % fit.support_)
print("Feature Ranking: %s" % fit.ranking_)`}
          />
          <h2>Embedded Methods</h2>
          <p>
            Embedded methods perform feature selection as part of the model
            training process. Examples include feature importance from
            tree-based models.
          </p>
          <CodeBlock
            code={`# Example of feature importance from a random forest
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X, y)
importances = model.feature_importances_
indices = np.argsort(importances)
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()`}
          />
        </Col>
      </Row>
    </Container>
  );
};

export default FeatureSelection;
