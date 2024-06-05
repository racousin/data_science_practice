import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";

const EnsembleMethods = () => {
  return (
    <Container fluid>
      <h1 className="my-4">Ensemble Methods</h1>
      <p>
        In this section, you will learn about ensemble methods, which are
        techniques for combining multiple models to improve performance.
      </p>
      <Row>
        <Col>
          <h2>Bagging and Boosting Techniques</h2>
          <p>
            Bagging (Bootstrap Aggregating) and boosting are two common ensemble
            techniques. Bagging involves training multiple models on different
            subsets of the data and combining their predictions. Boosting
            involves training multiple models sequentially, with each subsequent
            model focusing on the mistakes made by the previous models.
          </p>
          <h2>Random Forests and Gradient Boosting Machines (GBM)</h2>
          <p>
            Random forests are an ensemble method that combines multiple
            decision trees. GBM is a boosting method that combines multiple
            decision trees, with each subsequent tree focusing on the mistakes
            made by the previous trees.
          </p>
          <CodeBlock
            code={`# Example of a random forest
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)`}
          />
          <h2>Advanced Boosting Models (XGBoost, LightGBM, CatBoost)</h2>
          <p>
            XGBoost, LightGBM, and CatBoost are advanced boosting models that
            are designed to be fast, scalable, and accurate. They are
            particularly effective for tabular data and are widely used in data
            science competitions.
          </p>
          <CodeBlock
            code={`# Example of XGBoost
import xgboost as xgb

model = xgb.XGBClassifier(n_estimators=100)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)`}
          />
        </Col>
      </Row>
    </Container>
  );
};

export default EnsembleMethods;
