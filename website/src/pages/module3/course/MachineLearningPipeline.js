import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";

const MachineLearningPipeline = () => {
  return (
    <Container fluid>
      <h1 className="my-4">Machine Learning Pipeline</h1>
      <p>
        In this section, you will learn about machine learning pipelines and how
        to implement a simple one in Python.
      </p>
      <Row>
        <Col>
          <h2>Machine Learning Pipeline</h2>
          <p>
            A machine learning pipeline typically consists of the following
            steps:
          </p>
          <ol>
            <li>Data loading</li>
            <li>Data analysis</li>
            <li>Data preprocessing</li>
            <li>Model training</li>
            <li>Model serving</li>
            <li>Model evaluation</li>
          </ol>
          <h3>Example Pipeline</h3>
          <CodeBlock
            code={`print("Training pipeline:")
train_data = load_data("train.csv")
data_analysis(train_data)
ids, X_train, Y_train = data_preprocessing(train_data, 'SalePrice', 'Id')
model = train_model(X_train, Y_train)
Y_train_pred = serving_model(X_train, model)
eval_model(Y_train, Y_train_pred)

# Serving pipeline
print("\\nServing pipeline:")
test_data = load_data("test.csv")
data_analysis(test_data)
IDs_test, X_test, _ = data_preprocessing(test_data, id_col='Id')  # Assuming no Y_test available
Y_test_pred = serving_model(X_test)
serve_data(IDs_test, Y_test_pred, "predictions.csv")`}
          />
        </Col>
      </Row>
    </Container>
  );
};

export default MachineLearningPipeline;
