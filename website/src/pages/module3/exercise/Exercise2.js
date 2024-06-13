import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";

const Exercise2 = () => {
  return (
    <Container fluid>
      <h1 className="my-4">Exercise 2: Generating Predictions</h1>
      <p>
        In this exercise, you will generate predictions for a dataset and save
        them to a CSV file. Your predictions will be evaluated using the Mean
        Absolute Error (MAE) metric, and the error threshold will be 12000.
      </p>
      <Row>
        <Col>
          <h2>Instructions</h2>
          <ol>
            <li>
              Load your dataset (you can use any dataset you have or generate a
              simple one for this exercise).
            </li>
            <li>
              Generate predictions for your dataset. Ensure your predictions are
              reasonable and formatted correctly.
            </li>
            <li>
              Create a CSV file named <code>predictions.csv</code> with two
              columns:
              <ul>
                <li>
                  <code>id</code>: The identifier for each prediction.
                </li>
                <li>
                  <code>SalePrice</code>: The predicted values.
                </li>
              </ul>
            </li>
            <CodeBlock
              code={`id,SalePrice\n1,200000\n2,250000\n3,300000\n...`}
            />
            <li>
              Save the <code>predictions.csv</code> file in the{" "}
              <code>module3</code> directory under your username folder.
            </li>
            <li>
              Ensure your predictions file is in the correct format and contains
              the required columns.
            </li>
          </ol>
        </Col>
      </Row>
      <Row>
        <Col>
          <h2>Example Code</h2>
          <p>
            Here's a simple example to help you get started with generating
            predictions:
          </p>
          <CodeBlock
            code={`import pandas as pd

# Example dataset
data = {
    'id': [1, 2, 3],
    'feature1': [10, 20, 30],
    'feature2': [15, 25, 35]
}
df = pd.DataFrame(data)

# Generate predictions (this is just a simple example, replace with your model's predictions)
df['SalePrice'] = df['feature1'] * 10000

# Select required columns
predictions = df[['id', 'SalePrice']]

# Save to CSV
predictions.to_csv('module3/predictions.csv', index=False)
`}
          />
        </Col>
      </Row>
      <Row>
        <Col>
          <h2>Evaluation</h2>
          <p>
            Your predictions will be evaluated using the Mean Absolute Error
            (MAE) metric. The error threshold for this exercise is 12000. Ensure
            your predictions are accurate enough to meet this threshold.
          </p>
        </Col>
      </Row>
    </Container>
  );
};

export default Exercise2;
