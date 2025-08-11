import React from "react";
import { Row, Col } from 'react-bootstrap';
import { Container, Grid } from '@mantine/core';
import DataInteractionPanel from "components/DataInteractionPanel";
import CodeBlock from "components/CodeBlock";
const Exercise1 = () => {
  const trainDataUrl = process.env.PUBLIC_URL + "/modules/module3/exercise/module3_exercise_train.csv";
  const testDataUrl = process.env.PUBLIC_URL + "/modules/module3/exercise/module3_exercise_test.csv";
  const notebookUrl = process.env.PUBLIC_URL + "/modules/module3/exercise/module3_exercise1.ipynb";
  const notebookHtmlUrl = process.env.PUBLIC_URL + "/modules/module3/exercise/module3_exercise1.html";
  const notebookColabUrl = process.env.PUBLIC_URL + "website/public/modules/module3/exercise/module3_exercise1.ipynb";
  const metadata = {
    description: "This dataset includes various property metrics crucial for analyzing real estate sales, features, and pricing.",
    source: "Real Estate Transaction Records",
    target: "SalePrice",
    listData: [
      { name: "BedroomAbvGr", description: "Bedrooms above grade (does NOT include basement bedrooms)" },
      { name: "KitchenAbvGr", description: "Kitchens above grade" },
      { name: "TotRmsAbvGrd", description: "Total rooms above grade (does not include bathrooms)" },
      { name: "Fireplaces", description: "Number of fireplaces" },
      { name: "GarageYrBlt", description: "Year garage was built" },
      { name: "GarageCars", description: "Size of garage in car capacity" },
      { name: "GarageArea", description: "Size of garage in square feet" },
      { name: "WoodDeckSF", description: "Wood deck area in square feet" },
      { name: "OpenPorchSF", description: "Open porch area in square feet" },
      { name: "EnclosedPorch", description: "Enclosed porch area in square feet" },
      { name: "3SsnPorch", description: "Three season porch area in square feet" },
      { name: "ScreenPorch", description: "Screen porch area in square feet" },
      { name: "PoolArea", description: "Pool area in square feet" },
      { name: "MiscVal", description: "Value of miscellaneous feature" },
      { name: "MoSold", description: "Month Sold (MM)" },
      { name: "YrSold", description: "Year Sold (YYYY)" },
      { name: "SalePrice", description: "Price of sale (target variable)" },
    ],
  };
  return (
    <Container fluid>
      <h1 className="my-4">Exercise 1: Baseline Model Prediction</h1>
      <p>
        In this exercise, you will develop a baseline model and make predictions on the test dataset.
      </p>
      <Row>
        <Col>
          <h2>Overview</h2>
          <ul>
            <li>Load the dataset and explore its structure and statistics.</li>
            <li>Prepare the data for modeling based on your EDA findings.</li>
            <li>Develop a simple baseline model to predict house prices.</li>
            <li>Evaluate the model's performance using appropriate metrics.</li>
            <li>Generate predictions for the test dataset.</li>
          </ul>
        </Col>
      </Row>
      <Row>
        <Col>
          <h2>Expected Output</h2>
          <p>
            You are expected to submit two files:
          </p>
          <ol>
            <li>A Jupyter Notebook named <code>exercise1.ipynb</code> containing your baseline model development and evaluation.</li>
            <li>A CSV file named <code>submission.csv</code> with your predictions.</li>
          </ol>
        </Col>
      </Row>
      <Row>
        <Col>
          <h2>Submission Requirements</h2>
          <ol>
            <li>
              A CSV file named <code>submission.csv</code> with two columns:
              <ul>
                <li><code>id</code>: The identifier for each prediction.</li>
                <li><code>SalePrice</code>: The predicted values.</li>
              </ul>
            </li>
            <CodeBlock
              code={`id,SalePrice\n1,200000\n2,250000\n3,300000\n...`}
            />
            <li>
              Save both the <code>exercise1.ipynb</code> notebook and the <code>submission.csv</code> file in the <code>module3</code> directory under your username folder.
            </li>
          </ol>
        </Col>
      </Row>
      <Row>
        <Col>
          <h2>Evaluation</h2>
          <p>Your baseline model and predictions will be evaluated based on the following criteria:</p>
          <p>
            The error threshold for this exercise is a Mean Absolute Error (MAE) of 36000. Ensure your predictions are accurate enough to meet this threshold.
          </p>
        </Col>
      </Row>
      <Row>
        <DataInteractionPanel
          trainDataUrl={trainDataUrl}
          testDataUrl={testDataUrl}
          notebookUrl={notebookUrl}
          notebookHtmlUrl={notebookHtmlUrl}
          notebookColabUrl={notebookColabUrl}
          metadata={metadata}
        />
      </Row>
    </Container>
  );
};
export default Exercise1;