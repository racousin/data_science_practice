import React from "react";
import { Container, Grid } from '@mantine/core';
import DataInteractionPanel from "components/DataInteractionPanel";
import CodeBlock from "components/CodeBlock";
const Exercise0 = () => {
  const trainDataUrl = process.env.PUBLIC_URL + "/modules/module3/exercise/module3_exercise_train.csv";
  const testDataUrl = process.env.PUBLIC_URL + "/modules/module3/exercise/module3_exercise_test.csv";
  const notebookUrl = process.env.PUBLIC_URL + "/modules/module3/exercise/module3_exercise0.ipynb";
  const notebookHtmlUrl = process.env.PUBLIC_URL + "/modules/module3/exercise/module3_exercise0.html";
  const notebookColabUrl = process.env.PUBLIC_URL + "website/public/modules/module3/exercise/module3_exercise0.ipynb";
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
      <h1 className="my-4">Exercise 0: Exploratory Data Analysis (EDA)</h1>
      <p>
        In this exercise, you will perform exploratory data analysis (EDA) to understand the underlying patterns and relationships within the data, with a specific emphasis on the visual and graphical representation of these elements.
      </p>
      <Grid>
        <Grid.Col>
          <h2>Overview</h2>
          <ul>
            <li>Load the dataset and explore its structure and statistics.</li>
            <li>Create visualizations to understand the data distributions and relationships.</li>
            <li>Identify potential correlations between features and the target variable (SalePrice).</li>
            <li>Investigate any outliers or anomalies in the data.</li>
            <li>Summarize your findings and insights from the EDA process.</li>
          </ul>
        </Grid.Col>
      </Grid>
      <Grid>
        <Grid.Col>
          <h2>Expected Output</h2>
          <p>
            You are expected to submit a Jupyter Notebook named <code>exercise0.ipynb</code> containing your exploratory data analysis. The notebook should include:
          </p>
        </Grid.Col>
      </Grid>
      <Grid>
        <Grid.Col>
          <h2>Review PR Validation</h2>
          The reviewer will assess:
              <ul>
                <li>Thoroughness of the analysis</li>
                <li>Quality and relevance of visualizations</li>
                <li>Clarity of explanations and insights</li>
                <li>Identification of important patterns or relationships in the data</li>
                <li>Proper handling and discussion of data quality issues</li>
              </ul>
          <p>
            Remember, the goal of this exercise is to gain a deep understanding of the dataset and to communicate your findings effectively through visual and written means. The review process is designed to help you improve your data analysis and presentation skills.
          </p>
        </Grid.Col>
      </Grid>
      <Grid>
        <DataInteractionPanel
          trainDataUrl={trainDataUrl}
          testDataUrl={testDataUrl}
          notebookUrl={notebookUrl}
          notebookHtmlUrl={notebookHtmlUrl}
          notebookColabUrl={notebookColabUrl}
          metadata={metadata}
        />
      </Grid>
    </Container>
  );
};
export default Exercise0;