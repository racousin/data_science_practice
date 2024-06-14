import React from "react";
import { Container } from "react-bootstrap";
import DataInteractionPanel from "components/DataInteractionPanel";

const ExploratoryDataAnalysis = () => {
  const trainDataUrl =
    process.env.PUBLIC_URL + "/modules/module3/course/train.csv";
  const testDataUrl =
    process.env.PUBLIC_URL + "/modules/module3/course/test.csv";
  const notebookUrl =
    process.env.PUBLIC_URL + "/modules/module3/course/data_exploration.ipynb";
  const notebookHtmlUrl =
    process.env.PUBLIC_URL + "/modules/module3/course/data_exploration.html";

  return (
    <Container fluid>
      <h1 className="my-4">Exploratory Data Analysis</h1>
      <p>
        In this section, you will learn about exploratory data analysis and how
        to perform it using Python and Jupyter Notebooks.
      </p>
      <DataInteractionPanel
        trainDataUrl={trainDataUrl}
        testDataUrl={testDataUrl}
        notebookUrl={notebookUrl}
        notebookHtmlUrl={notebookHtmlUrl}
      />
    </Container>
  );
};

export default ExploratoryDataAnalysis;
