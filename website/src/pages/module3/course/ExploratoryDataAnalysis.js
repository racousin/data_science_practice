import React from "react";
import { Container } from "react-bootstrap";

const ExploratoryDataAnalysis = () => {
  const notebookURL = "https://nbviewer.jupyter.org/urls/path_to_your_notebook";

  return (
    <Container>
      <h1 className="my-4">Exploratory Data Analysis</h1>
      <p>
        In this section, you will learn about exploratory data analysis and how
        to perform it using Python and Jupyter Notebooks.
      </p>
      <iframe src={notebookURL} width="100%" height="800px"></iframe>
    </Container>
  );
};

export default ExploratoryDataAnalysis;
