import React from "react";
import { Container } from "react-bootstrap";

const ExploratoryDataAnalysis = () => {
  // Make sure the path here is correctly pointing to the public directory
  const iframeSrc = process.env.PUBLIC_URL + "/data_exploration.html";

  return (
    <Container fluid>
      <h1 className="my-4">Exploratory Data Analysis</h1>
      <p>
        In this section, you will learn about exploratory data analysis and how
        to perform it using Python and Jupyter Notebooks.
      </p>
      <iframe
        src={iframeSrc}
        style={{ width: "100%", height: "800px", border: "none" }}
      ></iframe>
    </Container>
  );
};

export default ExploratoryDataAnalysis;
