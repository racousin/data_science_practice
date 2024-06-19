import React from "react";
import { Container } from "react-bootstrap";
import DataInteractionPanel from "components/DataInteractionPanel";

const ExploratoryDataAnalysisAndModelBaseline = () => {
  const trainDataUrl =
    process.env.PUBLIC_URL + "/modules/module3/course/module3_course_train.csv";
  const testDataUrl =
    process.env.PUBLIC_URL + "/modules/module3/course/module3_course_test.csv";
  const requirementsUrl =
    process.env.PUBLIC_URL + "/modules/module3/course/module3_requirements.txt";
  const notebookUrl =
    process.env.PUBLIC_URL + "/modules/module3/course/module3_course.ipynb";
  const notebookHtmlUrl =
    process.env.PUBLIC_URL + "/modules/module3/course/module3_course.html";
  const notebookColabUrl =
    process.env.PUBLIC_URL +
    "website/public/modules/module3/course/module3_course.ipynb";

  return (
    <Container fluid>
      <h1 className="my-4">Exploratory Data Analysis</h1>
      <p>
        In this section, you will learn about exploratory data analysis and
        model baseline and how to perform it using Python and Jupyter Notebooks.
      </p>
      <DataInteractionPanel
        trainDataUrl={trainDataUrl}
        testDataUrl={testDataUrl}
        notebookUrl={notebookUrl}
        notebookHtmlUrl={notebookHtmlUrl}
        notebookColabUrl={notebookColabUrl}
        requirementsUrl={requirementsUrl}
      />
    </Container>
  );
};

export default ExploratoryDataAnalysisAndModelBaseline;
