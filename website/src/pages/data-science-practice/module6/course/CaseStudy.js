import React from 'react';
import { Container } from '@mantine/core';
import CodeBlock from "components/CodeBlock";
import DataInteractionPanel from "components/DataInteractionPanel";

const CaseStudy = () => {
  const trainDataUrl = process.env.PUBLIC_URL + "/modules/data-science-practice/module6/course/module6_course_train.csv";
  const testDataUrl = process.env.PUBLIC_URL + "/modules/data-science-practice/module6/course/module6_course_test.csv";
  const requirementsUrl = process.env.PUBLIC_URL + "/modules/data-science-practice/module6/course/module6_requirements.txt";
  const notebookUrl = process.env.PUBLIC_URL + "/modules/data-science-practice/module6/course/module6_course.ipynb";
  const notebookHtmlUrl = process.env.PUBLIC_URL + "/modules/data-science-practice/module6/course/module6_course.html";
  const notebookColabUrl = process.env.PUBLIC_URL + "website/public/modules/data-science-practice/module6/course/module6_course.ipynb";

  const metadata = {
    description: "This dataset contains information about environmental conditions and well-being scores in a workplace setting.",
    source: "Workplace Environmental Monitoring System",
    target: "Score",
    listData: [
      { name: "Temperature", description: "Ambient temperature" },
      { name: "Humidity", description: "Relative humidity" },
      { name: "Humex", description: "Humidex (feels-like temperature)" },
      { name: "CO2", description: "Carbon dioxide levels" },
      { name: "Bright", description: "Brightness level" },
      { name: "weekday_0 to weekday_6", description: "Binary indicators for each day of the week" },
      { name: "hour_sine_wave", description: "Cyclical encoding of hour" },
      { name: "Score", description: "Well-being score (0-4)", isTarget: true },
    ],
  };

  return (
    <Container fluid>
      <h1 className="my-4">Case Study</h1>
      <DataInteractionPanel
        trainDataUrl={trainDataUrl}
        testDataUrl={testDataUrl}
        notebookUrl={notebookUrl}
        notebookHtmlUrl={notebookHtmlUrl}
        notebookColabUrl={notebookColabUrl}
        requirementsUrl={requirementsUrl}
        metadata={metadata}
      />
    </Container>
  );
};

export default CaseStudy;
