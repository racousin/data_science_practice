import React from 'react';
import { Container, Title, Text, Stack, Group, Badge, Paper } from '@mantine/core';
import CodeBlock from "components/CodeBlock";
import DataInteractionPanel from "components/DataInteractionPanel";

const CaseStudy = () => {
  // URLs for data and notebook files
  const trainDataUrl = process.env.PUBLIC_URL + "/modules/data-science-practice/module8/course/module8_course_train.zip";
  const requirementsUrl = process.env.PUBLIC_URL + "/modules/data-science-practice/module8/course/module8_requirements.txt";
  const notebookUrl = process.env.PUBLIC_URL + "/modules/data-science-practice/module8/course/module8_course.ipynb";
  const notebookHtmlUrl = process.env.PUBLIC_URL + "/modules/data-science-practice/module8/course/module8_course.html";
  const notebookColabUrl = process.env.PUBLIC_URL + "website/public/modules/data-science-practice/module8/course/module8_course.ipynb";


  // Dataset metadata
  const metadata = {
    description: "The MNIST dataset consists of handwritten digit images, perfect for demonstrating deep learning concepts.",
    source: "Modified MNIST Dataset",
    target: "Digit label (0-9)",
    listData: [
      { name: "pixel_values", description: "784 pixel values (28x28 image flattened)" },
      { name: "label", description: "Target digit (0-9)", isTarget: true }
    ],
  };

  return (
    <Container fluid>
      <Stack spacing="xl">



        <DataInteractionPanel
          trainDataUrl={trainDataUrl}
          notebookUrl={notebookUrl}
          notebookHtmlUrl={notebookHtmlUrl}
          notebookColabUrl={notebookColabUrl}
          requirementsUrl={requirementsUrl}
          metadata={metadata}
        />
      </Stack>
    </Container>
  );
};

export default CaseStudy;