import React from 'react';
import { Container, Text, Title, List, Stack, Code } from '@mantine/core';
import DataInteractionPanel from 'components/DataInteractionPanel';
import CodeBlock from 'components/CodeBlock';

const Exercise1 = () => {
  const trainDataUrl = process.env.PUBLIC_URL + "/modules/data-science-practice/module7/exercise/module7_exercise_train.zip";
  const testDataUrl = process.env.PUBLIC_URL + "/modules/data-science-practice/module7/exercise/module7_exercise_test_features.csv";
  const notebookUrl = process.env.PUBLIC_URL + "/modules/data-science-practice/module7/exercise/module7_exercise1.ipynb";
  const notebookHtmlUrl = process.env.PUBLIC_URL + "/modules/data-science-practice/module7/exercise/module7_exercise1.html";
  const notebookColabUrl = process.env.PUBLIC_URL + "website/public/modules/data-science-practice/module7/exercise/module7_exercise1.ipynb";
  
  const metadata = {
    description: "The CIFAR-10 dataset consists of 60,000 32x32 color images divided into 10 classes, with 6,000 images per class. There are 50,000 training images and 10,000 test images.",
    source: "CIFAR-10 Dataset",
    target: "Image class label (0-9)",
    listData: [
      { name: "image", description: "32x32x3 RGB image values (3072 values: 1024 for each color channel)" },
      { name: "label", description: "Target class (0: airplane, 1: automobile, 2: bird, 3: cat, 4: deer, 5: dog, 6: frog, 7: horse, 8: ship, 9: truck)", isTarget: true }
    ],
  };

  return (
    <Container fluid className="p-4">
      <Stack spacing="lg">
        <Title order={1}>Exercise 1: CIFAR-10 Image Classification with PyTorch</Title>
        
        <Stack spacing="md">
          <Title order={2} id="overview">Overview</Title>
          <List>
            <List.Item>Explore and analyze the CIFAR-10 dataset structure</List.Item>
            <List.Item>Develop a CNN model using PyTorch for multi-class image classification</List.Item>
            <List.Item>Generate predictions and evaluate model performance</List.Item>
          </List>

          <Title order={2} id="expected-output">Expected Output</Title>
          <Text>Submit a Jupyter Notebook (<Code>exercise1.ipynb</Code>) containing:</Text>
          <List>
            <List.Item>CNN architecture implementation and training pipeline</List.Item>
            <List.Item>Data augmentation and preprocessing steps (optional)</List.Item>
            <List.Item>Model evaluation and performance analysis</List.Item>
          </List>

          <Title order={2} id="evaluation">Evaluation</Title>
          <Text>Create a <Code>submission.csv</Code> with predictions:</Text>
          <CodeBlock
            code={`index,label\n0,4\n1,7\n2,2\n3,1\n...`}
          />
          <Text>Target accuracy threshold: 80% on the test set</Text>
        </Stack>

        <DataInteractionPanel
          trainDataUrl={trainDataUrl}
          testDataUrl={testDataUrl}
          notebookUrl={notebookUrl}
          notebookHtmlUrl={notebookHtmlUrl}
          notebookColabUrl={notebookColabUrl}
          metadata={metadata}
        />
      </Stack>
    </Container>
  );
};

export default Exercise1;