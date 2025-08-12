import React from 'react';
import { Container, Text, Title, List, Stack, Code } from '@mantine/core';
import DataInteractionPanel from 'components/DataInteractionPanel';
import CodeBlock from 'components/CodeBlock';

const Exercise1 = () => {
  const trainDataUrl = process.env.PUBLIC_URL + "/modules/module7/exercise/module7_exercise_train.zip";
  const testDataUrl = process.env.PUBLIC_URL + "/modules/module7/exercise/module7_exercise_test_features.csv";
  const notebookUrl = process.env.PUBLIC_URL + "/modules/module7/exercise/module7_exercise1.ipynb";
  const notebookHtmlUrl = process.env.PUBLIC_URL + "/modules/module7/exercise/module7_exercise1.html";
  const notebookColabUrl = process.env.PUBLIC_URL + "website/public/modules/module7/exercise/module7_exercise1.ipynb";
  
  const metadata = {
    description: "The Fashion MNIST dataset contains grayscale images of clothing items, serving as a more challenging drop-in replacement for MNIST.",
    source: "Fashion MNIST Dataset",
    target: "Clothing item label (0-9)",
    listData: [
      { name: "pixel_values", description: "784 pixel values (28x28 image flattened)" },
      { name: "label", description: "Target class (0: T-shirt/top, 1: Trouser, 2: Pullover, 3: Dress, 4: Coat, 5: Sandal, 6: Shirt, 7: Sneaker, 8: Bag, 9: Ankle boot)", isTarget: true }
    ],
  };

  return (
    <Container fluid className="p-4">
      <Stack spacing="lg">
        <Title order={1}>Exercise 1: Fashion MNIST Classification with PyTorch</Title>
        
        <Stack spacing="md">
          <Title order={2} id="overview">Overview</Title>
          <List>
            <List.Item>Explore and analyze the Fashion MNIST dataset structure</List.Item>
            <List.Item>Develop a PyTorch neural network for classification</List.Item>
            <List.Item>Generate predictions and evaluate model performance</List.Item>
          </List>

          <Title order={2} id="expected-output">Expected Output</Title>
          <Text>Submit a Jupyter Notebook (<Code>exercise1.ipynb</Code>) containing:</Text>
          <List>
            <List.Item>PyTorch model implementation and training</List.Item>
          </List>

          <Title order={2} id="evaluation">Evaluation</Title>
          <Text>Create a <Code>submission.csv</Code> with predictions:</Text>
          <CodeBlock
            code={`index,label\n0,4\n1,7\n2,2\n3,1\n...`}
          />
          <Text>Target accuracy threshold: 75%</Text>
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