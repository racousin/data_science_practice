import React from 'react';
import { Container, Text, Title, List, Stack, Code } from '@mantine/core';
import DataInteractionPanel from 'components/DataInteractionPanel';
import CodeBlock from 'components/CodeBlock';

const Exercise1 = () => {
  const trainDataUrl = process.env.PUBLIC_URL + "/modules/data-science-practice/module7/exercise/module7_exercise1_train.zip";
  const testDataUrl = process.env.PUBLIC_URL + "/modules/data-science-practice/module7/exercise/X_test.pkl";
  const notebookUrl = process.env.PUBLIC_URL + "/modules/data-science-practice/module7/exercise/module7_exercise1.ipynb";
  const notebookHtmlUrl = process.env.PUBLIC_URL + "/modules/data-science-practice/module7/exercise/module7_exercise1.html";
  const notebookColabUrl = process.env.PUBLIC_URL + "website/public/modules/data-science-practice/module7/exercise/module7_exercise1.ipynb";
  
  const metadata = {
    description: "A boat detection dataset from satellite imagery containing images with annotated bounding boxes in YOLO format. Training data includes labels; test data is provided for prediction only.",
    source: "Kaggle Ship Detection Dataset",
    target: "Bounding box predictions (YOLO format)",
    listData: [
      { name: "X_train.pkl", description: "Training images as numpy arrays (list of image arrays)" },
      { name: "y_train.pkl", description: "Training labels in YOLO format: [class_id, x_center, y_center, width, height] (normalized 0-1)" },
      { name: "train_files.pkl", description: "Filenames corresponding to training images" },
      { name: "X_test.pkl", description: "Test images as numpy arrays (labels withheld for evaluation)", isTarget: true }
    ],
  };

  return (
    <Container fluid className="p-4">
      <Stack spacing="lg">
        <Title order={1}>Exercise 1: Boat Object Detection with YOLO</Title>

        <Stack spacing="md">
          <Title order={2} id="overview">Overview</Title>
          <List>
            <List.Item>Load and visualize the boat detection dataset with bounding boxes</List.Item>
            <List.Item>Fine-tune a YOLOv8 model for boat detection in satellite imagery</List.Item>
            <List.Item>Generate predictions on test set and evaluate performance</List.Item>
          </List>

          <Title order={2} id="expected-output">Expected Output</Title>
          <Text>Submit a Jupyter Notebook (<Code>exercise1.ipynb</Code>) containing:</Text>
          <List>
            <List.Item>YOLO model fine-tuning on the boat detection dataset</List.Item>
            <List.Item>Training configuration and hyperparameter tuning</List.Item>
            <List.Item>Visualization of predictions with bounding boxes</List.Item>
          </List>

          <Title order={2} id="evaluation">Evaluation</Title>
          <Text>Create a <Code>predictions.csv</Code> with bounding box predictions:</Text>
          <CodeBlock
            code={`image_id,box_idx,class_id,confidence,x_center,y_center,width,height\n0,0,0,0.95,0.5234,0.6123,0.1234,0.2345\n0,1,0,0.87,0.3456,0.4567,0.0987,0.1543\n1,0,0,0.92,0.7123,0.5432,0.1567,0.2234\n...`}
          />
          <Text>Target mAP50 threshold: 0.70 on the test set</Text>
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