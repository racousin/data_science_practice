import React from 'react';
import { Container, Title, Text, Stack, Group, Badge, Paper } from '@mantine/core';
import CodeBlock from "components/CodeBlock";
import DataInteractionPanel from "components/DataInteractionPanel";

const CaseStudy = () => {
  // URLs for data and notebook files
  const trainDataUrl = process.env.PUBLIC_URL + "/modules/deep-learning/course/mnist_train_sample.csv";
  const testDataUrl = process.env.PUBLIC_URL + "/modules/deep-learning/course/mnist_test_sample.csv";
  const requirementsUrl = process.env.PUBLIC_URL + "/modules/deep-learning/course/requirements.txt";
  const notebookUrl = process.env.PUBLIC_URL + "/modules/deep-learning/course/mnist_classification.ipynb";
  const notebookHtmlUrl = process.env.PUBLIC_URL + "/modules/deep-learning/course/mnist_classification.html";
  const notebookColabUrl = process.env.PUBLIC_URL + "website/public/modules/deep-learning/course/mnist_classification.ipynb";

  // Sample training metrics for visualization
  const trainingMetrics = [
    { epoch: 1, trainLoss: 2.3, valLoss: 2.4, trainAcc: 0.45, valAcc: 0.44 },
    { epoch: 2, trainLoss: 1.8, valLoss: 1.9, trainAcc: 0.65, valAcc: 0.63 },
    { epoch: 3, trainLoss: 1.4, valLoss: 1.5, trainAcc: 0.78, valAcc: 0.75 },
    { epoch: 4, trainLoss: 1.1, valLoss: 1.3, trainAcc: 0.85, valAcc: 0.82 },
    { epoch: 5, trainLoss: 0.9, valLoss: 1.2, trainAcc: 0.89, valAcc: 0.85 }
  ];

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
        <Title order={1} id="case-study">Deep Learning Case Study: Handwritten Digit Classification</Title>
        
        <Text>
          In this case study, we'll implement a complete deep learning pipeline for handwritten digit 
          classification using the MNIST dataset. We'll apply the concepts covered in previous sections
          and demonstrate best practices in model development, training, and evaluation.
        </Text>

        <Title order={2} id="case-objectives">Learning Objectives</Title>
        <Group>
          {[
            "Build a multi-layer neural network",
            "Implement proper data preprocessing",
            "Apply regularization techniques",
            "Perform hyperparameter tuning",
            "Evaluate model performance"
          ].map((objective, index) => (
            <Badge key={index} size="lg" variant="light" color="blue">
              {objective}
            </Badge>
          ))}
        </Group>

        <Title order={2} id="model-architecture">Model Architecture Overview</Title>
        <CodeBlock
          language="python"
          code={`
import torch
import torch.nn as nn

class MNISTClassifier(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        return self.network(x)

# Model initialization
model = MNISTClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)`}
        />


        <Title order={2} id="interactive-notebook">Interactive Notebook</Title>
        <Text mb="md">
          Explore the complete implementation in the interactive notebook below. The notebook includes
          detailed explanations, code comments, and visualizations to help you understand each step
          of the deep learning pipeline.
        </Text>

        <DataInteractionPanel
          trainDataUrl={trainDataUrl}
          testDataUrl={testDataUrl}
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