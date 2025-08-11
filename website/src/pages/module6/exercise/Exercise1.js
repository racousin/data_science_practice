import React from "react";
import { Container, Title, Text, List, Alert, Code, Space } from '@mantine/core';
import { IconInfoCircle, IconBulb } from '@tabler/icons-react';
import DataInteractionPanel from "components/DataInteractionPanel";
import CodeBlock from "components/CodeBlock";

const Exercise1 = () => {
  const trainDataUrl = process.env.PUBLIC_URL + "/modules/module6/exercise/module6_exercise_train.csv";
  const testDataUrl = process.env.PUBLIC_URL + "/modules/module6/exercise/module6_exercise_test.csv";
  const notebookUrl = process.env.PUBLIC_URL + "/modules/module6/exercise/module6_exercise.ipynb";
  const notebookHtmlUrl = process.env.PUBLIC_URL + "/modules/module6/exercise/module6_exercise.html";
  const notebookColabUrl = process.env.PUBLIC_URL + "website/public/modules/module6/exercise/module6_exercise.ipynb";

  const metadata = {
    description: "This dataset represents market movements at 20-minute intervals throughout the trading day, with the goal of predicting the end-of-day return.",
    source: "Financial Market Data",
    target: "end_of_day_return",
    listData: [
      { name: "09:30:00 to 15:10:00", description: "Market movements at 20-minute intervals" },
      { name: "end_of_day_return", description: "The target variable representing the market movement at the end of the day" },
    ],
  };

  return (
    <Container fluid>
      <Title order={1} mb="md">Exercise 1: Fine-tuning Models for Market Movement Prediction</Title>
      <Text>
        In this exercise, you will work with a dataset of market movements to predict the end-of-day return. Your task is to fine-tune various models, experiment with different target representations, and evaluate their performance.
      </Text>

      <Title order={2} mt="xl" mb="md" id="overview">Overview</Title>
      <List>
        <List.Item>Explore the preprocessed dataset and understand its structure.</List.Item>
        <List.Item>Implement and fine-tune multiple machine learning models.</List.Item>
        <List.Item>Experiment with different target representations (regression vs. classification).</List.Item>
        <List.Item>Evaluate models using appropriate metrics, focusing on weighted accuracy.</List.Item>
        <List.Item>Optimize model performance to achieve a weighted accuracy above 0.53.</List.Item>
      </List>

      <Alert icon={<IconInfoCircle size="1rem" />} title="Important Note" color="blue" mt="md">
        The data preprocessing has already been performed. Focus on model selection, hyperparameter tuning, and performance evaluation.
      </Alert>


      <Title order={2} mt="xl" mb="md" id="submission-requirements">Submission Requirements</Title>
      <List>
        <List.Item>
          A Jupyter Notebook named <Code>exercise1.ipynb</Code> containing your data exploration, model implementation, fine-tuning process, and evaluation.
        </List.Item>
        <List.Item>
          A CSV file named <Code>submission.csv</Code> with your predictions with two columns:
          <List withPadding>
            <List.Item><Code>index</Code></List.Item>
            <List.Item><Code>end_of_day_return</Code></List.Item>
          </List>
        </List.Item>
        <List.Item>
          Save both files in the <Code>module6</Code> directory under your username folder.
        </List.Item>
      </List>
      <List>

      </List>
      <Space h="xl" />

      <DataInteractionPanel
        trainDataUrl={trainDataUrl}
        testDataUrl={testDataUrl}
        notebookUrl={notebookUrl}
        notebookHtmlUrl={notebookHtmlUrl}
        notebookColabUrl={notebookColabUrl}
        metadata={metadata}
      />
    </Container>
  );
};

export default Exercise1;