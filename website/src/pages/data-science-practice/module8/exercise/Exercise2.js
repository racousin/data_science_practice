import React from 'react';
import { Container, Text, Title, Stack, List } from '@mantine/core';
import { Brain } from 'lucide-react';
import DataInteractionPanel from 'components/DataInteractionPanel';

const Exercise2 = () => {
  const notebookUrl = process.env.PUBLIC_URL + "/modules/data-science-practice/module8/exercise/module8_exercise2.ipynb";
  const notebookHtmlUrl = process.env.PUBLIC_URL + "/modules/data-science-practice/module8/exercise/module8_exercise2.html";
  const notebookColabUrl = process.env.PUBLIC_URL + "website/public/modules/data-science-practice/module8/exercise/module8_exercise2.ipynb";

  const metadata = {
    description: "Deep dive into Hugging Face ecosystem, zero-shot learning, and model embeddings.",
    source: "Guided Exercise",
    target: "Understanding zero-shot capabilities and embeddings extraction",
    listData: [
      { name: "module8_exercise2.ipynb", description: "Complete notebook with Part A (Zero-Shot Learning) and Part B (Embeddings)" }
    ],
  };

  return (
    <>
      <Container fluid>
        <Stack spacing="xl" className="p-6">
          <div className="flex items-center gap-3">
            <Brain size={32} className="text-blue-600" />
            <Title order={1}>Exercise 2: Zero-Shot Learning</Title>
          </div>

          <Text size="md" mb="md">
            Deep dive into Hugging Face ecosystem to leverage pre-trained models for zero-shot tasks and extract model embeddings.
          </Text>

          <Stack spacing="lg">
            <div>
              <Title order={2} mb="md">Objectives</Title>
              <List spacing="sm">
                <List.Item>Deep dive into Hugging Face ecosystem</List.Item>
                <List.Item>Leverage pre-trained models for zero-shot tasks</List.Item>
                <List.Item>Understand and extract model embeddings</List.Item>
              </List>
            </div>

            <div>
              <Title order={2} mb="md">Part A: Zero-Shot Learning</Title>
              <Text size="md" mb="sm">
                Explore the power of pre-trained models on tasks without fine-tuning.
              </Text>
              <List spacing="sm">
                <List.Item>Zero-shot classification without fine-tuning</List.Item>
                <List.Item>Zero-shot question answering</List.Item>
                <List.Item>Zero-shot sentiment analysis</List.Item>
                <List.Item>Zero-shot translation and summarization</List.Item>
                <List.Item>Mathematical reasoning capabilities</List.Item>
              </List>
            </div>

            <div>
              <Title order={2} mb="md">Part B: Embeddings</Title>
              <Text size="md" mb="sm">
                Learn to extract and work with contextual embeddings from transformers.
              </Text>
              <List spacing="sm">
                <List.Item>Loading and using pre-trained models</List.Item>
                <List.Item>Extracting contextual embeddings</List.Item>
                <List.Item>Sentence and token-level representations</List.Item>
                <List.Item>Similarity computation with embeddings</List.Item>
                <List.Item>Dimensionality reduction and visualization</List.Item>
              </List>
            </div>

            <div>
              <Title order={2} mb="md">Deliverables</Title>
              <List spacing="sm">
                <List.Item>Attention visualization dashboard</List.Item>
                <List.Item>Zero-shot task implementations</List.Item>
                <List.Item>Embedding extraction and analysis toolkit</List.Item>
                <List.Item>Comparative study of different pre-trained models</List.Item>
              </List>
            </div>
          </Stack>

          <DataInteractionPanel
            notebookUrl={notebookUrl}
            notebookHtmlUrl={notebookHtmlUrl}
            notebookColabUrl={notebookColabUrl}
            metadata={metadata}
            className="mt-6"
          />
        </Stack>
      </Container>
    </>
  );
};

export default Exercise2;
