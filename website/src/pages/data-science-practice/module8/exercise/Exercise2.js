import React from 'react';
import { Container, Text, Title, Stack, List } from '@mantine/core';
import { Zap } from 'lucide-react';
import DataInteractionPanel from 'components/DataInteractionPanel';

const Exercise2 = () => {
  const notebookUrl = process.env.PUBLIC_URL + "/modules/data-science-practice/module8/exercise/module8_exercise2.ipynb";
  const notebookHtmlUrl = process.env.PUBLIC_URL + "/modules/data-science-practice/module8/exercise/module8_exercise2.html";
  const notebookColabUrl = process.env.PUBLIC_URL + "website/public/modules/data-science-practice/module8/exercise/module8_exercise2.ipynb";

  return (
    <>
      <Container fluid>
        <Stack spacing="xl" className="p-6">
          <div className="flex items-center gap-3"> 
            <Zap size={24} />
            <Title order={1} className="text-2xl font-bold">Exercise 2: Zero-Shot Learning with Transformers</Title>
          </div>

          <Stack spacing="lg">
            {/* Part 1 */}
            <div>
              <Title order={2} className="text-xl font-semibold mb-4">Part 1: Introduction to Transformers Library and Hugging Face</Title>
              <Text className="text-gray-700 mb-4">
                Learn to use pre-trained transformer models through the Hugging Face ecosystem:
              </Text>
              <List spacing="sm" className="ml-6">
                <List.Item>Navigating the Hugging Face Model Hub</List.Item>
                <List.Item>Loading pre-trained models with Transformers</List.Item>
                <List.Item>Understanding model architecture and tokenization</List.Item>
                <List.Item>Managing model weights and computational resources</List.Item>
              </List>
            </div>

            {/* Part 2 */}
            <div>
              <Title order={2} className="text-xl font-semibold mb-4">Part 2: Zero-Shot Learning</Title>
              <Text className="text-gray-700 mb-4">
                Explore various zero-shot capabilities of transformer models:
              </Text>
              
              <Title order={3} className="text-lg font-medium mb-2 ml-3">Zero-Shot Classification</Title>
              <List spacing="sm" className="ml-6 mb-3">
                <List.Item>Classifying text without task-specific training</List.Item>
                <List.Item>Customizing label sets for different domains</List.Item>
              </List>
              
              <Title order={3} className="text-lg font-medium mb-2 ml-3">Zero-Shot Question Answering</Title>
              <List spacing="sm" className="ml-6 mb-3">
                <List.Item>Extracting answers from context without fine-tuning</List.Item>
                <List.Item>Evaluating answer relevance and accuracy</List.Item>
              </List>
              
              <Title order={3} className="text-lg font-medium mb-2 ml-3">Zero-Shot Sentiment Analysis</Title>
              <List spacing="sm" className="ml-6 mb-3">
                <List.Item>Detecting sentiment in various text types</List.Item>
                <List.Item>Analyzing sentiment intensity and nuance</List.Item>
              </List>
              
              <Title order={3} className="text-lg font-medium mb-2 ml-3">Zero-Shot Translation</Title>
              <List spacing="sm" className="ml-6 mb-3">
                <List.Item>Cross-lingual capabilities of multilingual models</List.Item>
                <List.Item>Handling low-resource languages</List.Item>
              </List>
              
              <Title order={3} className="text-lg font-medium mb-2 ml-3">Zero-Shot Text Summarization</Title>
              <List spacing="sm" className="ml-6 mb-3">
                <List.Item>Generating abstractive summaries without training</List.Item>
                <List.Item>Controlling summary length and focus</List.Item>
              </List>
              
              <Title order={3} className="text-lg font-medium mb-2 ml-3">Zero-Shot Mathematical Reasoning</Title>
              <List spacing="sm" className="ml-6">
                <List.Item>Using specialized models for mathematical tasks</List.Item>
                <List.Item>Evaluating reasoning capabilities and limitations</List.Item>
              </List>
            </div>
          </Stack>
        </Stack>
        <DataInteractionPanel
          notebookUrl={notebookUrl}
          notebookHtmlUrl={notebookHtmlUrl}
          notebookColabUrl={notebookColabUrl}
          className="mt-6"
        />
      </Container>
    </>
  );
};

export default Exercise2;