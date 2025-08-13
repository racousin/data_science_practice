import React from 'react';
import { Container, Text, Title, Stack, List } from '@mantine/core';
import { GitMerge } from 'lucide-react';
import DataInteractionPanel from 'components/DataInteractionPanel';

const Exercise3 = () => {
  const notebookUrl = process.env.PUBLIC_URL + "/modules/data-science-practice/module10/exercise/module10_exercise3.ipynb";
  const notebookHtmlUrl = process.env.PUBLIC_URL + "/modules/data-science-practice/module10/exercise/module10_exercise3.html";
  const notebookColabUrl = process.env.PUBLIC_URL + "website/public/modules/data-science-practice/module10/exercise/module10_exercise3.ipynb";

  return (
    <>
      <Container fluid>
        <Stack spacing="xl" className="p-6">
          <div className="flex items-center gap-3"> 
            <GitMerge size={24} />
            <Title order={1} className="text-2xl font-bold">Exercise 3: Fine-Tuning LLMs with LoRA</Title>
          </div>

          <Stack spacing="lg">
            {/* Part 1 */}
            <div>
              <Title order={2} className="text-xl font-semibold mb-4">Part 1: Introduction to LoRA (Low-Rank Adaptation)</Title>
              <Text className="text-gray-700 mb-4">
                Understand the core concepts behind efficient fine-tuning with LoRA:
              </Text>
              <List spacing="sm" className="ml-6">
                <List.Item>Principles of parameter-efficient fine-tuning</List.Item>
                <List.Item>How LoRA reduces trainable parameters</List.Item>
                <List.Item>Advantages over full fine-tuning methods</List.Item>
                <List.Item>Rank dimensionality and its effect on model adaptation</List.Item>
              </List>
            </div>

            {/* Part 2 */}
            <div>
              <Title order={2} className="text-xl font-semibold mb-4">Part 2: Setting Up LoRA Fine-Tuning Environment</Title>
              <Text className="text-gray-700 mb-4">
                Prepare your environment for efficient fine-tuning:
              </Text>
              <List spacing="sm" className="ml-6">
                <List.Item>Installing PEFT (Parameter-Efficient Fine-Tuning) library</List.Item>
                <List.Item>Setting up Hugging Face Transformers integration</List.Item>
                <List.Item>Configuring hardware acceleration (GPU/TPU)</List.Item>
                <List.Item>Managing memory constraints for large models</List.Item>
              </List>
            </div>

            {/* Part 3 */}
            <div>
              <Title order={2} className="text-xl font-semibold mb-4">Part 3: Preparing a Dataset for LoRA Fine-Tuning</Title>
              <Text className="text-gray-700 mb-4">
                Use the SST-2 dataset (Stanford Sentiment Treebank) for binary sentiment classification:
              </Text>
              <List spacing="sm" className="ml-6">
                <List.Item>Loading and exploring the SST-2 dataset structure</List.Item>
                <List.Item>Preprocessing text for sentiment classification</List.Item>
                <List.Item>Tokenization strategies for fine-tuning</List.Item>
                <List.Item>Creating train/validation splits for evaluation</List.Item>
              </List>
            </div>

            {/* Part 4 (Part 5 in your outline, but keeping sequential for consistency) */}
            <div>
              <Title order={2} className="text-xl font-semibold mb-4">Part 4: Training with LoRA</Title>
              <Text className="text-gray-700 mb-4">
                Implement and execute the LoRA fine-tuning process:
              </Text>
              <List spacing="sm" className="ml-6">
                <List.Item>Configuring LoRA hyperparameters (rank, alpha, dropout)</List.Item>
                <List.Item>Setting up the training loop and optimization</List.Item>
                <List.Item>Monitoring training progress and preventing overfitting</List.Item>
                <List.Item>Evaluating model performance on sentiment classification</List.Item>
                <List.Item>Merging and exporting LoRA weights for deployment</List.Item>
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

export default Exercise3;