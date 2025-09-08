import React from 'react';
import { Container, Text, Title, Stack, List } from '@mantine/core';
import { SplitSquareVertical } from 'lucide-react';
import DataInteractionPanel from 'components/DataInteractionPanel';

const Exercise0 = () => {
  const notebookUrl = process.env.PUBLIC_URL + "/modules/data-science-practice/module8/exercise/module8_exercise0.ipynb";
  const notebookHtmlUrl = process.env.PUBLIC_URL + "/modules/data-science-practice/module8/exercise/module8_exercise0.html";
  const notebookColabUrl = process.env.PUBLIC_URL + "website/public/modules/data-science-practice/module8/exercise/module8_exercise0.ipynb";

  return (
    <>
      <Container fluid>
        <Stack spacing="xl" className="p-6">
          <div className="flex items-center gap-3"> 
            <SplitSquareVertical size={24} />
            <Title order={1} className="text-2xl font-bold">Exercise 0: Tokenization Warmup</Title>
          </div>

          <Stack spacing="lg">
            {/* Tokenization Basics */}
            <div>
              <Title order={2} className="text-xl font-semibold mb-4">Part A: Tokenization Fundamentals</Title>
              <Text className="text-gray-700 mb-4">
                Understand the core concepts of text tokenization in NLP:
              </Text>
              <List spacing="sm" className="ml-6">
                <List.Item>Introduction to tokens and tokenization methods</List.Item>
                <List.Item>Word-level vs. subword-level tokenization</List.Item>
                <List.Item>Common algorithms (BPE, WordPiece, SentencePiece)</List.Item>
                <List.Item>Handling special tokens and out-of-vocabulary words</List.Item>
              </List>
            </div>

            {/* Implementation */}
            <div>
              <Title order={2} className="text-xl font-semibold mb-4">Part B: Implementing Tokenizers</Title>
              <Text className="text-gray-700 mb-4">
                Practice implementing and using tokenization techniques:
              </Text>
              <List spacing="sm" className="ml-6">
                <List.Item>Building a basic whitespace tokenizer</List.Item>
                <List.Item>Using pre-trained tokenizers from libraries</List.Item>
                <List.Item>Converting tokens to IDs and back to text</List.Item>
                <List.Item>Exploring token vocabularies and frequencies</List.Item>
              </List>
            </div>

            {/* Applications */}
            <div>
              <Title order={2} className="text-xl font-semibold mb-4">Part C: Tokenization in Practice</Title>
              <Text className="text-gray-700 mb-4">
                Apply tokenization to real-world NLP tasks:
              </Text>
              <List spacing="sm" className="ml-6">
                <List.Item>Preprocessing text for language models</List.Item>
                <List.Item>Analyzing token distribution in datasets</List.Item>
                <List.Item>Impact of tokenization choices on model performance</List.Item>
                <List.Item>Handling multilingual text and special characters</List.Item>
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

export default Exercise0;