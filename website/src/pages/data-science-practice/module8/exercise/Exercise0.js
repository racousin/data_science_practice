import React from 'react';
import { Container, Text, Title, Stack, List, Alert } from '@mantine/core';
import { FileText } from 'lucide-react';
import { IconAlertCircle } from '@tabler/icons-react';
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
            <FileText size={32} className="text-blue-600" />
            <Title order={1}>Exercise 0: Understanding Corpus, Tokens, and BPE Implementation</Title>
          </div>

          <Text size="md" mb="md">
            Build a foundational understanding of text processing for NLP by implementing a tokenizer from scratch.
            You will work with the TinyStories dataset to understand how text is transformed into tokens for language models.
          </Text>

          <Stack spacing="lg">
            <div>
              <Title order={2} mb="md">Objectives</Title>
              <List spacing="sm">
                <List.Item>Understand what a text corpus is and how it's structured</List.Item>
                <List.Item>Learn about tokens and different tokenization strategies</List.Item>
                <List.Item>Implement the Byte-Pair Encoding (BPE) algorithm from scratch</List.Item>
                <List.Item>Train a custom BPE tokenizer on real data</List.Item>
                <List.Item>Analyze tokenization characteristics and vocabulary distributions</List.Item>
              </List>
            </div>

            <div>
              <Title order={2} mb="md">Dataset</Title>
              <Text size="md" mb="sm">
                <strong>TinyStories</strong> (<code>roneneldan/TinyStories</code>)
              </Text>
              <List spacing="sm">
                <List.Item>Collection of short stories generated for language learning</List.Item>
                <List.Item>Simple vocabulary and sentence structures</List.Item>
                <List.Item>Ideal for understanding tokenization concepts</List.Item>
              </List>
            </div>

            <div>
              <Title order={2} mb="md">Key Topics</Title>
              <List spacing="sm">
                <List.Item>Corpus structure and text data collection</List.Item>
                <List.Item>Token types: characters, words, and subwords</List.Item>
                <List.Item>Tokenization strategies comparison</List.Item>
                <List.Item>BPE algorithm: merge rules and vocabulary construction</List.Item>
                <List.Item>Special tokens and handling unknown words</List.Item>
                <List.Item>Token frequency analysis</List.Item>
                <List.Item>Impact of vocabulary size on tokenization</List.Item>
              </List>
            </div>

            <div>
              <Title order={2} mb="md">Deliverables</Title>
              <List spacing="sm">
                <List.Item>BPE tokenizer implementation from scratch</List.Item>
                <List.Item>Trained tokenizer on TinyStories corpus subset</List.Item>
                <List.Item>Vocabulary analysis with frequency distributions</List.Item>
                <List.Item>Comparison of different vocabulary sizes</List.Item>
                <List.Item>Token statistics and insights</List.Item>
              </List>
            </div>

            <Alert icon={<IconAlertCircle />} color="blue" mt="md">
              This exercise provides the foundation for Exercise 1, where you'll use this tokenizer to build a GPT-like language model.
            </Alert>
          </Stack>

          <DataInteractionPanel
            notebookUrl={notebookUrl}
            notebookHtmlUrl={notebookHtmlUrl}
            notebookColabUrl={notebookColabUrl}
            className="mt-6"
          />
        </Stack>
      </Container>
    </>
  );
};

export default Exercise0;
