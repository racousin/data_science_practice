import React from 'react';
import { Container, Text, Title, Stack, List, Alert, Badge } from '@mantine/core';
import { Calculator } from 'lucide-react';
import { IconAlertCircle, IconStar } from '@tabler/icons-react';
import DataInteractionPanel from 'components/DataInteractionPanel';

const Exercise3 = () => {
  const notebookUrl = process.env.PUBLIC_URL + "/modules/data-science-practice/module8/exercise/module8_exercise3.ipynb";
  const notebookHtmlUrl = process.env.PUBLIC_URL + "/modules/data-science-practice/module8/exercise/module8_exercise3.html";
  const notebookColabUrl = process.env.PUBLIC_URL + "website/public/modules/data-science-practice/module8/exercise/module8_exercise3.ipynb";

  return (
    <>
      <Container fluid>
        <Stack spacing="xl" className="p-6">
          <div className="flex items-center gap-3">
            <Calculator size={32} className="text-blue-600" />
            <Title order={1}>Exercise 3: Mathematical Problem Solving</Title>
            <Badge color="yellow" size="lg" leftSection={<IconStar size={14} />}>
              Marked Exercise
            </Badge>
          </div>

          <Text size="md" mb="md">
            Apply LLMs to mathematical reasoning tasks. Choose your approach: prompting engineering or LoRA fine-tuning.
            You will evaluate model performance on mathematical problems and compare different strategies.
          </Text>

          <Stack spacing="lg">
            <div>
              <Title order={2} mb="md">Objectives</Title>
              <List spacing="sm">
                <List.Item>Apply LLMs to mathematical reasoning tasks</List.Item>
                <List.Item>Compare prompting strategies vs. fine-tuning approaches</List.Item>
                <List.Item>Implement LoRA fine-tuning for specialized tasks (optional)</List.Item>
                <List.Item>Evaluate model performance on mathematical problems</List.Item>
                <List.Item>Analyze trade-offs between different approaches</List.Item>
              </List>
            </div>

            <div>
              <Title order={2} mb="md">Dataset</Title>
              <Text size="md" mb="sm">
                <strong>Custom Math Dataset</strong>
              </Text>
              <List spacing="sm">
                <List.Item>Simple mathematical exercises (arithmetic, algebra, word problems)</List.Item>
                <List.Item>Generated problems with ground truth answers</List.Item>
                <List.Item>Structured format for evaluation</List.Item>
              </List>
            </div>

            <div>
              <Title order={2} mb="md">Choose Your Approach</Title>

              <Title order={3} mb="sm">Approach A: Prompting Engineering</Title>
              <List spacing="sm" mb="md">
                <List.Item>Design effective prompts for math problem solving</List.Item>
                <List.Item>Chain-of-thought prompting techniques</List.Item>
                <List.Item>Few-shot learning examples</List.Item>
                <List.Item>Prompt optimization strategies</List.Item>
              </List>

              <Title order={3} mb="sm">Approach B: Fine-Tuning with LoRA</Title>
              <List spacing="sm">
                <List.Item>Parameter-efficient fine-tuning with LoRA</List.Item>
                <List.Item>Dataset preparation for math problems</List.Item>
                <List.Item>Training loop implementation</List.Item>
                <List.Item>Hyperparameter optimization (rank, alpha, dropout)</List.Item>
                <List.Item>Model evaluation and comparison</List.Item>
              </List>
            </div>

            <div>
              <Title order={2} mb="md">Key Topics</Title>
              <List spacing="sm">
                <List.Item>Mathematical reasoning in LLMs</List.Item>
                <List.Item>Prompting techniques (zero-shot, few-shot, chain-of-thought)</List.Item>
                <List.Item>LoRA architecture and implementation</List.Item>
                <List.Item>PEFT (Parameter-Efficient Fine-Tuning) library</List.Item>
                <List.Item>Evaluation metrics for mathematical tasks</List.Item>
                <List.Item>Trade-offs: prompting vs. fine-tuning</List.Item>
              </List>
            </div>

            <div>
              <Title order={2} mb="md">Deliverables</Title>
              <List spacing="sm">
                <List.Item>Mathematical problem-solving system (prompting OR fine-tuning)</List.Item>
                <List.Item>Performance evaluation on test set</List.Item>
                <List.Item>Comparative analysis of chosen approach</List.Item>
                <List.Item>Error analysis and improvement strategies</List.Item>
                <List.Item>Report justifying approach selection</List.Item>
              </List>
            </div>

            <Alert icon={<IconAlertCircle />} color="yellow" mt="md">
              This is a marked exercise. Choose ONE approach (prompting or fine-tuning) and implement it thoroughly.
              Your choice should be justified based on resource constraints, task requirements, and expected performance.
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

export default Exercise3;
