import React from 'react';
import { Container, Text, Title, Stack, List, Badge } from '@mantine/core';
import { Calculator } from 'lucide-react';
import { IconStar } from '@tabler/icons-react';
import DataInteractionPanel from 'components/DataInteractionPanel';

const Exercise2 = () => {
  const trainDataUrl = process.env.PUBLIC_URL + "/modules/data-science-practice/module8/exercise/train.csv";
  const testDataUrl = process.env.PUBLIC_URL + "/modules/data-science-practice/module8/exercise/test.csv";
  const notebookUrl = process.env.PUBLIC_URL + "/modules/data-science-practice/module8/exercise/module8_exercise2.ipynb";
  const notebookHtmlUrl = process.env.PUBLIC_URL + "/modules/data-science-practice/module8/exercise/module8_exercise2.html";
  const notebookColabUrl = process.env.PUBLIC_URL + "website/public/modules/data-science-practice/module8/exercise/module8_exercise2.ipynb";

  const metadata = {
    description: "A mathematical problem-solving dataset containing diverse math problems across different categories with numerical solutions.",
    source: "Custom Math Problem Dataset",
    target: "Numerical solutions (accuracy with 2 decimal precision tolerance)",
    listData: [
      { name: "train.csv", description: "900 math problems with solutions (id, problem, solution, category columns)" },
      { name: "test.csv", description: "100 test problems without solutions (id, problem, category columns)", isTarget: true }
    ],
  };

  return (
    <>
      <Container fluid>
        <Stack spacing="xl" className="p-6">
          <div className="flex items-center gap-3">
            <Calculator size={32} className="text-blue-600" />
            <Title order={1}>Exercise 2: Mathematical Problem Solving with LLMs</Title>
            <Badge color="yellow" size="lg" leftSection={<IconStar size={14} />}>
              Marked Exercise
            </Badge>
          </div>

          <Text size="md" mb="md">
            Apply LLMs to solve mathematical reasoning tasks. Test different pre-trained models with various prompting
            strategies and optionally fine-tune with LoRA to improve performance on a diverse set of math problems.
          </Text>

          <Stack spacing="lg">
            <div>
              <Title order={2} mb="md">Objectives</Title>
              <List spacing="sm">
                <List.Item>Evaluate LLMs on mathematical reasoning tasks</List.Item>
                <List.Item>Design and test effective prompt engineering strategies</List.Item>
                <List.Item>Implement chain-of-thought and few-shot prompting</List.Item>
                <List.Item>Extract numerical answers from text outputs</List.Item>
                <List.Item>Optionally implement LoRA fine-tuning</List.Item>
                <List.Item>Achieve 70% accuracy with 2 decimal precision tolerance</List.Item>
              </List>
            </div>

            <div>
              <Title order={2} mb="md">Dataset</Title>
              <Text size="md" mb="sm">
                Custom Math Problem Dataset (1000 problems)
              </Text>
              <Text size="sm" c="dimmed" mb="sm">
                Generated using scripts/generate_math_dataset.py
              </Text>
              <List spacing="sm">
                <List.Item><strong>Training Set:</strong> 900 problems with solutions</List.Item>
                <List.Item><strong>Test Set:</strong> 100 problems for evaluation</List.Item>
                <List.Item><strong>Categories:</strong> Arithmetic, Algebra, Geometry, Percentages, Fractions, Word Problems</List.Item>
                <List.Item><strong>Solution Format:</strong> Numeric values (integers or decimals with 2 decimal precision)</List.Item>
              </List>
            </div>

            <div>
              <Title order={2} mb="md">Approach Options</Title>

              <Title order={3} mb="sm">Option A: Prompt Engineering (Recommended)</Title>
              <List spacing="sm" mb="md">
                <List.Item>Simple prompts: Direct problem statement</List.Item>
                <List.Item>Instruction-based: Add explicit solving instructions</List.Item>
                <List.Item>Chain-of-thought: Guide step-by-step reasoning</List.Item>
                <List.Item>Few-shot learning: Include training examples</List.Item>
                <List.Item>Compare effectiveness across strategies</List.Item>
              </List>

              <Title order={3} mb="sm">Option B: Fine-Tuning with LoRA (Advanced)</Title>
              <List spacing="sm">
                <List.Item>Parameter-efficient fine-tuning approach</List.Item>
                <List.Item>Adapt pre-trained models to math domain</List.Item>
                <List.Item>Configure LoRA hyperparameters (rank, alpha, dropout)</List.Item>
                <List.Item>Train and evaluate on math problems</List.Item>
                <List.Item>Compare with prompting approaches</List.Item>
              </List>
            </div>

            <div>
              <Title order={2} mb="md">Key Concepts</Title>
              <List spacing="sm">
                <List.Item>Mathematical reasoning in LLMs</List.Item>
                <List.Item>Prompt engineering techniques</List.Item>
                <List.Item>Zero-shot vs. few-shot learning</List.Item>
                <List.Item>Chain-of-thought prompting</List.Item>
                <List.Item>Regular expressions for number extraction</List.Item>
                <List.Item>Model evaluation with tolerance metrics</List.Item>
                <List.Item>LoRA and parameter-efficient fine-tuning</List.Item>
              </List>
            </div>

            <div>
              <Title order={2} mb="md">Evaluation Metric</Title>
              <Text size="md" mb="sm">
                <strong>Accuracy with 2 Decimal Precision Tolerance</strong>
              </Text>
              <List spacing="sm">
                <List.Item>Predictions are rounded to 2 decimal places</List.Item>
                <List.Item>Compared against ground truth solutions</List.Item>
                <List.Item>Target: 70% accuracy on test set (100 problems)</List.Item>
                <List.Item>Non-numeric predictions count as incorrect</List.Item>
              </List>
            </div>

            <div>
              <Title order={2} mb="md">Deliverables</Title>
              <List spacing="sm">
                <List.Item>Completed Jupyter notebook with implementation</List.Item>
                <List.Item>submission.csv with predictions (id, solution columns)</List.Item>
                <List.Item>Comparison of different prompting strategies</List.Item>
                <List.Item>Analysis of model performance by problem category</List.Item>
                <List.Item>Error analysis and improvement strategies</List.Item>
                <List.Item>Written answers to reflection questions</List.Item>
              </List>
            </div>

            <div>
              <Title order={2} mb="md">Suggested Models</Title>
              <List spacing="sm">
                <List.Item><strong>GPT-2:</strong> Fast baseline, limited math reasoning</List.Item>
                <List.Item><strong>TinyLlama-1.1B:</strong> Good balance of speed and capability</List.Item>
                <List.Item><strong>Phi-2:</strong> Better reasoning, requires more memory</List.Item>
                <List.Item>Compare multiple models to understand trade-offs</List.Item>
              </List>
            </div>
          </Stack>

          <DataInteractionPanel
            trainDataUrl={trainDataUrl}
            testDataUrl={testDataUrl}
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
