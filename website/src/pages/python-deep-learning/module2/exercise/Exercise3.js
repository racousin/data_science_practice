import React from 'react';
import { Container, Text, Title, Stack, List } from '@mantine/core';
import { Cpu } from 'lucide-react';
import DataInteractionPanel from 'components/DataInteractionPanel';

const Exercise3 = () => {
  const notebookUrl = process.env.PUBLIC_URL + "/modules/python-deep-learning/module2/exercises/exercise3.ipynb";
  const notebookHtmlUrl = process.env.PUBLIC_URL + "/modules/python-deep-learning/module2/exercises/exercise3.html";
  const notebookColabUrl = process.env.PUBLIC_URL + "website/public/modules/python-deep-learning/module2/exercises/exercise3.ipynb";

  return (
    <>
      <Container fluid>
        <Stack spacing="xl" className="p-6">
          <div className="flex items-center gap-3">
            <Cpu size={32} className="text-blue-600" />
            <Title order={1} className="text-2xl font-bold">Exercise 2.3: Optimizer Implementation</Title>
          </div>

          <Stack spacing="lg">
            {/* Part 1 */}
            <div className="border rounded-lg p-6 bg-gray-50">
              <Title order={2} className="text-xl font-semibold mb-4">Part 1: From-Scratch Optimizer Implementation</Title>
              <Text className="text-gray-700 mb-4">
                Implement modern optimizers from mathematical foundations:
              </Text>
              <List spacing="sm" className="ml-6">
                <List.Item>Build Adam optimizer with bias correction</List.Item>
                <List.Item>Implement RMSprop with exponential moving averages</List.Item>
                <List.Item>Create SGD with Nesterov momentum</List.Item>
                <List.Item>Compare convergence behavior on different loss landscapes</List.Item>
              </List>
            </div>

            {/* Part 2 */}
            <div className="border rounded-lg p-6 bg-gray-50">
              <Title order={2} className="text-xl font-semibold mb-4">Part 2: Optimizer Performance Analysis</Title>
              <Text className="text-gray-700 mb-4">
                Benchmark optimizers on challenging optimization problems:
              </Text>
              <List spacing="sm" className="ml-6">
                <List.Item>Test on Rosenbrock function (banana-shaped valley)</List.Item>
                <List.Item>Evaluate performance on high-dimensional quadratics</List.Item>
                <List.Item>Analyze convergence speed vs. final accuracy trade-offs</List.Item>
                <List.Item>Study behavior near local minima and saddle points</List.Item>
              </List>
            </div>

            {/* Part 3 */}
            <div className="border rounded-lg p-6 bg-gray-50">
              <Title order={2} className="text-xl font-semibold mb-4">Part 3: Custom Learning Rate Schedules</Title>
              <Text className="text-gray-700 mb-4">
                Design and implement advanced learning rate scheduling:
              </Text>
              <List spacing="sm" className="ml-6">
                <List.Item>Implement cosine annealing with warm restarts</List.Item>
                <List.Item>Create adaptive schedules based on loss plateaus</List.Item>
                <List.Item>Design polynomial decay with different powers</List.Item>
                <List.Item>Combine warm-up with various decay strategies</List.Item>
              </List>
            </div>

            {/* Part 4 */}
            <div className="border rounded-lg p-6 bg-gray-50">
              <Title order={2} className="text-xl font-semibold mb-4">Part 4: Second-Order Optimization</Title>
              <Text className="text-gray-700 mb-4">
                Explore Newton's method and quasi-Newton approaches:
              </Text>
              <List spacing="sm" className="ml-6">
                <List.Item>Implement Newton's method with Hessian computation</List.Item>
                <List.Item>Build L-BFGS with limited memory updates</List.Item>
                <List.Item>Compare first-order vs. second-order convergence rates</List.Item>
                <List.Item>Analyze computational cost vs. convergence trade-offs</List.Item>
              </List>
            </div>

            {/* Part 5 */}
            <div className="border rounded-lg p-6 bg-gray-50">
              <Title order={2} className="text-xl font-semibold mb-4">Part 5: Hyperparameter Sensitivity Analysis</Title>
              <Text className="text-gray-700 mb-4">
                Study the impact of optimizer hyperparameters:
              </Text>
              <List spacing="sm" className="ml-6">
                <List.Item>Grid search over learning rates and momentum values</List.Item>
                <List.Item>Analyze Adam's beta parameters and epsilon sensitivity</List.Item>
                <List.Item>Study batch size effects on optimizer performance</List.Item>
                <List.Item>Investigate optimizer stability near convergence</List.Item>
              </List>
            </div>

            {/* Part 6 */}
            <div className="border rounded-lg p-6 bg-gray-50">
              <Title order={2} className="text-xl font-semibold mb-4">Part 6: Custom Optimizer Design</Title>
              <Text className="text-gray-700 mb-4">
                Design novel optimization algorithms:
              </Text>
              <List spacing="sm" className="ml-6">
                <List.Item>Create hybrid optimizers combining multiple methods</List.Item>
                <List.Item>Implement optimizers with gradient noise injection</List.Item>
                <List.Item>Design problem-specific optimization heuristics</List.Item>
                <List.Item>Test custom optimizers on neural network training</List.Item>
              </List>
            </div>
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