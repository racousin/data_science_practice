import React from 'react';
import { Container, Text, Title, Stack, List } from '@mantine/core';
import { Network } from 'lucide-react';
import DataInteractionPanel from 'components/DataInteractionPanel';

const Exercise4 = () => {
  const notebookUrl = process.env.PUBLIC_URL + "/modules/data-science-practice/module8/exercise/module8_exercise4.ipynb";
  const notebookHtmlUrl = process.env.PUBLIC_URL + "/modules/data-science-practice/module8/exercise/module8_exercise4.html";
  const notebookColabUrl = process.env.PUBLIC_URL + "website/public/modules/data-science-practice/module8/exercise/module8_exercise4.ipynb";

  return (
    <>
      <Container fluid>
        <Stack spacing="xl" className="p-6">
          <div className="flex items-center gap-3">
            <Network size={32} className="text-blue-600" />
            <Title order={1}>Exercise 4: Agentic Systems with LangGraph</Title>
          </div>

          <Text size="md" mb="md">
            Build agent systems using LangGraph. Learn to create reasoning loops, implement tool use,
            and manage complex multi-step workflows with the Qwen model.
          </Text>

          <Stack spacing="lg">
            <div>
              <Title order={2} mb="md">Objectives</Title>
              <List spacing="sm">
                <List.Item>Understand agentic AI architectures</List.Item>
                <List.Item>Build agent systems with LangGraph</List.Item>
                <List.Item>Implement tool use and function calling</List.Item>
                <List.Item>Create reasoning and planning loops</List.Item>
                <List.Item>Manage agent state and memory</List.Item>
              </List>
            </div>

            <div>
              <Title order={2} mb="md">Model</Title>
              <Text size="md" mb="sm">
                <strong>Qwen</strong> (suitable for Colab environment)
              </Text>
              <List spacing="sm">
                <List.Item>Efficient model size for free GPU resources</List.Item>
                <List.Item>Strong tool-use capabilities</List.Item>
                <List.Item>Good reasoning performance</List.Item>
              </List>
            </div>

            <div>
              <Title order={2} mb="md">Part A: Introduction to Agentic Systems</Title>
              <List spacing="sm">
                <List.Item>Agent architectures and paradigms</List.Item>
                <List.Item>Tool use and function calling</List.Item>
                <List.Item>Reasoning and planning loops</List.Item>
                <List.Item>Memory and state management</List.Item>
              </List>
            </div>

            <div>
              <Title order={2} mb="md">Part B: LangGraph Fundamentals</Title>
              <List spacing="sm">
                <List.Item>Graph-based agent workflows</List.Item>
                <List.Item>Node and edge definitions</List.Item>
                <List.Item>State management across the graph</List.Item>
                <List.Item>Conditional routing and branching</List.Item>
                <List.Item>Building simple agent architectures</List.Item>
              </List>
            </div>

            <div>
              <Title order={2} mb="md">Part C: Implementing Tools</Title>
              <List spacing="sm">
                <List.Item>Defining custom tools for agents</List.Item>
                <List.Item>Tool schemas and descriptions</List.Item>
                <List.Item>Tool execution and result handling</List.Item>
                <List.Item>Error handling and retry mechanisms</List.Item>
              </List>
            </div>

            <div>
              <Title order={2} mb="md">Part D: Building Practical Agents</Title>
              <List spacing="sm">
                <List.Item>Simple task-solving agents</List.Item>
                <List.Item>Multi-step reasoning workflows</List.Item>
                <List.Item>Tool selection and orchestration</List.Item>
                <List.Item>Agent evaluation and testing</List.Item>
              </List>
            </div>

            <div>
              <Title order={2} mb="md">Deliverables</Title>
              <List spacing="sm">
                <List.Item>Working agent system with LangGraph</List.Item>
                <List.Item>Custom tool implementations</List.Item>
                <List.Item>Multi-step task examples</List.Item>
                <List.Item>Agent workflow visualizations</List.Item>
                <List.Item>Performance analysis and improvements</List.Item>
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

export default Exercise4;
