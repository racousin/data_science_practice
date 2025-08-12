import React from 'react';
import { Container, Text, Title, Stack } from '@mantine/core';
import DataInteractionPanel from 'components/DataInteractionPanel';
import CodeBlock from 'components/CodeBlock';

const Exercise0 = () => {
  const notebookUrl = process.env.PUBLIC_URL + "/modules/module7/exercise/module7_exercise0.ipynb";
  const notebookHtmlUrl = process.env.PUBLIC_URL + "/modules/module7/exercise/module7_exercise0.html";
  const notebookColabUrl = process.env.PUBLIC_URL + "website/public/modules/module7/exercise/module7_exercise0.ipynb";

  return (
    <>
    <Container fluid className="p-4">
      <Stack spacing="lg">
        <Title order={1}>Exercise 0: PyTorch Warmup</Title>
        
        <Stack spacing="md">
          <Title order={2} id="part-a">Part A: Function Definition and Gradients</Title>
          <Text>Learn to define functions and compute gradients using PyTorch's autograd functionality.</Text>
          
          <Title order={2} id="part-b">Part B: Learning to Approximate f(x,y,z)</Title>
          <Text>Build and train a neural network to approximate a 3-variable function.</Text>

          <Title order={2} id="part-c">Part C: Train with Checkpoint</Title>
          <Text>Implement model checkpointing to save and load training progress.</Text>

          <Title order={2} id="part-d">Part D: Device Management</Title>
          <Text>Practice managing model and data across CPU/GPU devices.</Text>
        </Stack>


      </Stack>
      
    </Container>
            <DataInteractionPanel
            notebookUrl={notebookUrl}
            notebookHtmlUrl={notebookHtmlUrl}
            notebookColabUrl={notebookColabUrl}
          />
          </>
  );
};

export default Exercise0;