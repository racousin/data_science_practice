import React from 'react';
import { Title, Text, Stack, List, Alert, Divider, Code } from '@mantine/core';
import CodeBlock from 'components/CodeBlock';
import 'katex/dist/katex.min.css';
import { InlineMath, BlockMath } from 'react-katex';

const SkipConnections = () => {
  return (
    <Stack spacing="md">
      <Text>
        Skip connections are a fundamental architectural pattern in neural networks that create shortcuts 
        to bypass one or more layers. They address key challenges in training deep networks by providing 
        alternative pathways for information and gradient flow. While residual connections (introduced by He et al. 
        in 2015) are the most well-known type of skip connection, the general principle has been applied in 
        various forms across different network architectures.
      </Text>
      <Text>
          <strong>Identity skip connections:</strong> The input is directly added to the output of a block 
          (<InlineMath>{`y = F(x) + x`}</InlineMath>). This is the core of residual connections.
        </Text>
      <Title order={3}>Core Principles of Skip Connections</Title>

      <List>
        <List.Item>They create alternative paths for information flow</List.Item>
        <List.Item>They help mitigate the vanishing gradient problem</List.Item>
        <List.Item>They make it easier for the network to learn identity mappings when needed</List.Item>
        <List.Item>They enable the training of much deeper architectures</List.Item>
      </List>


      <Title order={3}>Implementation with PyTorch</Title>
      <CodeBlock
        language="python"
        code={`
import torch
import torch.nn as nn

class AdditionSkipBlock(nn.Module):
    """Basic skip connection that adds the input to the output (residual-style)"""
    def __init__(self, input_dim, hidden_dim):
        super(AdditionSkipBlock, self).__init__()
        
        # Main path
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, input_dim)  # Output dim matches input for addition
        
    def forward(self, x):
        identity = x
        
        # Main path
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        
        # Add skip connection
        out = out + identity
        out = self.relu(out)
        
        return out`}/>

    </Stack>
  );
};

export default SkipConnections;