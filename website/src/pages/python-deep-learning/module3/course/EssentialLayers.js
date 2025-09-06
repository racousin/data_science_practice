import React from 'react';
import { Title, Text, Stack, Accordion, Container, List, Code } from '@mantine/core';
import CodeBlock from 'components/CodeBlock';
import 'katex/dist/katex.min.css';

// Import components
import Activation from './EssentialComponents/Activation';
import Dropout from './EssentialComponents/Dropout';
import CategoricalEmbeddings from './EssentialComponents/CategoricalEmbeddings';
import BatchNormalization from './EssentialComponents/BatchNormalization';
import ResidualConnections from './EssentialComponents/ResidualConnections';
import SkipConnections from './EssentialComponents/ResidualConnections';

const EssentialLayers = () => {
  return (
    <Container fluid>
      <Stack spacing="lg">
        <Title order={1}>Essential Layers</Title>
        
        
          <Title order={2}>Foundation Layers</Title>
          
          <Title order={3} mt="md">Linear/Fully Connected</Title>
          <Text>
            Core transformation layer performing matrix multiplication + bias.
          </Text>
          <CodeBlock language="python">{`nn.Linear(in_features=128, out_features=64)
# Performs: output = input @ weight.T + bias`}</CodeBlock>
          
          <Title order={3} mt="md">Convolution</Title>
          <Text>
            Spatial pattern detection in images. Local connectivity with shared weights.
          </Text>
          <CodeBlock language="python">{`nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
# Slides kernel across input, computing local feature maps`}</CodeBlock>
          
          <Title order={3} mt="md">Recurrent</Title>
          <Text>
            Process sequential data by maintaining hidden state across time steps.
          </Text>
          <CodeBlock language="python">{`nn.LSTM(input_size=128, hidden_size=256, num_layers=2)
# Process sequences with memory of previous inputs`}</CodeBlock>
          
          <Title order={3} mt="md">Attention</Title>
          <Text>
            Dynamic focus mechanism that relates different positions in sequences.
          </Text>
          <CodeBlock language="python">{`nn.MultiheadAttention(embed_dim=512, num_heads=8)
# Computes weighted importance between sequence elements`}</CodeBlock>
        

        
          <Title order={2}>Activation Functions</Title>
          
          <Activation/>
        
          <Title order={2}>Regularization Layers</Title>
          
          <Title order={3} mt="md">Dropout</Title>
          <Dropout/>
          <Title order={3} mt="md">Batch Normalization</Title>
<BatchNormalization/>
          <Title order={2}>Specialized Components</Title>
          
          <Title order={3} mt="md">Embeddings</Title>
<CategoricalEmbeddings/>
          
          <Title order={3} mt="md">Skip Connections</Title>
<SkipConnections/>
        
      </Stack>
    </Container>
  );
};
export default EssentialLayers;