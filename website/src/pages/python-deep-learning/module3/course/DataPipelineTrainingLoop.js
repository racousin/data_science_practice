import React from 'react';
import { Container, Title, Text, List, Code, Stack } from '@mantine/core';
import CodeBlock from 'components/CodeBlock';
import 'katex/dist/katex.min.css';
import { InlineMath, BlockMath } from 'react-katex';


import WeightInitialization from './EssentialComponents/WeightInitialization';
import Optimization from './EssentialComponents/Optimization';
import EarlyStopping from './EssentialComponents/EarlyStopping';
import CustomLoss from './EssentialComponents/CustomLoss';
import ReduceLROnPlateau from './EssentialComponents/ReduceLROnPlateau';

const DataPipelineTrainingLoop = () => {
  return (
    <Container fluid>
      <Stack spacing="lg">
        <Title order={1}>Data Pipeline & Training Loop</Title>
        
        
          <Title order={2}>Core Components</Title>
          
          <Title order={3} mt="md">nn.Module</Title>
          <Text>
            Base class for all neural network components. Handles parameters and gradients automatically.
          </Text>
          <WeightInitialization/>
          
          <Title order={3} mt="md">Optimizers</Title>
          <Optimization/>
          
          <Title order={3} mt="md">Loss Functions</Title>
          <CustomLoss/>
        

        
          <Title order={2}>Data Pipeline Components</Title>
          
          <Title order={3} mt="md">Dataset</Title>
          <Text>
            Container that defines how to access your data. Implement <Code>__len__</Code> and <Code>__getitem__</Code> methods.
          </Text>
          <CodeBlock language="python">{`class CustomDataset(torch.utils.data.Dataset):
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]`}</CodeBlock>
          
          <Title order={3} mt="md">DataLoader</Title>
          <Text>
            Handles batching, shuffling, and parallel loading efficiently.
          </Text>
          <CodeBlock language="python">{`dataloader = DataLoader(dataset, 
                        batch_size=32, 
                        shuffle=True, 
                        num_workers=4)`}</CodeBlock>
          
          <Title order={3} mt="md">Batch Size</Title>
          <Text>
            Number of samples processed together. Balance between memory usage and training speed.
          </Text>
        

        
          <Title order={2}>Data Splits</Title>
          
          <Title order={3} mt="md">Training Set</Title>
          <Text>
            Model learns patterns from this data. Typically 60-80% of total data.
          </Text>
          
          <Title order={3} mt="md">Validation Set</Title>
          <Text>
            Tune hyperparameters, monitor overfitting, make early stopping decisions. Typically 10-20% of data.
          </Text>
          
          <Title order={3} mt="md">Test Set</Title>
          <Text>
            Final performance evaluation. Never touched during training. Typically 10-20% of data.
          </Text>
        

        
          <Title order={2}>Training Concepts</Title>
          
          <Title order={3} mt="md">Epoch</Title>
          <Text>
            One complete pass through the entire training dataset.
          </Text>
          
          <Title order={3} mt="md">Training Loop</Title>
          <Text>
            Core training cycle: Forward pass → Calculate loss → Backward pass → Update weights
          </Text>
          <CodeBlock language="python">{`for epoch in range(num_epochs):
    for batch in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()`}</CodeBlock>
          
          <Title order={3} mt="md">Evaluation Mode</Title>
          <Text>
            Disable dropout and batch normalization updates during validation.
          </Text>
          <CodeBlock language="python">{`model.eval()
with torch.no_grad():
    # validation code`}</CodeBlock>
        

        
          <Title order={2}>Callbacks</Title>
          
          <Title order={3} mt="md">Early Stopping</Title>
          <EarlyStopping/>
          
          <Title order={3} mt="md">Learning Rate Scheduling</Title>
          <ReduceLROnPlateau/>
        
      </Stack>
    </Container>
  );
};

export default DataPipelineTrainingLoop;