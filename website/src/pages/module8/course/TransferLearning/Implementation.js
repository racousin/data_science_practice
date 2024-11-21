import React from 'react';
import { Container, Stack, Title, Text } from '@mantine/core';
import CodeBlock from 'components/CodeBlock';
import 'katex/dist/katex.min.css';
import { BlockMath } from 'react-katex';
import { Book, GitBranch, Settings, Brain } from 'lucide-react';

const Implementation = () => {
    return (
      <Stack spacing="md">
        <Text>
          Implementation involves careful consideration of architecture modification
          and layer freezing strategies.
        </Text>
  
        <CodeBlock
          language="python"
          code={`
  import torch.nn as nn
  
  def modify_model_head(model, num_classes, dropout_rate=0.2):
      """Modify the classification head of the model"""
      in_features = model.fc.in_features
      model.fc = nn.Sequential(
          nn.Dropout(dropout_rate),
          nn.Linear(in_features, 512),
          nn.ReLU(),
          nn.Dropout(dropout_rate),
          nn.Linear(512, num_classes)
      )
      return model
  
  def freeze_layers(model, num_layers_to_freeze):
      """Freeze specified number of layers from the beginning"""
      for i, (name, param) in enumerate(model.named_parameters()):
          if i < num_layers_to_freeze:
              param.requires_grad = False
      return model`}
        />
      </Stack>
    );
  };

export default Implementation