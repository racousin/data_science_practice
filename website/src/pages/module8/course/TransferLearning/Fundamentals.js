import React from 'react';
import { Container, Stack, Title, Text } from '@mantine/core';
import CodeBlock from 'components/CodeBlock';
import 'katex/dist/katex.min.css';
import { BlockMath } from 'react-katex';
import { Book, GitBranch, Settings, Brain } from 'lucide-react';

const Fundamentals = () => {
    return (
      <Stack spacing="md">
        <Text>
          Transfer learning leverages pre-trained models to solve new tasks with limited data.
          The process involves:
        </Text>
        
  
        <CodeBlock
          language="python"
          code={`
  import torch
  import torchvision.models as models
  
  def load_pretrained_model(model_name='resnet50', num_classes=1000):
      """Load a pre-trained model with specified architecture"""
      if model_name == 'resnet50':
          model = models.resnet50(weights='ResNet50_Weights.IMAGENET1K_V2')
      elif model_name == 'vit_b_16':
          model = models.vit_b_16(weights='ViT_B_16_Weights.IMAGENET1K_V1')
      
      return model`}
        />
      </Stack>
    );
  };

export default Fundamentals