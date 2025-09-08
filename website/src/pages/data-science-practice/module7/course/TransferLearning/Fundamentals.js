import React from 'react';
import { Stack, Title, Text, Alert, Paper } from '@mantine/core';
import CodeBlock from 'components/CodeBlock';
import { AlertTriangle, Brain } from 'lucide-react';

const Fundamentals = () => {
  return (
    <Stack spacing="lg">
      <Text size="lg">
        Transfer learning leverages pre-trained models to solve new tasks with limited data. 
        This approach is particularly powerful when working with limited datasets or computational resources.
      </Text>

      <Title order={3} mt="md">Loading Pre-trained Models</Title>
      <Text>
        PyTorch provides various pre-trained models through torchvision. Each model has specific input 
        requirements and feature dimensions. Let's explore loading different architectures:
      </Text>

      <CodeBlock
        language="python"
        code={`
import torch
import torchvision.models as models
from torchvision.models import ResNet50_Weights, ViT_B_16_Weights

# ResNet-50
# Input: (batch_size, 3, 224, 224)
# Feature dimension: 2048
resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
print(f"ResNet-50 last layer: {resnet.fc}")  # Linear(in_features=2048, out_features=1000)

# Vision Transformer (ViT)
# Input: (batch_size, 3, 224, 224)
# Feature dimension: 768
vit = models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
print(f"ViT last layer: {vit.heads}")  # Linear(in_features=768, out_features=1000)

# EfficientNet
# Input: (batch_size, 3, 224, 224)
# Feature dimension: 1280
efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
print(f"EfficientNet last layer: {efficientnet.classifier}")  # Sequential with final Linear(1280, 1000)
])`}
      />

      <Text icon={<AlertTriangle size={16} />}>
        All these models were pre-trained on ImageNet with 1000 classes. The input images must be 
        preprocessed to match the original training data format (normalized).
      </Text>

      <Title order={3} mt="lg">Modifying Model Architecture</Title>
      <Text>
        For transfer learning, we often need to modify the model's architecture to match our task. 
        Here are common modification patterns:
      </Text>

      <CodeBlock
        language="python"
        code={`
import torch.nn as nn

# 1. Simple replacement of the final layer
model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
num_classes = 10
model.fc = nn.Linear(2048, num_classes)

# 2. Adding complexity to the classifier
model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
model.fc = nn.Sequential(
    nn.Linear(2048, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, num_classes)
)

# 3. Adding a custom head with batch normalization
model = models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
model.heads = nn.Sequential(
    nn.Linear(768, 1024),
    nn.BatchNorm1d(1024),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(1024, num_classes)
)
`}
      />

<Text>
        In convolutional neural networks (CNNs), `fc` stands for "fully connected" layer. 
        It's the final layer that takes the extracted features and maps them to class predictions:
      </Text>
      <Text>
        When we modify model.fc, we're essentially replacing the final classification layer 
        while keeping all the learned feature extractors (convolutional layers) intact. This 
        is the core principle of transfer learning - reusing learned features while adapting 
        the final layer for a new task.
      </Text>
    </Stack>
  );
};

export default Fundamentals;