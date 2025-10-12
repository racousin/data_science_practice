import React, { useState } from 'react';
import { Container, Tabs, Title, Text, Stack, Group, ThemeIcon, Paper, List } from '@mantine/core';
import { Maximize, Filter, Contrast, Binary, Sparkles, Target, RotateCw, Database, Shield } from 'lucide-react';
import GeometricTransforms from './Enhancement/GeometricTransforms';
import FilteringMethods from './Enhancement/FilteringMethods';
import ContrastEnhancement from './Enhancement/ContrastEnhancement';
import CodeBlock from 'components/CodeBlock';

export default function Enhancement() {
  const [activeTab, setActiveTab] = useState('geometry');

  return (
    <Container size="lg" className="py-8">
      <div data-slide>
        <Title order={2} mb="md">PyTorch Data Transformations</Title>

        <Text mb="md">
          PyTorch provides a powerful transformation pipeline through torchvision.transforms. Start by importing the necessary modules:
        </Text>

        <CodeBlock
          language="python"
          code={`from torchvision import transforms
from PIL import Image`}
        />

        <Text mt="lg" mb="md">
          Load an image using PIL, which PyTorch transforms expect as input:
        </Text>

        <CodeBlock
          language="python"
          code={`image = Image.open('ship.jpg')
print(f"Original size: {image.size}")`}
        />

        <Text mt="lg" mb="md">
          Define a transformation pipeline using transforms.Compose. This allows chaining multiple operations:
        </Text>

        <CodeBlock
          language="python"
          code={`transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])`}
        />

        <Text mt="lg" mb="md">
          Apply the transformation pipeline to your image:
        </Text>

        <CodeBlock
          language="python"
          code={`tensor_image = transform(image)
print(f"Tensor shape: {tensor_image.shape}")`}
        />

        <Text mt="lg" mb="md">
          Common transformations for data augmentation:
        </Text>

        <List spacing="xs" mb="md">
          <List.Item><Text component="span" fw={500}>RandomHorizontalFlip</Text> - Randomly flip images horizontally</List.Item>
          <List.Item><Text component="span" fw={500}>RandomRotation</Text> - Apply random rotations within a degree range</List.Item>
          <List.Item><Text component="span" fw={500}>ColorJitter</Text> - Randomly change brightness, contrast, saturation</List.Item>
          <List.Item><Text component="span" fw={500}>Normalize</Text> - Standardize pixel values using mean and standard deviation</List.Item>
        </List>

        <CodeBlock
          language="python"
          code={`augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])`}
        />
      </div>

      <Stack spacing="xl" mb="xl">
        <div>
          <Title className="text-3xl font-bold mb-4">Image Enhancement in Deep Learning</Title>
          <Text className="text-gray-700 mb-6">
            Image enhancement techniques are essential for addressing fundamental limitations of Convolutional Neural Networks (CNNs)
            and overcoming common data challenges in deep learning projects.
          </Text>
          
          <Paper withBorder p="md" className="bg-blue-50 mb-6">
            <Stack spacing="sm">
              <Text fw={500}>Why Enhancement Matters:</Text>
              <Text size="sm">
                CNNs are not inherently invariant to many image transformations. While they handle translation well due to their 
                convolutional nature, they struggle with rotations, scaling, and perspective changes. Without proper enhancement 
                and augmentation, a model might fail to recognize the same object when viewed from different angles or distances.
              </Text>
              <Text size="sm">
                Additionally, deep learning projects often start with limited datasets. Image enhancement techniques allow us to 
                generate new, valid training samples from existing data, effectively expanding our dataset while maintaining 
                semantic meaning.
              </Text>
            </Stack>
          </Paper>
        </div>

        <Group className="gap-8">
          <div className="flex items-start gap-3">
            <ThemeIcon size="lg" radius="md" variant="light" color="blue">
              <RotateCw size={18} />
            </ThemeIcon>
            <div>
              <Text size="sm" fw={500}>Transformation Invariance</Text>
              <Text size="sm" c="dimmed">Helps models recognize objects despite rotation, scale, or perspective changes</Text>
            </div>
          </div>

          <div className="flex items-start gap-3">
            <ThemeIcon size="lg" radius="md" variant="light" color="green">
              <Database size={18} />
            </ThemeIcon>
            <div>
              <Text size="sm" fw={500}>Data Augmentation</Text>
              <Text size="sm" c="dimmed">Expands small datasets through controlled transformations</Text>
            </div>
          </div>

          <div className="flex items-start gap-3">
            <ThemeIcon size="lg" radius="md" variant="light" color="grape">
              <Shield size={18} />
            </ThemeIcon>
            <div>
              <Text size="sm" fw={500}>Robustness</Text>
              <Text size="sm" c="dimmed">Improves model resilience to real-world image variations</Text>
            </div>
          </div>
        </Group>
      </Stack>

      <Tabs value={activeTab} onChange={setActiveTab}>
        <Tabs.List>
          <Tabs.Tab value="geometry" leftSection={<Maximize size={16} />}>
            Geometric Transforms
          </Tabs.Tab>
          <Tabs.Tab value="filtering" leftSection={<Filter size={16} />}>
            Filtering Methods
          </Tabs.Tab>
          <Tabs.Tab value="contrast" leftSection={<Contrast size={16} />}>
            Contrast Enhancement
          </Tabs.Tab>
        </Tabs.List>

        <Tabs.Panel value="geometry" pt="xl">
          <GeometricTransforms />
        </Tabs.Panel>
        <Tabs.Panel value="filtering" pt="xl">
          <FilteringMethods />
        </Tabs.Panel>
        <Tabs.Panel value="contrast" pt="xl">
          <ContrastEnhancement />
        </Tabs.Panel>
      </Tabs>
    </Container>
  );
}