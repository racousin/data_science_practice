import React, { useState } from 'react';
import { Container, Tabs, Title, Text, Stack, Group, ThemeIcon, Paper } from '@mantine/core';
import { Maximize, Filter, Contrast, Binary, Sparkles, Target, RotateCw, Database, Shield } from 'lucide-react';
import GeometricTransforms from './Enhancement/GeometricTransforms';
import FilteringMethods from './Enhancement/FilteringMethods';
import ContrastEnhancement from './Enhancement/ContrastEnhancement';

export default function Enhancement() {
  const [activeTab, setActiveTab] = useState('geometry');

  return (
    <Container size="lg" className="py-8">
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