import React, { useState } from 'react';
import { Container, Title, Text, Stack, Tabs, Paper, Code, Alert, Slider, Group, Button } from '@mantine/core';
import { BlockMath } from 'react-katex';
import { AlertCircle, Image as ImageIcon, Maximize, Rotate, Contrast, Filter } from 'lucide-react';

const CodeBlock = ({ code }) => (
  <Paper p="md" withBorder className="bg-slate-50 overflow-x-auto">
    <Code block className="text-sm">{code}</Code>
  </Paper>
);

const TechniqueSection = ({ title, formula, code, description, children }) => (
  <Stack className="space-y-4">
    <Title order={3} className="text-xl font-semibold">{title}</Title>
    <Text className="text-gray-700">{description}</Text>
    {formula && (
      <div className="py-2">
        <BlockMath>{formula}</BlockMath>
      </div>
    )}
    {code && <CodeBlock code={code} />}
    {children}
  </Stack>
);

export default function Enhancement() {
  const [activeTab, setActiveTab] = useState('geometry');

  const geometricCode = `
def apply_geometric_transforms(image, angle=30, scale=1.2):
    """Apply geometric transformations to an image"""
    if not torch.is_tensor(image):
        image = TF.to_tensor(image)
        
    # Rotation
    rotated = TF.rotate(image, angle)
    
    # Scaling
    scaled = TF.resize(image, 
                      [int(image.shape[1] * scale),
                       int(image.shape[2] * scale)])
    
    # Perspective transform
    width, height = image.shape[2], image.shape[1]
    startpoints = [[0, 0], [width-1, 0], [0, height-1], [width-1, height-1]]
    endpoints = [[width*0.1, height*0.1], [width*0.9, height*0.1],
                [width*0.2, height*0.9], [width*0.8, height*0.9]]
    
    perspective = TF.perspective(image, startpoints, endpoints)
    
    return rotated, scaled, perspective`;

  const filteringCode = `
def apply_enhancement_filters(image):
    """Apply various enhancement filters"""
    if not torch.is_tensor(image):
        image = TF.to_tensor(image)
    
    # Gaussian blur
    blurred = TF.gaussian_blur(image, kernel_size=3, sigma=1.0)
    
    # Unsharp masking
    unsharp_mask = image - blurred
    sharpened = image + 1.5 * unsharp_mask
    
    # Median filtering for noise reduction
    median = torch.median(image.unfold(1, 3, 1).unfold(2, 3, 1), dim=-1)[0]
    
    return blurred, sharpened, median`;

  const contrastCode = `
def enhance_contrast(image, clip_limit=2.0, grid_size=8):
    """Apply advanced contrast enhancement"""
    if not torch.is_tensor(image):
        image = TF.to_tensor(image)
    
    # Convert to LAB color space
    lab = rgb_to_lab(image)
    l_channel = lab[0]
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(grid_size, grid_size)
    )
    enhanced_l = clahe.apply(
        (l_channel.numpy() * 255).astype(np.uint8)
    )
    
    # Reconstruct image
    lab[0] = torch.from_numpy(enhanced_l).float() / 255
    enhanced = lab_to_rgb(lab)
    
    return enhanced`;

  return (
    <Container size="lg" className="py-8">
      <Stack spacing="xl">
        <div>
          <Title order={1} className="text-3xl font-bold mb-4">
            Image Enhancement Techniques
          </Title>
          <Alert 
            icon={<AlertCircle size={16} />}
            title="Best Practices"
            className="mb-6"
          >
            Always validate enhanced images through both quantitative metrics and visual inspection.
            Consider maintaining the original aspect ratio and using consistent normalization.
          </Alert>
        </div>

        <Tabs value={activeTab} onChange={setActiveTab} className="w-full">
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

          <Tabs.Panel value="geometry" className="pt-4">
            <TechniqueSection
              title="Geometric Transformations"
              description="Geometric transformations modify the spatial arrangement of pixels while preserving their intensity values."
              formula={`T(x,y) = \\begin{bmatrix} 
                a & b & c \\\\
                d & e & f \\\\
                g & h & 1
              \\end{bmatrix}
              \\begin{bmatrix}
                x \\\\ y \\\\ 1
              \\end{bmatrix}`}
              code={geometricCode}
            />
          </Tabs.Panel>

          <Tabs.Panel value="filtering" className="pt-4">
            <TechniqueSection
              title="Enhancement Filters"
              description="Spatial filtering operations for noise reduction and feature enhancement."
              formula={`G(x,y) = \\sum_{s=-a}^{a}\\sum_{t=-b}^{b} w(s,t)f(x+s,y+t)`}
              code={filteringCode}
            />
          </Tabs.Panel>

          <Tabs.Panel value="contrast" className="pt-4">
            <TechniqueSection
              title="Contrast Enhancement"
              description="Advanced contrast enhancement techniques including adaptive histogram equalization."
              formula={`h(v) = round(\\frac{cdf(v) - cdf_{min}}{(M × N) - cdf_{min}} × (L-1))`}
              code={contrastCode}
            >
              <Paper p="md" withBorder className="mt-4">
                <Stack spacing="md">
                  <Text size="sm" className="font-medium">Adjust Parameters:</Text>
                  <Group>
                    <Text size="sm" className="w-24">Clip Limit:</Text>
                    <Slider 
                      defaultValue={2}
                      min={1}
                      max={5}
                      step={0.1}
                      className="flex-1"
                    />
                  </Group>
                  <Group>
                    <Text size="sm" className="w-24">Grid Size:</Text>
                    <Slider 
                      defaultValue={8}
                      min={4}
                      max={16}
                      step={2}
                      className="flex-1"
                    />
                  </Group>
                </Stack>
              </Paper>
            </TechniqueSection>
          </Tabs.Panel>
        </Tabs>
      </Stack>
    </Container>
  );
}