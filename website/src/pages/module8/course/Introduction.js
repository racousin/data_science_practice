import React from 'react';
import { Container, Title, Text, Stack, Image } from '@mantine/core';
import CodeBlock from 'components/CodeBlock';
import 'katex/dist/katex.min.css';
import { InlineMath, BlockMath } from 'react-katex';

export default function Introduction() {
  const basicImageCode = `
import torch
import torchvision.transforms as T
from PIL import Image

def load_and_preprocess(image_path):
    # Load image
    img = Image.open(image_path)
    
    # Define transformations
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                   std=[0.229, 0.224, 0.225])
    ])
    
    # Apply transformations
    return transform(img)`;

  return (
    <Container size="lg">
      <Stack spacing="xl">
        <Title order={1} id="fundamentals">Introduction</Title>
        
        <div>
          <Title order={2}>Core Concepts</Title>
          <Text>
            Image processing in deep learning involves transforming raw pixel data into meaningful features. 
            A digital image is represented as a tensor with shape (C, H, W) where:
          </Text>
          <BlockMath>
            {`Image_{tensor} \\in \\mathbb{R}^{C \\times H \\times W}`}
          </BlockMath>
          <Text>
            • C: Number of channels (3 for RGB, 1 for grayscale)
            • H: Height of the image
            • W: Width of the image
          </Text>
        </div>

        <div id="digital-images">
          <Title order={2}>Digital Image Representation</Title>
          <Text>
            Each pixel in a digital image is quantized into discrete intensity values:
            • 8-bit images: <InlineMath>{`[0, 255]`}</InlineMath>
            • Normalized values: <InlineMath>{`[0, 1]`}</InlineMath>
          </Text>
          <CodeBlock
            language="python"
            code={basicImageCode}
          />
        </div>

        <div id="preprocessing">
          <Title order={2}>Image Preprocessing Techniques</Title>
          <Text>
            Key preprocessing steps for deep learning:
            1. Resizing: Standardize image dimensions
            2. Normalization: Scale pixel values to improve training stability
            3. Data augmentation: Increase dataset variability
          </Text>
          <BlockMath>
            {`Normalized_{pixel} = \\frac{pixel - mean}{std}`}
          </BlockMath>
        </div>
        
        <div>
          <Title order={2}>Best Practices</Title>
          <Text>
            • Maintain aspect ratio during resizing
            • Use dataset statistics for normalization
            • Consider memory constraints when batching
            • Validate image preprocessing pipeline
          </Text>
        </div>
      </Stack>
    </Container>
  );
}