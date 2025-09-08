import React, { useState } from 'react';
import { Paper, Title, Text, Stack, Group, Button, Image } from '@mantine/core';
import { BlockMath } from 'react-katex';
import { Maximize, RotateCw, Move, FlipHorizontal } from 'lucide-react';
import CodeBlock from 'components/CodeBlock';

const GeometricTransforms = () => {
  const [activeTransform, setActiveTransform] = useState('rotate');

  const rotationCode = `
import torch
import torchvision.transforms.functional as F
from PIL import Image

def apply_rotation(image_path, angle=30):
    """
    Apply rotation to an image
    Args:
        image_path: Path to the input image
        angle: Rotation angle in degrees (positive = counterclockwise)
    """
    # Load image
    image = Image.open(image_path)
    
    # Convert to tensor for PyTorch operations
    tensor_image = F.to_tensor(image)
    
    # Apply rotation
    rotated = F.rotate(tensor_image, angle)
    
    return rotated`;

  const flipCode = `
import torch
import torchvision.transforms.functional as F
from PIL import Image

def apply_flip(image_path, horizontal=True):
    """
    Apply flip transformation to an image
    Args:
        image_path: Path to the input image
        horizontal: If True, flip horizontally; if False, flip vertically
    """
    # Load image
    image = Image.open(image_path)
    
    # Convert to tensor for PyTorch operations
    tensor_image = F.to_tensor(image)
    
    # Apply flip
    if horizontal:
        flipped = F.hflip(tensor_image)
    else:
        flipped = F.vflip(tensor_image)
    
    return flipped`;

  const scaleCode = `
import torch
import torchvision.transforms.functional as F
from PIL import Image

def apply_scaling(image_path, scale_factor=1.2):
    """
    Apply scaling to an image
    Args:
        image_path: Path to the input image
        scale_factor: Factor to scale the image (>1 enlarges, <1 shrinks)
    """
    # Load image
    image = Image.open(image_path)
    
    # Get original dimensions
    width, height = image.size
    
    # Calculate new dimensions
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    
    # Apply scaling using PIL for better interpolation
    scaled = image.resize((new_width, new_height), Image.Resampling.BICUBIC)
    
    # Convert to tensor for PyTorch pipeline
    tensor_scaled = F.to_tensor(scaled)
    
    return tensor_scaled`;

  const perspectiveCode = `
import cv2
import numpy as np
import torch
from PIL import Image

def apply_perspective(image_path, intensity=0.2):
    """
    Apply perspective transform to an image
    Args:
        image_path: Path to the input image
        intensity: Controls the strength of perspective effect (0-1)
    """
    # Load image with OpenCV for perspective transform
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    
    # Define the 4 source points
    src_points = np.float32([
        [0, 0],
        [width-1, 0],
        [0, height-1],
        [width-1, height-1]
    ])
    
    # Define the 4 destination points with perspective effect
    dst_points = np.float32([
        [width*intensity, height*intensity],
        [width*(1-intensity), height*intensity],
        [0, height],
        [width, height]
    ])
    
    # Calculate perspective transform matrix
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    
    # Apply the transform
    result = cv2.warpPerspective(image, matrix, (width, height))
    
    # Convert back to RGB (from BGR) and then to PIL Image
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(result_rgb)
    
    # Convert to tensor for PyTorch pipeline
    tensor_result = F.to_tensor(pil_image)
    
    return tensor_result`;

  const transformContent = {
    rotate: {
      code: rotationCode,
      image: "/assets/data-science-practice/module7/rotate.png"
    },
    flip: {
      code: flipCode,
      image: "/assets/data-science-practice/module7/flip.png"
    },
    scale: {
      code: scaleCode,
      image: "/assets/data-science-practice/module7/scale.png"
    },
    perspective: {
      code: perspectiveCode,
      image: "/assets/data-science-practice/module7/perspective.png"
    }
  };

  return (
    <Stack spacing="xl" className="w-full">
      <div className="space-y-4">
        <Title order={2} className="text-2xl font-bold">Geometric Transformations</Title>
        <Text className="text-gray-700">
          Geometric transformations modify the spatial arrangement of pixels while preserving their intensity values. 
          These transformations are fundamental for data augmentation and image alignment tasks.
        </Text>
      </div>

      <Paper p="lg" withBorder className="bg-slate-50">
        <Stack spacing="md">
          <Title order={3} className="text-lg font-semibold">Transformation Matrix</Title>
          <BlockMath>{`T(x,y) = \\begin{bmatrix} 
            a & b & c \\\\
            d & e & f \\\\
            g & h & 1
          \\end{bmatrix}
          \\begin{bmatrix}
            x \\\\ y \\\\ 1
          \\end{bmatrix}`}</BlockMath>
        </Stack>
      </Paper>

      <Paper p="lg" withBorder>
        <Stack spacing="md">
          <Group justify="space-between">
            <Text className="font-medium">Available Transformations:</Text>
            <Group>
              <Button 
                leftSection={<RotateCw size={16} />} 
                variant={activeTransform === 'rotate' ? 'filled' : 'light'}
                onClick={() => setActiveTransform('rotate')}
              >
                Rotate
              </Button>
              <Button 
                leftSection={<FlipHorizontal size={16} />} 
                variant={activeTransform === 'flip' ? 'filled' : 'light'}
                onClick={() => setActiveTransform('flip')}
              >
                Flip
              </Button>
              <Button 
                leftSection={<Maximize size={16} />} 
                variant={activeTransform === 'scale' ? 'filled' : 'light'}
                onClick={() => setActiveTransform('scale')}
              >
                Scale
              </Button>
              <Button 
                leftSection={<Move size={16} />} 
                variant={activeTransform === 'perspective' ? 'filled' : 'light'}
                onClick={() => setActiveTransform('perspective')}
              >
                Perspective
              </Button>
            </Group>
          </Group>

          <div className="rounded-lg overflow-hidden">
            <Image
              src={transformContent[activeTransform].image}
              alt={`${activeTransform} transformation`}
              className="w-full object-contain"
            />
          </div>

          <CodeBlock 
            language="python"
            code={transformContent[activeTransform].code}
          />
        </Stack>
      </Paper>
    </Stack>
  );
};

export default GeometricTransforms;