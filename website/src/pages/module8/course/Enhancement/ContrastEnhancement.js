import React, { useState } from 'react';
import { Paper, Title, Text, Stack, Group, Button } from '@mantine/core';
import { BlockMath } from 'react-katex';
import { Contrast, BarChart2, Gauge } from 'lucide-react';
import CodeBlock from 'components/CodeBlock';

const ContrastEnhancement = () => {
  const [activeMethod, setActiveMethod] = useState('clahe');

  const claheCode = `
import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF

def apply_clahe(image_path, clip_limit=2.0, grid_size=(8, 8)):
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)
    Args:
        image_path: Path to the input image
        clip_limit: Threshold for contrast limiting
        grid_size: Size of grid for histogram equalization
    """
    # Load image
    image = Image.open(image_path)
    tensor_image = TF.to_tensor(image)
    
    # Convert to LAB color space
    np_image = (tensor_image.numpy() * 255).astype(np.uint8)
    np_image = np_image.transpose(1, 2, 0)
    lab_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2LAB)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    lab_image[:,:,0] = clahe.apply(lab_image[:,:,0])
    
    # Convert back to RGB
    enhanced = cv2.cvtColor(lab_image, cv2.COLOR_LAB2RGB)
    enhanced = torch.from_numpy(enhanced.transpose(2, 0, 1)) / 255.0
    
    return enhanced`;

  const histogramCode = `
import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF

def apply_histogram_equalization(image_path, per_channel=True):
    """
    Apply histogram equalization
    Args:
        image_path: Path to the input image
        per_channel: Whether to equalize each RGB channel separately
    """
    # Load image
    image = Image.open(image_path)
    tensor_image = TF.to_tensor(image)
    
    # Convert to numpy array
    np_image = (tensor_image.numpy() * 255).astype(np.uint8)
    enhanced = np.zeros_like(np_image)
    
    if per_channel:
        # Apply to each channel separately
        for i in range(3):
            enhanced[i] = cv2.equalizeHist(np_image[i])
    else:
        # Convert to YUV and apply to Y channel only
        image_yuv = cv2.cvtColor(np_image.transpose(1,2,0), 
                                cv2.COLOR_RGB2YUV)
        image_yuv[:,:,0] = cv2.equalizeHist(image_yuv[:,:,0])
        enhanced = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2RGB)
        enhanced = enhanced.transpose(2,0,1)
    
    return torch.from_numpy(enhanced) / 255.0`;

  const adaptiveCode = `
import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF

def apply_adaptive_gamma(image_path, block_size=16):
    """
    Apply adaptive gamma correction based on local luminance
    Args:
        image_path: Path to the input image
        block_size: Size of local blocks for adaptation
    """
    # Load image
    image = Image.open(image_path)
    tensor_image = TF.to_tensor(image)
    
    # Convert to numpy array
    np_image = (tensor_image.numpy() * 255).astype(np.uint8)
    height, width = np_image.shape[1:]
    enhanced = np.zeros_like(np_image)
    
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            # Extract block
            block = np_image[:,
                           i:min(i+block_size, height),
                           j:min(j+block_size, width)]
            
            # Calculate local mean luminance
            luminance = np.mean(block)
            
            # Adapt gamma based on luminance
            gamma = 2.0 - (luminance / 128.0)  # Higher gamma for darker regions
            gamma = np.clip(gamma, 0.5, 2.5)
            
            # Apply gamma correction to block
            enhanced[:,i:i+block_size,j:j+block_size] = \
                np.power(block / 255.0, gamma) * 255
    
    return torch.from_numpy(enhanced) / 255.0`;

  const methodCodes = {
    clahe: claheCode,
    histogram: histogramCode,
    adaptive: adaptiveCode
  };

  const methodFormulas = {
    clahe: `P_i = \\frac{N_{i,clip}}{M_{region}} \\times (L-1)`,
    histogram: `s_k = T(r_k) = (L-1)\\sum_{j=0}^k p_r(r_j)`,
    adaptive: `I_{out} = I_{in}^{\\gamma(x,y)}, \\gamma(x,y) = 2 - \\frac{L(x,y)}{128}`
  };

  return (
    <Stack spacing="xl" className="w-full">
      <div className="space-y-4">
        <Title order={2} className="text-2xl font-bold">Contrast Enhancement</Title>
        <Text className="text-gray-700">
          Contrast enhancement techniques improve image visibility by optimizing the intensity 
          distribution, either globally or adaptively based on local image characteristics.
        </Text>
      </div>

      <Paper p="lg" withBorder className="bg-slate-50">
        <Stack spacing="md">
          <Title order={3} className="text-lg font-semibold">Enhancement Formula</Title>
          <BlockMath>{methodFormulas[activeMethod]}</BlockMath>
        </Stack>
      </Paper>

      <Paper p="lg" withBorder>
        <Stack spacing="md">
          <Group justify="space-between">
            <Text className="font-medium">Enhancement Methods:</Text>
            <Group>
              <Button 
                leftSection={<Contrast size={16} />} 
                variant={activeMethod === 'clahe' ? 'filled' : 'light'}
                onClick={() => setActiveMethod('clahe')}
              >
                CLAHE
              </Button>
              <Button 
                leftSection={<BarChart2 size={16} />} 
                variant={activeMethod === 'histogram' ? 'filled' : 'light'}
                onClick={() => setActiveMethod('histogram')}
              >
                Histogram
              </Button>
              <Button 
                leftSection={<Gauge size={16} />} 
                variant={activeMethod === 'adaptive' ? 'filled' : 'light'}
                onClick={() => setActiveMethod('adaptive')}
              >
                Adaptive Gamma
              </Button>
            </Group>
          </Group>
          <CodeBlock 
            language="python"
            code={methodCodes[activeMethod]}
          />
        </Stack>
      </Paper>
    </Stack>
  );
};

export default ContrastEnhancement;