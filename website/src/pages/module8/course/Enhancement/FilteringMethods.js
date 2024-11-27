import React, { useState } from 'react';
import { Paper, Title, Text, Stack, Group, Button } from '@mantine/core';
import { BlockMath } from 'react-katex';
import { Waves, MinusCircle, Sparkles } from 'lucide-react';
import CodeBlock from 'components/CodeBlock';

const FilteringMethods = () => {
  const [activeFilter, setActiveFilter] = useState('gaussian');

  const gaussianCode = `
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np

def apply_gaussian_blur(image_path, kernel_size=3, sigma=1.0):
    """
    Apply Gaussian blur to an image
    Args:
        image_path: Path to the input image
        kernel_size: Size of the Gaussian kernel (odd number)
        sigma: Standard deviation of the Gaussian kernel
    """
    # Load image
    image = Image.open(image_path)
    tensor_image = TF.to_tensor(image)
    
    # Add batch dimension required for conv2d
    tensor_image = tensor_image.unsqueeze(0)
    
    # Create 2D Gaussian kernel
    x = torch.linspace(-kernel_size//2, kernel_size//2, kernel_size)
    x, y = torch.meshgrid(x, x)
    gaussian_kernel = torch.exp(-(x**2 + y**2)/(2*sigma**2))
    gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
    
    # Prepare kernel for each channel
    gaussian_kernel = gaussian_kernel.expand(3, 1, kernel_size, kernel_size)
    
    # Apply convolution
    blurred = F.conv2d(tensor_image, gaussian_kernel.to(tensor_image.device),
                      padding=kernel_size//2, groups=3)
    
    return blurred.squeeze(0)  # Remove batch dimension`;

  const medianCode = `
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF

def apply_median_filter(image_path, kernel_size=3):
    """
    Apply median filter for noise reduction
    Args:
        image_path: Path to the input image
        kernel_size: Size of the median filter window
    """
    # Load image
    image = Image.open(image_path)
    tensor_image = TF.to_tensor(image)
    
    # Convert to numpy for easier median computation
    np_image = tensor_image.numpy()
    filtered = np.zeros_like(np_image)
    
    # Apply median filter to each channel
    pad = kernel_size // 2
    padded = np.pad(np_image, ((0,0), (pad,pad), (pad,pad)), mode='reflect')
    
    for c in range(3):  # Process each channel
        for i in range(np_image.shape[1]):
            for j in range(np_image.shape[2]):
                window = padded[c, 
                              i:i+kernel_size,
                              j:j+kernel_size]
                filtered[c,i,j] = np.median(window)
    
    return torch.from_numpy(filtered)`;

  const sharpenCode = `
import torch
import torchvision.transforms.functional as TF
from PIL import Image

def apply_sharpening(image_path, amount=1.5):
    """
    Apply unsharp masking for image sharpening
    Args:
        image_path: Path to the input image
        amount: Strength of sharpening effect
    """
    # Load image
    image = Image.open(image_path)
    tensor_image = TF.to_tensor(image)
    
    # Create Gaussian blur kernel
    kernel_size = 3
    sigma = 1.0
    x = torch.linspace(-kernel_size//2, kernel_size//2, kernel_size)
    x, y = torch.meshgrid(x, x)
    gaussian_kernel = torch.exp(-(x**2 + y**2)/(2*sigma**2))
    gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
    gaussian_kernel = gaussian_kernel.expand(3, 1, kernel_size, kernel_size)
    
    # Add dimensions for convolution
    tensor_image = tensor_image.unsqueeze(0)
    
    # Create blurred image
    blurred = torch.nn.functional.conv2d(
        tensor_image,
        gaussian_kernel.to(tensor_image.device),
        padding=kernel_size//2,
        groups=3
    )
    
    # Apply unsharp masking
    sharpened = tensor_image + amount * (tensor_image - blurred)
    
    # Clip values to valid range
    sharpened = torch.clamp(sharpened, 0, 1)
    
    return sharpened.squeeze(0)  # Remove batch dimension`;

  const filterCodes = {
    gaussian: gaussianCode,
    median: medianCode,
    sharpen: sharpenCode
  };

  const filterFormulas = {
    gaussian: `G(x,y) = \\frac{1}{2\\pi\\sigma^2}e^{-\\frac{x^2+y^2}{2\\sigma^2}}`,
    median: `M(x,y) = median\\{f(x-k,y-l)|(k,l)\\in W\\}`,
    sharpen: `S(x,y) = I(x,y) + \\lambda(I(x,y) - G(x,y))`
  };

  return (
    <Stack spacing="xl" className="w-full">
      <div className="space-y-4">
        <Title order={2} className="text-2xl font-bold">Enhancement Filters</Title>
        <Text className="text-gray-700">
          Image enhancement filters modify pixel values based on their local neighborhoods,
          enabling noise reduction, feature enhancement, and detail preservation.
        </Text>
      </div>

      <Paper p="lg" withBorder className="bg-slate-50">
        <Stack spacing="md">
          <Title order={3} className="text-lg font-semibold">Filter Kernel</Title>
          <BlockMath>{filterFormulas[activeFilter]}</BlockMath>
        </Stack>
      </Paper>

      <Paper p="lg" withBorder>
        <Stack spacing="md">
          <Group justify="space-between">
            <Text className="font-medium">Available Filters:</Text>
            <Group>
              <Button 
                leftSection={<Waves size={16} />} 
                variant={activeFilter === 'gaussian' ? 'filled' : 'light'}
                onClick={() => setActiveFilter('gaussian')}
              >
                Gaussian
              </Button>
              <Button 
                leftSection={<MinusCircle size={16} />} 
                variant={activeFilter === 'median' ? 'filled' : 'light'}
                onClick={() => setActiveFilter('median')}
              >
                Median
              </Button>
              <Button 
                leftSection={<Sparkles size={16} />} 
                variant={activeFilter === 'sharpen' ? 'filled' : 'light'}
                onClick={() => setActiveFilter('sharpen')}
              >
                Sharpen
              </Button>
            </Group>
          </Group>
          <CodeBlock 
            language="python"
            code={filterCodes[activeFilter]}
          />
        </Stack>
      </Paper>
    </Stack>
  );
};

export default FilteringMethods;