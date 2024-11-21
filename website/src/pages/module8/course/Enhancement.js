import React from 'react';
import { Container, Title, Text, Stack } from '@mantine/core';
import CodeBlock from 'components/CodeBlock';
import 'katex/dist/katex.min.css';
import { BlockMath } from 'react-katex';

export default function Enhancement() {
  const filteringCode = `
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

def apply_gaussian_blur(image, kernel_size=3, sigma=1.0):
    # Ensure input is a torch tensor
    if not isinstance(image, torch.Tensor):
        image = TF.to_tensor(image)
    
    # Add batch dimension if needed
    if image.dim() == 3:
        image = image.unsqueeze(0)
    
    # Apply Gaussian blur
    blurred = F.gaussian_blur(
        image,
        kernel_size=[kernel_size, kernel_size],
        sigma=[sigma, sigma]
    )
    
    return blurred.squeeze(0)`;

  const histogramCode = `
def adaptive_histogram_equalization(image, clip_limit=2.0, tile_size=8):
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
    if not torch.is_tensor(image):
        image = TF.to_tensor(image)
    
    # Convert to LAB color space
    image_np = image.permute(1, 2, 0).numpy()
    lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
    l_channel = lab[:,:,0]
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(tile_size, tile_size)
    )
    cl = clahe.apply(l_channel)
    
    # Merge channels
    lab[:,:,0] = cl
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    return torch.from_numpy(enhanced).permute(2, 0, 1)`;

  const noiseReductionCode = `
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 64, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 2, stride=2),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)`;

  return (
    <Container size="lg">
      <Stack spacing="xl">
        <Title order={1} id="filtering">Image Enhancement Techniques</Title>

        <div>
          <Title order={2}>Image Filtering</Title>
          <Text>
            Spatial filtering improves image quality through convolution operations:
          </Text>
          <BlockMath>
            {`G(x,y) = \\sum_{s=-a}^{a}\\sum_{t=-b}^{b} w(s,t)f(x+s,y+t)`}
          </BlockMath>
          <CodeBlock
            language="python"
            code={filteringCode}
          />
        </div>

        <div id="histogram">
          <Title order={2}>Histogram Equalization</Title>
          <Text>
            Adaptive histogram equalization enhances local contrast:
          </Text>
          <BlockMath>
            {`h(v) = round(\\frac{cdf(v) - cdf_{min}}{(M × N) - cdf_{min}} × (L-1))`}
          </BlockMath>
          <CodeBlock
            language="python"
            code={histogramCode}
          />
        </div>

        <div id="noise-reduction">
          <Title order={2}>Noise Reduction Methods</Title>
          <Text>
            Deep learning-based denoising using autoencoders:
          </Text>
          <CodeBlock
            language="python"
            code={noiseReductionCode}
          />
          <Text>
            Key techniques:
            • Gaussian denoising
            • Non-local means
            • Deep learning approaches
            • Bilateral filtering
          </Text>
        </div>

        <div>
          <Title order={2}>Practical Applications</Title>
          <Text>
            • Medical image enhancement
            • Satellite imagery processing
            • Photography post-processing
            • Document image enhancement
            • Real-time video processing
          </Text>
        </div>
      </Stack>
    </Container>
  );
}