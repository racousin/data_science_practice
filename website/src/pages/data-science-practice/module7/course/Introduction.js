import React from 'react';
import { Container, Title, Text, Stack, Group, Box, Image, List } from '@mantine/core';
import CodeBlock from 'components/CodeBlock';
import 'katex/dist/katex.min.css';
import { InlineMath, BlockMath } from 'react-katex';
import ConvolutionExplanation from './Introduction/ConvolutionExplanation.js'

const TaskBox = ({ title, formula, description }) => (
  <Box p="md" >
    <Title order={4} mb="sm">{title}</Title>
    <Stack spacing="xs">
      <BlockMath>{formula}</BlockMath>
      <Text size="sm">{description}</Text>
    </Stack>
  </Box>
);

const Introduction = () => {
  // Example code showing tensor representation
  const tensorCode = `
import torch
import numpy as np
from PIL import Image

# Load and convert image to tensor
image = Image.open('sample.jpg')
tensor = torch.from_numpy(np.array(image))

# Shape: [Height, Width, Channels]
print(f"Image shape: {tensor.shape}")

# Access RGB values of pixel at position (0,0)
pixel = tensor[0, 0]
print(f"RGB values at (0,0): {pixel}")`;

  // Example code showing convolution operation
  const convolutionCode = `
import torch
import torch.nn.functional as F

# Define a simple edge detection kernel
edge_kernel = torch.tensor([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
]).float().unsqueeze(0).unsqueeze(0)

# Apply convolution
def apply_convolution(image_tensor):
    # Ensure image is in the right shape [1, 1, H, W]
    if len(image_tensor.shape) == 2:
        image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)
    
    # Apply convolution
    return F.conv2d(image_tensor, edge_kernel, padding=1)`;

  return (
    <Container size="lg">
      <Title order={1} id="introduction" mb="xl">Understanding Images in Deep Learning</Title>

      {/* Slide 1: Digital Image Representation */}
      <div data-slide>
        <Title order={2} id="digital-representation" mb="md">Digital Image Representation</Title>

        <Box p="md" mb="md" >
          <Image src="/assets/data-science-practice/module7/gray.png" alt="Grayscale representation" mb="sm" />
          <Text size="sm">
            Single channel grayscale image: each pixel represents intensity from 0 (black) to max value 8 (white)
          </Text>
        </Box>

        <Text mb="md">
          At its core, a digital image is a structured grid of numerical values. Each point in this grid,
          called a pixel, contains information about color intensity.
        </Text>
      </div>

      {/* Slide 2: Color Depth */}
      <div data-slide>
        <Title order={2} mb="md">Color Depth</Title>

        <Box p="md" mb="md" >
          <Image src="/assets/data-science-practice/module7/bit-depth-representation.png" alt="Bit depth visualization" mb="sm" />
          <Text size="sm" mb="xs">Bit depth determines the range of possible values:</Text>
          <List size="sm">
            <List.Item>2-bit: 2 levels (0-1)</List.Item>
            <List.Item>8-bit: 256 levels (0-255)</List.Item>
            <List.Item>16-bit: 65,536 levels (0-65535)</List.Item>
          </List>
        </Box>
      </div>

      {/* Slide 3: Color Channels */}
      <div data-slide>
        <Title order={2} mb="md">Color Channels</Title>

        <Box p="md" mb="md" >
          <Image src="/assets/data-science-practice/module7/rgb.png" alt="RGB channel separation" mb="sm" />
          <Text size="sm">
            RGB images consist of three channels, each representing the intensity of Red, Green, and Blue components (usually on 8-bit)
          </Text>
        </Box>
      </div>

      {/* Slide 4: Extended Channel Applications */}
      <div data-slide>
        <Title order={2} mb="md">Extended Channel Applications</Title>

        <Box p="md" mb="md" >
          <Image src="/assets/data-science-practice/module7/satelite.png" alt="Multi-spectral satellite imagery" mb="sm" />
          <Text size="sm" mb="xs">Images can contain multiple layers beyond RGB:</Text>
          <List size="sm">
            <List.Item>Satellite imagery: Near Infrared, Thermal, Altitude, etc.</List.Item>
            <List.Item>Medical imaging: Different sensor readings</List.Item>
          </List>
        </Box>

        <Text mb="md">
          Mathematically, a general image is represented as a H × W × C tensor, where:
        </Text>

        <List mb="md">
          <List.Item>H: Height in pixels</List.Item>
          <List.Item>W: Width in pixels</List.Item>
          <List.Item>C: Number of channels</List.Item>
        </List>

        <BlockMath>
          {`I \\in \\mathbb{R}^{H \\times W \\times C}`}
        </BlockMath>
      </div>

      {/* Slide 5: ML Tasks Overview */}
      <div data-slide>
        <Title order={2} id="ml-tasks" mb="md">Machine Learning Tasks with Images</Title>

        <Text mb="md">
          Images can serve as both input and output in various machine learning tasks.
        </Text>
      </div>

      {/* Slide 6: Image to Value Tasks */}
      <div data-slide>
        <Title order={3} mb="md">Image to Value Tasks</Title>

        <Group grow mb="md">
          <TaskBox
            title="Classification"
            formula={`f: \\mathbb{R}^{H \\times W \\times C} \\rightarrow \\{1,\\ldots,K\\}`}
            description="Maps images to K discrete categories. Examples: object recognition, disease diagnosis, quality control."
          />
          <TaskBox
            title="Regression"
            formula={`f: \\mathbb{R}^{H \\times W \\times C} \\rightarrow \\mathbb{R}^n`}
            description="Predicts continuous values from images. Examples: age estimation, pose estimation, depth prediction."
          />
        </Group>
      </div>

      {/* Slide 7: Image to Structured Output */}
      <div data-slide>
        <Title order={3} mb="md">Image to Structured Output</Title>

        <Group grow mb="md">
          <TaskBox
            title="Object Detection"
            formula={`f: \\mathbb{R}^{H \\times W \\times C} \\rightarrow \\{(b_i, c_i, p_i)\\}_{i=1}^N`}
            description="Detects objects and their locations. Output includes bounding boxes (b), class labels (c), and confidence scores (p)."
          />
          <TaskBox
            title="Image Segmentation"
            formula={`f: \\mathbb{R}^{H \\times W \\times C} \\rightarrow \\{0,\\ldots,K\\}^{H \\times W}`}
            description="Assigns a class label to each pixel. Output is a label map of same height and width as input."
          />
        </Group>
      </div>

      {/* Slide 8: Image to Image Tasks */}
      <div data-slide>
        <Title order={3} mb="md">Image to Image Tasks</Title>

        <Group grow mb="md">
          <TaskBox
            title="Image Translation"
            formula={`f: \\mathbb{R}^{H \\times W \\times C} \\rightarrow \\mathbb{R}^{H \\times W \\times C}`}
            description="Converts images from one domain to another. Examples: day-to-night, young-to-old, sketch-to-photo."
          />
          <TaskBox
            title="Image Compression"
            formula={`f: \\mathbb{R}^{H \\times W \\times C} \\rightarrow \\mathbb{R}^d, \\quad g: \\mathbb{R}^d \\rightarrow \\mathbb{R}^{H \\times W \\times C}`}
            description="Transforms image into compact latent representation (f) and reconstructs it (g)."
          />
        </Group>
      </div>

      {/* Slide 9: Value to Image Tasks */}
      <div data-slide>
        <Title order={3} mb="md">Value/Sequence to Image Tasks</Title>

        <Group grow mb="md">
          <TaskBox
            title="Image Generation from Description"
            formula={`f: \\mathcal{S} \\rightarrow \\mathbb{R}^{H \\times W \\times C}`}
            description="Generates images from text descriptions or other sequential inputs. Examples: text-to-image generation, DALL-E, Stable Diffusion."
          />
          <TaskBox
            title="Noise to Image (GANs/Diffusion)"
            formula={`f: \\mathbb{R}^d \\rightarrow \\mathbb{R}^{H \\times W \\times C}`}
            description="Generates images from random noise using generative models. Examples: StyleGAN, DDPM."
          />
        </Group>
      </div>



      {/* Slide 10: Introduction to Convolutions */}
      <div data-slide>
        <Title order={2} id="convolutions" mb="md">Leveraging Spatial Structure with Convolutions</Title>

        <Text mb="md">
          While we could flatten an image into a 1D vector (
          <InlineMath>{`\\mathbb{R}^{H \\times W \\times C} \\rightarrow \\mathbb{R}^{HWC}`}</InlineMath>
          ), this loses crucial spatial information. Convolutions preserve and exploit this structure.
        </Text>

        <Text mb="md">
          A convolution operation slides a kernel (small matrix) across the image, computing weighted sums
          of local regions:
        </Text>

        <BlockMath>
          {`(f * g)[n] = \\sum_{m=-\\infty}^{\\infty} f[m]g[n-m]`}
        </BlockMath>
      </div>

      {/* Slide 11: Convolution Code Example */}
      <div data-slide>
        <Title order={3} mb="md">Convolution Implementation</Title>

        <CodeBlock
          language="python"
          code={convolutionCode}
        />
      </div>

      {/* Slide 12: 2D Convolution Formula */}
      <div data-slide>
        <Title order={3} mb="md">2D Convolution Operation</Title>

        <Image src="/assets/data-science-practice/module7/conv2.png" alt="conv" mb="md" />

        <Text mb="md">
          Concretely, a 2D convolution is the operation that computes a weighted sum between
          two matrices: an input matrix (typically an image or feature map) and a kernel
          (also called a filter). The operation is defined as:
        </Text>

        <Box p="md" mb="md" >
          <BlockMath>
            {`G[i, j, c_{out}] = \\sigma\\left(\\sum_{c_{in}=0}^{C_{in}-1} \\sum_{k=0}^{K_h-1} \\sum_{l=0}^{K_w-1} F[s_h \\cdot i + k - p_h, s_w \\cdot j + l - p_w, c_{in}] \\cdot K[k, l, c_{in}, c_{out}] + b[c_{out}]\\right)`}
          </BlockMath>

          <Text size="sm" mt="md" mb="xs">where:</Text>

          <List size="sm">
            <List.Item><InlineMath>{"G[i,j,c_{out}]"}</InlineMath> is the output at position <InlineMath>{"(i,j)"}</InlineMath> for output channel <InlineMath>{"c_{out}"}</InlineMath></List.Item>
            <List.Item><InlineMath>{"F"}</InlineMath> is the input tensor (with padding applied)</List.Item>
            <List.Item><InlineMath>K</InlineMath> is the kernel/filter tensor</List.Item>
            <List.Item><InlineMath>\sigma</InlineMath> is the activation function (e.g., ReLU, sigmoid)</List.Item>
            <List.Item><InlineMath>{"C_{in}"}</InlineMath> is the number of input channels</List.Item>
            <List.Item><InlineMath>K_h, K_w</InlineMath> are kernel height and width</List.Item>
            <List.Item><InlineMath>s_h, s_w</InlineMath> are stride values for height and width directions</List.Item>
            <List.Item><InlineMath>{"p_h, p_w"}</InlineMath> are padding values for height and width</List.Item>
            <List.Item><InlineMath>{"b[c_{out}]"}</InlineMath> is the bias term for output channel <InlineMath>{"c_{out}"}</InlineMath></List.Item>
          </List>
        </Box>
      </div>

      {/* Slide 13: CNN Architecture */}
      <div data-slide>
        <Title order={2} id="cnn-architecture" mb="md">Understanding Convolutional Neural Networks</Title>

        <Text mb="md">
          Convolutional Neural Networks (CNNs) are specialized deep learning models that process grid-like data using convolutional layers. Unlike traditional neural networks, CNNs automatically learn spatial hierarchies of features through multiple layers of convolutions.
        </Text>

        <Image src="/assets/data-science-practice/module7/cnn-network.jpg" alt="conv" mb="md" />

        <Text mb="md">
          The backpropagation process in CNNs follows the same principles as in fully connected networks, but with kernel weight sharing and local connectivity constraints.
        </Text>
      </div>

      {/* Slide 14: Feature Learning Hierarchy */}
      <div data-slide>
        <Title order={3} mb="md">Feature Learning Hierarchy</Title>

        <Text mb="md">
          Each convolutional layer learns progressively more complex features:
        </Text>

        <Group grow mb="md">
          <Box p="md" >
            <Text weight={600} mb="xs">Early Layers</Text>
            <Text size="sm">Detect basic features like edges and corners</Text>
          </Box>

          <Box p="md" >
            <Text weight={600} mb="xs">Middle Layers</Text>
            <Text size="sm">Combine basic features into patterns and textures</Text>
          </Box>

          <Box p="md" >
            <Text weight={600} mb="xs">Deep Layers</Text>
            <Text size="sm">Recognize complex objects and their arrangements</Text>
          </Box>
        </Group>
      </div>
      {/* Slide 15: Parameter Efficiency */}
      <div data-slide>
        <Title order={2} id="efficiency" mb="md">Convolutional vs. Fully Connected Layers</Title>

        <Text mb="md">Key advantages of convolutions:</Text>

        <List mb="md">
          <List.Item>Preserve spatial relationships between pixels</List.Item>
          <List.Item>Parameter sharing reduces model complexity</List.Item>
        </List>

        <Text mb="md">
          Consider an image of size 224×224×3. A fully connected layer would require in first layer:
        </Text>

        <BlockMath>
          {`224 \\times 224 \\times 3 \\times N_{neurons} = 150,528 \\times N_{neurons}`}
        </BlockMath>

        <Text mb="md">
          In contrast, a convolutional layer with 64 filters of size 3×3 only needs in first layer:
        </Text>

        <BlockMath>
          {`3 \\times 3 \\times 3 \\times 64 = 1,728`}
        </BlockMath>

        <Text>
          parameters, while maintaining the ability to detect features anywhere in the image.
        </Text>
      </div>
    </Container>
  );
}

export default Introduction;