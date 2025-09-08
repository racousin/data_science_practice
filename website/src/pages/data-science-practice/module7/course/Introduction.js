import React from 'react';
import { Container, Title, Text, Stack, Group, Box, Image } from '@mantine/core';
import CodeBlock from 'components/CodeBlock';
import 'katex/dist/katex.min.css';
import { InlineMath, BlockMath } from 'react-katex';
import ConvolutionExplanation from './Introduction/ConvolutionExplanation.js'

const TaskBox = ({ title, formula, description }) => (
  <Box className="bg-white p-4 rounded-lg shadow-sm border border-gray-200">
    <Title order={4} className="mb-2">{title}</Title>
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
      <Stack spacing="xl">
        <Title order={1} id="introduction">Understanding Images in Deep Learning</Title>
        
        {/* Digital Image Representation */}
        <div>
          <Title order={2} id="digital-representation">Digital Image Representation</Title>
          <Group spacing="xs" className="bg-gray-100 p-4 rounded-lg">
            <Stack spacing={0}>
              <Image src="/assets/data-science-practice/module7/gray.png" alt="Grayscale representation" />
              <Text size="sm" mt="xs">
                Single channel grayscale image: each pixel represents intensity from 0 (black) to max value 8 (white)
              </Text>
            </Stack>
          </Group>
          
          <Text>
            At its core, a digital image is a structured grid of numerical values. Each point in this grid,
            called a pixel, contains information about color intensity.
          </Text>

          {/* Bit Depth Explanation */}
          <Title order={3}>Color Depth</Title>
          <Group spacing="xs" className="bg-gray-100 p-4 rounded-lg">
            <Stack spacing={0}>
              <Image src="/assets/data-science-practice/module7/bit-depth-representation.png" alt="Bit depth visualization" />
              <Text size="sm" mt="xs">
                Bit depth determines the range of possible values:
                • 2-bit: 2 levels (0-1)
                • 8-bit: 256 levels (0-255)
                • 16-bit: 65,536 levels (0-65535)
              </Text>
            </Stack>
          </Group>

          {/* Color Channels */}
          <Title order={3}>Color Channels</Title>
          <Group spacing="xs" className="bg-gray-100 p-4 rounded-lg">
            <Stack spacing={0}>
              <Image src="/assets/data-science-practice/module7/rgb.png" alt="RGB channel separation" />
              <Text size="sm" mt="xs">
                RGB images consist of three channels, each representing the intensity of Red, Green, and Blue components (usually on 8-bit)
              </Text>
            </Stack>
          </Group>

          {/* Multi-spectral Imagery */}
          <Title order={3}>Extended Channel Applications</Title>
          <Group spacing="xs" className="bg-gray-100 p-4 rounded-lg">
            <Stack spacing={0}>
              <Image src="/assets/data-science-practice/module7/satelite.png" alt="Multi-spectral satellite imagery" />
              <Text size="sm" mt="xs">
                Images can contain multiple layers beyond RGB:
                • Satellite imagery: Near Infrared, Thermal, Altitude, etc.
                • Medical imaging: Different sensor readings
              </Text>
            </Stack>
          </Group>

          <Text>
            Mathematically, a general image is represented as a H × W × C tensor, where:
            • H: Height in pixels
            • W: Width in pixels
            • C: Number of channels
          </Text>
          
          <BlockMath>
            {`I \\in \\mathbb{R}^{H \\times W \\times C}`}
          </BlockMath>
        </div>

        {/* ML Formalism */}

        <Title order={2} id="ml-tasks">Machine Learning Tasks with Images</Title>
      <Stack spacing="xl" className="mt-4">
        <Text>
          Images can serve as both input and output in various machine learning tasks. 
        </Text>

        {/* Image to Value Tasks */}
        <div>
          <Title order={3} className="mb-3">Image to Value Tasks</Title>
          <Group grow>
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
        {/* Image to Structured Output */}
        <div>
          <Title order={3} className="mb-3">Image to Structured Output</Title>
          <Group grow>
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

        {/* Image to Image Tasks */}
        <div>
          <Title order={3} className="mb-3">Image to Image Tasks</Title>
          <Group grow>
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
        {/* Value to Image Tasks */}
        <div>
          <Title order={3} className="mb-3">Value/Sequence to Image Tasks</Title>
          <Group grow>
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

      </Stack>



        {/* Convolutions */}
        <div>
        <Title order={2} id="convolutions">Leveraging Spatial Structure with Convolutions</Title>
          <Stack spacing="md">
            <Text>
              While we could flatten an image into a 1D vector (
              <InlineMath>{`\\mathbb{R}^{H \\times W \\times C} \\rightarrow \\mathbb{R}^{HWC}`}</InlineMath>
              ), this loses crucial spatial information. Convolutions preserve and exploit this structure.
            </Text>

            <Text>
              A convolution operation slides a kernel (small matrix) across the image, computing weighted sums
              of local regions:
            </Text>
            
            <BlockMath>
              {`(f * g)[n] = \\sum_{m=-\\infty}^{\\infty} f[m]g[n-m]`}
            </BlockMath>

            <CodeBlock
              language="python"
              code={convolutionCode}
            />

          <Image src="/assets/data-science-practice/module7/conv2.png" alt="conv" />
          <Text>
            Concertly, a 2D convolution is the operation that computes a weighted sum between 
            two matrices: an input matrix (typically an image or feature map) and a kernel 
            (also called a filter).       The operation is defined as:
          </Text>
          
          <div className="p-4 bg-white rounded-md">
            <BlockMath>
              {`G[i, j, c_{out}] = \\sigma\\left(\\sum_{c_{in}=0}^{C_{in}-1} \\sum_{k=0}^{K_h-1} \\sum_{l=0}^{K_w-1} F[s_h \\cdot i + k - p_h, s_w \\cdot j + l - p_w, c_{in}] \\cdot K[k, l, c_{in}, c_{out}] + b[c_{out}]\\right)`}
            </BlockMath>
            <Text className="text-sm text-gray-600 mt-2">
              where:
            </Text>
            <ul className="list-disc ml-6 text-sm text-gray-600">
              <li><InlineMath>{"G[i,j,c_{out}]"}</InlineMath> is the output at position <InlineMath>{"(i,j)"}</InlineMath> for output channel <InlineMath>{"c_{out}"}</InlineMath></li>
              <li><InlineMath>{"F"}</InlineMath> is the input tensor (with padding applied)</li>
              <li><InlineMath>K</InlineMath> is the kernel/filter tensor</li>
              <li><InlineMath>\sigma</InlineMath> is the activation function (e.g., ReLU, sigmoid)</li>
              <li><InlineMath>{"C_{in}"}</InlineMath> is the number of input channels</li>
              <li><InlineMath>K_h, K_w</InlineMath> are kernel height and width</li>
              <li><InlineMath>s_h, s_w</InlineMath> are stride values for height and width directions</li>
              <li><InlineMath>{"p_h, p_w"}</InlineMath> are padding values for height and width</li>
              <li><InlineMath>{"b[c_{out}]"}</InlineMath> is the bias term for output channel <InlineMath>{"c_{out}"}</InlineMath></li>
            </ul>
          </div>

          </Stack>
        </div>

        <div>
        <Title order={2} id="cnn-architecture">Understanding Convolutional Neural Networks</Title>
      <Stack spacing="md" className="mt-4">
        <Text>
          Convolutional Neural Networks (CNNs) are specialized deep learning models that process grid-like data using convolutional layers. Unlike traditional neural networks, CNNs automatically learn spatial hierarchies of features through multiple layers of convolutions.
        </Text>

        <Image src="/assets/data-science-practice/module7/cnn-network.jpg" alt="conv" />

        <Text>
          The backpropagation process in CNNs follows the same principles as in fully connected networks, but with kernel weight sharing and local connectivity constraints.
        </Text>

        <Text>
          Each convolutional layer learns progressively more complex features:
        </Text>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 my-4">
          <div className="bg-white p-4">
            <Text className="font-semibold">Early Layers</Text>
            <Text className="text-sm">Detect basic features like edges and corners</Text>
          </div>
          <div className="bg-white p-4">
            <Text className="font-semibold">Middle Layers</Text>
            <Text className="text-sm">Combine basic features into patterns and textures</Text>
          </div>
          <div className="bg-white p-4">
            <Text className="font-semibold">Deep Layers</Text>
            <Text className="text-sm">Recognize complex objects and their arrangements</Text>
          </div>
        </div>


        </Stack>
        </div>
        {/* Parameter Efficiency */}
        <div>

          
        <Title order={2} id="efficiency">Convolutional vs. Fully Connected Layers</Title>
          <Text>
              Key advantages of convolutions:
            </Text>
            <Stack spacing="xs">
              <Text>• Preserve spatial relationships between pixels</Text>
              <Text>• Parameter sharing reduces model complexity</Text>
            </Stack>
          <Stack spacing="md">
            <Text>
              Consider an image of size 224×224×3. A fully connected layer would require in first layer:
            </Text>
            <BlockMath>
              {`224 \\times 224 \\times 3 \\times N_{neurons} = 150,528 \\times N_{neurons}`}
            </BlockMath>
            <Text>
              In contrast, a convolutional layer with 64 filters of size 3×3 only needs in first layer:
            </Text>
            <BlockMath>
              {`3 \\times 3 \\times 3 \\times 64 = 1,728`}
            </BlockMath>
            <Text>
              parameters, while maintaining the ability to detect features anywhere in the image.
            </Text>
          </Stack>
        </div>
      </Stack>
    </Container>
  );
}

export default Introduction;