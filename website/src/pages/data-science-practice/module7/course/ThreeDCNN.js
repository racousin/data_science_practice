import React from 'react';
import { Container, Title, Text, Box, List, Group, Image } from '@mantine/core';
import CodeBlock from 'components/CodeBlock';
import 'katex/dist/katex.min.css';
import { InlineMath, BlockMath } from 'react-katex';
import { Link } from 'react-router-dom';

const ThreeDCNN = () => {
  return (
    <Container size="lg">
      <Title order={1} id="3d-cnn" mb="xl">3D Convolutional Neural Networks</Title>

      {/* Slide 1: Introduction */}
      <div data-slide>
        <Title order={2} mb="md">From 2D to 3D Convolutions</Title>

        <Box mb="md">
          <Image
            src="/assets/data-science-practice/module7/2d-vs-3d-convolution.png"
            alt="Comparison of 2D and 3D convolution operations"
            mb="sm"
          />
          <Text size="sm">
            2D convolutions operate on spatial dimensions (H×W), 3D convolutions add temporal/depth dimension
          </Text>
        </Box>

        <Text mb="md">
          While 2D CNNs process spatial information in images, 3D CNNs extend convolution operations
          to three dimensions, enabling the processing of volumetric data (medical scans) or temporal
          sequences (videos).
        </Text>

        <Text mb="md">
          The key difference: 3D convolutions capture spatiotemporal features by sliding kernels
          across height, width, and time/depth dimensions simultaneously.
        </Text>
      </div>

      {/* Slide 2: Applications */}
      <div data-slide>
        <Title order={2} mb="md">Main Applications</Title>

        <Group grow mb="md">
          <Box p="md">
            <Text weight={600} mb="xs">Video Understanding</Text>
            <List size="sm">
              <List.Item>Action recognition in videos</List.Item>
              <List.Item>Video classification</List.Item>
              <List.Item>Temporal action detection</List.Item>
              <List.Item>Video captioning</List.Item>
            </List>
          </Box>

          <Box p="md">
            <Text weight={600} mb="xs">Medical Imaging</Text>
            <List size="sm">
              <List.Item>3D organ segmentation</List.Item>
              <List.Item>Tumor detection in CT/MRI scans</List.Item>
              <List.Item>Volumetric analysis</List.Item>
              <List.Item>Disease progression tracking</List.Item>
            </List>
          </Box>
        </Group>

        <Box p="md" mt="md">
          <Text weight={600} mb="xs">Other Applications</Text>
          <List size="sm">
            <List.Item>3D object detection from LiDAR point clouds</List.Item>
            <List.Item>Gesture recognition</List.Item>
            <List.Item>Molecular structure analysis</List.Item>
            <List.Item>Climate and weather prediction</List.Item>
          </List>
        </Box>
      </div>

      {/* Slide 3: Data Representation */}
      <div data-slide>
        <Title order={2} mb="md">Data Representation</Title>

        <Text mb="md" weight={500}>Video Data:</Text>
        <Text mb="md">
          Videos are represented as 5D tensors with dimensions:
        </Text>
        <BlockMath>
          {`X \\in \\mathbb{R}^{B \\times C \\times T \\times H \\times W}`}
        </BlockMath>

        <List mb="md">
          <List.Item>B: Batch size</List.Item>
          <List.Item>C: Number of channels (typically 3 for RGB)</List.Item>
          <List.Item>T: Temporal dimension (number of frames)</List.Item>
          <List.Item>H, W: Spatial dimensions (height, width)</List.Item>
        </List>

        <Text mb="md" weight={500}>Medical Volumetric Data:</Text>
        <Text mb="md">
          CT/MRI scans represented as 4D tensors:
        </Text>
        <BlockMath>
          {`X \\in \\mathbb{R}^{C \\times D \\times H \\times W}`}
        </BlockMath>

        <List mb="md">
          <List.Item>C: Channels (modality, e.g., different MRI sequences)</List.Item>
          <List.Item>D: Depth (number of slices)</List.Item>
          <List.Item>H, W: Spatial dimensions of each slice</List.Item>
        </List>
      </div>

      {/* Slide 4: 3D Convolution Operation */}
      <div data-slide>
        <Title order={2} mb="md">3D Convolution Operation</Title>

        <Box mb="md">
          <Image
            src="/assets/data-science-practice/module7/3d-convolution-operation.png"
            alt="3D convolution kernel sliding across spatiotemporal volume"
            mb="sm"
          />
          <Text size="sm">
            3D convolution kernel slides across all three dimensions simultaneously
          </Text>
        </Box>

        <Text mb="md">
          A 3D convolution extends the 2D operation to include the temporal or depth dimension:
        </Text>

        <BlockMath>
          {`G[i,j,t,c_{out}] = \\sigma\\left(\\sum_{c_{in}} \\sum_{k,l,\\tau} F[i+k, j+l, t+\\tau, c_{in}] \\cdot K[k,l,\\tau,c_{in},c_{out}] + b[c_{out}]\\right)`}
        </BlockMath>

        <Text mb="md">where:</Text>
        <List mb="md">
          <List.Item><InlineMath>{'K[k,l,\\tau,c_{in},c_{out}]'}</InlineMath>: 3D kernel of size <InlineMath>{'K_h \\times K_w \\times K_t'}</InlineMath></List.Item>
          <List.Item><InlineMath>{'\\tau'}</InlineMath>: Temporal dimension index</List.Item>
          <List.Item>The kernel captures both spatial and temporal patterns</List.Item>
        </List>

        <CodeBlock
          language="python"
          code={`import torch.nn as nn

# 3D convolution layer
conv3d = nn.Conv3d(
    in_channels=3,      # RGB
    out_channels=64,    # Feature maps
    kernel_size=(3,3,3), # temporal × height × width
    stride=(1,2,2),     # Different strides per dimension
    padding=(1,1,1)
)`}
        />
      </div>

      {/* Slide 5: Popular Architectures */}
      <div data-slide>
        <Title order={2} mb="md">3D CNN Architectures</Title>

        <Text mb="md" weight={500}>C3D (3D ConvNets):</Text>
        <Text mb="md">
          Applies 3D convolutions throughout the network. Uses 3×3×3 kernels consistently.
        </Text>

        <CodeBlock
          language="python"
          code={`# C3D-style architecture
model = nn.Sequential(
    nn.Conv3d(3, 64, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool3d(kernel_size=(1,2,2))  # Pool spatially, preserve time
)`}
        />

        <Text mb="md" weight={500}>I3D (Inflated 3D):</Text>
        <Text mb="md">
          Inflates 2D filters from pre-trained models (e.g., ImageNet) into 3D by repeating
          weights along the temporal dimension, enabling transfer learning.
        </Text>

        <Text mb="md" weight={500}>(2+1)D Convolutions:</Text>
        <Text mb="md">
          Factorizes 3D convolutions into separate spatial (2D) and temporal (1D) operations,
          reducing parameters while maintaining performance.
        </Text>

        <BlockMath>
          {`\\text{Conv3D} \\approx \\text{Conv2D}_{spatial} + \\text{Conv1D}_{temporal}`}
        </BlockMath>
      </div>

      {/* Slide 6: Computational Considerations */}
      <div data-slide>
        <Title order={2} mb="md">Computational Challenges</Title>

        <Text mb="md">
          3D CNNs are significantly more expensive than 2D CNNs:
        </Text>

        <Text mb="md" weight={500}>Parameter Count:</Text>
        <Text mb="md">
          A 3D kernel has more parameters than its 2D counterpart:
        </Text>
        <BlockMath>
          {`\\text{Params}_{3D} = K_t \\times K_h \\times K_w \\times C_{in} \\times C_{out}`}
        </BlockMath>

        <Text mb="md">
          For a 3×3×3 kernel vs 3×3 kernel with 64 input/output channels:
        </Text>
        <List mb="md">
          <List.Item>2D: 3 × 3 × 64 × 64 = 36,864 parameters</List.Item>
          <List.Item>3D: 3 × 3 × 3 × 64 × 64 = 110,592 parameters (3× more)</List.Item>
        </List>

        <Text mb="md" weight={500}>Solutions:</Text>
        <List mb="md">
          <List.Item>Factorized convolutions: (2+1)D reduces parameters</List.Item>
          <List.Item>Depthwise separable 3D convolutions</List.Item>
          <List.Item>Mixed precision training</List.Item>
          <List.Item>Temporal downsampling: process fewer frames</List.Item>
        </List>
      </div>

      {/* Slide 7: Metrics */}
      <div data-slide>
        <Title order={2} mb="md">Evaluation Metrics</Title>

        <Text mb="md">
          Metrics depend on the task:
        </Text>

        <Text mb="md" weight={500}>Video Classification:</Text>
        <List mb="md">
          <List.Item>Top-1/Top-5 accuracy</List.Item>
          <List.Item>Per-class accuracy</List.Item>
          <List.Item>Confusion matrices</List.Item>
        </List>

        <Text mb="md" weight={500}>Action Detection:</Text>
        <List mb="md">
          <List.Item>Temporal IoU (tIoU): Intersection over union of temporal segments</List.Item>
          <List.Item>Mean Average Precision (mAP) at various tIoU thresholds</List.Item>
        </List>

        <Text mb="md" weight={500}>Medical Volumetric Segmentation:</Text>
        <List mb="md">
          <List.Item>3D Dice coefficient</List.Item>
          <List.Item>Volumetric IoU</List.Item>
          <List.Item>Hausdorff distance (measures boundary accuracy)</List.Item>
        </List>
      </div>

      {/* Slide 8: Alternative Approaches */}
      <div data-slide>
        <Title order={2} mb="md">Alternative Temporal Modeling Approaches</Title>

        <Text mb="md">
          Beyond 3D CNNs, other architectures model temporal information:
        </Text>

        <Text mb="md" weight={500}>RNN/LSTM-Based (Legacy):</Text>
        <Text mb="md">
          Process video frame-by-frame through recurrent connections. Less common now due to
          training difficulties and sequential processing limitations.
        </Text>
        <Text mb="md">
          <Link to="/data-science-practice/module8/course" style={{ color: '#228be6' }}>
            See Module 8 (NLP/Sequences) for details on RNNs and LSTMs
          </Link>
        </Text>

        <Text mb="md" weight={500}>Transformer-Based (Current Trend):</Text>
        <Text mb="md">
          Apply self-attention mechanisms over space and time. Examples: ViViT, TimeSformer,
          Video Swin Transformer. These models treat video frames as sequences of patches.
        </Text>
        <Text mb="md">
          <Link to="/data-science-practice/module8/course" style={{ color: '#228be6' }}>
            See Module 8 for attention mechanisms and transformers
          </Link>
        </Text>

        <Text mb="md" weight={500}>Hybrid Approaches:</Text>
        <Text mb="md">
          Combine 3D CNNs for local spatiotemporal features with attention for long-range
          dependencies, achieving best of both worlds.
        </Text>
      </div>

      {/* Slide 9: Implementation Example */}
      <div data-slide>
        <Title order={2} mb="md">Practical Implementation</Title>

        <Text mb="md">Basic 3D CNN for video classification:</Text>

        <CodeBlock
          language="python"
          code={`import torch
import torch.nn as nn

class Simple3DCNN(nn.Module):
    def __init__(self, num_classes=101):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1,2,2))
        )`}
        />

        <Text mb="md">Continue with more layers:</Text>

        <CodeBlock
          language="python"
          code={`        # More 3D conv layers
        self.features.extend([
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2)  # Pool across all dims
        ])

        self.classifier = nn.Linear(128 * 4 * 7 * 7, num_classes)`}
        />

        <Text mb="md">Process video clips:</Text>

        <CodeBlock
          language="python"
          code={`# Input: video clip [batch, channels, frames, height, width]
video = torch.randn(4, 3, 16, 112, 112)  # 4 clips, 16 frames

model = Simple3DCNN(num_classes=101)
output = model(video)  # [4, 101] class scores`}
        />
      </div>

      {/* Slide 10: Best Practices */}
      <div data-slide>
        <Title order={2} mb="md">Best Practices and Tips</Title>

        <Text mb="md" weight={500}>Data Preprocessing:</Text>
        <List mb="md">
          <List.Item>Sample fixed-length clips from videos (e.g., 16 or 32 frames)</List.Item>
          <List.Item>Apply temporal augmentation: random cropping in time</List.Item>
          <List.Item>Normalize across spatial and temporal dimensions</List.Item>
        </List>

        <Text mb="md" weight={500}>Training Strategies:</Text>
        <List mb="md">
          <List.Item>Transfer learning: Initialize from 2D models when possible (I3D approach)</List.Item>
          <List.Item>Use smaller batch sizes due to memory constraints</List.Item>
          <List.Item>Gradient accumulation to simulate larger batches</List.Item>
          <List.Item>Mixed precision training (FP16) to reduce memory usage</List.Item>
        </List>

        <Text mb="md" weight={500}>Architecture Choices:</Text>
        <List mb="md">
          <List.Item>Start with (2+1)D for parameter efficiency</List.Item>
          <List.Item>Use temporal stride to reduce computation</List.Item>
          <List.Item>Consider two-stream networks: spatial + temporal pathways</List.Item>
        </List>
      </div>

    </Container>
  );
};

export default ThreeDCNN;
