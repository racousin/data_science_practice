import React from 'react';
import { Container, Title, Text, Stack, Box, Image, List, Group, Flex } from '@mantine/core';
import CodeBlock from 'components/CodeBlock';
import 'katex/dist/katex.min.css';
import { InlineMath, BlockMath } from 'react-katex';

const Segmentation = () => {
  return (
    <Container size="lg">
      <Title order={1} id="segmentation" mb="xl">Image Segmentation</Title>

      {/* Slide 1: Introduction */}
      <div data-slide>
        <Title order={2} mb="md">Introduction and Applications</Title>

        <Flex direction="column" align="center" mb="md">
          <Image
            src="/assets/data-science-practice/module7/segmentation-intro.png"
            alt="Image segmentation visualization showing different segmentation masks"
            style={{ maxWidth: 'min(600px, 70vw)', height: 'auto' }}
            fluid
            mb="sm"
          />
          <Text size="sm">
            Image segmentation assigns a class label to every pixel in the image
          </Text>
        </Flex>

        <Text mb="md">
          Image segmentation partitions an image into meaningful regions by assigning a label
          to every pixel. This dense prediction task provides pixel-level understanding of
          scene content.
        </Text>

        <Title order={3} mb="sm">Main Applications:</Title>
        <List mb="md">
          <List.Item>Medical imaging: tumor delineation, organ segmentation</List.Item>
          <List.Item>Autonomous driving: road, vehicle, pedestrian segmentation</List.Item>
          <List.Item>Satellite imagery: land use classification, building footprints</List.Item>
          <List.Item>Video editing: background removal, object isolation</List.Item>
          <List.Item>Agriculture: crop monitoring, disease detection</List.Item>
        </List>
      </div>

      {/* Slide 2: Types of Segmentation */}
      <div data-slide>
        <Title order={2} mb="md">Types of Segmentation</Title>

        <Flex direction="column" align="center" mb="md">
          <Image
            src="/assets/data-science-practice/module7/segmentation-types-comparison.jpg"
            alt="Comparison of semantic, instance, and panoptic segmentation"
            style={{ maxWidth: 'min(600px, 70vw)', height: 'auto' }}
            fluid
            mb="sm"
          />
          <Text size="sm">
            Comparison: semantic (classes only), instance (individual objects), panoptic (both)
          </Text>
        </Flex>

        <Group grow mb="md">
          <Box p="md">
            <Text weight={600} mb="xs">Semantic Segmentation</Text>
            <Text size="sm" mb="xs">
              Classifies each pixel into predefined categories. All instances of the same class
              receive the same label.
            </Text>
            <BlockMath>
              {`f: \\mathbb{R}^{H \\times W \\times C} \\rightarrow \\{0,\\ldots,K\\}^{H \\times W}`}
            </BlockMath>
          </Box>

          <Box p="md">
            <Text weight={600} mb="xs">Instance Segmentation</Text>
            <Text size="sm" mb="xs">
              Distinguishes between different instances of the same class. Each object instance
              receives a unique identifier.
            </Text>
            <BlockMath>
              {`f: \\mathbb{R}^{H \\times W \\times C} \\rightarrow \\{0,\\ldots,N\\}^{H \\times W}`}
            </BlockMath>
          </Box>
        </Group>

        <Box p="md" mt="md">
          <Text weight={600} mb="xs">Panoptic Segmentation</Text>
          <Text size="sm">
            Combines semantic and instance segmentation. Assigns both semantic category and
            instance ID to each pixel, handling both thing classes (countable objects) and
            stuff classes (uncountable regions).
          </Text>
        </Box>
      </div>

      {/* Slide 3: Problem Formulation */}
      <div data-slide>
        <Title order={2} mb="md">Problem Formulation: Semantic Segmentation</Title>

        <Text mb="md">
          Given an input image, predict a label map of the same spatial dimensions:
        </Text>

        <BlockMath>
          {`f: \\mathbb{R}^{H \\times W \\times C} \\rightarrow \\{0,\\ldots,K\\}^{H \\times W}`}
        </BlockMath>

        <Text mb="md">where:</Text>
        <List mb="md">
          <List.Item>Input: Image of height H, width W, channels C</List.Item>
          <List.Item>Output: Label map with K+1 classes (including background)</List.Item>
          <List.Item>Each pixel <InlineMath>{'(i,j)'}</InlineMath> receives exactly one label</List.Item>
        </List>

        <Text mb="md">
          The model typically outputs class probabilities per pixel:
        </Text>

        <BlockMath>
          {`\\hat{y}_{i,j,k} = P(\\text{class } k \\mid \\text{pixel } (i,j))`}
        </BlockMath>
      </div>

      {/* Slide 4: Data Format */}
      <div data-slide>
        <Title order={2} mb="md">Data Format: X and Y</Title>

        <Title order={3} mb="sm">Input (X):</Title>
        <Text mb="md">
          Images in standard formats, typically normalized. Common sizes:
          <InlineMath>{'\\mathbb{R}^{3 \\times 512 \\times 512}'}</InlineMath> or
          <InlineMath>{'\\mathbb{R}^{3 \\times 1024 \\times 2048}'}</InlineMath>.
        </Text>

        <Title order={3} mb="sm">Ground Truth (Y):</Title>
        <Text mb="md">
          Label maps matching input spatial dimensions. Each value represents a class index.
        </Text>

        <CodeBlock
          language="python"
          code={`# Example segmentation data
import torch

image = torch.randn(3, 512, 512)  # RGB image
mask = torch.randint(0, 21, (512, 512))  # 21 classes

# Mask values: 0 = background, 1-20 = object classes
# Each pixel gets exactly one label`}
        />
      </div>

      {/* Slide 5: Evaluation Metrics */}
      <div data-slide>
        <Title order={2} mb="md">Evaluation Metrics</Title>

        <Title order={3} mb="sm">Pixel Accuracy:</Title>
        <BlockMath>
          {`\\text{Accuracy} = \\frac{\\sum_{i,j} \\mathbb{1}[y_{i,j} = \\hat{y}_{i,j}]}{H \\times W}`}
        </BlockMath>

        <Title order={3} mb="sm">Intersection over Union (IoU):</Title>
        <Text mb="md">Per-class metric measuring overlap between prediction and ground truth:</Text>
        <BlockMath>
          {`\\text{IoU}_k = \\frac{|P_k \\cap G_k|}{|P_k \\cup G_k|}`}
        </BlockMath>

        <Title order={3} mb="sm">Mean IoU (mIoU):</Title>
        <Text mb="md">Average IoU across all classes:</Text>
        <BlockMath>
          {`\\text{mIoU} = \\frac{1}{K} \\sum_{k=1}^K \\text{IoU}_k`}
        </BlockMath>

        <Title order={3} mb="sm">Dice Coefficient:</Title>
        <Text mb="md">Commonly used in medical imaging:</Text>
        <BlockMath>
          {`\\text{Dice}_k = \\frac{2|P_k \\cap G_k|}{|P_k| + |G_k|}`}
        </BlockMath>

                <Flex direction="column" align="center" mb="md">
                  <Image
                    src="/assets/data-science-practice/module7/dicevsiou.png"
                    alt="Dice vs IoU comparison"
                    style={{ maxWidth: 'min(600px, 70vw)', height: 'auto' }}
                    fluid
                    mb="sm"
                  />
                  <Text size="sm">
                    Dice vs IoU: Comparison of two popular segmentation metrics
                  </Text>
              </Flex>
            </div>

      {/* Slide 6: Loss Functions */}
      <div data-slide>
        <Title order={2} mb="md">Loss Functions</Title>

        <Title order={3} mb="sm">Cross-Entropy Loss:</Title>
        <Text mb="md">
          Standard loss for pixel-wise classification:
        </Text>
        <BlockMath>
          {`\\mathcal{L}_{CE} = -\\frac{1}{HW} \\sum_{i=1}^H \\sum_{j=1}^W \\sum_{k=0}^K y_{i,j,k} \\log(\\hat{y}_{i,j,k})`}
        </BlockMath>

        <Title order={3} mb="sm">Weighted Cross-Entropy:</Title>
        <Text mb="md">
          Addresses class imbalance by weighting classes differently:
        </Text>
        <BlockMath>
          {`\\mathcal{L}_{WCE} = -\\frac{1}{HW} \\sum_{i,j} \\sum_{k=0}^K w_k \\cdot y_{i,j,k} \\log(\\hat{y}_{i,j,k})`}
        </BlockMath>

        <Title order={3} mb="sm">Dice Loss:</Title>
        <Text mb="md">
          Directly optimizes the Dice coefficient:
        </Text>
        <BlockMath>
          {`\\mathcal{L}_{Dice} = 1 - \\frac{2\\sum_{i,j} p_{i,j} g_{i,j}}{\\sum_{i,j} p_{i,j} + \\sum_{i,j} g_{i,j}}`}
        </BlockMath>

        <Title order={3} mb="sm">Focal Loss:</Title>
        <Text mb="md">
          Focuses on hard-to-classify pixels:
        </Text>
        <BlockMath>
          {`\\mathcal{L}_{Focal} = -\\frac{1}{HW} \\sum_{i,j,k} (1-\\hat{y}_{i,j,k})^\\gamma y_{i,j,k} \\log(\\hat{y}_{i,j,k})`}
        </BlockMath>
      </div>

      {/* Slide 7: FCN Architecture */}
      <div data-slide>
        <Title order={2} mb="md">Fully Convolutional Networks (FCN)</Title>

        <Text mb="md">
          FCN pioneered end-to-end learning for semantic segmentation by replacing fully
          connected layers with convolutional layers.
        </Text>

        <List mb="md">
          <List.Item>Encoder: Downsampling path extracts features</List.Item>
          <List.Item>Decoder: Upsampling path recovers spatial resolution</List.Item>
          <List.Item>Skip connections: Combine features from multiple scales</List.Item>
        </List>

        <CodeBlock
          language="python"
          code={`# FCN architecture
features = encoder(image)  # Downsample

# Upsample to original resolution
upsampled = decoder(features)

# Pixel-wise classification
output = classifier(upsampled)  # [B, K, H, W]`}
        />

        <Text mt="md">
          The output is a probability distribution over classes for each pixel.
        </Text>
      </div>

      {/* Slide 8: U-Net */}
      <div data-slide>
        <Title order={2} mb="md">U-Net Architecture</Title>

        <Flex direction="column" align="center" mb="md">
          <Image
            src="/assets/data-science-practice/module7/unet-architecture.png"
            alt="U-Net architecture diagram showing encoder-decoder with skip connections"
            style={{ maxWidth: 'min(600px, 70vw)', height: 'auto' }}
            fluid
            mb="sm"
          />
          <Text size="sm">
            U-Net architecture: symmetric encoder-decoder with skip connections for precise localization
          </Text>
        </Flex>

        <Text mb="md">
          U-Net is a widely-used architecture, particularly in medical imaging, featuring
          a symmetric encoder-decoder structure with skip connections.
        </Text>

        <List mb="md">
          <List.Item>Contracting path: Captures context through downsampling</List.Item>
          <List.Item>Expanding path: Enables precise localization through upsampling</List.Item>
          <List.Item>Skip connections: Concatenate encoder features with decoder features</List.Item>
        </List>

        <CodeBlock
          language="python"
          code={`# U-Net forward pass
# Encoder
e1 = encoder_block1(x)
e2 = encoder_block2(e1)
e3 = encoder_block3(e2)
e4 = encoder_block4(e3)

# Bottleneck
bottleneck = center_block(e4)

# Decoder with skip connections
d4 = decoder_block4(bottleneck, e4)
d3 = decoder_block3(d4, e3)
d2 = decoder_block2(d3, e2)
d1 = decoder_block1(d2, e1)

output = final_conv(d1)`}
        />
      </div>

      {/* Slide 9: DeepLab */}
      <div data-slide>
        <Title order={2} mb="md">DeepLab Series</Title>

        <Flex direction="column" align="center" mb="md">
          <Image
            src="/assets/data-science-practice/module7/deeplab-aspp.png"
            alt="DeepLab ASPP module with multiple atrous convolutions"
            style={{ maxWidth: 'min(600px, 70vw)', height: 'auto' }}
            fluid
            mb="sm"
          />
          <Text size="sm">
            DeepLab ASPP module: parallel atrous convolutions at different rates for multi-scale features
          </Text>
        </Flex>

        <Text mb="md">
          DeepLab introduced atrous (dilated) convolutions and spatial pyramid pooling for
          multi-scale segmentation.
        </Text>

        <Title order={3} mb="sm">Key Components:</Title>
        <List mb="md">
          <List.Item>Atrous convolutions: Expand receptive field without reducing resolution</List.Item>
          <List.Item>Atrous Spatial Pyramid Pooling (ASPP): Captures multi-scale context</List.Item>
          <List.Item>Encoder-decoder structure with feature alignment</List.Item>
        </List>

        <CodeBlock
          language="python"
          code={`# DeepLabv3+ ASPP module
def aspp(features):
    # Multiple parallel atrous convolutions
    conv1x1 = conv_1x1(features)
    conv3x3_r6 = atrous_conv(features, rate=6)
    conv3x3_r12 = atrous_conv(features, rate=12)
    conv3x3_r18 = atrous_conv(features, rate=18)
    pool = global_avg_pool(features)

    # Concatenate and fuse
    return concat([conv1x1, conv3x3_r6,
                   conv3x3_r12, conv3x3_r18, pool])`}
        />
      </div>

      {/* Slide 10: Transformer-Based Segmentation */}
      <div data-slide>
        <Title order={2} mb="md">Transformer-Based Segmentation</Title>

        <Text mb="md">
          Recent architectures leverage transformers for global context modeling.
        </Text>

        <Title order={3} mb="sm">SegFormer:</Title>
        <Text mb="md">
          Efficient transformer architecture combining hierarchical features.
        </Text>

        <CodeBlock
          language="python"
          code={`# SegFormer architecture
# Multi-scale transformer encoder
f1 = transformer_block1(x)    # 1/4 resolution
f2 = transformer_block2(f1)   # 1/8 resolution
f3 = transformer_block3(f2)   # 1/16 resolution
f4 = transformer_block4(f3)   # 1/32 resolution

# Lightweight all-MLP decoder
features = [f1, f2, f3, f4]
fused = mlp_decoder(features)
output = segmentation_head(fused)`}
        />

        <Title order={3} mb="sm">Mask2Former:</Title>
        <Text mb="md">
          Universal architecture for semantic, instance, and panoptic segmentation using
          masked attention.
        </Text>
      </div>

      {/* Slide 11: Instance Segmentation */}
      <div data-slide>
        <Title order={2} mb="md">Instance Segmentation</Title>

        <Flex direction="column" align="center" mb="md">
          <Image
            src="/assets/data-science-practice/module7/mask-rcnn-architecture.webp"
            alt="Mask R-CNN architecture with mask prediction branch"
            style={{ maxWidth: 'min(600px, 70vw)', height: 'auto' }}
            fluid
            mb="sm"
          />
          <Text size="sm">
            Mask R-CNN architecture: adds mask prediction branch to Faster R-CNN
          </Text>
        </Flex>

        <Text mb="md">
          Instance segmentation extends object detection by predicting pixel-level masks
          for each object instance.
        </Text>

        <Title order={3} mb="sm">Mask R-CNN:</Title>
        <Text mb="md">
          Extends Faster R-CNN by adding a mask prediction branch in parallel with the
          bounding box branch.
        </Text>

        <CodeBlock
          language="python"
          code={`# Mask R-CNN architecture
features = backbone(image)

# Region proposals
proposals = rpn(features)

# RoI features
roi_features = roi_align(features, proposals)

# Parallel heads
classes, boxes = detection_head(roi_features)
masks = mask_head(roi_features)  # [N, K, 28, 28]

# Resize masks to original resolution`}
        />

        <Text mt="md">
          Each detected instance gets a binary segmentation mask aligned to the bounding box.
        </Text>
      </div>

      {/* Slide 12: State-of-the-Art Models */}
      <div data-slide>
        <Title order={2} mb="md">State-of-the-Art Models</Title>

        <Text mb="md">Best models by task and requirements:</Text>

        <Group grow mb="md">
          <Box p="md">
            <Text weight={600} mb="xs">Semantic Segmentation</Text>
            <List size="sm">
              <List.Item>SegFormer</List.Item>
              <List.Item>DeepLabv3+</List.Item>
              <List.Item>HRNet</List.Item>
            </List>
            <Text size="sm" mt="xs">mIoU: 80-85% on Cityscapes</Text>
          </Box>

          <Box p="md">
            <Text weight={600} mb="xs">Instance Segmentation</Text>
            <List size="sm">
              <List.Item>Mask2Former</List.Item>
              <List.Item>Mask R-CNN</List.Item>
              <List.Item>SOLOv2</List.Item>
            </List>
            <Text size="sm" mt="xs">AP: 45-50% on COCO</Text>
          </Box>

          <Box p="md">
            <Text weight={600} mb="xs">Medical Imaging</Text>
            <List size="sm">
              <List.Item>nnU-Net (self-configuring)</List.Item>
              <List.Item>U-Net++</List.Item>
              <List.Item>TransUNet</List.Item>
            </List>
            <Text size="sm" mt="xs">Dice: 85-95% on medical datasets</Text>
          </Box>
        </Group>

        <Text mt="md">
          Model selection depends on domain, computational constraints, and required accuracy.
          Medical applications favor U-Net variants, while general vision tasks use transformer
          or DeepLab-based models.
        </Text>
      </div>

      {/* Slide 13: Implementation Example */}
      <div data-slide>
        <Title order={2} mb="md">Practical Implementation</Title>

        <CodeBlock
          language="python"
          code={`import torch
from torchvision.models.segmentation import deeplabv3_resnet50

# Load pre-trained model
model = deeplabv3_resnet50(pretrained=True)
model.eval()`}
        />

        <Text mb="md">Prepare input and run segmentation:</Text>

        <CodeBlock
          language="python"
          code={`from PIL import Image
from torchvision import transforms

# Load and preprocess
image = Image.open('image.jpg')
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
input_tensor = transform(image).unsqueeze(0)`}
        />

        <Text mb="md">Get predictions:</Text>

        <CodeBlock
          language="python"
          code={`with torch.no_grad():
    output = model(input_tensor)['out']

# Get class predictions per pixel
predictions = output.argmax(1).squeeze(0)
# Shape: [H, W], values in [0, num_classes-1]`}
        />
      </div>

    </Container>
  );
};

export default Segmentation;
