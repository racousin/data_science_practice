import React from 'react';
import { Container, Title, Text, Box, List, Group, Image, Flex } from '@mantine/core';
import CodeBlock from 'components/CodeBlock';
import DataInteractionPanel from 'components/DataInteractionPanel';
import 'katex/dist/katex.min.css';
import { InlineMath, BlockMath } from 'react-katex';
import { Code } from 'lucide-react';

const ObjectDetection = () => {
  // Notebook URLs
  const notebookUrl = process.env.PUBLIC_URL + "/modules/data-science-practice/module7/course/module7_course_detection.ipynb";
  const notebookHtmlUrl = process.env.PUBLIC_URL + "/modules/data-science-practice/module7/course/module7_course_detection.html";
  const notebookColabUrl = process.env.PUBLIC_URL + "website/public/modules/data-science-practice/module7/course/module7_course_detection.ipynb";

  const metadata = {
    description: "Practical object detection using YOLOv8 - from pre-trained detection to fine-tuning on custom pedestrian dataset.",
    source: "COCO Dataset / Penn-Fudan Pedestrian Detection",
    target: "Object detection (bounding boxes + class labels)",
    listData: [
      { name: "Pre-trained YOLOv8", description: "Immediate detection on sample images (80 classes)" },
      { name: "Custom Fine-tuning", description: "Train pedestrian detector on Penn-Fudan dataset" },
      { name: "Deployment Ready", description: "Export to ONNX/TorchScript for production" }
    ],
  };

  return (
    <Container size="lg">
      <Title order={1} id="object-detection" mb="xl">Object Detection</Title>

      {/* Slide 1: Introduction */}
      <div data-slide>
        <Title order={2} mb="md">Introduction and Applications</Title>

        <Flex direction="column" align="center" mb="md">
          <Image
            src="/assets/data-science-practice/module7/object-detection-examples.jpeg"
            alt="Object detection examples showing bounding boxes on various objects"
            style={{ maxWidth: 'min(600px, 70vw)', height: 'auto' }}
            fluid
            mb="sm"
          />
          <Text size="sm">
            Object detection example: multiple objects detected with bounding boxes and class labels
          </Text>
        </Flex>

        <Text mb="md">
          Object detection extends image classification by not only identifying what objects are present
          in an image, but also localizing where they appear. This task is fundamental to numerous
          applications in computer vision.
        </Text>

        <Title order={3} mb="sm">Main Applications:</Title>
        <List mb="md">
          <List.Item>Autonomous vehicles: detecting pedestrians, vehicles, traffic signs</List.Item>
          <List.Item>Surveillance systems: identifying persons or suspicious objects</List.Item>
          <List.Item>Medical imaging: locating tumors or abnormalities</List.Item>
          <List.Item>Retail: automated checkout systems, inventory management</List.Item>
          <List.Item>Robotics: object manipulation and navigation</List.Item>
        </List>
      </div>

      {/* Slide 2: Problem Formulation */}
      <div data-slide>
        <Title order={2} mb="md">Problem Formulation</Title>

        <Text mb="md">
          Object detection maps an input image to a variable-length set of detections:
        </Text>

        <BlockMath>
          {`f: \\mathbb{R}^{H \\times W \\times C} \\rightarrow \\{(b_i, c_i, p_i)\\}_{i=1}^N`}
        </BlockMath>

        <Text mb="md">Each detection consists of:</Text>
        <List mb="md">
          <List.Item>
            <InlineMath>{'b_i \\in \\mathbb{R}^4'}</InlineMath>: Bounding box coordinates (x, y, width, height) or (x_min, y_min, x_max, y_max)
          </List.Item>
          <List.Item>
            <InlineMath>{'c_i \\in \\{1,\\ldots,K\\}'}</InlineMath>: Class label from K object categories
          </List.Item>
          <List.Item>
            <InlineMath>{'p_i \\in [0,1]'}</InlineMath>: Confidence score
          </List.Item>
        </List>

        <Text mb="md">
          The number of detections N varies per image and is determined by the model.
        </Text>
      </div>

      {/* Slide 3: Data Format */}
      <div data-slide>
        <Title order={2} mb="md">Data Format: X and Y</Title>

        <Title order={3} mb="sm">Input (X):</Title>
        <Text mb="md">
          Images of fixed or variable size, typically normalized to a standard input size.
          Common formats: <InlineMath>{'\\mathbb{R}^{3 \\times H \\times W}'}</InlineMath> with H, W in {`{224, 416, 512, 640}`}.
        </Text>

        <Title order={3} mb="sm">Ground Truth (Y):</Title>
        <Text mb="md">
          For each image, a set of annotations consisting of:
        </Text>
        <List mb="md">
          <List.Item>Bounding boxes in pixel coordinates</List.Item>
          <List.Item>Class labels for each bounding box</List.Item>
          <List.Item>Optional: difficulty flags, occlusion indicators</List.Item>
        </List>

        <CodeBlock
          language="python"
          code={`# Example ground truth format
annotations = {
    'boxes': [[50, 30, 200, 180], [300, 100, 450, 280]],
    'labels': [1, 3],  # class indices
    'image_id': 12345
}`}
        />
      </div>

      {/* Slide 4: Evaluation Metrics */}
      <div data-slide>
        <Title order={2} mb="md">Evaluation Metrics</Title>

        <Text mb="md">
          Object detection evaluation requires matching predicted boxes with ground truth boxes
          and assessing both localization and classification accuracy.
        </Text>

        <Flex direction="column" align="center" mb="md">
          <Image
            src="/assets/data-science-practice/module7/iou-visualization.ppm"
            alt="IoU visualization showing intersection and union of bounding boxes"
            style={{ maxWidth: 'min(600px, 70vw)', height: 'auto' }}
            fluid
            mb="sm"
          />
          <Text size="sm">
            IoU (Intersection over Union) visualization: ratio of overlapping area to total area
          </Text>
        </Flex>

        <Title order={3} mb="sm">Intersection over Union (IoU):</Title>
        <BlockMath>
          {`\\text{IoU}(B_{pred}, B_{gt}) = \\frac{\\text{Area}(B_{pred} \\cap B_{gt})}{\\text{Area}(B_{pred} \\cup B_{gt})}`}
        </BlockMath>

        <Text mb="md">
          A prediction is considered correct if IoU exceeds a threshold (typically 0.5).
        </Text>

        <Title order={3} mb="sm">Average Precision (AP):</Title>
        <Text mb="md">
          Computed from the precision-recall curve for each class. Mean Average Precision (mAP)
          averages AP across all classes. Variants include:
        </Text>
        <List mb="md">
          <List.Item>AP@0.5: IoU threshold at 0.5</List.Item>
          <List.Item>AP@0.75: IoU threshold at 0.75</List.Item>
          <List.Item>AP@[0.5:0.95]: Average across IoU thresholds from 0.5 to 0.95</List.Item>
        </List>
      </div>

      {/* Slide 5: Loss Functions */}
      <div data-slide>
        <Title order={2} mb="md">Loss Functions</Title>

        <Text mb="md">
          Object detection models typically optimize a multi-task loss combining localization
          and classification objectives:
        </Text>

        <BlockMath>
          {`\\mathcal{L}_{total} = \\lambda_{cls} \\mathcal{L}_{cls} + \\lambda_{box} \\mathcal{L}_{box} + \\lambda_{obj} \\mathcal{L}_{obj}`}
        </BlockMath>

        <Title order={3} mb="sm">Classification Loss:</Title>
        <Text mb="md">
          Cross-entropy loss for predicting object class:
        </Text>
        <BlockMath>
          {`\\mathcal{L}_{cls} = -\\sum_{i=1}^N \\sum_{k=1}^K y_{i,k} \\log(\\hat{p}_{i,k})`}
        </BlockMath>

        <Title order={3} mb="sm">Localization Loss:</Title>
        <Text mb="md">
          Measures difference between predicted and ground truth bounding boxes:
        </Text>
        <BlockMath>
          {`\\mathcal{L}_{box} = \\sum_{i=1}^N \\mathbb{1}_{obj}^{(i)} \\text{smooth}_{L1}(b_i - \\hat{b}_i)`}
        </BlockMath>

        <Title order={3} mb="sm">Objectness Loss:</Title>
        <Text mb="md">
          Binary loss indicating whether a region contains an object:
        </Text>
        <BlockMath>
          {`\\mathcal{L}_{obj} = -\\sum_{i=1}^N [y_i \\log(\\hat{o}_i) + (1-y_i) \\log(1-\\hat{o}_i)]`}
        </BlockMath>
      </div>

      {/* Slide 6: Two-Stage Detectors */}
      <div data-slide>
        <Title order={2} mb="md">Model Architecture: Two-Stage Detectors</Title>

        <Text mb="md">
          Two-stage detectors first generate region proposals, then classify and refine them.
        </Text>

        <Flex direction="column" align="center" mb="md">
          <Image
            src="/assets/data-science-practice/module7/faster-rcnn-architecture.jpg"
            alt="Faster R-CNN architecture diagram"
            style={{ maxWidth: 'min(600px, 70vw)', height: 'auto' }}
            fluid
            mb="sm"
          />
          <Text size="sm">
            Faster R-CNN architecture: RPN generates proposals, then ROI pooling and classification
          </Text>
        </Flex>

        <Title order={3} mb="sm">R-CNN Family (R-CNN, Fast R-CNN, Faster R-CNN):</Title>

        <List mb="md">
          <List.Item>Stage 1: Region Proposal Network (RPN) generates candidate boxes</List.Item>
          <List.Item>Stage 2: Each proposal is classified and its box refined</List.Item>
        </List>

        <CodeBlock
          language="python"
          code={`# Simplified Faster R-CNN forward pass
features = backbone(image)  # Extract features

# Stage 1: RPN generates proposals
proposals = rpn(features)  # ~2000 region proposals

# Stage 2: Classification and refinement
roi_features = roi_pooling(features, proposals)
classes, boxes = rcnn_head(roi_features)`}
        />

        <Text mt="md">
          Advantages: High accuracy. Disadvantages: Slower inference due to two-stage process.
        </Text>
      </div>

      {/* Slide 7: One-Stage Detectors */}
      <div data-slide>
        <Title order={2} mb="md">Model Architecture: One-Stage Detectors</Title>

        <Text mb="md">
          One-stage detectors predict classes and boxes directly from feature maps in a single pass.
        </Text>

        <Flex direction="column" align="center" mb="md">
          <Image
            src="/assets/data-science-practice/module7/yolo-grid.webp"
            alt="YOLO grid-based detection visualization"
            style={{ maxWidth: 'min(600px, 70vw)', height: 'auto' }}
            fluid
            mb="sm"
          />
          <Text size="sm">
            YOLO architecture: image divided into grid cells, each predicting bounding boxes
          </Text>
        </Flex>

        <Title order={3} mb="sm">YOLO (You Only Look Once):</Title>
        <Text mb="md">
          Divides image into grid cells. Each cell predicts bounding boxes and class probabilities.
        </Text>

        <CodeBlock
          language="python"
          code={`# YOLO detection
features = backbone(image)

# Direct prediction of boxes and classes
predictions = detection_head(features)
# Shape: [batch, num_anchors, grid_h, grid_w, 5+num_classes]
# 5 = (x, y, w, h, objectness)`}
        />

        <Title order={3} mb="sm">SSD (Single Shot Detector):</Title>
        <Text mb="md">
          Uses multiple feature maps at different scales for multi-scale detection.
        </Text>

        <Text mt="md">
          Advantages: Fast inference. Disadvantages: May sacrifice accuracy for speed.
        </Text>
      </div>

      {/* Slide 8: Anchor-Free Detectors */}
      <div data-slide>
        <Title order={2} mb="md">Modern Approaches: Anchor-Free Detectors</Title>

        <Text mb="md">
          Recent architectures eliminate predefined anchor boxes, predicting object centers
          and sizes directly.
        </Text>

        <Title order={3} mb="sm">FCOS (Fully Convolutional One-Stage):</Title>
        <List mb="md">
          <List.Item>Predicts object center location</List.Item>
          <List.Item>Regresses distances to bounding box edges</List.Item>
          <List.Item>Uses centerness score to suppress low-quality detections</List.Item>
        </List>

        <CodeBlock
          language="python"
          code={`# FCOS prediction per pixel
class_logits = cls_head(features)  # [B, K, H, W]
box_regression = box_head(features)  # [B, 4, H, W]
centerness = center_head(features)  # [B, 1, H, W]

# Convert to bounding boxes
boxes = decode_boxes(box_regression, locations)`}
        />
      </div>

      {/* Slide 9: Transformer-Based Detectors */}
      <div data-slide>
        <Title order={2} mb="md">Transformer-Based Detectors</Title>

        <Flex direction="column" align="center" mb="md">
          <Image
            src="/assets/data-science-practice/module7/detr-architecture.png"
            alt="DETR architecture with transformer encoder-decoder"
            style={{ maxWidth: 'min(600px, 70vw)', height: 'auto' }}
            fluid
            mb="sm"
          />
          <Text size="sm">
            DETR architecture: CNN backbone, transformer encoder-decoder, and parallel prediction heads
          </Text>
        </Flex>

        <Title order={3} mb="sm">DETR (DEtection TRansformer):</Title>
        <Text mb="md">
          Formulates object detection as a set prediction problem using transformers.
        </Text>

        <List mb="md">
          <List.Item>CNN backbone extracts features</List.Item>
          <List.Item>Transformer encoder processes feature map</List.Item>
          <List.Item>Transformer decoder outputs fixed set of predictions</List.Item>
          <List.Item>Bipartite matching between predictions and ground truth</List.Item>
        </List>

        <CodeBlock
          language="python"
          code={`# DETR architecture
features = cnn_backbone(image)
memory = transformer_encoder(features)

# Fixed set of learned object queries
queries = learned_queries  # [num_queries, dim]
predictions = transformer_decoder(queries, memory)

# Each prediction: class + box
classes = cls_head(predictions)  # [num_queries, K]
boxes = box_head(predictions)  # [num_queries, 4]`}
        />
      </div>

      {/* Slide 10: Best Models */}
      <div data-slide>
        <Title order={2} mb="md">State-of-the-Art Models</Title>

        <Text mb="md">
          Current best-performing models balance accuracy and speed:
        </Text>

        <Group grow mb="md">
          <Box p="md">
            <Text weight={600} mb="xs">High Accuracy</Text>
            <List size="sm">
              <List.Item>Cascade R-CNN with ResNeXt backbone</List.Item>
              <List.Item>DINO (transformer-based)</List.Item>
              <List.Item>Co-DETR</List.Item>
            </List>
            <Text size="sm" mt="xs">mAP: 55-60% on COCO</Text>
          </Box>

          <Box p="md">
            <Text weight={600} mb="xs">Balanced</Text>
            <List size="sm">
              <List.Item>YOLOv8</List.Item>
              <List.Item>EfficientDet</List.Item>
              <List.Item>RT-DETR</List.Item>
            </List>
            <Text size="sm" mt="xs">mAP: 45-55%, Real-time capable</Text>
          </Box>

          <Box p="md">
            <Text weight={600} mb="xs">High Speed</Text>
            <List size="sm">
              <List.Item>YOLOv8-nano/small</List.Item>
              <List.Item>MobileNet-SSD</List.Item>
              <List.Item>NanoDet</List.Item>
            </List>
            <Text size="sm" mt="xs">{`>100 FPS on GPU`}</Text>
          </Box>
        </Group>

        <Text mt="md">
          Choice depends on application requirements: accuracy-critical applications favor
          two-stage or transformer models, while real-time applications prefer optimized
          one-stage detectors.
        </Text>
      </div>

      {/* Slide 11: Implementation Example */}
      <div data-slide>
        <Title order={2} mb="md">Practical Implementation</Title>

        <CodeBlock
          language="python"
          code={`import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F

# Load pre-trained model
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()`}
        />

        <Text mb="md">Prepare input:</Text>

        <CodeBlock
          language="python"
          code={`from PIL import Image

image = Image.open('image.jpg')
image_tensor = F.to_tensor(image)`}
        />

        <Text mb="md">Run detection:</Text>

        <CodeBlock
          language="python"
          code={`with torch.no_grad():
    predictions = model([image_tensor])

boxes = predictions[0]['boxes']
labels = predictions[0]['labels']
scores = predictions[0]['scores']`}
        />

        <Text mt="md">
          Filter predictions by confidence threshold and apply non-maximum suppression (NMS)
          to remove overlapping detections.
        </Text>
      </div>

      {/* Slide 12: Practical Application */}
      <div data-slide>
        <Title order={2} mb="md">
          <Code className="inline-block mr-2" size={24} />
          Practical Application: YOLOv8 Object Detection
        </Title>
        <Text mb="md">
          This notebook demonstrates a complete, production-ready object detection workflow using YOLOv8.
          Start with pre-trained detection, then fine-tune on a custom pedestrian dataset - fast training and excellent results.
        </Text>
        <DataInteractionPanel
          notebookUrl={notebookUrl}
          notebookHtmlUrl={notebookHtmlUrl}
          notebookColabUrl={notebookColabUrl}
          metadata={metadata}
        />
      </div>

    </Container>
  );
};

export default ObjectDetection;
