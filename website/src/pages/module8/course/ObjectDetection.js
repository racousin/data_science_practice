import React from 'react';
import { Container, Title, Text, Stack } from '@mantine/core';
import CodeBlock from 'components/CodeBlock';
import 'katex/dist/katex.min.css';
import { BlockMath } from 'react-katex';

export default function ObjectDetection() {
  const rcnnCode = `
class FasterRCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = models.resnet50(weights='ResNet50_Weights.IMAGENET1K_V2')
        in_features = self.backbone.fc.in_features
        
        # Remove fully connected layer
        self.backbone = nn.Sequential(
            *list(self.backbone.children())[:-2]
        )
        
        # RPN (Region Proposal Network)
        self.rpn = nn.Sequential(
            nn.Conv2d(2048, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 4 * 9, 1)  # 4 coordinates * 9 anchors
        )
        
        # ROI pooling and classification
        self.roi_pool = nn.AdaptiveMaxPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(2048 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes * 4)  # class-specific box regression
        )`;

  const yoloCode = `
class YOLOv5(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = nn.Sequential(
            # Darknet-inspired backbone
            ConvBlock(3, 32, kernel_size=6, stride=2),
            ConvBlock(32, 64, kernel_size=3, stride=2),
            ResBlock(64),
            ConvBlock(64, 128, kernel_size=3, stride=2),
            ResBlock(128),
            ConvBlock(128, 256, kernel_size=3, stride=2),
            ResBlock(256),
            ConvBlock(256, 512, kernel_size=3, stride=2),
        )
        
        self.head = nn.Sequential(
            nn.Conv2d(512, 512, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 3 * (5 + num_classes), 1)  # 3 scales, 5 box params
        )
    
    def forward(self, x):
        feat = self.backbone(x)
        return self.head(feat).reshape(x.shape[0], 3, -1, feat.shape[-2], feat.shape[-1])`;

  const evaluationCode = `
def calculate_iou(box1, box2):
    """Calculate Intersection over Union"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    return intersection / (area1 + area2 - intersection)

def calculate_map(pred_boxes, true_boxes, iou_threshold=0.5):
    """Calculate mean Average Precision"""
    aps = []
    for c in range(num_classes):
        preds = [b for b in pred_boxes if b['class'] == c]
        gts = [b for b in true_boxes if b['class'] == c]
        ap = average_precision(preds, gts, iou_threshold)
        aps.append(ap)
    return sum(aps) / len(aps)`;

  return (
    <Container size="lg">
      <Stack spacing="xl">
        <Title order={1}>Object Detection</Title>

        <div id="rcnn-family">
          <Title order={2}>R-CNN Family</Title>
          <Text>
            Region-based CNN architectures evolution:
            • R-CNN: Region proposals + CNN
            • Fast R-CNN: RoI pooling
            • Faster R-CNN: Region Proposal Network
          </Text>
          <BlockMath>
            {`L = L_{cls} + \\lambda L_{box}`}
          </BlockMath>
          <CodeBlock
            language="python"
            code={rcnnCode}
          />
        </div>

        <div id="yolo">
          <Title order={2}>YOLO Architecture</Title>
          <Text>
            Single-stage detection approach:
            • Grid-based prediction
            • Multiple scales
            • Anchor boxes
          </Text>
          <CodeBlock
            language="python"
            code={yoloCode}
          />
        </div>

        <div id="metrics">
          <Title order={2}>Evaluation Metrics</Title>
          <Text>
            Key metrics for object detection:
            • Intersection over Union (IoU)
            • Precision and Recall
            • Mean Average Precision (mAP)
          </Text>
          <CodeBlock
            language="python"
            code={evaluationCode}
          />
          <BlockMath>
            {`mAP = \\frac{1}{|C|} \\sum_{c \\in C} AP_c`}
          </BlockMath>
        </div>
      </Stack>
    </Container>
  );
}