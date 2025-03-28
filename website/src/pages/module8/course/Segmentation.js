import React from 'react';
import { Container, Title, Text, Stack } from '@mantine/core';
import CodeBlock from 'components/CodeBlock';
import 'katex/dist/katex.min.css';
import { BlockMath } from 'react-katex';

export default function Segmentation() {
  const semanticCode = `
class UNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.encoder = nn.ModuleList([
            self._make_layer(3, 64),
            self._make_layer(64, 128),
            self._make_layer(128, 256),
            self._make_layer(256, 512),
        ])
        
        self.decoder = nn.ModuleList([
            self._make_decoder_layer(512, 256),
            self._make_decoder_layer(256, 128),
            self._make_decoder_layer(128, 64),
            nn.Conv2d(64, n_classes, 1)
        ])
    
    def _make_layer(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def _make_decoder_layer(self, in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2),
            self._make_layer(out_ch, out_ch)
        )`;

  const instanceCode = `
class MaskRCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = models.resnet50(weights='ResNet50_Weights.IMAGENET1K_V2')
        self.fpn = FeaturePyramidNetwork()
        self.rpn = RegionProposalNetwork()
        self.box_head = BoundingBoxHead(num_classes)
        self.mask_head = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 2, 2),
            nn.ReLU(),
            nn.ConvTranspose2d(256, num_classes, 2, 2)
        )
    
    def forward(self, x):
        features = self.fpn(self.backbone(x))
        proposals = self.rpn(features)
        boxes = self.box_head(features, proposals)
        masks = self.mask_head(features, boxes)
        return boxes, masks`;

  const evaluationCode = `
def evaluate_segmentation(pred_mask, true_mask):
    """Calculate segmentation metrics"""
    intersection = torch.logical_and(pred_mask, true_mask).sum()
    union = torch.logical_or(pred_mask, true_mask).sum()
    
    # IoU
    iou = intersection / union
    
    # Dice coefficient
    dice = 2 * intersection / (pred_mask.sum() + true_mask.sum())
    
    # Pixel accuracy
    accuracy = (pred_mask == true_mask).float().mean()
    
    return {
        'iou': iou.item(),
        'dice': dice.item(),
        'accuracy': accuracy.item()
    }`;

  return (
    <Container size="lg">
      <Stack spacing="xl">
        <Title order={1}>Image Segmentation</Title>

        <div id="semantic">
          <Title order={2}>Semantic Segmentation</Title>
          <Text>
            Pixel-wise classification using U-Net architecture:
          </Text>
          <BlockMath>
            {`L = -\\frac{1}{N}\\sum_{i=1}^N\\sum_{c=1}^C y_{ic}\\log(p_{ic})`}
          </BlockMath>
          <CodeBlock
            language="python"
            code={semanticCode}
          />
        </div>

        <div id="instance">
          <Title order={2}>Instance Segmentation</Title>
          <Text>
            Mask R-CNN implementation for instance-level segmentation:
          </Text>
          <CodeBlock
            language="python"
            code={instanceCode}
          />
        </div>

        <div id="evaluation">
          <Title order={2}>Performance Evaluation</Title>
          <Text>
            Key metrics for segmentation evaluation:
            • Intersection over Union (IoU)
            • Dice coefficient
            • Pixel accuracy
            • Mean IoU across classes
          </Text>
          <CodeBlock
            language="python"
            code={evaluationCode}
          />
          <BlockMath>
            {`Dice = \\frac{2|X \\cap Y|}{|X| + |Y|}`}
          </BlockMath>
        </div>
      </Stack>
    </Container>
  );
}