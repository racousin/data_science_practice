import React from 'react';
import { Container, Title, Text, Stack, Grid, Paper, Code, List } from '@mantine/core';
import { InlineMath, BlockMath } from 'react-katex';
import 'katex/dist/katex.min.css';

const ConvolutionalNetworks = () => {
  return (
    <Container size="xl" py="xl">
      <Stack spacing="xl">
        
        {/* Slide 1: Title and Introduction */}
        <div data-slide className="min-h-[500px] flex flex-col justify-center">
          <Title order={1} className="text-center mb-8">
            Convolutional Neural Networks
          </Title>
          <Text size="xl" className="text-center mb-6">
            Deep Learning for Visual Data
          </Text>
          <div className="max-w-3xl mx-auto">
            <Paper className="p-6 bg-blue-50">
              <Text size="lg" mb="md">
                Convolutional Neural Networks (CNNs) are specialized architectures designed for processing 
                grid-like data such as images. They leverage spatial locality and parameter sharing to 
                efficiently learn visual patterns.
              </Text>
              <List>
                <List.Item>Convolutional layers and feature maps</List.Item>
                <List.Item>Pooling operations and spatial reduction</List.Item>
                <List.Item>CNN architectures and design patterns</List.Item>
                <List.Item>Transfer learning and pre-trained models</List.Item>
              </List>
            </Paper>
          </div>
        </div>

        {/* Slide 2: Convolution Fundamentals */}
        <div data-slide className="min-h-[500px]" id="convolution-fundamentals">
          <Title order={2} className="mb-6">Convolution Operation</Title>
          
          <Paper className="p-6 bg-gray-50 mb-6">
            <Text size="lg">
              Convolution applies a learnable filter (kernel) across the input to produce feature maps.
              This operation preserves spatial relationships while reducing the number of parameters.
            </Text>
            
            <Text className="mt-4">
              <strong>Mathematical Definition:</strong> For input <InlineMath>{`\\mathbf{X} \\in \\mathbb{R}^{H \\times W}`}</InlineMath> and kernel <InlineMath>{`\\mathbf{K} \\in \\mathbb{R}^{k \\times k}`}</InlineMath>:
            </Text>
            <BlockMath>{`(\\mathbf{X} * \\mathbf{K})_{i,j} = \\sum_{m=0}^{k-1} \\sum_{n=0}^{k-1} \\mathbf{X}_{i+m,j+n} \\cdot \\mathbf{K}_{m,n}`}</BlockMath>
          </Paper>
          
          <Grid gutter="lg">
            <Grid.Col span={6}>
              <Paper className="p-4 bg-green-50">
                <Title order={4} mb="sm">Basic Convolution Layer</Title>
                <Code block language="python">{`import torch
import torch.nn as nn
import torch.nn.functional as F

# Basic 2D Convolution
conv_layer = nn.Conv2d(
    in_channels=3,    # RGB input
    out_channels=64,  # Number of filters
    kernel_size=3,    # 3x3 filter
    stride=1,         # Step size
    padding=1         # Zero padding
)

# Input: (batch_size, 3, 224, 224)
# Output: (batch_size, 64, 224, 224)

# Manual convolution
def conv2d_manual(input_tensor, kernel, stride=1, padding=0):
    if padding > 0:
        input_tensor = F.pad(input_tensor, (padding,) * 4)
    
    batch_size, in_channels, in_height, in_width = input_tensor.shape
    out_channels, _, kernel_height, kernel_width = kernel.shape
    
    out_height = (in_height - kernel_height) // stride + 1
    out_width = (in_width - kernel_width) // stride + 1
    
    output = torch.zeros(batch_size, out_channels, out_height, out_width)
    
    for i in range(0, out_height * stride, stride):
        for j in range(0, out_width * stride, stride):
            region = input_tensor[:, :, i:i+kernel_height, j:j+kernel_width]
            output[:, :, i//stride, j//stride] = torch.sum(
                region.unsqueeze(1) * kernel.unsqueeze(0), dim=(2, 3, 4)
            )
    
    return output`}</Code>
              </Paper>
            </Grid.Col>
            
            <Grid.Col span={6}>
              <Paper className="p-4 bg-blue-50">
                <Title order={4} mb="sm">Convolution Parameters</Title>
                <Code block language="python">{`# Different convolution configurations
conv_configs = [
    # Standard convolution
    nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
    
    # Stride 2 for downsampling
    nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
    
    # 1x1 convolution for channel reduction
    nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0),
    
    # Large kernel for receptive field
    nn.Conv2d(64, 32, kernel_size=7, stride=1, padding=3),
    
    # Dilated convolution
    nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=2, dilation=2)
]

# Calculate output size
def conv_output_size(input_size, kernel_size, stride, padding, dilation=1):
    effective_kernel = dilation * (kernel_size - 1) + 1
    return (input_size + 2 * padding - effective_kernel) // stride + 1

# Example: 224x224 input, 3x3 kernel, stride=1, padding=1
output_size = conv_output_size(224, 3, 1, 1)  # 224`}</Code>
              </Paper>
            </Grid.Col>
          </Grid>
        </div>

        {/* Slide 3: Pooling Operations */}
        <div data-slide className="min-h-[500px]" id="pooling-operations">
          <Title order={2} className="mb-6">Pooling Operations</Title>
          
          <Grid gutter="lg">
            <Grid.Col span={6}>
              <Paper className="p-4 bg-purple-50">
                <Title order={4} mb="sm">Max and Average Pooling</Title>
                <Code block language="python">{`# Max pooling
max_pool = nn.MaxPool2d(
    kernel_size=2, 
    stride=2, 
    padding=0
)

# Average pooling
avg_pool = nn.AvgPool2d(
    kernel_size=2, 
    stride=2, 
    padding=0
)

# Adaptive pooling (output size independent of input)
adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))

# Global average pooling
global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

# Example usage
x = torch.randn(32, 64, 56, 56)
max_pooled = max_pool(x)  # (32, 64, 28, 28)
avg_pooled = avg_pool(x)  # (32, 64, 28, 28)
adaptive_pooled = adaptive_pool(x)  # (32, 64, 7, 7)
global_pooled = global_avg_pool(x)  # (32, 64, 1, 1)`}</Code>
              </Paper>
            </Grid.Col>
            
            <Grid.Col span={6}>
              <Paper className="p-4 bg-orange-50">
                <Title order={4} mb="sm">Custom Pooling Operations</Title>
                <Code block language="python">{`class StochasticPooling(nn.Module):
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
    
    def forward(self, x):
        if self.training:
            # Stochastic pooling during training
            return F.avg_pool2d(x, self.kernel_size, self.stride)
        else:
            # Max pooling during inference
            return F.max_pool2d(x, self.kernel_size, self.stride)

class MixedPooling(nn.Module):
    def __init__(self, kernel_size, stride, alpha=0.5):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.alpha = alpha
    
    def forward(self, x):
        max_pool = F.max_pool2d(x, self.kernel_size, self.stride)
        avg_pool = F.avg_pool2d(x, self.kernel_size, self.stride)
        return self.alpha * max_pool + (1 - self.alpha) * avg_pool`}</Code>
              </Paper>
            </Grid.Col>
          </Grid>
        </div>

        {/* Slide 4: CNN Architectures */}
        <div data-slide className="min-h-[500px]" id="cnn-architectures">
          <Title order={2} className="mb-6">Classic CNN Architectures</Title>
          
          <Grid gutter="lg">
            <Grid.Col span={12}>
              <Paper className="p-4 bg-yellow-50 mb-4">
                <Title order={4} mb="sm">LeNet-5 Style Architecture</Title>
                <Code block language="python">{`class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv_layers = nn.Sequential(
            # First convolutional block
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Second convolutional block
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.classifier(x)
        return x`}</Code>
              </Paper>
            </Grid.Col>
            
            <Grid.Col span={6}>
              <Paper className="p-4 bg-green-50">
                <Title order={4} mb="sm">VGG-Style Architecture</Title>
                <Code block language="python">{`class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_convs):
        super().__init__()
        layers = []
        for i in range(num_convs):
            layers.extend([
                nn.Conv2d(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1
                ),
                nn.ReLU()
            ])
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)

class VGG(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.features = nn.Sequential(
            VGGBlock(3, 64, 2),    # VGG Block 1
            VGGBlock(64, 128, 2),  # VGG Block 2
            VGGBlock(128, 256, 3), # VGG Block 3
            VGGBlock(256, 512, 3), # VGG Block 4
            VGGBlock(512, 512, 3), # VGG Block 5
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )`}</Code>
              </Paper>
            </Grid.Col>
            
            <Grid.Col span={6}>
              <Paper className="p-4 bg-blue-50">
                <Title order={4} mb="sm">ResNet Block</Title>
                <Code block language="python">{`class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out`}</Code>
              </Paper>
            </Grid.Col>
          </Grid>
        </div>

        {/* Slide 5: Modern CNN Techniques */}
        <div data-slide className="min-h-[500px]" id="modern-techniques">
          <Title order={2} className="mb-6">Modern CNN Techniques</Title>
          
          <Grid gutter="lg">
            <Grid.Col span={6}>
              <Paper className="p-4 bg-red-50">
                <Title order={4} mb="sm">Separable Convolutions</Title>
                <Code block language="python">{`class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        # Depthwise convolution
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=False
        )
        # Pointwise convolution
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.depthwise(x)))
        x = F.relu(self.bn2(self.pointwise(x)))
        return x

# Group convolution example
group_conv = nn.Conv2d(64, 128, kernel_size=3, padding=1, groups=4)`}</Code>
              </Paper>
            </Grid.Col>
            
            <Grid.Col span={6}>
              <Paper className="p-4 bg-purple-50">
                <Title order={4} mb="sm">Squeeze-and-Excitation</Title>
                <Code block language="python">{`class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.shape
        # Global pooling
        y = self.global_pool(x).view(b, c)
        # Excitation
        y = self.fc(y).view(b, c, 1, 1)
        # Scale original features
        return x * y.expand_as(x)

class SEBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//4, 1, bias=False),
            nn.BatchNorm2d(out_channels//4),
            nn.ReLU(),
            nn.Conv2d(out_channels//4, out_channels//4, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels//4),
            nn.ReLU(),
            nn.Conv2d(out_channels//4, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.se_block = SEBlock(out_channels)
        
    def forward(self, x):
        out = self.conv_block(x)
        out = self.se_block(out)
        return F.relu(out)`}</Code>
              </Paper>
            </Grid.Col>
          </Grid>
        </div>

        {/* Slide 6: Transfer Learning */}
        <div data-slide className="min-h-[500px]" id="transfer-learning">
          <Title order={2} className="mb-6">Transfer Learning with Pre-trained Models</Title>
          
          <Grid gutter="lg">
            <Grid.Col span={12}>
              <Paper className="p-4 bg-indigo-50 mb-4">
                <Title order={4} mb="sm">Using Torchvision Pre-trained Models</Title>
                <Code block language="python">{`import torchvision.models as models
import torch.nn as nn

# Load pre-trained ResNet
model = models.resnet50(pretrained=True)

# Freeze all parameters
for param in model.parameters():
    param.requires_grad = False

# Replace classifier for new task
num_classes = 10  # Your number of classes
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Option 1: Fine-tune only the classifier
# Option 2: Fine-tune last few layers
for param in model.layer4.parameters():
    param.requires_grad = True

# Option 3: Fine-tune entire network with lower learning rate
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

# Custom classifier head
class CustomHead(nn.Module):
    def __init__(self, in_features, num_classes, dropout=0.5):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.classifier(x)

# Replace with custom head
model.fc = CustomHead(model.fc.in_features, num_classes)`}</Code>
              </Paper>
            </Grid.Col>
            
            <Grid.Col span={6}>
              <Paper className="p-4 bg-green-50">
                <Title order={4} mb="sm">Different Learning Rates</Title>
                <Code block language="python">{`# Different learning rates for different parts
backbone_params = []
classifier_params = []

for name, param in model.named_parameters():
    if 'fc' in name:  # Classifier parameters
        classifier_params.append(param)
    else:  # Backbone parameters
        backbone_params.append(param)

optimizer = torch.optim.SGD([
    {'params': backbone_params, 'lr': 1e-4},
    {'params': classifier_params, 'lr': 1e-3}
], momentum=0.9)`}</Code>
              </Paper>
            </Grid.Col>
            
            <Grid.Col span={6}>
              <Paper className="p-4 bg-yellow-50">
                <Title order={4} mb="sm">Progressive Unfreezing</Title>
                <Code block language="python">{`def unfreeze_layers(model, num_layers_to_unfreeze):
    # Get all layer names in reverse order
    layer_names = [name for name, _ in model.named_parameters()]
    layer_names.reverse()
    
    # Unfreeze specified number of layers
    unfrozen_count = 0
    for name in layer_names:
        param = dict(model.named_parameters())[name]
        param.requires_grad = True
        unfrozen_count += 1
        if unfrozen_count >= num_layers_to_unfreeze:
            break

# Progressive training schedule
def progressive_training(model, train_loader, epochs_per_stage):
    stages = [10, 20, 30]  # Number of layers to unfreeze
    
    for stage, num_layers in enumerate(stages):
        print(f"Stage {stage + 1}: Unfreezing {num_layers} layers")
        unfreeze_layers(model, num_layers)
        
        # Train for specified epochs
        for epoch in range(epochs_per_stage):
            train_one_epoch(model, train_loader)`}</Code>
              </Paper>
            </Grid.Col>
          </Grid>
        </div>

        {/* Slide 7: Data Augmentation */}
        <div data-slide className="min-h-[500px]" id="data-augmentation">
          <Title order={2} className="mb-6">Data Augmentation for CNNs</Title>
          
          <Paper className="p-4 bg-gray-50">
            <Code block language="python">{`import torchvision.transforms as transforms
import torch

# Standard augmentation pipeline
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Advanced augmentation techniques
class MixUp(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, x, y):
        if self.alpha > 0:
            lam = torch.distributions.beta.Beta(self.alpha, self.alpha).sample()
        else:
            lam = 1
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size)
        
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

class CutMix(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, x, y):
        lam = torch.distributions.beta.Beta(self.alpha, self.alpha).sample()
        batch_size = x.size(0)
        index = torch.randperm(batch_size)
        
        _, _, H, W = x.shape
        cut_rat = torch.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        cx = torch.randint(W, (1,))
        cy = torch.randint(H, (1,))
        
        bbx1 = torch.clamp(cx - cut_w // 2, 0, W)
        bby1 = torch.clamp(cy - cut_h // 2, 0, H)
        bbx2 = torch.clamp(cx + cut_w // 2, 0, W)
        bby2 = torch.clamp(cy + cut_h // 2, 0, H)
        
        x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        
        return x, y, y[index], lam`}</Code>
          </Paper>
        </div>

      </Stack>
    </Container>
  );
};

export default ConvolutionalNetworks;