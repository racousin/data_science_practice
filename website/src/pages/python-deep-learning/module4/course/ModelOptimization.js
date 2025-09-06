import React from 'react';
import { Container, Title, Text, Stack, Grid, Paper, Code, List, Flex, Image } from '@mantine/core';
import { InlineMath, BlockMath } from 'react-katex';
import CodeBlock from '../../../../components/CodeBlock';
const ModelOptimization = () => {
  return (
    <Container size="xl" py="xl">
      <Stack spacing="xl">
        
        
        {/* Slide 1: Title and Introduction */}
        <div data-slide className="min-h-[500px] flex flex-col justify-center">
          <Title order={1} className="text-center mb-8">
            Model Optimization
          </Title>
          <Text size="xl" className="text-center mb-6">
            Efficient Deep Learning for Production
          </Text>
          <div className="max-w-3xl mx-auto">
            <Paper className="p-6 bg-blue-50">
              <Text size="lg" mb="md">
                Model optimization involves reducing computational requirements while maintaining performance.
                This includes techniques for quantization, pruning, distillation, and efficient architectures
                to deploy models on resource-constrained devices.
              </Text>
              <List>
                <List.Item>Model compression and pruning techniques</List.Item>
                <List.Item>Quantization and mixed-precision training</List.Item>
                <List.Item>Knowledge distillation</List.Item>
                <List.Item>Efficient architectures and mobile deployment</List.Item>
              </List>
            </Paper>
          </div>
        </div>

        {/* Slide 2: Model Pruning */}
        <div data-slide className="min-h-[500px]" id="model-pruning">
          <Title order={2} mb="xl">Model Pruning</Title>
          
            <Paper mb="xl">
            <Text size="lg">
              Pruning removes unimportant weights or neurons from neural networks to reduce model size
              and computational requirements while maintaining accuracy.
            </Text>
          </Paper>
          
          <Grid gutter="lg">
            <Grid.Col span={6}>
              <Paper className="p-4 bg-green-50">
                <Title order={4} mb="sm">Magnitude-based Pruning</Title>
                <Code block language="python">{`import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

def magnitude_prune_model(model, amount=0.3):
    """Prune model based on weight magnitude"""
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            prune.l1_unstructured(module, name='weight', amount=amount)
    return model

def structured_prune_model(model, amount=0.2):
    """Structured pruning - remove entire channels/filters"""
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # Prune entire filters based on L2 norm
            prune.ln_structured(
                module, name='weight', amount=amount, n=2, dim=0
            )
        elif isinstance(module, nn.Linear):
            # Prune neurons
            prune.ln_structured(
                module, name='weight', amount=amount, n=2, dim=0
            )
    return model

# Custom pruning method
class CustomPruningMethod(prune.BasePruningMethod):
    PRUNING_TYPE = 'unstructured'
    
    def __init__(self, threshold):
        self.threshold = threshold
    
    def compute_mask(self, tensor, default_mask):
        return torch.abs(tensor) > self.threshold

def apply_custom_pruning(module, threshold=0.1):
    CustomPruningMethod.apply(module, 'weight', threshold=threshold)

# Example usage
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

# Apply magnitude pruning
pruned_model = magnitude_prune_model(model, amount=0.4)

# Check sparsity
def check_sparsity(model):
    total_params = 0
    zero_params = 0
    
    for module in model.modules():
        if hasattr(module, 'weight_mask'):
            total_params += module.weight_mask.numel()
            zero_params += (module.weight_mask == 0).sum().item()
    
    sparsity = 100. * zero_params / total_params
    print(f"Model sparsity: {sparsity:.2f}%")

check_sparsity(pruned_model)`}</Code>
              </Paper>
            </Grid.Col>
            
            <Grid.Col span={6}>
              <Paper className="p-4 bg-blue-50">
                <Title order={4} mb="sm">Iterative Pruning</Title>
                <Code block language="python">{`class IterativePruner:
    def __init__(self, model, dataloader, criterion, device):
        self.model = model
        self.dataloader = dataloader
        self.criterion = criterion
        self.device = device
        self.original_accuracy = self.evaluate()
    
    def evaluate(self):
        """Evaluate model accuracy"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.dataloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        return 100 * correct / total
    
    def iterative_prune(self, target_sparsity=0.9, step_size=0.1, max_accuracy_drop=2.0):
        """Iteratively prune the model"""
        current_sparsity = 0
        
        while current_sparsity < target_sparsity:
            # Prune one step
            for module in self.model.modules():
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    prune.l1_unstructured(module, name='weight', amount=step_size)
            
            current_sparsity += step_size
            
            # Evaluate accuracy
            accuracy = self.evaluate()
            accuracy_drop = self.original_accuracy - accuracy
            
            print(f"Sparsity: {current_sparsity:.1%}, Accuracy: {accuracy:.2f}%, Drop: {accuracy_drop:.2f}%")
            
            # Stop if accuracy drops too much
            if accuracy_drop > max_accuracy_drop:
                print(f"Stopping pruning at {current_sparsity:.1%} sparsity")
                break
            
            # Fine-tune after pruning
            self.fine_tune(epochs=5)
    
    def fine_tune(self, epochs=5, lr=1e-4):
        """Fine-tune model after pruning"""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        for epoch in range(epochs):
            self.model.train()
            for data, target in self.dataloader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                optimizer.step()

# Gradient-based pruning
class GradientPruning:
    @staticmethod
    def compute_importance_scores(model, dataloader, criterion, device):
        """Compute importance scores based on gradients"""
        importance_scores = {}
        
        # Collect gradients
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if name not in importance_scores:
                        importance_scores[name] = torch.zeros_like(param)
                    importance_scores[name] += param.grad.abs()
            
            model.zero_grad()
        
        return importance_scores
    
    @staticmethod
    def prune_by_importance(model, importance_scores, sparsity=0.5):
        """Prune parameters with low importance scores"""
        for name, param in model.named_parameters():
            if name in importance_scores:
                importance = importance_scores[name]
                threshold = torch.quantile(importance.flatten(), sparsity)
                mask = importance > threshold
                param.data *= mask.float()`}</Code>
              </Paper>
            </Grid.Col>
          </Grid>
        </div>

        {/* Slide 3: Quantization */}
        <div data-slide className="min-h-[500px]" id="quantization">
          <Title order={2} mb="xl">Model Quantization</Title>
          
          <Grid gutter="lg">
            <Grid.Col span={6}>
              <Paper className="p-4 bg-purple-50">
                <Title order={4} mb="sm">Post-Training Quantization</Title>
                <Code block language="python">{`import torch.quantization as quantization

def post_training_quantize(model, calibration_loader):
    """Apply post-training quantization"""
    # Set quantization config
    model.qconfig = quantization.get_default_qconfig('fbgemm')
    
    # Prepare model for quantization
    model_prepared = quantization.prepare(model, inplace=False)
    
    # Calibrate with representative data
    model_prepared.eval()
    with torch.no_grad():
        for data, _ in calibration_loader:
            model_prepared(data)
    
    # Convert to quantized model
    model_quantized = quantization.convert(model_prepared, inplace=False)
    
    return model_quantized

# Dynamic quantization (weights only)
def dynamic_quantize_model(model):
    """Apply dynamic quantization to linear layers"""
    quantized_model = quantization.quantize_dynamic(
        model,
        {nn.Linear, nn.Conv2d}, 
        dtype=torch.qint8
    )
    return quantized_model

# Custom quantization
class QuantizedLinear(nn.Module):
    def __init__(self, linear_layer, bits=8):
        super().__init__()
        self.bits = bits
        
        # Quantize weights
        weight = linear_layer.weight.data
        self.scale, self.zero_point = self.calculate_qparams(weight)
        self.quantized_weight = self.quantize_tensor(weight)
        
        # Keep bias in float
        self.bias = linear_layer.bias
        
    def calculate_qparams(self, tensor):
        """Calculate quantization parameters"""
        min_val = tensor.min()
        max_val = tensor.max()
        
        qmin = -(2 ** (self.bits - 1))
        qmax = 2 ** (self.bits - 1) - 1
        
        scale = (max_val - min_val) / (qmax - qmin)
        zero_point = qmin - min_val / scale
        zero_point = torch.clamp(zero_point, qmin, qmax).round()
        
        return scale, zero_point
    
    def quantize_tensor(self, tensor):
        """Quantize tensor"""
        return torch.clamp(
            (tensor / self.scale + self.zero_point).round(),
            -(2 ** (self.bits - 1)),
            2 ** (self.bits - 1) - 1
        ).to(torch.int8)
    
    def dequantize_tensor(self, quantized_tensor):
        """Dequantize tensor"""
        return (quantized_tensor.float() - self.zero_point) * self.scale
    
    def forward(self, x):
        # Dequantize weights for computation
        weight = self.dequantize_tensor(self.quantized_weight)
        return F.linear(x, weight, self.bias)

def replace_linear_with_quantized(model, bits=8):
    """Replace linear layers with quantized versions"""
    for name, child in model.named_children():
        if isinstance(child, nn.Linear):
            setattr(model, name, QuantizedLinear(child, bits))
        else:
            replace_linear_with_quantized(child, bits)`}</Code>
              </Paper>
            </Grid.Col>
            
            <Grid.Col span={6}>
              <Paper className="p-4 bg-orange-50">
                <Title order={4} mb="sm">Quantization-Aware Training</Title>
                <Code block language="python">{`class QATModel(nn.Module):
    """Quantization-Aware Training Model"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.quant = quantization.QuantStub()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dequant = quantization.DeQuantStub()
    
    def forward(self, x):
        x = self.quant(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.dequant(x)
        return x

def train_with_qat(model, train_loader, num_epochs=10):
    """Train model with quantization awareness"""
    # Set quantization config
    model.qconfig = quantization.get_default_qat_qconfig('fbgemm')
    
    # Prepare for QAT
    model_prepared = quantization.prepare_qat(model, inplace=False)
    
    optimizer = torch.optim.Adam(model_prepared.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        model_prepared.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model_prepared(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        
        # Switch to eval mode periodically to update quantization parameters
        if epoch % 3 == 0:
            model_prepared.eval()
    
    # Convert to fully quantized model
    model_quantized = quantization.convert(model_prepared.eval(), inplace=False)
    return model_quantized

# Mixed precision training
from torch.cuda.amp import autocast, GradScaler

def mixed_precision_training(model, train_loader, num_epochs=10):
    """Train with mixed precision"""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scaler = GradScaler()
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        for data, target in train_loader:
            optimizer.zero_grad()
            
            # Forward pass with autocast
            with autocast():
                output = model(data)
                loss = criterion(output, target)
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
    
    return model`}</Code>
              </Paper>
            </Grid.Col>
          </Grid>
        </div>

        {/* Slide 4: Knowledge Distillation */}
        <div data-slide className="min-h-[500px]" id="knowledge-distillation">
          <Title order={2} mb="xl">Knowledge Distillation</Title>
          
          <Grid gutter="lg">
            <Grid.Col span={12}>
              <Paper className="p-4 bg-red-50 mb-4">
                <Title order={4} mb="sm">Teacher-Student Framework</Title>
                <Code block language="python">{`class KnowledgeDistillationLoss(nn.Module):
    def __init__(self, alpha=0.7, temperature=4):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.cross_entropy = nn.CrossEntropyLoss()
    
    def forward(self, student_logits, teacher_logits, targets):
        # Distillation loss (soft targets)
        distillation_loss = self.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=1),
            F.softmax(teacher_logits / self.temperature, dim=1)
        ) * (self.temperature ** 2)
        
        # Standard cross-entropy loss (hard targets)
        student_loss = self.cross_entropy(student_logits, targets)
        
        # Combine losses
        total_loss = self.alpha * distillation_loss + (1 - self.alpha) * student_loss
        
        return total_loss

def train_student_model(teacher_model, student_model, train_loader, num_epochs=20):
    """Train student model using knowledge distillation"""
    teacher_model.eval()  # Teacher in eval mode
    student_model.train()
    
    optimizer = torch.optim.Adam(student_model.parameters(), lr=0.001)
    kd_loss = KnowledgeDistillationLoss(alpha=0.7, temperature=4)
    
    for epoch in range(num_epochs):
        for data, targets in train_loader:
            optimizer.zero_grad()
            
            # Get teacher predictions (no gradients needed)
            with torch.no_grad():
                teacher_logits = teacher_model(data)
            
            # Get student predictions
            student_logits = student_model(data)
            
            # Compute distillation loss
            loss = kd_loss(student_logits, teacher_logits, targets)
            
            loss.backward()
            optimizer.step()
    
    return student_model

# Feature-based distillation
class FeatureDistillationLoss(nn.Module):
    def __init__(self, student_channels, teacher_channels):
        super().__init__()
        # Adapt student features to match teacher dimensions
        self.adapter = nn.Conv2d(student_channels, teacher_channels, 1)
        self.mse = nn.MSELoss()
    
    def forward(self, student_features, teacher_features):
        adapted_features = self.adapter(student_features)
        return self.mse(adapted_features, teacher_features)

class TeacherStudentModel(nn.Module):
    def __init__(self, teacher_model, student_model):
        super().__init__()
        self.teacher = teacher_model
        self.student = student_model
        
        # Hooks to extract intermediate features
        self.teacher_features = []
        self.student_features = []
        
        # Register hooks
        self.teacher.layer3.register_forward_hook(self.save_teacher_features)
        self.student.layer2.register_forward_hook(self.save_student_features)
        
        # Feature distillation loss
        self.feature_loss = FeatureDistillationLoss(
            student_channels=64,  # Student feature channels
            teacher_channels=256  # Teacher feature channels
        )
    
    def save_teacher_features(self, module, input, output):
        self.teacher_features.append(output)
    
    def save_student_features(self, module, input, output):
        self.student_features.append(output)
    
    def forward(self, x):
        # Clear previous features
        self.teacher_features = []
        self.student_features = []
        
        with torch.no_grad():
            teacher_output = self.teacher(x)
        
        student_output = self.student(x)
        
        # Compute feature distillation loss
        feature_loss = self.feature_loss(
            self.student_features[0], 
            self.teacher_features[0]
        )
        
        return student_output, teacher_output, feature_loss`}</Code>
              </Paper>
            </Grid.Col>
          </Grid>
        </div>

        {/* Slide 5: Efficient Architectures */}
        <div data-slide className="min-h-[500px]" id="efficient-architectures">
          <Title order={2} mb="xl">Efficient Architectures</Title>
          
          <Grid gutter="lg">
            <Grid.Col span={6}>
              <Paper className="p-4 bg-indigo-50">
                <Title order={4} mb="sm">MobileNet Architecture</Title>
                <Code block language="python">{`class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        
        # Depthwise convolution
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        
        # Pointwise convolution
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.depthwise(x)))
        x = F.relu(self.bn2(self.pointwise(x)))
        return x

class MobileNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expansion=6):
        super().__init__()
        hidden_dim = in_channels * expansion
        self.use_shortcut = stride == 1 and in_channels == out_channels
        
        self.conv = nn.Sequential(
            # Expansion layer
            nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            
            # Depthwise convolution
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            
            # Pointwise linear projection
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        
    def forward(self, x):
        if self.use_shortcut:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0):
        super().__init__()
        
        # First layer
        input_channel = int(32 * width_mult)
        self.features = [nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
                        nn.BatchNorm2d(input_channel),
                        nn.ReLU6(inplace=True)]
        
        # MobileNet blocks configuration
        # [expansion, out_channels, repeats, stride]
        configs = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        
        # Build inverted residual blocks
        for expansion, out_channels, repeats, stride in configs:
            out_channels = int(out_channels * width_mult)
            for i in range(repeats):
                s = stride if i == 0 else 1
                self.features.append(MobileNetBlock(input_channel, out_channels, s, expansion))
                input_channel = out_channels
        
        # Last layer
        last_channel = int(1280 * width_mult)
        self.features.extend([
            nn.Conv2d(input_channel, last_channel, 1, bias=False),
            nn.BatchNorm2d(last_channel),
            nn.ReLU6(inplace=True)
        ])
        
        self.features = nn.Sequential(*self.features)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(last_channel, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x`}</Code>
              </Paper>
            </Grid.Col>
            
            <Grid.Col span={6}>
              <Paper p="md">
                <Title order={4} mb="sm">EfficientNet Architecture</Title>
                <Code block language="python">{`class SqueezeExcitation(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.global_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expansion_ratio, se_ratio=0.25):
        super().__init__()
        
        expanded_channels = in_channels * expansion_ratio
        self.use_residual = stride == 1 and in_channels == out_channels
        
        layers = []
        
        # Expansion phase
        if expansion_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, expanded_channels, 1, bias=False),
                nn.BatchNorm2d(expanded_channels),
                nn.SiLU(inplace=True)
            ])
        
        # Depthwise convolution
        layers.extend([
            nn.Conv2d(expanded_channels, expanded_channels, kernel_size, stride, 
                     kernel_size//2, groups=expanded_channels, bias=False),
            nn.BatchNorm2d(expanded_channels),
            nn.SiLU(inplace=True)
        ])
        
        # Squeeze and Excitation
        if se_ratio > 0:
            layers.append(SqueezeExcitation(expanded_channels, int(1/se_ratio)))
        
        # Output projection
        layers.extend([
            nn.Conv2d(expanded_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        
        self.conv = nn.Sequential(*layers)
        
    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)

def compound_scaling(base_channels, base_depth, base_resolution, phi):
    """EfficientNet compound scaling"""
    alpha = 1.2  # depth scaling factor
    beta = 1.1   # width scaling factor
    gamma = 1.15 # resolution scaling factor
    
    depth_factor = alpha ** phi
    width_factor = beta ** phi
    resolution_factor = gamma ** phi
    
    new_channels = int(base_channels * width_factor)
    new_depth = int(base_depth * depth_factor)
    new_resolution = int(base_resolution * resolution_factor)
    
    return new_channels, new_depth, new_resolution

# Neural Architecture Search inspired block
class NASBlock(nn.Module):
    def __init__(self, in_channels, out_channels, operations=['conv3x3', 'conv5x5', 'maxpool']):
        super().__init__()
        self.operations = nn.ModuleList()
        
        for op in operations:
            if op == 'conv3x3':
                self.operations.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, 3, padding=1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU()
                    )
                )
            elif op == 'conv5x5':
                self.operations.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, 5, padding=2),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU()
                    )
                )
            elif op == 'maxpool':
                self.operations.append(
                    nn.Sequential(
                        nn.MaxPool2d(3, stride=1, padding=1),
                        nn.Conv2d(in_channels, out_channels, 1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU()
                    )
                )
        
        # Learnable architecture weights
        self.alpha = nn.Parameter(torch.randn(len(operations)))
    
    def forward(self, x):
        weights = F.softmax(self.alpha, dim=0)
        output = sum(w * op(x) for w, op in zip(weights, self.operations))
        return output`}</Code>
              </Paper>
            </Grid.Col>
          </Grid>
        </div>

        {/* Slide 6: Model Compression Tools */}
        <div data-slide className="min-h-[500px]" id="compression-tools">
          <Title order={2} mb="xl">Model Compression Tools and Frameworks</Title>
          
          <Grid gutter="lg">
            <Grid.Col span={6}>
              <Paper className="p-4 bg-green-50">
                <Title order={4} mb="sm">TensorFlow Lite Conversion</Title>
                <Code block language="python">{`# TensorFlow Lite optimization
import tensorflow as tf

def convert_to_tflite(model, representative_dataset=None):
    """Convert PyTorch/TF model to TensorFlow Lite"""
    
    converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
    
    # Optimization options
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Post-training quantization
    if representative_dataset:
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    
    # Convert model
    tflite_model = converter.convert()
    
    return tflite_model

# ONNX optimization
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

def optimize_onnx_model(model_path, output_path):
    """Optimize ONNX model"""
    
    # Load model
    model = onnx.load(model_path)
    
    # Dynamic quantization
    quantized_model = quantize_dynamic(
        model_path,
        output_path,
        weight_type=QuantType.QUInt8
    )
    
    return quantized_model

# PyTorch Mobile optimization
def optimize_for_mobile(model):
    """Optimize PyTorch model for mobile deployment"""
    
    # Script the model
    scripted_model = torch.jit.script(model)
    
    # Optimize for mobile
    optimized_model = torch.utils.mobile_optimizer.optimize_for_mobile(
        scripted_model,
        optimization_blocklist={'conv_bn_fusion'}  # Optional blocklist
    )
    
    return optimized_model`}</Code>
              </Paper>
            </Grid.Col>
            
            <Grid.Col span={6}>
              <Paper className="p-4 bg-blue-50">
                <Title order={4} mb="sm">Model Profiling and Analysis</Title>
                <Code block language="python">{`import time
import torch.profiler

def profile_model(model, input_tensor, num_runs=100):
    """Profile model performance"""
    
    # Warmup
    for _ in range(10):
        _ = model(input_tensor)
    
    # Time measurement
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(num_runs):
        _ = model(input_tensor)
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs
    return avg_time

def detailed_profiling(model, input_tensor):
    """Detailed profiling with PyTorch profiler"""
    
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        with torch.profiler.record_function("model_inference"):
            output = model(input_tensor)
    
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    return prof

def calculate_model_size(model):
    """Calculate model size in MB"""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb

def calculate_flops(model, input_shape):
    """Calculate FLOPs (requires torchprofile)"""
    try:
        from torchprofile import profile_macs
        
        input_tensor = torch.randn(input_shape)
        macs = profile_macs(model, input_tensor)
        flops = 2 * macs  # Multiply-accumulate = 2 operations
        
        return flops
    except ImportError:
        print("Install torchprofile for FLOP calculation")
        return None

# Compression ratio analysis
class CompressionAnalyzer:
    def __init__(self, original_model, compressed_model):
        self.original_model = original_model
        self.compressed_model = compressed_model
    
    def analyze(self):
        """Analyze compression results"""
        
        # Size comparison
        original_size = calculate_model_size(self.original_model)
        compressed_size = calculate_model_size(self.compressed_model)
        size_ratio = original_size / compressed_size
        
        # Speed comparison
        dummy_input = torch.randn(1, 3, 224, 224)
        original_time = profile_model(self.original_model, dummy_input)
        compressed_time = profile_model(self.compressed_model, dummy_input)
        speed_up = original_time / compressed_time
        
        results = {
            'original_size_mb': original_size,
            'compressed_size_mb': compressed_size,
            'compression_ratio': size_ratio,
            'original_inference_time': original_time,
            'compressed_inference_time': compressed_time,
            'speedup': speed_up
        }
        
        return results`}</Code>
              </Paper>
            </Grid.Col>
          </Grid>
        </div>

        {/* Slide 7: Production Deployment */}
        <div data-slide className="min-h-[500px]" id="production-deployment">
          <Title order={2} mb="xl">Production Deployment Strategies</Title>
          
          <Grid gutter="lg">
            <Grid.Col span={6}>
              <Paper className="p-4 bg-purple-50">
                <Title order={4} mb="sm">Edge Deployment</Title>
                <List spacing="sm">
                  <List.Item><strong>Model Size:</strong> Keep models under 10MB for mobile devices</List.Item>
                  <List.Item><strong>Latency:</strong> Target sub-100ms inference for real-time applications</List.Item>
                  <List.Item><strong>Memory:</strong> Optimize for limited RAM (1-4GB typical)</List.Item>
                  <List.Item><strong>Power:</strong> Consider battery life constraints</List.Item>
                  <List.Item><strong>Hardware:</strong> Leverage specialized chips (NPU, DSP)</List.Item>
                </List>
                
                <Paper className="p-3 bg-white mt-4">
                  <Title order={5} className="mb-2">Mobile Optimization Checklist</Title>
                  <List size="sm">
                    <List.Item>Use quantized models (INT8 or FP16)</List.Item>
                    <List.Item>Apply pruning to reduce model size</List.Item>
                    <List.Item>Optimize input preprocessing</List.Item>
                    <List.Item>Use mobile-specific architectures</List.Item>
                    <List.Item>Test on target devices</List.Item>
                    <List.Item>Monitor thermal throttling</List.Item>
                  </List>
                </Paper>
              </Paper>
            </Grid.Col>
            
            <Grid.Col span={6}>
              <Paper p="md">
                <Title order={4} mb="sm">Cloud Deployment</Title>
                <Code block language="python">{`# Model serving with batching
class BatchedModelServer:
    def __init__(self, model, max_batch_size=32, timeout_ms=50):
        self.model = model
        self.max_batch_size = max_batch_size
        self.timeout_ms = timeout_ms
        self.batch_queue = []
        
    async def predict(self, input_data):
        """Batched prediction with timeout"""
        future = asyncio.Future()
        
        self.batch_queue.append((input_data, future))
        
        # Trigger batch processing if queue is full or timeout
        if len(self.batch_queue) >= self.max_batch_size:
            await self._process_batch()
        else:
            # Set timeout to process partial batch
            asyncio.create_task(self._timeout_handler())
        
        return await future
    
    async def _process_batch(self):
        """Process current batch"""
        if not self.batch_queue:
            return
        
        inputs, futures = zip(*self.batch_queue)
        self.batch_queue = []
        
        # Stack inputs and run inference
        batch_input = torch.stack(inputs)
        with torch.no_grad():
            batch_output = self.model(batch_input)
        
        # Return results to futures
        for i, future in enumerate(futures):
            future.set_result(batch_output[i])
    
    async def _timeout_handler(self):
        """Handle batch timeout"""
        await asyncio.sleep(self.timeout_ms / 1000)
        if self.batch_queue:
            await self._process_batch()

# Model versioning and A/B testing
class ModelRegistry:
    def __init__(self):
        self.models = {}
        self.traffic_splits = {}
    
    def register_model(self, name, version, model):
        """Register a model version"""
        key = f"{name}:{version}"
        self.models[key] = model
    
    def set_traffic_split(self, name, splits):
        """Set traffic split between versions"""
        # splits: {'v1': 0.8, 'v2': 0.2}
        self.traffic_splits[name] = splits
    
    def get_model(self, name, user_id=None):
        """Get model based on traffic splitting"""
        if name not in self.traffic_splits:
            # Default to latest version
            versions = [k for k in self.models.keys() if k.startswith(f"{name}:")]
            if versions:
                latest_key = max(versions)
                return self.models[latest_key]
        
        # A/B testing based on user_id
        splits = self.traffic_splits[name]
        if user_id:
            hash_val = hash(str(user_id)) % 100
            cumsum = 0
            for version, percentage in splits.items():
                cumsum += percentage * 100
                if hash_val < cumsum:
                    return self.models[f"{name}:{version}"]
        
        # Fallback to random selection
        import random
        version = random.choices(
            list(splits.keys()), 
            weights=list(splits.values())
        )[0]
        return self.models[f"{name}:{version}"]`}</Code>
              </Paper>
            </Grid.Col>
          </Grid>
        </div>

      </Stack>
    </Container>
  );
};

export default ModelOptimization;