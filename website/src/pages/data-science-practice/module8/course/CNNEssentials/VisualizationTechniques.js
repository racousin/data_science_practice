import React from 'react';
import { Text, Stack, List, Grid, Table } from '@mantine/core';
import CodeBlock from 'components/CodeBlock';
import { BlockMath, InlineMath } from 'react-katex';

const VisualizationTechniques = () => {
  const activationVisualizationCode = `
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class FeatureVisualizer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.activations = {}
        
        # Register hooks for feature visualization
        def get_activation(name):
            def hook(model, input, output):
                self.activations[name] = output.detach()
            return hook
        
        # Register hooks for each layer you want to visualize
        for name, layer in model.named_modules():
            if isinstance(layer, nn.Conv2d):
                layer.register_forward_hook(get_activation(name))
    
    def plot_feature_maps(self, x, layer_name, num_features=8):
        # Forward pass
        self.model(x)
        
        # Get activations for specified layer
        activations = self.activations[layer_name]
        
        # Plot feature maps
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        for idx, ax in enumerate(axes.flat):
            if idx < num_features:
                ax.imshow(activations[0, idx].cpu(), cmap='viridis')
            ax.axis('off')
            
        plt.suptitle(f'Feature Maps for {layer_name}')
        plt.tight_layout()
        return fig`;

  const gradCamCode = `
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        def save_gradients(grad):
            self.gradients = grad
        
        def save_activations(module, input, output):
            self.activations = output
        
        target_layer.register_forward_hook(save_activations)
        target_layer.register_backward_hook(lambda m, i, o: save_gradients(o[0]))
    
    def generate_cam(self, input_image, target_class):
        # Forward pass
        model_output = self.model(input_image)
        
        # Zero all gradients
        self.model.zero_grad()
        
        # Backward pass with respect to target class
        output_for_class = model_output[0, target_class]
        output_for_class.backward()
        
        # Get weights
        weights = torch.mean(self.gradients, dim=(2, 3))
        
        # Generate weighted combination of forward activation maps
        cam = torch.zeros(self.activations.shape[2:], dtype=torch.float32)
        for i, w in enumerate(weights[0]):
            cam += w * self.activations[0, i]
        
        # Apply ReLU and normalize
        cam = F.relu(cam)
        cam = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            size=input_image.shape[2:],
            mode='bilinear',
            align_corners=False
        )
        
        cam = cam - cam.min()
        cam = cam / cam.max()
        
        return cam.squeeze()`;

  const saliencyMapCode = `
def compute_saliency_maps(model, image, target_class):
    # Prepare image for gradient computation
    image.requires_grad_()
    
    # Forward pass
    output = model(image)
    
    # Zero gradients
    model.zero_grad()
    
    # Backward pass for target class
    output[0, target_class].backward()
    
    # Get gradients with respect to input
    saliency, _ = torch.max(image.grad.abs(), dim=1)
    
    return saliency

def visualize_saliency(image, saliency_map):
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(image.permute(1, 2, 0).cpu())
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(saliency_map.cpu(), cmap='hot')
    plt.title('Saliency Map')
    plt.axis('off')
    
    plt.tight_layout()
    return plt.gcf()`;

  const filterVisualizationCode = `
def visualize_filters(model, layer_idx, num_filters=8):
    # Get the conv layer
    conv_layer = list(model.modules())[layer_idx]
    
    if not isinstance(conv_layer, nn.Conv2d):
        raise ValueError("Selected layer is not a Conv2d layer")
    
    # Get filters
    filters = conv_layer.weight.data.cpu()
    
    # Normalize filter values
    n_filters = min(filters.shape[0], num_filters)
    
    # Plot filters
    fig = plt.figure(figsize=(12, 8))
    for i in range(n_filters):
        ax = fig.add_subplot(2, 4, i + 1)
        
        # Plot each channel of the filter
        if filters.shape[1] == 3:  # RGB filters
            # Combine RGB channels
            img = filters[i].permute(1, 2, 0)
            # Normalize to [0, 1]
            img = (img - img.min()) / (img.max() - img.min())
            ax.imshow(img)
        else:  # Single channel filters
            ax.imshow(filters[i][0], cmap='gray')
        
        ax.axis('off')
        ax.set_title(f'Filter {i+1}')
    
    plt.tight_layout()
    return fig

# Interactive filter visualization with input image
def apply_filter(image, filter_weights):
    # Ensure inputs are on the same device
    device = filter_weights.device
    image = image.to(device)
    
    # Apply filter
    conv = F.conv2d(
        image.unsqueeze(0),
        filter_weights.unsqueeze(0),
        padding='same'
    )
    
    return conv.squeeze()`;

  const attentionVisualizationCode = `
class AttentionVisualizer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.attention_maps = {}
        
        def get_attention(name):
            def hook(module, input, output):
                self.attention_maps[name] = output.detach()
            return hook
        
        # Register hooks for attention layers
        for name, module in model.named_modules():
            if "attention" in name.lower():
                module.register_forward_hook(get_attention(name))
    
    def visualize_attention(self, x, layer_name):
        # Forward pass
        self.model(x)
        
        # Get attention weights
        attention = self.attention_maps[layer_name]
        
        # Average attention heads if multiple
        if len(attention.shape) == 4:  # [batch, heads, seq_len, seq_len]
            attention = attention.mean(dim=1)
        
        return attention[0]  # Return first batch item

def plot_attention_map(attention_map, image_size):
    # Reshape attention to image dimensions
    h, w = image_size
    attention_image = attention_map.view(h, w)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(attention_image.cpu(), cmap='viridis')
    plt.colorbar()
    plt.title('Attention Map')
    plt.axis('off')
    return plt.gcf()`;

  return (
    <Stack spacing="md">
      <Text>
        CNN visualization techniques help understand how the network processes information
        and makes decisions. This section covers various visualization methods and their
        implementations.
      </Text>

      <Text weight={700}>1. Feature Map Visualization</Text>

      <Text>
        Visualizing intermediate activations helps understand what features each layer learns:
      </Text>

      <CodeBlock
        language="python"
        code={activationVisualizationCode}
      />

      <Text weight={700}>2. Grad-CAM (Gradient-weighted Class Activation Mapping)</Text>

      <Text>
        Grad-CAM uses gradients flowing into the final convolutional layer to highlight important regions:
      </Text>

      <BlockMath>
        {`L^c_{Grad-CAM} = ReLU\\left(\\sum_k \\alpha^c_k A^k\\right)`}
      </BlockMath>

      <Text>
        where <InlineMath>\alpha^c_k</InlineMath> represents the importance weights and
        <InlineMath>A^k</InlineMath> represents the feature maps.
      </Text>

      <CodeBlock
        language="python"
        code={gradCamCode}
      />

      <Text weight={700}>3. Saliency Maps</Text>

      <Text>
        Saliency maps show which input pixels contribute most to the classification:
      </Text>

      <CodeBlock
        language="python"
        code={saliencyMapCode}
      />

      <Text weight={700}>4. Filter Visualization</Text>

      <Text>
        Visualizing learned filters helps understand low-level feature detection:
      </Text>

      <CodeBlock
        language="python"
        code={filterVisualizationCode}
      />

      <Text weight={700}>5. Attention Visualization</Text>

      <Text>
        For networks with attention mechanisms, visualizing attention weights:
      </Text>

      <CodeBlock
        language="python"
        code={attentionVisualizationCode}
      />

      <Text weight={700}>6. Comparison of Visualization Techniques</Text>

      <Table>
        <thead>
          <tr>
            <th>Technique</th>
            <th>Pros</th>
            <th>Cons</th>
            <th>Best Use Case</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Feature Maps</td>
            <td>Direct visualization of learned features</td>
            <td>Can be hard to interpret</td>
            <td>Understanding layer behavior</td>
          </tr>
          <tr>
            <td>Grad-CAM</td>
            <td>Class-specific localization</td>
            <td>Limited to convolutional features</td>
            <td>Object localization explanation</td>
          </tr>
          <tr>
            <td>Saliency Maps</td>
            <td>Shows input importance</td>
            <td>Can be noisy</td>
            <td>Input sensitivity analysis</td>
          </tr>
          <tr>
            <td>Filter Visualization</td>
            <td>Shows learned patterns</td>
            <td>Only useful for early layers</td>
            <td>Understanding feature detectors</td>
          </tr>
          <tr>
            <td>Attention Maps</td>
            <td>Shows focus areas</td>
            <td>Only for attention-based models</td>
            <td>Understanding model attention</td>
          </tr>
        </tbody>
      </Table>

      <Text weight={700}>7. Best Practices for Visualization</Text>

      <List>
        <List.Item>
          <strong>Data Preprocessing:</strong>
          <List withPadding>
            <List.Item>Normalize visualizations appropriately</List.Item>
            <List.Item>Consider colormap choice for interpretability</List.Item>
            <List.Item>Use consistent scales across comparisons</List.Item>
          </List>
        </List.Item>

        <List.Item>
          <strong>Interpretation Guidelines:</strong>
          <List withPadding>
            <List.Item>Compare multiple samples</List.Item>
            <List.Item>Consider layer depth when interpreting</List.Item>
            <List.Item>Validate patterns across different inputs</List.Item>
          </List>
        </List.Item>

        <List.Item>
          <strong>Technical Considerations:</strong>
          <List withPadding>
            <List.Item>Handle batch normalization carefully</List.Item>
            <List.Item>Consider gradient accumulation for stability</List.Item>
            <List.Item>Use appropriate resolution for visualizations</List.Item>
          </List>
        </List.Item>
      </List>

      <Text weight={700}>8. Evaluation Metrics for Visualizations</Text>

      <Text>
        When evaluating visualization quality, consider:
      </Text>

      <List>
        <List.Item>Localization accuracy (for Grad-CAM)</List.Item>
        <List.Item>Consistency across similar inputs</List.Item>
        <List.Item>Correlation with human attention (for attention maps)</List.Item>
        <List.Item>Robustness to input variations</List.Item>
      </List>
    </Stack>
  );
};

export default VisualizationTechniques;