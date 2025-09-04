import React from 'react';
import { Container, Title, Text, Stack, Paper, Code, List, Alert, Flex, Image, Table, Badge, Group, Divider } from '@mantine/core';
import CodeBlock from 'components/CodeBlock';

const MonitoringVisualization = () => {
  return (
    <Container size="xl">
      <Stack spacing="xl">
        <div>
          <Title order={1} mb="xl">
            Monitoring & Visualization with TensorBoard
          </Title>
          
          <Paper className="p-6 bg-blue-50 mb-6">
            <Text>
              TensorBoard is TensorFlow's visualization toolkit, fully compatible with PyTorch. 
              It provides insights into model training through interactive visualizations of metrics, 
              model graphs, embeddings, and more.
            </Text>
          </Paper>

          <Stack spacing="xl" mt="xl">
            {/* Section 1: Setting up TensorBoard */}
            <div>
              <Title order={2} mb="md">1. Setting up TensorBoard with PyTorch</Title>
              
              <Title order={3} mt="lg" mb="sm">Basic Setup</Title>
              <Text mb="sm">Import SummaryWriter to start logging:</Text>
              <CodeBlock language="python" code={`from torch.utils.tensorboard import SummaryWriter

# Create a writer for logging
writer = SummaryWriter('runs/experiment_1')`} />
              
              <Text mt="md" mb="sm">Log your model architecture:</Text>
              <CodeBlock language="python" code={`# Define model
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)`} />
              
              <Text mt="md" mb="sm">Add the model graph to TensorBoard:</Text>
              <CodeBlock language="python" code={`# Create dummy input and log graph
dummy_input = torch.randn(1, 784)
writer.add_graph(model, dummy_input)
writer.close()`} />

                <Text mt="md" mb="sm">Launching TensorBoard:</Text>
                <CodeBlock code={`tensorboard --logdir=runs`} />
                <Text size="sm" mt="sm">Then navigate to http://localhost:6006</Text>
                                <Text mt="md" mb="sm">Launching TensorBoard in colab:</Text>
                <CodeBlock language="python" code={`%load_ext tensorboard`} />

    

            </div>

            {/* Section 2: Logging Scalars */}
            <div>
              <Title order={2} mb="md">2. Logging Training Metrics</Title>
              
              <Title order={3} mt="lg" mb="sm">Scalar Metrics</Title>
              <Text mb="sm">Log individual metrics during training:</Text>
              <CodeBlock language="python" code={`# Log single scalar values
writer.add_scalar('Loss/train', train_loss, epoch)
writer.add_scalar('Loss/validation', val_loss, epoch)
writer.add_scalar('Accuracy/validation', accuracy, epoch)`} />
              
              <Text mt="md" mb="sm">Compare multiple metrics on the same plot:</Text>
              <CodeBlock language="python" code={`# Log multiple scalars together
writer.add_scalars('Loss', {
    'train': train_loss,
    'validation': val_loss
}, epoch)`} />
              
              <Text mt="md" mb="sm">Complete training loop example:</Text>
              <CodeBlock language="python" code={`for epoch in range(num_epochs):
    # Training step
    train_loss = train_epoch(model, train_loader)
    val_loss, val_acc = validate(model, val_loader)
    
    # Log metrics
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/val', val_loss, epoch)
    writer.add_scalar('Accuracy/val', val_acc, epoch)`} />

              <Flex direction="column" align="center" mt="md" mb="md">
                <Image
                  src="/assets/python-deep-learning/module3/tensorboard_scalars.png"
                  alt="TensorBoard Scalars Dashboard"
                  style={{ maxWidth: 'min(800px, 90vw)', height: 'auto' }}
                  fluid
                />
                <Text size="sm" c="dimmed" mt="xs">TensorBoard Scalars Visualization</Text>
              </Flex>
            </div>

            {/* Section 3: Visualizing Model Architecture */}
            <div>
              <Title order={2} mb="md">3. Model Architecture Visualization</Title>
              
              <Text mb="sm">Define a CNN architecture:</Text>
              <CodeBlock language="python" code={`class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)`} />
              
              <Text mt="md" mb="sm">Visualize the model architecture:</Text>
              <CodeBlock language="python" code={`model = ConvNet()
writer = SummaryWriter('runs/model_viz')

# Create dummy input and add graph
dummy_input = torch.randn(1, 1, 28, 28)
writer.add_graph(model, dummy_input)`} />

              <Flex direction="column" align="center" mt="md" mb="md">
                <Image
                  src="/assets/python-deep-learning/module3/tensorboard_graph.png"
                  alt="TensorBoard Model Graph"
                  style={{ maxWidth: 'min(800px, 90vw)', height: 'auto' }}
                  fluid
                />
                <Text size="sm" c="dimmed" mt="xs">Model Architecture in TensorBoard</Text>
              </Flex>
            </div>

            {/* Section 4: Histograms and Distributions */}
            <div>
              <Title order={2} mb="md">4. Weight and Gradient Distributions</Title>
              
              <Text mb="sm">Log weight distributions during training:</Text>
              <CodeBlock language="python" code={`# Log weights and gradients for each layer
for name, param in model.named_parameters():
    if param.grad is not None:
        writer.add_histogram(f'{name}/weights', param.data, epoch)
        writer.add_histogram(f'{name}/gradients', param.grad.data, epoch)`} />
              
              <Text mt="md" mb="sm">Monitor gradient norms to detect vanishing/exploding gradients:</Text>
              <CodeBlock language="python" code={`# Calculate and log gradient norms
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.data.norm(2)
        writer.add_scalar(f'GradNorm/{name}', grad_norm, epoch)`} />
              
              <Text mt="md" mb="sm">Track activation distributions:</Text>
              <CodeBlock language="python" code={`# Hook to capture activations
def get_activation(name):
    def hook(model, input, output):
        writer.add_histogram(f'Activations/{name}', output, epoch)
    return hook

# Register hooks
model.fc1.register_forward_hook(get_activation('fc1'))
model.fc2.register_forward_hook(get_activation('fc2'))`} />

              <Flex direction="column" align="center" mt="md" mb="md">
                <Image
                  src="/assets/python-deep-learning/module3/tensorboard_histograms.png"
                  alt="TensorBoard Histograms"
                  style={{ maxWidth: 'min(800px, 90vw)', height: 'auto' }}
                  fluid
                />
                <Text size="sm" c="dimmed" mt="xs">Weight and Gradient Distributions</Text>
              </Flex>
            </div>

            {/* Section 5: Image Visualization */}
            <div>
              <Title order={2} mb="md">5. Image and Tensor Visualization</Title>
              
              <Title order={3} mt="lg" mb="sm">Logging Images</Title>
              <Text mb="sm">Log single images:</Text>
              <CodeBlock language="python" code={`# Single image (CHW format)
img = torch.randn(3, 224, 224)
writer.add_image('Sample_Image', img, epoch)`} />
              
              <Text mt="md" mb="sm">Create image grids for batch visualization:</Text>
              <CodeBlock language="python" code={`# Multiple images in a grid
images = batch_images[:16]  # Take first 16 images
grid = torchvision.utils.make_grid(images, nrow=4, normalize=True)
writer.add_image('Batch_Samples', grid, epoch)`} />
              
              <Text mt="md" mb="sm">Visualize feature maps using hooks:</Text>
              <CodeBlock language="python" code={`def hook_fn(module, input, output):
    # Visualize first 16 channels
    feat = output[0, :16, :, :].unsqueeze(1)
    grid = torchvision.utils.make_grid(feat, nrow=4, normalize=True)
    writer.add_image(f'Features/{module.__class__.__name__}', grid, epoch)

# Register hook on convolution layers
for layer in model.modules():
    if isinstance(layer, nn.Conv2d):
        layer.register_forward_hook(hook_fn)`} />

              <Flex direction="column" align="center" mt="md" mb="md">
                <Image
                  src="/assets/python-deep-learning/module3/tensorboard_images.png"
                  alt="TensorBoard Image Visualization"
                  style={{ maxWidth: 'min(800px, 90vw)', height: 'auto' }}
                  fluid
                />
                <Text size="sm" c="dimmed" mt="xs">Image and Feature Map Visualization</Text>
              </Flex>
            </div>

            {/* Section 6: Embeddings Visualization */}
            <div>
              <Title order={2} mb="md">6. Embedding Projector</Title>
              
              <Text mb="sm">Log embeddings with metadata:</Text>
              <CodeBlock language="python" code={`# Extract embeddings from model
embeddings = model.get_embeddings(data)  # Shape: [N, embed_dim]
labels = ['Class_' + str(i) for i in targets]

# Add to TensorBoard
writer.add_embedding(
    embeddings,
    metadata=labels,
    global_step=epoch,
    tag='feature_embeddings'
)`} />
              
              <Text mt="md" mb="sm">Include images with embeddings:</Text>
              <CodeBlock language="python" code={`# Add embeddings with thumbnail images
writer.add_embedding(
    embeddings,
    metadata=labels,
    label_img=images,  # Thumbnail images for each point
    tag='visual_embeddings'
)`} />
              
              <Text mt="md" mb="sm">Visualize word embeddings:</Text>
              <CodeBlock language="python" code={`# For NLP models
word_embeddings = model.embedding.weight.data
vocab = ['word1', 'word2', ...]  # Your vocabulary

writer.add_embedding(
    word_embeddings[:1000],  # Top 1000 words
    metadata=vocab[:1000],
    tag='word_embeddings'
)`} />

              <Flex direction="column" align="center" mt="md" mb="md">
                <Image
                  src="/assets/python-deep-learning/module3/tensorboard_embeddings.png"
                  alt="TensorBoard Embeddings Projector"
                  style={{ maxWidth: 'min(800px, 90vw)', height: 'auto' }}
                  fluid
                />
                <Text size="sm" c="dimmed" mt="xs">3D Embedding Visualization with t-SNE/PCA</Text>
              </Flex>
            </div>

            {/* Section 7: Custom Visualizations */}
            <div>
              <Title order={2} mb="md">7. Custom Visualizations with Matplotlib</Title>
              
              <Text mb="sm">Create confusion matrix visualization:</Text>
              <CodeBlock language="python" code={`import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
writer.add_figure('Confusion_Matrix', fig, epoch)`} />
              
              <Text mt="md" mb="sm">Plot learning curves:</Text>
              <CodeBlock language="python" code={`fig, ax = plt.subplots()
ax.plot(train_losses, label='Training')
ax.plot(val_losses, label='Validation')
ax.set_xlabel('Epochs')
ax.set_ylabel('Loss')
ax.legend()
writer.add_figure('Learning_Curves', fig, epoch)`} />
              
              <Text mt="md" mb="sm">Visualize attention weights:</Text>
              <CodeBlock language="python" code={`# For transformer models
fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(attention_weights, cmap='hot', cbar=True)
ax.set_xlabel('Keys')
ax.set_ylabel('Queries')
writer.add_figure('Attention_Weights', fig, epoch)`} />

              <Flex direction="column" align="center" mt="md" mb="md">
                <Image
                  src="/assets/python-deep-learning/module3/tensorboard_custom.png"
                  alt="Custom Visualizations in TensorBoard"
                  style={{ maxWidth: 'min(800px, 90vw)', height: 'auto' }}
                  fluid
                />
                <Text size="sm" c="dimmed" mt="xs">Custom Matplotlib Figures in TensorBoard</Text>
              </Flex>
            </div>

            {/* Section 8: Hyperparameter Tuning */}
            <div>
              <Title order={2} mb="md">8. Hyperparameter Tuning Visualization</Title>
              
              <Text mb="sm">Define hyperparameter search space:</Text>
              <CodeBlock language="python" code={`hparam_dict = {
    'learning_rate': [0.001, 0.01, 0.1],
    'batch_size': [32, 64, 128],
    'hidden_size': [128, 256, 512],
    'dropout': [0.2, 0.3, 0.5]
}`} />
              
              <Text mt="md" mb="sm">Log hyperparameters with metrics:</Text>
              <CodeBlock language="python" code={`# After training with specific hyperparameters
writer.add_hparams(
    hparam_dict={'lr': 0.01, 'batch_size': 64},
    metric_dict={'accuracy': 0.95, 'loss': 0.15}
)`} />
              
              <Text mt="md" mb="sm">Grid search with TensorBoard logging:</Text>
              <CodeBlock language="python" code={`for lr in [0.001, 0.01, 0.1]:
    for batch_size in [32, 64, 128]:
        # Train with these hyperparameters
        metrics = train_model(lr, batch_size)
        
        # Log results
        writer.add_hparams(
            {'lr': lr, 'batch_size': batch_size},
            {'best_accuracy': metrics['acc']}
        )`} />

              <Flex direction="column" align="center" mt="md" mb="md">
                <Image
                  src="/assets/python-deep-learning/module3/tensorboard_hparams.png"
                  alt="TensorBoard HParams Dashboard"
                  style={{ maxWidth: 'min(800px, 90vw)', height: 'auto' }}
                  fluid
                />
                <Text size="sm" c="dimmed" mt="xs">Hyperparameter Tuning Visualization</Text>
              </Flex>
            </div>

            {/* Section 9: Profiling */}
            <div>
              <Title order={2} mb="md">9. Performance Profiling</Title>
              
              <Text mb="sm">Basic profiling setup:</Text>
              <CodeBlock language="python" code={`import torch.profiler as profiler

with profiler.profile(
    activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
    on_trace_ready=profiler.tensorboard_trace_handler('./runs/profiling')
) as prof:
    model(input_batch)
    prof.step()`} />
              
              <Text mt="md" mb="sm">Profile specific operations:</Text>
              <CodeBlock language="python" code={`with profiler.record_function("custom_operation"):
    # Your expensive operation
    output = model.forward(input)
    loss = criterion(output, target)`} />
              
              <Text mt="md" mb="sm">Memory profiling:</Text>
              <CodeBlock language="python" code={`# Track memory usage
print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

# Clear cache if needed
torch.cuda.empty_cache()`} />

              <Flex direction="column" align="center" mt="md" mb="md">
                <Image
                  src="/assets/python-deep-learning/module3/tensorboard_profiler.png"
                  alt="TensorBoard PyTorch Profiler"
                  style={{ maxWidth: 'min(800px, 90vw)', height: 'auto' }}
                  fluid
                />
                <Text size="sm" c="dimmed" mt="xs">PyTorch Profiler in TensorBoard</Text>
              </Flex>
            </div>
          </Stack>
        </div>
      </Stack>
    </Container>
  );
};

export default MonitoringVisualization;