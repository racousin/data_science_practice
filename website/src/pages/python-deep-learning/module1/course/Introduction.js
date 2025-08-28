import React from 'react';
import { Flex, Container, Title, Text, Stack, Grid, Image, Paper, List } from '@mantine/core';
import { InlineMath, BlockMath } from 'react-katex';
import 'katex/dist/katex.min.css';
import CodeBlock from 'components/CodeBlock';

const Introduction = () => {
  return (
    <Container size="xl" className="py-6">
      <Stack spacing="xl">
        
          <div data-slide>
            <Title order={1} mb="xl">
              Historical Context and Applications
            </Title>
            <Flex direction="column" align="center">
              <Image
                src="/assets/python-deep-learning/module1/intro.jpg"
                alt="Yutong Liu & The Bigger Picture"
                w={{ base: 400, sm: 600, md: 800 }}
                h="auto"
                fluid
              />
              <Text component="p" ta="center" mt="xs">
                Yutong Liu & The Bigger Picture
              </Text>
            </Flex>
          </div>
          <div data-slide>    <Title order={2} className="mb-6" id="introduction">
      Introduction to Deep Learning
    </Title></div>
<div data-slide>
<Title order={3} className="mb-4">What is Deep Learning?</Title>
    
    <Flex direction="column" align="center">
      <Image
        src="/assets/python-deep-learning/module1/ai_segmentation.png"
        alt="AI Fields Segmentation Diagram"
        w={{ base: 400, sm: 600, md: 800 }}
        h="auto"
        fluid
      />
      <Text component="p" ta="center" mt="xs" size="sm" c="dimmed">
        col_jung
      </Text>
    </Flex>

    {/* Field Definitions */}
<div className="mt-8 mb-8">
  <Grid>
    <Grid.Col span={{ base: 12, md: 4 }}>
        <Title order={4} className="mb-3">
          AI (Artificial Intelligence)
        </Title>
        <Text size="sm" className="mb-3">
          Automation involving algorithms to reproduce cognitive capabilities such as reasoning, perception, and decision-making.
        </Text>
        <Text size="xs" c="dimmed" fs="italic">
          Example: Expert systems, rule-based engines, calculators - human-created logical systems that have existed for decades
        </Text>
    </Grid.Col>
    
    <Grid.Col span={{ base: 12, md: 4 }}>
        <Title order={4} className="mb-3">
          ML (Machine Learning)
        </Title>
        <Text size="sm" className="mb-3">
          Statistical algorithms that automatically learn patterns from data without being explicitly programmed for each task.
        </Text>
        <Text size="xs" c="dimmed" fs="italic">
          Example: Linear regression for house prices, decision trees for loan approval, clustering for customer segmentation
        </Text>
    </Grid.Col>
    
    <Grid.Col span={{ base: 12, md: 4 }}>
        <Title order={4} className="mb-3">
          DL (Deep Learning)
        </Title>
        <Text size="sm" className="mb-3">
          Neural networks with multiple layers (millions of parameters) that can learn complex representations from raw data.
        </Text>
        <Text size="xs" c="dimmed" fs="italic">
          Example: Protein folding prediction (AlphaFold), language modeling (GPT), computer vision for medical imaging
        </Text>
    </Grid.Col>
  </Grid>
</div>
</div>
<div data-slide>
            <Paper className="p-6 bg-gray-50 mb-6">
              <Title order={3} className="mb-4">The Deep Learning Revolution</Title>
              <Text className="mb-4">
                The convergence of three critical factors enabled the deep learning revolution:
              </Text>
              
              <Grid gutter="lg">
                <Grid.Col span={4}>
                  <Paper className="p-4 bg-yellow-50 h-full">
                    <Title order={4} className="mb-3">1. Big Data</Title>
                    <Text size="sm">
                      The internet era generated massive datasets: billions of images, text documents, 
                      videos, and user interactions. This data explosion provided the fuel needed to train 
                      complex models that require millions of examples to learn robust patterns.
                    </Text>
                  </Paper>
                </Grid.Col>
                
                <Grid.Col span={4}>
                  <Paper className="p-4 bg-green-50 h-full">
                    <Title order={4} className="mb-3">2. Computational Power</Title>
                    <Text size="sm">
                      GPUs originally designed for gaming proved perfect for neural network computations. 
                      A single modern GPU can perform trillions of operations per second, enabling training 
                      of models with billions of parameters in days rather than years.
                    </Text>
                  </Paper>
                </Grid.Col>
                
                <Grid.Col span={4}>
                  <Paper className="p-4 bg-blue-50 h-full">
                    <Title order={4} className="mb-3">3. Algorithmic Innovation</Title>
                    <Text size="sm">
                      Breakthrough techniques like ReLU activation, batch normalization, dropout, and 
                      attention mechanisms solved critical training challenges. These innovations made 
                      it practical to train networks with dozens or even hundreds of layers.
                    </Text>
                  </Paper>
                </Grid.Col>
              </Grid>
            </Paper>
          
        </div>

        {/* Historical Evolution */}
        <div data-slide>
            <Title order={2} className="mb-6" id="history">
              Historical Evolution of Deep Learning
            </Title>
                      <div data-slide>

            <Flex direction="column" align="center">
              <Image
                src="/assets/python-deep-learning/module1/ai_history.png"
                alt="Yutong Liu & The Bigger Picture"
                w={{ base: 400, sm: 600, md: 800 }}
                h="auto"
                fluid
              />
              <Text component="p" ta="center" mt="xs">
                wikipedia
              </Text>
            </Flex>
          </div>
            <Paper className="p-6 bg-gradient-to-r from-purple-50 to-pink-50 mb-8">
              <Title order={3} className="mb-4">Timeline of Major Milestones</Title>
              
              <div className="space-y-6">
                {/* 1940s-1960s: The Birth of Neural Networks */}
                <div className="border-l-4 border-purple-500 pl-6">
                  <Title order={4} className="mb-2">1940s-1960s: The Birth of Neural Networks</Title>
                  <Grid gutter="lg">
                    <Grid.Col span={12}>
                      <List>
                        <List.Item>
                          <strong>1943 - McCulloch-Pitts Neuron:</strong> Warren McCulloch and Walter Pitts create the first 
                          mathematical model of a neuron, showing how neurons might perform logical computations.
                          <BlockMath>{`y = \\begin{cases} 1 & \\text{if } \\sum_{i} w_i x_i \\geq \\theta \\\\ 0 & \\text{otherwise} \\end{cases}`}</BlockMath>
                        </List.Item>
                        <List.Item>
                          <strong>1958 - Perceptron:</strong> Frank Rosenblatt develops the perceptron, the first algorithm 
                          that could learn from data. It could solve linearly separable problems through iterative weight updates.
                        </List.Item>
                        <List.Item>
                          <strong>1969 - Perceptrons Book:</strong> Minsky and Papert publish "Perceptrons", highlighting limitations 
                          like the XOR problem, leading to the first "AI Winter" and reduced funding for neural network research.
                        </List.Item>
                      </List>
                    </Grid.Col>
                  </Grid>
                </div>

                {/* 1980s-1990s: The Backpropagation Era */}
                <div className="border-l-4 border-blue-500 pl-6">
                  <Title order={4} className="mb-2">1980s-1990s: The Backpropagation Era</Title>
                  <List>
                    <List.Item>
                      <strong>1986 - Backpropagation Popularized:</strong> Rumelhart, Hinton, and Williams demonstrate 
                      backpropagation's effectiveness, enabling training of multi-layer networks. This algorithm computes 
                      gradients efficiently using the chain rule:
                      <BlockMath>{`\\frac{\\partial L}{\\partial w_{ij}} = \\frac{\\partial L}{\\partial a_j} \\cdot \\frac{\\partial a_j}{\\partial z_j} \\cdot \\frac{\\partial z_j}{\\partial w_{ij}}`}</BlockMath>
                    </List.Item>
                    <List.Item>
                      <strong>1989 - Universal Approximation Theorem:</strong> Cybenko proves that neural networks with one 
                      hidden layer can approximate any continuous function, providing theoretical foundation for deep learning.
                    </List.Item>
                    <List.Item>
                      <strong>1989 - Convolutional Neural Networks:</strong> Yann LeCun develops LeNet for handwritten digit 
                      recognition, introducing convolutional layers that exploit spatial structure in images.
                    </List.Item>
                    <List.Item>
                      <strong>1997 - LSTM:</strong> Hochreiter and Schmidhuber introduce Long Short-Term Memory networks, 
                      solving the vanishing gradient problem in recurrent neural networks.
                    </List.Item>
                  </List>
                </div>

                {/* 2006-2012: The Deep Learning Renaissance */}
                <div className="border-l-4 border-green-500 pl-6">
                  <Title order={4} className="mb-2">2006-2012: The Deep Learning Renaissance</Title>
                  <List>
                    <List.Item>
                      <strong>2006 - Deep Belief Networks:</strong> Geoffrey Hinton introduces layer-wise pretraining using 
                      Restricted Boltzmann Machines, making it practical to train deep networks for the first time.
                    </List.Item>
                    <List.Item>
                      <strong>2009 - ImageNet Dataset:</strong> Fei-Fei Li creates ImageNet with 14 million labeled images 
                      across 20,000 categories, providing the benchmark that would drive computer vision progress.
                    </List.Item>
                    <List.Item>
                      <strong>2012 - AlexNet Breakthrough:</strong> Alex Krizhevsky's CNN wins ImageNet with 15.3% error rate 
                      (vs 26.2% for second place), using GPUs, ReLU activations, and dropout. This marked the definitive 
                      arrival of deep learning.
                      <CodeBlock language="python" code={`# AlexNet Architecture (simplified)
model = nn.Sequential(
    nn.Conv2d(3, 96, kernel_size=11, stride=4),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Conv2d(96, 256, kernel_size=5, padding=2),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    # ... more layers
    nn.Linear(4096, 1000)  # 1000 ImageNet classes
)`} />
                    </List.Item>
                  </List>
                </div>

                {/* 2014-2017: The Architecture Revolution */}
                <div className="border-l-4 border-orange-500 pl-6">
                  <Title order={4} className="mb-2">2014-2017: The Architecture Revolution</Title>
                  <List>
                    <List.Item>
                      <strong>2014 - Generative Adversarial Networks:</strong> Ian Goodfellow introduces GANs, enabling 
                      unprecedented image generation through adversarial training between generator and discriminator networks.
                    </List.Item>
                    <List.Item>
                      <strong>2015 - ResNet:</strong> Kaiming He introduces residual connections, enabling training of networks 
                      with hundreds of layers by solving the degradation problem:
                      <BlockMath>{`F(x) + x`}</BlockMath>
                    </List.Item>
                    <List.Item>
                      <strong>2017 - Transformer Architecture:</strong> Vaswani et al. publish "Attention is All You Need", 
                      introducing self-attention mechanism that would revolutionize NLP:
                      <BlockMath>{`\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V`}</BlockMath>
                    </List.Item>
                  </List>
                </div>

                {/* 2018-Present: The Scale Era */}
                <div className="border-l-4 border-red-500 pl-6">
                  <Title order={4} className="mb-2">2018-Present: The Scale Era</Title>
                  <List>
                    <List.Item>
                      <strong>2018 - BERT:</strong> Google's Bidirectional Encoder Representations from Transformers achieves 
                      state-of-the-art on 11 NLP tasks through self-supervised pretraining on massive text corpora.
                    </List.Item>
                    <List.Item>
                      <strong>2020 - GPT-3:</strong> OpenAI releases 175-billion parameter model showing emergent capabilities 
                      like few-shot learning, code generation, and reasoning without task-specific training.
                    </List.Item>
                    <List.Item>
                      <strong>2022 - Diffusion Models:</strong> DALL-E 2, Midjourney, and Stable Diffusion democratize 
                      AI art generation through denoising diffusion probabilistic models.
                    </List.Item>
                    <List.Item>
                      <strong>2023-2024 - Multimodal Foundation Models:</strong> GPT-4V, Gemini, and Claude demonstrate 
                      understanding across text, images, code, and audio, approaching artificial general intelligence.
                    </List.Item>
                  </List>
                </div>
              </div>
            </Paper>
        </div>

        {/* Real-World Applications */}
        <div data-slide>
            <Title order={2} className="mb-6" id="applications">
              Real-World Applications
            </Title>
            
            <Text size="lg" className="mb-6">
              Deep learning has transformed virtually every field it has touched, enabling capabilities 
              that seemed like science fiction just a decade ago.
            </Text>

            {/* Computer Vision Applications */}
            <Paper className="p-6 bg-blue-50 mb-6">
              <Title order={3} className="mb-4">Computer Vision</Title>
              
              <Grid gutter="lg">
                <Grid.Col span={6}>
                  <Paper className="p-4 bg-white">
                    <Title order={4} className="mb-3">Medical Imaging</Title>
                    <List size="sm">
                      <List.Item><strong>Cancer Detection:</strong> CNNs match or exceed radiologist performance in detecting breast cancer, skin cancer, and lung nodules</List.Item>
                      <List.Item><strong>Retinal Disease:</strong> Diabetic retinopathy screening preventing blindness in millions</List.Item>
                      <List.Item><strong>Medical Segmentation:</strong> Precise tumor boundary detection for radiation therapy planning</List.Item>
                      <List.Item><strong>Drug Discovery:</strong> Predicting molecular properties from chemical structures</List.Item>
                    </List>
                  </Paper>
                </Grid.Col>
                
                <Grid.Col span={6}>
                  <Paper className="p-4 bg-white">
                    <Title order={4} className="mb-3">Autonomous Systems</Title>
                    <List size="sm">
                      <List.Item><strong>Self-Driving Cars:</strong> Real-time object detection, lane detection, path planning</List.Item>
                      <List.Item><strong>Drone Navigation:</strong> Obstacle avoidance and terrain mapping</List.Item>
                      <List.Item><strong>Industrial Robotics:</strong> Quality control, defect detection, assembly verification</List.Item>
                      <List.Item><strong>Agriculture:</strong> Crop disease detection, yield prediction, precision farming</List.Item>
                    </List>
                  </Paper>
                </Grid.Col>
              </Grid>

              <CodeBlock language="python" code={`# Example: Object Detection with Pre-trained Model
import torch
import torchvision

# Load pre-trained Faster R-CNN model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Detect objects in image
predictions = model(image_tensor)
boxes = predictions[0]['boxes']  # Bounding boxes
labels = predictions[0]['labels']  # Class labels  
scores = predictions[0]['scores']  # Confidence scores`} />
            </Paper>

            {/* Natural Language Processing */}
            <Paper className="p-6 bg-green-50 mb-6">
              <Title order={3} className="mb-4">Natural Language Processing</Title>
              
              <Grid gutter="lg">
                <Grid.Col span={4}>
                  <Paper className="p-4 bg-white">
                    <Title order={4} className="mb-2">Language Understanding</Title>
                    <List size="sm">
                      <List.Item>Machine Translation (Google Translate)</List.Item>
                      <List.Item>Sentiment Analysis</List.Item>
                      <List.Item>Named Entity Recognition</List.Item>
                      <List.Item>Question Answering Systems</List.Item>
                    </List>
                  </Paper>
                </Grid.Col>
                
                <Grid.Col span={4}>
                  <Paper className="p-4 bg-white">
                    <Title order={4} className="mb-2">Language Generation</Title>
                    <List size="sm">
                      <List.Item>ChatGPT and Conversational AI</List.Item>
                      <List.Item>Code Generation (GitHub Copilot)</List.Item>
                      <List.Item>Content Creation</List.Item>
                      <List.Item>Summarization</List.Item>
                    </List>
                  </Paper>
                </Grid.Col>
                
                <Grid.Col span={4}>
                  <Paper className="p-4 bg-white">
                    <Title order={4} className="mb-2">Speech Processing</Title>
                    <List size="sm">
                      <List.Item>Speech Recognition (Siri, Alexa)</List.Item>
                      <List.Item>Text-to-Speech Synthesis</List.Item>
                      <List.Item>Voice Cloning</List.Item>
                      <List.Item>Real-time Translation</List.Item>
                    </List>
                  </Paper>
                </Grid.Col>
              </Grid>
            </Paper>

            {/* Scientific Applications */}
            <Paper className="p-6 bg-yellow-50 mb-6">
              <Title order={3} className="mb-4">Scientific Breakthroughs</Title>
              
              <Grid gutter="lg">
                <Grid.Col span={6}>
                  <Paper className="p-4 bg-white">
                    <Title order={4} className="mb-3">AlphaFold: Protein Structure Prediction</Title>
                    <Text size="sm" className="mb-3">
                      DeepMind's AlphaFold solved the 50-year protein folding problem, predicting 3D structures 
                      from amino acid sequences with atomic accuracy. This breakthrough accelerates drug discovery 
                      and our understanding of biological processes.
                    </Text>
                    <BlockMath>{`\\text{RMSD} < 1.0 \\text{ Å for } 95\\% \\text{ of proteins}`}</BlockMath>
                  </Paper>
                </Grid.Col>
                
                <Grid.Col span={6}>
                  <Paper className="p-4 bg-white">
                    <Title order={4} className="mb-3">Climate and Weather Prediction</Title>
                    <Text size="sm" className="mb-3">
                      Neural networks now outperform traditional numerical weather prediction models, providing 
                      accurate forecasts at a fraction of the computational cost. Applications include:
                    </Text>
                    <List size="sm">
                      <List.Item>GraphCast: 10-day weather forecasting</List.Item>
                      <List.Item>Climate change modeling</List.Item>
                      <List.Item>Extreme weather event prediction</List.Item>
                    </List>
                  </Paper>
                </Grid.Col>
              </Grid>
            </Paper>

            {/* Generative AI */}
            <Paper className="p-6 bg-purple-50 mb-6">
              <Title order={3} className="mb-4">Generative AI Revolution</Title>
              
              <Grid gutter="lg">
                <Grid.Col span={12}>
                  <div className="grid grid-cols-3 gap-4">
                    <Paper className="p-4 bg-white">
                      <Title order={4} className="mb-2">Image Generation</Title>
                      <List size="sm">
                        <List.Item>DALL-E, Midjourney, Stable Diffusion</List.Item>
                        <List.Item>Photorealistic face generation</List.Item>
                        <List.Item>Style transfer and artistic creation</List.Item>
                        <List.Item>Image editing and inpainting</List.Item>
                      </List>
                    </Paper>
                    
                    <Paper className="p-4 bg-white">
                      <Title order={4} className="mb-2">Video & Animation</Title>
                      <List size="sm">
                        <List.Item>Text-to-video generation (Runway, Pika)</List.Item>
                        <List.Item>Deepfakes and face swapping</List.Item>
                        <List.Item>Motion capture and animation</List.Item>
                        <List.Item>Video enhancement and restoration</List.Item>
                      </List>
                    </Paper>
                    
                    <Paper className="p-4 bg-white">
                      <Title order={4} className="mb-2">Audio & Music</Title>
                      <List size="sm">
                        <List.Item>Music generation (MuseNet, Jukebox)</List.Item>
                        <List.Item>Voice synthesis and cloning</List.Item>
                        <List.Item>Audio enhancement and separation</List.Item>
                        <List.Item>Real-time audio effects</List.Item>
                      </List>
                    </Paper>
                  </div>
                </Grid.Col>
              </Grid>
            </Paper>
        </div>

        {/* Data in Deep Learning */}
        <div data-slide>
            <Title order={2} className="mb-6" id="data">
              Data: The Fuel of Deep Learning
            </Title>
            
            <Paper className="p-6 bg-gradient-to-r from-indigo-50 to-purple-50 mb-6">
              <Title order={3} className="mb-4">The Critical Role of Data</Title>
              <Text size="lg" className="mb-4">
                Data is the foundation upon which all deep learning systems are built. The quality, quantity, 
                and diversity of training data directly determine model performance and capabilities.
              </Text>
              
              <Grid gutter="lg">
                <Grid.Col span={6}>
                  <Paper className="p-4 bg-white">
                    <Title order={4} className="mb-3">Data Requirements</Title>
                    <List>
                      <List.Item>
                        <strong>Volume:</strong> Deep networks typically need thousands to millions of examples
                        <BlockMath>{`N \\propto d^2`}</BlockMath>
                        <Text size="sm" c="dimmed">Sample complexity grows with model capacity</Text>
                      </List.Item>
                      <List.Item>
                        <strong>Quality:</strong> Clean labels, consistent annotations, minimal noise
                      </List.Item>
                      <List.Item>
                        <strong>Diversity:</strong> Representative of real-world distribution
                      </List.Item>
                      <List.Item>
                        <strong>Balance:</strong> Adequate representation of all classes/scenarios
                      </List.Item>
                    </List>
                  </Paper>
                </Grid.Col>
                
                <Grid.Col span={6}>
                  <Paper className="p-4 bg-white">
                    <Title order={4} className="mb-3">Data Challenges</Title>
                    <List>
                      <List.Item>
                        <strong>Annotation Cost:</strong> Manual labeling is expensive and time-consuming
                      </List.Item>
                      <List.Item>
                        <strong>Privacy Concerns:</strong> GDPR, HIPAA compliance for sensitive data
                      </List.Item>
                      <List.Item>
                        <strong>Bias and Fairness:</strong> Historical biases encoded in training data
                      </List.Item>
                      <List.Item>
                        <strong>Distribution Shift:</strong> Training and deployment distributions differ
                      </List.Item>
                    </List>
                  </Paper>
                </Grid.Col>
              </Grid>
            </Paper>

            {/* Major Datasets */}
            <Paper className="p-6 bg-gray-50 mb-6">
              <Title order={3} className="mb-4">Landmark Datasets in Deep Learning</Title>
              
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                  <thead>
                    <tr style={{ backgroundColor: '#f8f9fa' }}>
                      <th style={{ border: '1px solid #dee2e6', padding: '12px', textAlign: 'left' }}>Dataset</th>
                      <th style={{ border: '1px solid #dee2e6', padding: '12px', textAlign: 'left' }}>Domain</th>
                      <th style={{ border: '1px solid #dee2e6', padding: '12px', textAlign: 'left' }}>Size</th>
                      <th style={{ border: '1px solid #dee2e6', padding: '12px', textAlign: 'left' }}>Impact</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td style={{ border: '1px solid #dee2e6', padding: '8px' }}><strong>ImageNet</strong></td>
                      <td style={{ border: '1px solid #dee2e6', padding: '8px' }}>Computer Vision</td>
                      <td style={{ border: '1px solid #dee2e6', padding: '8px' }}>14M images, 20K classes</td>
                      <td style={{ border: '1px solid #dee2e6', padding: '8px' }}>Enabled CNN revolution, transfer learning</td>
                    </tr>
                    <tr>
                      <td style={{ border: '1px solid #dee2e6', padding: '8px' }}><strong>COCO</strong></td>
                      <td style={{ border: '1px solid #dee2e6', padding: '8px' }}>Object Detection</td>
                      <td style={{ border: '1px solid #dee2e6', padding: '8px' }}>330K images, 80 classes</td>
                      <td style={{ border: '1px solid #dee2e6', padding: '8px' }}>Standard for detection, segmentation</td>
                    </tr>
                    <tr>
                      <td style={{ border: '1px solid #dee2e6', padding: '8px' }}><strong>Common Crawl</strong></td>
                      <td style={{ border: '1px solid #dee2e6', padding: '8px' }}>NLP</td>
                      <td style={{ border: '1px solid #dee2e6', padding: '8px' }}>Petabytes of web text</td>
                      <td style={{ border: '1px solid #dee2e6', padding: '8px' }}>Training data for GPT, BERT</td>
                    </tr>
                    <tr>
                      <td style={{ border: '1px solid #dee2e6', padding: '8px' }}><strong>LibriSpeech</strong></td>
                      <td style={{ border: '1px solid #dee2e6', padding: '8px' }}>Speech</td>
                      <td style={{ border: '1px solid #dee2e6', padding: '8px' }}>1000 hours audio</td>
                      <td style={{ border: '1px solid #dee2e6', padding: '8px' }}>ASR model development</td>
                    </tr>
                    <tr>
                      <td style={{ border: '1px solid #dee2e6', padding: '8px' }}><strong>LAION-5B</strong></td>
                      <td style={{ border: '1px solid #dee2e6', padding: '8px' }}>Multimodal</td>
                      <td style={{ border: '1px solid #dee2e6', padding: '8px' }}>5.85B image-text pairs</td>
                      <td style={{ border: '1px solid #dee2e6', padding: '8px' }}>Enabled open-source DALL-E alternatives</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </Paper>

            {/* Data Processing Pipeline */}
            <Paper className="p-6 bg-blue-50 mb-6">
              <Title order={3} className="mb-4">Data Processing Pipeline</Title>
              
              <CodeBlock language="python" code={`import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

# Data Augmentation Pipeline
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomRotation(degrees=15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet statistics
                        std=[0.229, 0.224, 0.225])
])

# Custom Dataset Class
class CustomDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.samples = self.load_samples()
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample

# DataLoader for Batch Processing
dataset = CustomDataset('path/to/data', transform=transform)
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True  # Faster GPU transfer
)`} />
            </Paper>

            {/* Data Efficiency Techniques */}
            <Paper className="p-6 bg-green-50">
              <Title order={3} className="mb-4">Modern Data Efficiency Techniques</Title>
              
              <Grid gutter="lg">
                <Grid.Col span={6}>
                  <Paper className="p-4 bg-white">
                    <Title order={4} className="mb-3">Self-Supervised Learning</Title>
                    <Text size="sm" className="mb-3">
                      Learning from unlabeled data through pretext tasks:
                    </Text>
                    <List size="sm">
                      <List.Item><strong>Contrastive Learning:</strong> SimCLR, MoCo - learning by comparing augmented views</List.Item>
                      <List.Item><strong>Masked Prediction:</strong> BERT (masked language), MAE (masked images)</List.Item>
                      <List.Item><strong>Generative Pretraining:</strong> GPT learns by predicting next tokens</List.Item>
                    </List>
                  </Paper>
                </Grid.Col>
                
                <Grid.Col span={6}>
                  <Paper className="p-4 bg-white">
                    <Title order={4} className="mb-3">Data Augmentation</Title>
                    <Text size="sm" className="mb-3">
                      Artificially expanding dataset diversity:
                    </Text>
                    <List size="sm">
                      <List.Item><strong>Classical:</strong> Rotations, flips, crops, color jittering</List.Item>
                      <List.Item><strong>Advanced:</strong> MixUp, CutMix, AutoAugment</List.Item>
                      <List.Item><strong>Synthetic:</strong> GANs for generating training data</List.Item>
                    </List>
                  </Paper>
                </Grid.Col>
              </Grid>
              
              <Grid gutter="lg" className="mt-4">
                <Grid.Col span={6}>
                  <Paper className="p-4 bg-white">
                    <Title order={4} className="mb-3">Transfer Learning</Title>
                    <Text size="sm" className="mb-3">
                      Leveraging pre-trained models:
                    </Text>
                    <List size="sm">
                      <List.Item><strong>Feature Extraction:</strong> Use pre-trained features as-is</List.Item>
                      <List.Item><strong>Fine-tuning:</strong> Adapt pre-trained weights to new task</List.Item>
                      <List.Item><strong>Domain Adaptation:</strong> Bridge distribution gaps</List.Item>
                    </List>
                  </Paper>
                </Grid.Col>
                
                <Grid.Col span={6}>
                  <Paper className="p-4 bg-white">
                    <Title order={4} className="mb-3">Few-Shot Learning</Title>
                    <Text size="sm" className="mb-3">
                      Learning from limited examples:
                    </Text>
                    <List size="sm">
                      <List.Item><strong>Meta-Learning:</strong> Learning to learn from few examples</List.Item>
                      <List.Item><strong>Prototypical Networks:</strong> Classification via prototype matching</List.Item>
                      <List.Item><strong>Prompt Engineering:</strong> Leveraging large models with examples</List.Item>
                    </List>
                  </Paper>
                </Grid.Col>
              </Grid>
            </Paper>
        </div>

        {/* Summary */}
        <div data-slide>
    <Title order={2} className="mb-6">Part 1 Summary: Historical Context and Applications</Title>
            
            <Grid gutter="lg">
              <Grid.Col span={6}>
                <Paper className="p-6 bg-gradient-to-br from-blue-50 to-blue-100 h-full">
                  <Title order={3} className="mb-4">Key Historical Insights</Title>
                  <List spacing="md">
                    <List.Item>Deep learning evolved through multiple waves, each driven by algorithmic breakthroughs</List.Item>
                    <List.Item>The 2012 AlexNet moment marked the definitive arrival of the deep learning era</List.Item>
                    <List.Item>Modern progress driven by scale: bigger models, more data, more compute</List.Item>
                    <List.Item>Transformers and attention mechanisms revolutionized both NLP and computer vision</List.Item>
                  </List>
                </Paper>
              </Grid.Col>
              
              <Grid.Col span={6}>
                <Paper className="p-6 bg-gradient-to-br from-green-50 to-green-100 h-full">
                  <Title order={3} className="mb-4">Application Landscape</Title>
                  <List spacing="md">
                    <List.Item>Deep learning has achieved superhuman performance in many domains</List.Item>
                    <List.Item>Real-world deployment spans healthcare, autonomous systems, science, and creativity</List.Item>
                    <List.Item>Generative AI represents a paradigm shift in human-computer interaction</List.Item>
                    <List.Item>Data quality and quantity remain critical factors for success</List.Item>
                  </List>
                </Paper>
              </Grid.Col>
            </Grid>
            
            <Paper className="p-6 bg-gradient-to-r from-purple-50 to-pink-50 mt-6">
              <Title order={3} className="mb-4 text-center">The Deep Learning Revolution Continues</Title>
              <Text size="lg" className="text-center">
                We are witnessing the early stages of a technological revolution that will reshape society. 
                Understanding the mathematical foundations and practical techniques of deep learning is essential 
                for participating in this transformation.
              </Text>
            </Paper>
        </div>

      </Stack>
    </Container>
  );
};

export default Introduction;