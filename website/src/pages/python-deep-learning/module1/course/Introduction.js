import React from 'react';
import { Flex, Container, Title, Text, Stack, Grid, Image, Paper, List } from '@mantine/core';
import { InlineMath, BlockMath } from 'react-katex';
import 'katex/dist/katex.min.css';
import CodeBlock from 'components/CodeBlock';

const Introduction = () => {
  return (
    <Container size="xl" py="xl">
      <Stack spacing="xl">
        
          <div data-slide>
            <Title order={1} mb="xl">
              Historical Context and Applications
            </Title>
            <Flex direction="column" align="center">
              <Image
                src="/assets/python-deep-learning/module1/im0.png"
                alt="Yutong Liu & The Bigger Picture"
                style={{ maxWidth: 'min(800px, 90vw)', height: 'auto' }}
                fluid
              />
              <Text component="p" ta="center" mt="xs">
                A simple multi layer perceptron
              </Text>
            </Flex>
          </div>
          <div data-slide>    <Title order={2} mb="xl" id="introduction">
      Introduction to Deep Learning
    </Title></div>
<div data-slide>
<Title order={3} mb="md">What is Deep Learning?</Title>
    
    <Flex direction="column" align="center">
      <Image
        src="/assets/python-deep-learning/module1/ai_segmentation.png"
        alt="AI Fields Segmentation Diagram"
        style={{ maxWidth: 'min(800px, 90vw)', height: 'auto' }}
        fluid
      />
      <Text component="p" ta="center" mt="xs" size="sm" c="dimmed">
        https://medium.com/geekculture/ai-revolution-your-fast-paced-introduction-to-machine-learning-914ce9b6ddf
      </Text>
    </Flex>

    {/* Field Definitions */}
<div className="mt-8 mb-8">
  <Grid>
    <Grid.Col span={{ base: 12, md: 4 }}>
        <Title order={4} mb="sm">
          AI (Artificial Intelligence)
        </Title>
        <Text size="sm" mb="md">
          Automation involving algorithms to reproduce cognitive capabilities such as reasoning, perception, and decision-making.
        </Text>
        <Text size="xs" c="dimmed" fs="italic">
          Example: Expert systems, rule-based engines, calculators - human-created logical systems that have existed for decades
        </Text>
    </Grid.Col>
    
    <Grid.Col span={{ base: 12, md: 4 }}>
        <Title order={4} mb="sm">
          ML (Machine Learning)
        </Title>
        <Text size="sm" mb="md">
          Statistical algorithms that automatically learn patterns from data without being explicitly programmed for each task.
        </Text>
        <Text size="xs" c="dimmed" fs="italic">
          Example: Linear regression for house prices, decision trees for loan approval, clustering for customer segmentation
        </Text>
    </Grid.Col>
    
    <Grid.Col span={{ base: 12, md: 4 }}>
        <Title order={4} mb="sm">
          DL (Deep Learning)
        </Title>
        <Text size="sm" mb="md">
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
              <Title order={3} mb="md">The Deep Learning Revolution</Title>
                      
        <Flex direction="column" align="center" mb="md">

        
                                      <Image
        src="/assets/python-deep-learning/module1/big_data0.png"
        alt="AI Fields Segmentation Diagram"
        style={{ maxWidth: 'min(800px, 90vw)', height: 'auto' }}
        fluid
      />
      <Text component="p" ta="center" mt="xs" size="sm" c="dimmed">
        Source: https://trends.google.fr/trends/explore?date=all&q=deep%20learning&hl=en
      </Text>
              <Text className="mb-4">
                The convergence of three critical factors enabled the deep learning revolution:
              </Text>
              </Flex>
              </div>
              <div data-slide>
                        <Flex direction="column" align="center">
      <Image
        src="/assets/python-deep-learning/module1/bigdata1.png"
        alt="AI Fields Segmentation Diagram"
        style={{ maxWidth: 'min(800px, 90vw)', height: 'auto' }}
        fluid
      />
      <Text component="p" ta="center" mt="xs" size="sm" c="dimmed">
        Source: https://www.domo.com/learn/datanever-sleeps-7
      </Text>
    </Flex>
                    <Title order={4} mb="sm">1. Big Data</Title>
                    <Text size="sm">
                      The internet era generated massive datasets: billions of images, text documents, 
                      videos, and user interactions. This data explosion provided the fuel needed to train 
                      complex models that require millions of examples to learn robust patterns.
                    </Text>

</div>
<div data-slide>
<Flex direction="column" align="center">
                        <Image
        src="/assets/python-deep-learning/module1/bigdata2.png"
        alt="AI Fields Segmentation Diagram"
        style={{ maxWidth: 'min(800px, 90vw)', height: 'auto' }}
        fluid
      />
      </Flex>
      <Text component="p" ta="center" mt="xs" size="sm" c="dimmed">
        Source: https://www.researchgate.net/figure/The-exponential-progress-of-computing-power-from-1900-to-2013-with-projections-into_fig1_335422453
      </Text>
                    <Title order={4} mb="sm">2. Computational Power</Title>
                    
                    <Text size="sm">
                      GPUs originally designed for gaming proved perfect for neural network computations. 
                      A single modern GPU can perform trillions of operations per second, enabling training 
                      of models with billions of parameters in days rather than years.
                    </Text>
</div>
<div data-slide>
              <Grid gutter="lg">
                <Grid.Col span={6}>
                                          <Image
        src="/assets/python-deep-learning/module1/bigdata3.png"
        alt="AI Fields Segmentation Diagram"
        style={{ maxWidth: 'min(600px, 90vw)', height: 'auto' }}
        fluid
      />
            <Text component="p" ta="center" mt="xs" size="sm" c="dimmed">
        Source: https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)      </Text>
                </Grid.Col>
                <Grid.Col span={6}>
                                          <Image
        src="/assets/python-deep-learning/module1/Pytorch_logo.png"
        alt="AI Fields Segmentation Diagram"
        style={{ maxWidth: 'min(600px, 90vw)', height: 'auto' }}
        fluid
      />
            <Text component="p" ta="center" mt="xs" size="sm" c="dimmed">
        Source: https://pytorch.org/     </Text>
                </Grid.Col>
              </Grid>




                    <Title order={4} mb="sm">3. Algorithmic Innovation</Title>
                    <Text size="sm">
                      Breakthrough techniques like ReLU activation, batch normalization, dropout, and 
                      attention mechanisms solved critical training challenges. These innovations made 
                      it practical to train networks with dozens or even hundreds of layers. Open source code with big communities.
                    </Text>

</div>
          

        {/* Historical Evolution */}
        <div data-slide>
            <Title order={2} mb="xl" id="history">
              Historical Evolution of Deep Learning
            </Title>
                     

            <Flex direction="column" align="center">
              <Image
                src="/assets/python-deep-learning/module1/ai_history.png"
                alt="Yutong Liu & The Bigger Picture"
                style={{ maxWidth: 'min(800px, 90vw)', height: 'auto' }}
                fluid
              />
              <Text component="p" ta="center" mt="xs">
                https://en.wikipedia.org/wiki/Timeline_of_artificial_intelligence
              </Text>
            </Flex>
          </div>
          <div data-slide>
                        <Flex direction="column" align="center">
              <Image
                src="/assets/python-deep-learning/module1/agi_buble.png"
                alt="Yutong Liu & The Bigger Picture"
                style={{ maxWidth: 'min(800px, 90vw)', height: 'auto' }}
                fluid
              />
              <Text component="p" ta="center" mt="xs">
                https://marketoonist.com/2023/04/navigating-ai-hype.html
              </Text>
            </Flex>
            Artificial general intelligence (AGI) refers to the hypothetical intelligence of a machine that possesses the ability to understand or learn any intellectual task that a human being can. However, deep learning excels at pattern recognition tasks but struggles with theoretical breakthroughs in physics, proving mathematical theorems like the Riemann Hypothesis, and understanding consciousness - areas requiring new abstractions rather than data pattern matching, while also being highly energy-intensive with massive computational resource requirements.
          </div>
<div data-slide>
              <Title order={3} mb="md">Timeline of Major Milestones</Title>

                  <Title order={4} className="mb-2">1940s-1960s: The Birth of Neural Networks</Title>
                  <Grid gutter="lg">
                    <Grid.Col span={12}>
                      <List>
                        <List.Item>
                          <strong>1943 - McCulloch-Pitts Neuron:</strong> Warren McCulloch and Walter Pitts create the first 
                          mathematical model of a neuron, showing how neurons might perform logical computations.
                          <BlockMath>{`y = \\begin{cases} 1 & \\text{if } \\sum_{i} w_i x_i \\geq \\theta \\\\ 0 & \\text{otherwise} \\end{cases}`}</BlockMath>
                          <a href="https://www.cs.cmu.edu/~epxing/Class/10715/reading/McCulloch.and.Pitts.pdf" target="_blank" rel="noopener noreferrer">[Original Paper]</a>
                        </List.Item>
                        <List.Item>
                          <strong>1958 - Perceptron:</strong> Frank Rosenblatt develops the perceptron, the first algorithm 
                          that could learn from data. It could solve linearly separable problems through iterative weight updates.
                          <a href="https://www.semanticscholar.org/paper/The-perceptron:-a-probabilistic-model-for-storage-Rosenblatt/5d11aad09f65431b5d3cb1d85328743c9e53ba96" target="_blank" rel="noopener noreferrer">[Rosenblatt 1958]</a>
                        </List.Item>
                        <List.Item>
                          <strong>1969 - Perceptrons Book:</strong> Minsky and Papert publish "Perceptrons", highlighting limitations 
                          like the XOR problem, leading to the first "AI Winter" and reduced funding for neural network research.
                          <a href="https://mitpress.mit.edu/9780262630221/perceptrons/" target="_blank" rel="noopener noreferrer">[MIT Press]</a>
                        </List.Item>
                      </List>
                    </Grid.Col>
                  </Grid>
                </div>
<div data-slide>
                {/* 1980s-1990s: The Backpropagation Era */}
                <div className="border-l-4 border-blue-500 pl-6">
                  <Title order={4} className="mb-2">1980s-1990s: The Backpropagation Era</Title>
                  <List>
                    <List.Item>
                      <strong>1986 - Backpropagation Popularized:</strong> Rumelhart, Hinton, and Williams demonstrate 
                      backpropagation's effectiveness, enabling training of multi-layer networks. This algorithm computes 
                      gradients efficiently using the chain rule:
                      <BlockMath>{`\\frac{\\partial L}{\\partial w_{ij}} = \\frac{\\partial L}{\\partial a_j} \\cdot \\frac{\\partial a_j}{\\partial z_j} \\cdot \\frac{\\partial z_j}{\\partial w_{ij}}`}</BlockMath>
                      <a href="https://www.nature.com/articles/323533a0" target="_blank" rel="noopener noreferrer">[Nature Paper]</a>
                    </List.Item>
                    <List.Item>
                      <strong>1989 - Universal Approximation Theorem:</strong> Cybenko proves that neural networks with one 
                      hidden layer can approximate any continuous function, providing theoretical foundation for deep learning.
                      <a href="https://arxiv.org/html/2407.12895v1" target="_blank" rel="noopener noreferrer">[Survey on UAT]</a> | 
                      <a href="https://www.sciencedirect.com/science/article/abs/pii/0893608089900208" target="_blank" rel="noopener noreferrer">[Hornik et al.]</a>
                    </List.Item>
                    <List.Item>
                      <strong>1989 - Convolutional Neural Networks:</strong> Yann LeCun develops LeNet for handwritten digit 
                      recognition, introducing convolutional layers that exploit spatial structure in images.
                      <a href="http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf" target="_blank" rel="noopener noreferrer">[LeCun et al. 1998]</a> | 
                      <a href="http://yann.lecun.com/exdb/publis/pdf/lecun-89e.pdf" target="_blank" rel="noopener noreferrer">[1989 Original]</a>
                    </List.Item>
                    <List.Item>
                      <strong>1997 - LSTM:</strong> Hochreiter and Schmidhuber introduce Long Short-Term Memory networks, 
                      solving the vanishing gradient problem in recurrent neural networks.
                      <a href="https://dl.acm.org/doi/10.1162/neco.1997.9.8.1735" target="_blank" rel="noopener noreferrer">[Neural Computation]</a> | 
                      <a href="https://www.bioinf.jku.at/publications/older/2604.pdf" target="_blank" rel="noopener noreferrer">[Original PDF]</a>
                    </List.Item>
                  </List>
                </div>
</div>
<div data-slide>
                {/* 2006-2012: The Deep Learning Renaissance */}
                <div className="border-l-4 border-green-500 pl-6">
                  <Title order={4} className="mb-2">2006-2012: The Deep Learning Renaissance</Title>
                  <List>
                    <List.Item>
                      <strong>2006 - Deep Belief Networks:</strong> Geoffrey Hinton introduces layer-wise pretraining using 
                      Restricted Boltzmann Machines, making it practical to train deep networks for the first time.
                      <a href="https://www.cs.toronto.edu/~hinton/" target="_blank" rel="noopener noreferrer">[Hinton's Publications]</a>
                    </List.Item>
                    <List.Item>
                      <strong>2009 - ImageNet Dataset:</strong> Fei-Fei Li creates ImageNet with 14 million labeled images 
                      across 20,000 categories, providing the benchmark that would drive computer vision progress.
                      <a href="http://www.image-net.org/" target="_blank" rel="noopener noreferrer">[ImageNet]</a>
                    </List.Item>
                    <List.Item>
                      <strong>2012 - AlexNet Breakthrough:</strong> Alex Krizhevsky's CNN wins ImageNet with 15.3% error rate 
                      (vs 26.2% for second place), using GPUs, ReLU activations, and dropout. This marked the definitive 
                      arrival of deep learning.
                      <a href="https://proceedings.neurips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf" target="_blank" rel="noopener noreferrer">[NIPS 2012]</a> | 
                      <a href="https://dl.acm.org/doi/10.1145/3065386" target="_blank" rel="noopener noreferrer">[Communications ACM]</a>
                    </List.Item>
                  </List>
                </div>
</div>
<div data-slide>
                {/* 2014-2017: The Architecture Revolution */}
                <div className="border-l-4 border-orange-500 pl-6">
                  <Title order={4} className="mb-2">2014-2017: The Architecture Revolution</Title>
                  <List>
                    <List.Item>
                      <strong>2014 - Generative Adversarial Networks:</strong> Ian Goodfellow introduces GANs, enabling 
                      unprecedented image generation through adversarial training between generator and discriminator networks.
                      <a href="https://arxiv.org/abs/1406.2661" target="_blank" rel="noopener noreferrer">[arXiv:1406.2661]</a>
                    </List.Item>
                    <List.Item>
                      <strong>2015 - ResNet:</strong> Kaiming He introduces residual connections, enabling training of networks 
                      with hundreds of layers by solving the degradation problem:
                      <BlockMath>{`F(x) + x`}</BlockMath>
                      <a href="https://arxiv.org/abs/1512.03385" target="_blank" rel="noopener noreferrer">[arXiv:1512.03385]</a>
                    </List.Item>
                    <List.Item>
                      <strong>2017 - Transformer Architecture:</strong> Vaswani et al. publish "Attention is All You Need", 
                      introducing self-attention mechanism that would revolutionize NLP:
                      <BlockMath>{`\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V`}</BlockMath>
                      <a href="https://arxiv.org/abs/1706.03762" target="_blank" rel="noopener noreferrer">[arXiv:1706.03762]</a> | 
                      <a href="https://papers.nips.cc/paper/7181-attention-is-all-you-need" target="_blank" rel="noopener noreferrer">[NIPS 2017]</a>
                    </List.Item>
                  </List>
                </div>
</div>
<div data-slide>
                {/* 2018-Present: The Scale Era */}
                <div className="border-l-4 border-red-500 pl-6">
                  <Title order={4} className="mb-2">2018-Present: The Scale Era</Title>
                  <List>
                    <List.Item>
                      <strong>2018 - BERT:</strong> Google's Bidirectional Encoder Representations from Transformers achieves 
                      state-of-the-art on 11 NLP tasks through self-supervised pretraining on massive text corpora.
                      <a href="https://arxiv.org/abs/1810.04805" target="_blank" rel="noopener noreferrer">[arXiv:1810.04805]</a>
                    </List.Item>
                    <List.Item>
                      <strong>2020 - GPT-3:</strong> OpenAI releases 175-billion parameter model showing emergent capabilities 
                      like few-shot learning, code generation, and reasoning without task-specific training.
                      <a href="https://arxiv.org/abs/2005.14165" target="_blank" rel="noopener noreferrer">[arXiv:2005.14165]</a>
                    </List.Item>
                    <List.Item>
                      <strong>2022 - Diffusion Models:</strong> DALL-E 2, Midjourney, and Stable Diffusion democratize 
                      AI art generation through denoising diffusion probabilistic models.
                      <a href="https://arxiv.org/abs/2006.11239" target="_blank" rel="noopener noreferrer">[DDPM arXiv:2006.11239]</a> | 
                      <a href="https://arxiv.org/abs/2112.10752" target="_blank" rel="noopener noreferrer">[DALL-E 2 arXiv:2112.10752]</a>
                    </List.Item>
                    <List.Item>
                      <strong>2023-2024 - Multimodal Foundation Models:</strong> GPT-4V, Gemini, and Claude demonstrate 
                      understanding across text, images, code, and audio, approaching artificial general intelligence.
                      <a href="https://arxiv.org/abs/2303.08774" target="_blank" rel="noopener noreferrer">[GPT-4 arXiv:2303.08774]</a> | 
                      <a href="https://arxiv.org/abs/2312.11805" target="_blank" rel="noopener noreferrer">[Gemini arXiv:2312.11805]</a>
                    </List.Item>
                  </List>
                </div>
              </div>
                   {/* Real-World Applications */}
        <div data-slide>

      <Title order={2} mb="xl" id="introduction">Real-World Applications</Title>
      
      <Text size="lg" className="mb-6">
        Deep learning has revolutionized virtually every field it has touched, enabling capabilities 
        that seemed impossible just a decade ago.
      </Text>
</div>
<div data-slide>
      {/* Computer Vision */}
      <Paper className="p-6 mb-6">
        <Title order={3} className="mb-2">Computer Vision</Title>
        <Text size="sm" className="mb-4" color="dimmed">
          Enabling machines to understand and interpret visual information from the world
        </Text>
        
        <Flex direction="column" align="center" mb="md">
          <Image
            src="/assets/python-deep-learning/module1/object_detetcion.png"
            alt="Computer Vision Applications"
            style={{ maxWidth: 'min(800px, 90vw)', height: 'auto' }}
            fluid
          />
        </Flex>
                          <Text component="p" ta="center" mt="xs">
            Source: ImageNet
          </Text>
        
        <List>
          <List.Item>
            <strong>Medical Imaging:</strong> Cancer detection, radiology AI, pathology analysis
            <a href="https://www.nature.com/articles/s41591-020-0842-3" target="_blank" rel="noopener noreferrer">[Nature Medicine 2020]</a>
          </List.Item>
          <List.Item>
            <strong>Autonomous Vehicles:</strong> Self-driving cars, drones, robotic navigation
            <a href="https://arxiv.org/abs/2308.05731" target="_blank" rel="noopener noreferrer">[Survey 2023]</a>
          </List.Item>
          <List.Item><strong>Face Recognition:</strong> Security systems, device authentication, photo organization</List.Item>
          <List.Item><strong>Industrial QC:</strong> Defect detection, assembly verification, quality control</List.Item>
          <List.Item><strong>Agriculture:</strong> Crop monitoring, disease detection, yield prediction</List.Item>
        </List>
      </Paper>
</div>
<div data-slide>
      {/* Natural Language Processing */}
      <Paper className="p-6 mb-6">
        <Title order={3} className="mb-2">Natural Language Processing</Title>
        <Text size="sm" className="mb-4" color="dimmed">
          Understanding, generating, and translating human language
        </Text>
        
        <Flex direction="column" align="center" mb="md">
          <Image
            src="/assets/python-deep-learning/module1/ai_chatbot.png"
            alt="NLP Applications"
            style={{ maxWidth: 'min(800px, 90vw)', height: 'auto' }}
            fluid
          />
        </Flex>
                  <Text component="p" ta="center" mt="xs">
            Source: https://openai.com/
          </Text>
        <List>
          <List.Item>
            <strong>Language Models:</strong> ChatGPT, Claude, Gemini for conversation and assistance
            <a href="https://arxiv.org/abs/2303.08774" target="_blank" rel="noopener noreferrer">[GPT-4]</a>
            <a href="https://arxiv.org/abs/2312.11805" target="_blank" rel="noopener noreferrer">[Gemini]</a>
          </List.Item>
          <List.Item>
            <strong>Machine Translation:</strong> Real-time translation across 100+ languages
            <a href="https://arxiv.org/abs/1706.03762" target="_blank" rel="noopener noreferrer">[Transformer]</a>
          </List.Item>
          <List.Item>
            <strong>Code Generation:</strong> GitHub Copilot, automated programming assistance
            <a href="https://arxiv.org/abs/2107.03374" target="_blank" rel="noopener noreferrer">[Codex]</a>
          </List.Item>
          <List.Item><strong>Text Analytics:</strong> Sentiment analysis, summarization, information extraction</List.Item>
          <List.Item>
            <strong>Search Engines:</strong> Semantic search, question answering, knowledge retrieval
            <a href="https://arxiv.org/abs/1810.04805" target="_blank" rel="noopener noreferrer">[BERT]</a>
          </List.Item>
        </List>
      </Paper>

</div>
<div data-slide>
      {/* Signal Processing */}
      <Paper className="p-6 mb-6">
        <Title order={3} className="mb-2">Signal Processing</Title>
        <Text size="sm" className="mb-4" color="dimmed">
          Analyzing and transforming temporal data from sensors and recordings
        </Text>
        
        <Flex direction="column" align="center" mb="md">
          <Image
            src="/assets/python-deep-learning/module1/signal_processing.png"
            alt="Signal Processing Applications"
            style={{ maxWidth: 'min(800px, 90vw)', height: 'auto' }}
            fluid
          />
          <Text component="p" ta="center" mt="xs">
            Source: https://www.edgeimpulse.com/blog/dsp-key-embedded-ml/
          </Text>
        </Flex>
        
        <List>
          <List.Item>
            <strong>Speech Recognition:</strong> Voice assistants, transcription, real-time captioning
            <a href="https://arxiv.org/abs/2212.04356" target="_blank" rel="noopener noreferrer">[Whisper]</a>
          </List.Item>
          <List.Item>
            <strong>Audio Synthesis:</strong> Text-to-speech, voice cloning, music generation
            <a href="https://arxiv.org/abs/2301.11325" target="_blank" rel="noopener noreferrer">[MusicLM]</a>
          </List.Item>
          <List.Item><strong>Time Series:</strong> Stock prediction, weather forecasting, demand planning</List.Item>
          <List.Item><strong>Healthcare Signals:</strong> ECG/EEG analysis, vital sign monitoring</List.Item>
          <List.Item><strong>IoT Analytics:</strong> Predictive maintenance, anomaly detection, sensor fusion</List.Item>
        </List>
      </Paper>
</div>
<div data-slide>
      {/* Agent Systems */}
      <Paper className="p-6 mb-6">
        <Title order={3} className="mb-2">Agent Systems</Title>
        <Text size="sm" className="mb-4" color="dimmed">
          Intelligent agents that can perceive, decide, and act in complex environments
        </Text>
        
        <Flex direction="column" align="center" mb="md">
          <Image
            src="/assets/python-deep-learning/module1/agentic.gif"
            alt="Agent Systems Applications"
            style={{ maxWidth: 'min(800px, 90vw)', height: 'auto' }}
            fluid
          />
          <Text component="p" ta="center" mt="xs">
            Source: https://blog.dailydoseofds.com/p/rag-vs-agentic-rag
          </Text>
        </Flex>
        
        <List>
          <List.Item>
            <strong>Game AI:</strong> AlphaGo, StarCraft II, Dota 2, chess engines
            <a href="https://www.nature.com/articles/nature16961" target="_blank" rel="noopener noreferrer">[AlphaGo]</a>
            <a href="https://www.nature.com/articles/s41586-019-1724-z" target="_blank" rel="noopener noreferrer">[AlphaStar]</a>
          </List.Item>
          <List.Item><strong>Robotics:</strong> Manipulation, grasping, assembly, warehouse automation</List.Item>
          <List.Item><strong>Navigation:</strong> Path planning, SLAM, obstacle avoidance</List.Item>
          <List.Item>
            <strong>Multi-Agent Systems:</strong> Swarm robotics, traffic optimization, resource allocation
          </List.Item>
          <List.Item>
            <strong>Recommendation:</strong> Content suggestions, personalization, ranking systems
            <a href="https://arxiv.org/abs/1606.07792" target="_blank" rel="noopener noreferrer">[Deep Neural Networks for YouTube]</a>
          </List.Item>
        </List>
      </Paper>

</div>
<div data-slide>
      {/* Generative AI */}
      <Paper className="p-6 mb-6">
        <Title order={3} className="mb-2">Generative AI</Title>
        <Text size="sm" className="mb-4" color="dimmed">
          Creating new content across text, images, audio, and video modalities
        </Text>
        
        <Flex direction="column" align="center" mb="md">
          <Image
            src="/assets/python-deep-learning/module1/not_exist.jpg"
            alt="Generative AI Applications"
            style={{ maxWidth: 'min(800px, 90vw)', height: 'auto' }}
            fluid
          />
          <Text component="p" ta="center" mt="xs">
            https://this-person-does-not-exist.com/en
          </Text>
        </Flex>
        
        <List>
          <List.Item>
            <strong>Image Generation:</strong> DALL-E, Midjourney, Stable Diffusion art creation
            <a href="https://arxiv.org/abs/2112.10752" target="_blank" rel="noopener noreferrer">[DALL-E 2]</a>
            <a href="https://arxiv.org/abs/2112.10741" target="_blank" rel="noopener noreferrer">[Stable Diffusion]</a>
          </List.Item>
          <List.Item>
            <strong>Video Synthesis:</strong> Sora, Runway, animation and film production
            <a href="https://openai.com/research/video-generation-models-as-world-simulators" target="_blank" rel="noopener noreferrer">[Sora]</a>
          </List.Item>
          <List.Item>
            <strong>3D Generation:</strong> NeRF, 3D model creation, virtual environments
            <a href="https://arxiv.org/abs/2003.08934" target="_blank" rel="noopener noreferrer">[NeRF]</a>
          </List.Item>
          <List.Item>
            <strong>Music Composition:</strong> AI composers, style transfer, sound design
            <a href="https://arxiv.org/abs/2306.05284" target="_blank" rel="noopener noreferrer">[MusicGen]</a>
          </List.Item>
          <List.Item><strong>Synthetic Data:</strong> Training data generation, privacy-preserving datasets</List.Item>
        </List>
      </Paper>

</div>
<div data-slide>
      {/* Scientific & Specialized Domains */}
      <Paper className="p-6 mb-6">
        <Title order={3} className="mb-2">Scientific & Specialized Domains</Title>
        <Text size="sm" className="mb-4" color="dimmed">
          Accelerating scientific discovery and solving domain-specific challenges
        </Text>
        
        
        <List>
          <List.Item>
            <strong>Protein Folding:</strong> AlphaFold revolutionizing structural biology
            <a href="https://www.nature.com/articles/s41586-021-03819-2" target="_blank" rel="noopener noreferrer">[AlphaFold 2]</a>
          </List.Item>
          <List.Item>
            <strong>Drug Discovery:</strong> Molecular design, clinical trial optimization
            <a href="https://www.nature.com/articles/s41587-022-01618-2" target="_blank" rel="noopener noreferrer">[Nature Biotech]</a>
          </List.Item>
          <List.Item>
            <strong>Climate Science:</strong> Weather prediction, climate modeling, carbon tracking
            <a href="https://www.science.org/doi/10.1126/science.adi2336" target="_blank" rel="noopener noreferrer">[GraphCast]</a>
          </List.Item>
          <List.Item>
            <strong>Materials Science:</strong> Crystal structure prediction, property optimization
            <a href="https://www.nature.com/articles/s41586-022-05761-3" target="_blank" rel="noopener noreferrer">[GNoME]</a>
          </List.Item>
          <List.Item><strong>Astronomy:</strong> Exoplanet detection, galaxy classification, data analysis</List.Item>
        </List>
      </Paper>
    </div>
<div data-slide>
      <Title order={2} mb="xl" id="data">Data: The Fuel of Deep Learning</Title>
      
      <Text size="lg" className="mb-6">
        Data is the foundation upon which all deep learning systems are built. The quality, quantity, 
        and diversity of training data directly determine model performance and capabilities.
      </Text>
</div>
<div data-slide>
      {/* Core Data Requirements */}
      <Paper className="p-6 mb-6">
        <Title order={3} className="mb-2">Core Data Requirements</Title>
        
        
        <List>
          <List.Item>
            <strong>Volume:</strong> Deep networks typically require thousands to millions of training examples. 
            The sample complexity grows quadratically with model dimensionality (N ∝ d²). While simple tasks 
            might work with 1000s of examples, complex vision or NLP models often need millions.
          </List.Item>
          <List.Item>
            <strong>Quality:</strong> Clean, accurate labels are crucial - a 5% label error rate can reduce 
            model accuracy by 10-20%. Consistent annotation guidelines, inter-annotator agreement, and 
            quality control processes directly impact final model performance.
          </List.Item>
          <List.Item>
            <strong>Diversity:</strong> Training data must capture the full spectrum of real-world scenarios. 
            Models trained on limited demographics, lighting conditions, or contexts fail to generalize. 
            Geographic, temporal, and domain diversity are essential.
          </List.Item>
          <List.Item>
            <strong>Balance:</strong> Class imbalance is a major issue - a dataset with 99% negative and 1% 
            positive examples will bias predictions. Techniques like oversampling, undersampling, or weighted 
            losses are needed to ensure fair representation.
          </List.Item>
          <List.Item>
            <strong>Temporal Consistency:</strong> Data distributions shift over time (concept drift). Models 
            trained on 2020 data may fail on 2024 scenarios. Continuous retraining and monitoring are 
            essential for production systems.
          </List.Item>
        </List>
      </Paper>
</div>
<div data-slide>
      {/* Data Challenges */}
      <Paper className="p-6 mb-6">
        <Title order={3} className="mb-2">Key Challenges</Title>
        <Text size="sm" className="mb-4" color="dimmed">
          Critical issues that teams face when building real-world deep learning systems
        </Text>
        
        <List>
          <List.Item>
            <strong>Annotation Cost:</strong> Labeling ImageNet's 14M images took 2.5 years and millions of 
            dollars. Medical imaging annotation requires expert radiologists at $100+/hour. Active learning 
            and semi-supervised methods can reduce costs by 50-90%.
          </List.Item>
          <List.Item>
            <strong>Privacy & Compliance:</strong> GDPR fines can reach 4% of global revenue. Healthcare data 
            requires HIPAA compliance. Techniques like federated learning, differential privacy, and synthetic 
            data generation help navigate regulatory requirements.
          </List.Item>
          <List.Item>
            <strong>Bias & Fairness:</strong> Face recognition systems show 35% higher error rates on darker 
            skin tones. Language models perpetuate gender stereotypes. Careful dataset curation, bias metrics, 
            and fairness constraints are essential for ethical AI.
          </List.Item>
          <List.Item>
            <strong>Distribution Shift:</strong> Models trained in labs fail in production - self-driving cars 
            trained in California struggle in snow. Domain adaptation, robust training, and continuous 
            monitoring help bridge the gap between training and deployment.
          </List.Item>
          <List.Item>
            <strong>Long Tail:</strong> Rare events (0.01% of data) often matter most - detecting rare diseases, 
            fraud, or safety-critical failures. Techniques like focal loss, hard negative mining, and synthetic 
            data augmentation address imbalanced distributions.
          </List.Item>
        </List>
      </Paper>
</div>
<div data-slide>
      {/* Open Source Datasets */}
      <Paper className="p-6 mb-6">
        <Title order={3} className="mb-2">Open Source Dataset Repositories</Title>
        <Text size="sm" className="mb-4" color="dimmed">
          Major platforms providing free datasets for research and development
        </Text>
        
        <List>
          <List.Item>
            <strong>Hugging Face Datasets:</strong> 70,000+ datasets across all modalities with standardized loading
            <a href="https://huggingface.co/datasets" target="_blank" rel="noopener noreferrer">[huggingface.co/datasets]</a>
          </List.Item>
          <List.Item>
            <strong>Kaggle:</strong> Competition datasets with kernels, discussions, and leaderboards
            <a href="https://www.kaggle.com/datasets" target="_blank" rel="noopener noreferrer">[kaggle.com/datasets]</a>
          </List.Item>
          <List.Item>
            <strong>Google Dataset Search:</strong> Search engine for datasets across the web
            <a href="https://datasetsearch.research.google.com" target="_blank" rel="noopener noreferrer">[datasetsearch.research.google.com]</a>
          </List.Item>
          <List.Item>
            <strong>UCI ML Repository:</strong> Classic datasets for machine learning research since 1987
            <a href="https://archive.ics.uci.edu" target="_blank" rel="noopener noreferrer">[archive.ics.uci.edu]</a>
          </List.Item>
          <List.Item>
            <strong>OpenML:</strong> Collaborative platform with 20,000+ datasets and experiments
            <a href="https://www.openml.org" target="_blank" rel="noopener noreferrer">[openml.org]</a>
          </List.Item>
          <List.Item>
            <strong>AWS Open Data:</strong> Large-scale datasets on AWS S3 (satellite, genomics, climate)
            <a href="https://registry.opendata.aws" target="_blank" rel="noopener noreferrer">[registry.opendata.aws]</a>
          </List.Item>
        </List>
      </Paper>
    </div>
      </Stack>
    </Container>
  );
};

export default Introduction;