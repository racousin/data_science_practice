import React from 'react';
import { Container, Title, Text, Stack, Box, Image, List, Group } from '@mantine/core';
import CodeBlock from 'components/CodeBlock';
import 'katex/dist/katex.min.css';
import { InlineMath, BlockMath } from 'react-katex';

const GenerativeModel = () => {
  return (
    <Container size="lg">
      <Title order={1} id="generative-models" mb="xl">Generative Models for Images</Title>

      {/* Slide 1: Introduction */}
      <div data-slide>
        <Title order={2} mb="md">Introduction to Generative Models</Title>

        <Box mb="md">
          <Image
            src="/assets/data-science-practice/module7/generative-models-examples.png"
            alt="Examples of generated images from different generative models"
            mb="sm"
          />
          <Text size="sm">
            Generated images from various generative models: VAE, GAN, Flow, and Diffusion
          </Text>
        </Box>

        <Text mb="md">
          Generative models learn the underlying probability distribution of data to generate
          new samples that resemble the training data. Unlike discriminative models that learn
          <InlineMath>{'P(y|x)'}</InlineMath>, generative models learn <InlineMath>{'P(x)'}</InlineMath> or
          <InlineMath>{'P(x,y)'}</InlineMath>.
        </Text>

        <Text mb="md" weight={500}>Main Applications:</Text>
        <List mb="md">
          <List.Item>Image synthesis: creating realistic images from noise or descriptions</List.Item>
          <List.Item>Data augmentation: generating training samples for other models</List.Item>
          <List.Item>Image editing: modifying images while preserving realism</List.Item>
          <List.Item>Anomaly detection: identifying out-of-distribution samples</List.Item>
          <List.Item>Compression: learning compact representations of data</List.Item>
        </List>
      </div>

      {/* Slide 2: Overview of Generative Model Types */}
      <div data-slide>
        <Title order={2} mb="md">Types of Generative Models</Title>

        <Text mb="md">
          Four main families of generative models have emerged, each with different approaches
          to learning data distributions:
        </Text>

        <Group grow mb="md">
          <Box p="md">
            <Text weight={600} mb="xs">Variational Autoencoders (VAE)</Text>
            <Text size="sm">Learn explicit probabilistic latent representations through
            variational inference</Text>
          </Box>

          <Box p="md">
            <Text weight={600} mb="xs">Generative Adversarial Networks (GAN)</Text>
            <Text size="sm">Learn through adversarial training between generator and
            discriminator</Text>
          </Box>
        </Group>

        <Group grow mb="md">
          <Box p="md">
            <Text weight={600} mb="xs">Normalizing Flows</Text>
            <Text size="sm">Learn invertible transformations with tractable likelihood
            computation</Text>
          </Box>

          <Box p="md">
            <Text weight={600} mb="xs">Diffusion Models</Text>
            <Text size="sm">Learn to reverse a gradual noising process through denoising</Text>
          </Box>
        </Group>
      </div>

      {/* Slide 3: VAE - Introduction */}
      <div data-slide>
        <Title order={2} mb="md">Variational Autoencoders (VAE)</Title>

        <Box mb="md">
          <Image
            src="/assets/data-science-practice/module7/vae-architecture.png"
            alt="VAE architecture showing encoder, latent space, and decoder"
            mb="sm"
          />
          <Text size="sm">
            VAE architecture: encoder maps to latent distribution, decoder reconstructs from samples
          </Text>
        </Box>

        <Text mb="md">
          VAEs combine deep learning with variational inference to learn a continuous latent
          space representation. They consist of an encoder that maps data to latent distributions
          and a decoder that reconstructs data from latent samples.
        </Text>

        <Text mb="md" weight={500}>Core Objective:</Text>
        <Text mb="md">
          Learn parameters to maximize the evidence lower bound (ELBO):
        </Text>

        <BlockMath>
          {`\\mathcal{L}(\\theta, \\phi; x) = \\mathbb{E}_{q_\\phi(z|x)}[\\log p_\\theta(x|z)] - D_{KL}(q_\\phi(z|x) \\| p(z))`}
        </BlockMath>

        <Text mb="md">where:</Text>
        <List mb="md">
          <List.Item><InlineMath>{'q_\\phi(z|x)'}</InlineMath>: encoder (inference network)</List.Item>
          <List.Item><InlineMath>{'p_\\theta(x|z)'}</InlineMath>: decoder (generative network)</List.Item>
          <List.Item><InlineMath>{'p(z)'}</InlineMath>: prior distribution, typically <InlineMath>{'\\mathcal{N}(0, I)'}</InlineMath></List.Item>
          <List.Item><InlineMath>{'D_{KL}'}</InlineMath>: Kullback-Leibler divergence</List.Item>
        </List>
      </div>

      {/* Slide 4: VAE - Architecture and Loss */}
      <div data-slide>
        <Title order={2} mb="md">VAE Architecture and Loss</Title>

        <Text mb="md">
          The VAE loss consists of two terms:
        </Text>

        <BlockMath>
          {`\\mathcal{L}_{VAE} = \\mathcal{L}_{recon} + \\beta \\cdot \\mathcal{L}_{KL}`}
        </BlockMath>

        <Text mb="md" weight={500}>Reconstruction Loss:</Text>
        <Text mb="md">Measures how well the decoder reconstructs the input:</Text>
        <BlockMath>
          {`\\mathcal{L}_{recon} = -\\mathbb{E}_{q_\\phi(z|x)}[\\log p_\\theta(x|z)] \\approx \\|x - \\hat{x}\\|^2`}
        </BlockMath>

        <Text mb="md" weight={500}>KL Divergence:</Text>
        <Text mb="md">Regularizes latent space to match the prior:</Text>
        <BlockMath>
          {`\\mathcal{L}_{KL} = D_{KL}(q_\\phi(z|x) \\| p(z)) = \\frac{1}{2}\\sum_{j=1}^d (1 + \\log\\sigma_j^2 - \\mu_j^2 - \\sigma_j^2)`}
        </BlockMath>

        <Text mt="md">
          The parameter <InlineMath>{'\\beta'}</InlineMath> controls the trade-off between reconstruction
          and regularization (<InlineMath>{'\\beta'}</InlineMath>-VAE).
        </Text>
      </div>

      {/* Slide 5: VAE - Implementation */}
      <div data-slide>
        <Title order={2} mb="md">VAE Implementation</Title>

        <CodeBlock
          language="python"
          code={`import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU()
        )`}
        />

        <Text mb="md">Reparameterization trick enables backpropagation:</Text>

        <CodeBlock
          language="python"
          code={`# Latent distribution parameters
self.fc_mu = nn.Linear(64*8*8, latent_dim)
self.fc_logvar = nn.Linear(64*8*8, latent_dim)

def reparameterize(self, mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std`}
        />

        <Text mb="md">Decoder reconstructs from latent code:</Text>

        <CodeBlock
          language="python"
          code={`# Decoder
self.decoder = nn.Sequential(
    nn.ConvTranspose2d(latent_dim, 64, 4, 2, 1),
    nn.ReLU(),
    nn.ConvTranspose2d(64, 32, 4, 2, 1),
    nn.ReLU(),
    nn.ConvTranspose2d(32, 3, 4, 2, 1),
    nn.Sigmoid()
)`}
        />
      </div>

      {/* Slide 6: GAN - Introduction */}
      <div data-slide>
        <Title order={2} mb="md">Generative Adversarial Networks (GAN)</Title>

        <Box mb="md">
          <Image
            src="/assets/data-science-practice/module7/gan-architecture.webp"
            alt="GAN architecture with generator and discriminator networks"
            mb="sm"
          />
          <Text size="sm">
            GAN architecture: generator creates fake samples, discriminator classifies real vs fake
          </Text>
        </Box>

        <Text mb="md">
          GANs learn to generate data through adversarial training between two networks:
          a generator G that creates fake samples and a discriminator D that distinguishes
          real from fake samples.
        </Text>

        <Text mb="md" weight={500}>Min-Max Objective:</Text>
        <BlockMath>
          {`\\min_G \\max_D V(D,G) = \\mathbb{E}_{x \\sim p_{data}}[\\log D(x)] + \\mathbb{E}_{z \\sim p_z}[\\log(1-D(G(z)))]`}
        </BlockMath>

        <List mb="md">
          <List.Item>Discriminator maximizes its ability to classify real vs fake</List.Item>
          <List.Item>Generator minimizes discriminator's ability to detect fakes</List.Item>
          <List.Item>At equilibrium, <InlineMath>{'D(x) = 0.5'}</InlineMath> everywhere</List.Item>
        </List>
      </div>

      {/* Slide 7: GAN - Training and Loss */}
      <div data-slide>
        <Title order={2} mb="md">GAN Training Procedure</Title>

        <Text mb="md">Training alternates between updating discriminator and generator:</Text>

        <Text mb="md" weight={500}>Discriminator Loss:</Text>
        <BlockMath>
          {`\\mathcal{L}_D = -\\mathbb{E}_{x \\sim p_{data}}[\\log D(x)] - \\mathbb{E}_{z \\sim p_z}[\\log(1-D(G(z)))]`}
        </BlockMath>

        <Text mb="md" weight={500}>Generator Loss (non-saturating):</Text>
        <BlockMath>
          {`\\mathcal{L}_G = -\\mathbb{E}_{z \\sim p_z}[\\log D(G(z))]`}
        </BlockMath>

        <Text mb="md">
          Common challenges in GAN training include mode collapse (generator produces limited
          variety), vanishing gradients, and training instability.
        </Text>

        <Text mb="md" weight={500}>Improved GAN variants:</Text>
        <List mb="md">
          <List.Item>DCGAN: Uses convolutional architectures with architectural guidelines</List.Item>
          <List.Item>WGAN: Wasserstein loss for more stable training</List.Item>
          <List.Item>StyleGAN: Progressive growing and style-based generation</List.Item>
        </List>
      </div>

      {/* Slide 8: GAN - Implementation */}
      <div data-slide>
        <Title order={2} mb="md">GAN Implementation</Title>

        <CodeBlock
          language="python"
          code={`class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0),
            nn.BatchNorm2d(512),
            nn.ReLU()`}
        />

        <Text mb="md">Training step alternates between networks:</Text>

        <CodeBlock
          language="python"
          code={`# Train discriminator
real_loss = criterion(D(real_images), real_labels)
fake_images = G(noise)
fake_loss = criterion(D(fake_images.detach()), fake_labels)
d_loss = real_loss + fake_loss
d_loss.backward()`}
        />

        <Text mb="md">Then update generator:</Text>

        <CodeBlock
          language="python"
          code={`# Train generator
fake_images = G(noise)
g_loss = criterion(D(fake_images), real_labels)
g_loss.backward()`}
        />
      </div>

      {/* Slide 9: Normalizing Flows */}
      <div data-slide>
        <Title order={2} mb="md">Normalizing Flows</Title>

        <Box mb="md">
          <Image
            src="/assets/data-science-practice/module7/normalizing-flow-diagram.png"
            alt="Normalizing flow transformation from base to data distribution"
            mb="sm"
          />
          <Text size="sm">
            Normalizing flow: invertible transformations between simple and complex distributions
          </Text>
        </Box>

        <Text mb="md">
          Normalizing flows learn an invertible transformation between a simple base
          distribution (e.g., Gaussian) and the data distribution, enabling exact likelihood
          computation.
        </Text>

        <Text mb="md" weight={500}>Key Property:</Text>
        <Text mb="md">
          Through a sequence of invertible transformations <InlineMath>{'f = f_1 \\circ f_2 \\circ \\ldots \\circ f_K'}</InlineMath>:
        </Text>

        <BlockMath>
          {`\\log p_X(x) = \\log p_Z(f^{-1}(x)) + \\sum_{k=1}^K \\log \\left| \\det \\frac{\\partial f_k^{-1}}{\\partial f_{k-1}^{-1}} \\right|`}
        </BlockMath>

        <List mb="md">
          <List.Item>Base distribution: <InlineMath>{'z \\sim p_Z'}</InlineMath> (typically Gaussian)</List.Item>
          <List.Item>Transformation: <InlineMath>{'x = f(z)'}</InlineMath> is invertible</List.Item>
          <List.Item>Jacobian determinant accounts for volume change</List.Item>
        </List>

        <Text mb="md">
          Loss is negative log-likelihood:
        </Text>

        <BlockMath>
          {`\\mathcal{L} = -\\mathbb{E}_{x \\sim p_{data}}[\\log p_X(x)]`}
        </BlockMath>
      </div>

      {/* Slide 10: Flow Architectures */}
      <div data-slide>
        <Title order={2} mb="md">Flow-Based Model Architectures</Title>

        <Text mb="md" weight={500}>Coupling Layers (RealNVP, Glow):</Text>
        <Text mb="md">
          Split input into two parts, transform one conditioned on the other:
        </Text>

        <BlockMath>
          {`\\begin{aligned} y_{1:d} &= x_{1:d} \\\\ y_{d+1:D} &= x_{d+1:D} \\odot \\exp(s(x_{1:d})) + t(x_{1:d}) \\end{aligned}`}
        </BlockMath>

        <Text mb="md">
          where <InlineMath>{'s'}</InlineMath> and <InlineMath>{'t'}</InlineMath> are neural networks
          computing scale and translation.
        </Text>

        <CodeBlock
          language="python"
          code={`# Coupling layer implementation
x1, x2 = torch.chunk(x, 2, dim=1)

# Forward pass
scale = scale_net(x1)
shift = shift_net(x1)
y2 = x2 * torch.exp(scale) + shift
y = torch.cat([x1, y2], dim=1)

# Log determinant
log_det = scale.sum(dim=[1,2,3])`}
        />
      </div>

      {/* Slide 11: Diffusion Models - Introduction */}
      <div data-slide>
        <Title order={2} mb="md">Diffusion Models</Title>

        <Box mb="md">
          <Image
            src="/assets/data-science-practice/module7/diffusion-process.ppm"
            alt="Diffusion model forward and reverse process"
            mb="sm"
          />
          <Text size="sm">
            Diffusion process: forward noising (fixed) and reverse denoising (learned)
          </Text>
        </Box>

        <Text mb="md">
          Diffusion models learn to generate data by reversing a gradual noising process.
          They have achieved state-of-the-art results in image generation, surpassing GANs
          in sample quality.
        </Text>

        <Text mb="md" weight={500}>Forward Process (Fixed):</Text>
        <Text mb="md">
          Gradually adds Gaussian noise over T timesteps:
        </Text>

        <BlockMath>
          {`q(x_t | x_{t-1}) = \\mathcal{N}(x_t; \\sqrt{1-\\beta_t} x_{t-1}, \\beta_t I)`}
        </BlockMath>

        <Text mb="md">
          With closed form:
        </Text>

        <BlockMath>
          {`q(x_t | x_0) = \\mathcal{N}(x_t; \\sqrt{\\bar{\\alpha}_t} x_0, (1-\\bar{\\alpha}_t) I)`}
        </BlockMath>

        <Text mb="md">
          where <InlineMath>{'\\bar{\\alpha}_t = \\prod_{s=1}^t (1-\\beta_s)'}</InlineMath>.
        </Text>
      </div>

      {/* Slide 12: Diffusion - Reverse Process */}
      <div data-slide>
        <Title order={2} mb="md">Diffusion Reverse Process</Title>

        <Text mb="md" weight={500}>Reverse Process (Learned):</Text>
        <Text mb="md">
          Learn to denoise by predicting the reverse transition:
        </Text>

        <BlockMath>
          {`p_\\theta(x_{t-1}|x_t) = \\mathcal{N}(x_{t-1}; \\mu_\\theta(x_t, t), \\Sigma_\\theta(x_t, t))`}
        </BlockMath>

        <Text mb="md">
          The model <InlineMath>{'\\epsilon_\\theta(x_t, t)'}</InlineMath> predicts the noise
          added at each timestep.
        </Text>

        <Text mb="md" weight={500}>Training Objective (Simplified):</Text>
        <BlockMath>
          {`\\mathcal{L}_{simple} = \\mathbb{E}_{t, x_0, \\epsilon}[\\|\\epsilon - \\epsilon_\\theta(x_t, t)\\|^2]`}
        </BlockMath>

        <Text mb="md">
          At inference, start from noise <InlineMath>{'x_T \\sim \\mathcal{N}(0,I)'}</InlineMath> and
          iteratively denoise to generate <InlineMath>{'x_0'}</InlineMath>.
        </Text>
      </div>

      {/* Slide 13: Diffusion - Implementation */}
      <div data-slide>
        <Title order={2} mb="md">Diffusion Model Implementation</Title>

        <CodeBlock
          language="python"
          code={`# Training
def train_step(x0, model):
    # Sample timestep and noise
    t = torch.randint(0, T, (batch_size,))
    noise = torch.randn_like(x0)

    # Forward process
    alpha_bar_t = alpha_bar[t]
    xt = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * noise

    # Predict noise
    predicted_noise = model(xt, t)
    loss = F.mse_loss(predicted_noise, noise)
    return loss`}
        />

        <Text mb="md">Sampling (generation):</Text>

        <CodeBlock
          language="python"
          code={`@torch.no_grad()
def sample(model, shape):
    # Start from pure noise
    x = torch.randn(shape)

    # Iteratively denoise
    for t in reversed(range(T)):
        # Predict noise
        predicted_noise = model(x, t)
        # Compute denoised x
        x = denoise_step(x, predicted_noise, t)

    return x`}
        />
      </div>

      {/* Slide 14: Conditional Generation */}
      <div data-slide>
        <Title order={2} mb="md">Conditional Generation</Title>

        <Text mb="md">
          All generative model families can be conditioned on additional information
          (class labels, text, images) to control generation.
        </Text>

        <Text mb="md" weight={500}>Conditional VAE:</Text>
        <Text mb="md">
          Condition encoder and decoder on label <InlineMath>{'y'}</InlineMath>:
          <InlineMath>{'q_\\phi(z|x,y)'}</InlineMath> and <InlineMath>{'p_\\theta(x|z,y)'}</InlineMath>
        </Text>

        <Text mb="md" weight={500}>Conditional GAN:</Text>
        <Text mb="md">
          Both generator and discriminator receive conditioning:
          <InlineMath>{'G(z,y)'}</InlineMath> and <InlineMath>{'D(x,y)'}</InlineMath>
        </Text>

        <Text mb="md" weight={500}>Classifier-Free Guidance (Diffusion):</Text>
        <Text mb="md">
          Train both conditional and unconditional models jointly:
        </Text>
        <BlockMath>
          {`\\tilde{\\epsilon}_\\theta(x_t, t, y) = \\epsilon_\\theta(x_t, t, \\emptyset) + s \\cdot (\\epsilon_\\theta(x_t, t, y) - \\epsilon_\\theta(x_t, t, \\emptyset))`}
        </BlockMath>

        <Text mt="md">
          where <InlineMath>{'s'}</InlineMath> is guidance scale controlling adherence to condition.
        </Text>
      </div>

      {/* Slide 15: Comparison and Best Models */}
      <div data-slide>
        <Title order={2} mb="md">Model Comparison and State-of-the-Art</Title>

        <Group grow mb="md">
          <Box p="md">
            <Text weight={600} mb="xs">VAE</Text>
            <Text size="sm" mb="xs">Advantages: Stable training, explicit latent space</Text>
            <Text size="sm">Disadvantages: Blurry samples</Text>
            <Text size="sm" weight={500} mt="xs">Best: VQ-VAE-2</Text>
          </Box>

          <Box p="md">
            <Text weight={600} mb="xs">GAN</Text>
            <Text size="sm" mb="xs">Advantages: Sharp, high-quality samples</Text>
            <Text size="sm">Disadvantages: Training instability, mode collapse</Text>
            <Text size="sm" weight={500} mt="xs">Best: StyleGAN3, BigGAN</Text>
          </Box>
        </Group>

        <Group grow mb="md">
          <Box p="md">
            <Text weight={600} mb="xs">Flows</Text>
            <Text size="sm" mb="xs">Advantages: Exact likelihood, invertible</Text>
            <Text size="sm">Disadvantages: Architectural constraints</Text>
            <Text size="sm" weight={500} mt="xs">Best: Glow, Flow++</Text>
          </Box>

          <Box p="md">
            <Text weight={600} mb="xs">Diffusion</Text>
            <Text size="sm" mb="xs">Advantages: SOTA quality, stable training</Text>
            <Text size="sm">Disadvantages: Slow sampling</Text>
            <Text size="sm" weight={500} mt="xs">Best: Stable Diffusion, DALL-E 3</Text>
          </Box>
        </Group>

        <Text mt="md">
          Current state-of-the-art in image generation is dominated by diffusion models,
          particularly for conditional generation tasks like text-to-image (Stable Diffusion,
          Midjourney, DALL-E 3). GANs remain competitive for specific domains requiring
          fast inference.
        </Text>
      </div>

      {/* Slide 16: Evaluation Metrics */}
      <div data-slide>
        <Title order={2} mb="md">Evaluation Metrics</Title>

        <Text mb="md">
          Generative models are evaluated using multiple metrics assessing different aspects
          of sample quality and diversity.
        </Text>

        <Text mb="md" weight={500}>Fréchet Inception Distance (FID):</Text>
        <Text mb="md">
          Measures distance between feature distributions of real and generated images:
        </Text>
        <BlockMath>
          {`FID = \\|\\mu_r - \\mu_g\\|^2 + \\text{Tr}(\\Sigma_r + \\Sigma_g - 2(\\Sigma_r \\Sigma_g)^{1/2})`}
        </BlockMath>

        <Text mb="md" weight={500}>Inception Score (IS):</Text>
        <Text mb="md">
          Evaluates sample quality and diversity based on classifier predictions:
        </Text>
        <BlockMath>
          {`IS = \\exp(\\mathbb{E}_x[D_{KL}(p(y|x) \\| p(y))])`}
        </BlockMath>

        <Text mb="md" weight={500}>Precision and Recall:</Text>
        <Text mb="md">
          Separately measure sample quality (precision) and diversity (recall).
          Lower FID is better; higher IS is better.
        </Text>
      </div>

    </Container>
  );
};

export default GenerativeModel;
