import React from 'react';
import { Container, Title, Text, Stack, Grid, Paper, Code, List } from '@mantine/core';

const GenerativeModels = () => {
  return (
    <Container size="xl" py="xl">
      <Stack spacing="xl">
        
        {/* Slide 1: Title and Introduction */}
        <div data-slide className="min-h-[500px] flex flex-col justify-center">
          <Title order={1} className="text-center mb-8">
            Generative Models
          </Title>
          <Text size="xl" className="text-center mb-6">
            Creating New Data with Deep Learning
          </Text>
          <div className="max-w-3xl mx-auto">
            <Paper className="p-6 bg-blue-50">
              <Text size="lg" mb="md">
                Generative models learn to create new data samples that resemble the training data.
                They capture the underlying data distribution and can generate novel, realistic samples
                across various domains including images, text, and audio.
              </Text>
              <List>
                <List.Item>Variational Autoencoders (VAEs)</List.Item>
                <List.Item>Generative Adversarial Networks (GANs)</List.Item>
                <List.Item>Normalizing Flows and Diffusion Models</List.Item>
                <List.Item>Applications in data synthesis and augmentation</List.Item>
              </List>
            </Paper>
          </div>
        </div>

        {/* Slide 2: Autoencoders */}
        <div data-slide className="min-h-[500px]" id="autoencoders">
          <Title order={2} mb="xl">Autoencoders</Title>
          
          <Paper className="p-6 bg-gray-50 mb-6">
            <Text size="lg">
              Autoencoders learn efficient representations by compressing input data through an encoder
              and reconstructing it through a decoder. The bottleneck layer captures essential features.
            </Text>
          </Paper>
          
          <Grid gutter="lg">
            <Grid.Col span={6}>
              <Paper className="p-4 bg-green-50">
                <Title order={4} mb="sm">Basic Autoencoder</Title>
                <Code block language="python">{`import torch
import torch.nn as nn
import torch.nn.functional as F

class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super().__init__()
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        decoder_layers.extend([
            nn.Linear(prev_dim, input_dim),
            nn.Sigmoid()  # For images normalized to [0,1]
        ])
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

# Example usage
autoencoder = Autoencoder(
    input_dim=784,  # MNIST flattened
    hidden_dims=[512, 256, 128],
    latent_dim=32
)`}</Code>
              </Paper>
            </Grid.Col>
            
            <Grid.Col span={6}>
              <Paper className="p-4 bg-blue-50">
                <Title order={4} mb="sm">Convolutional Autoencoder</Title>
                <Code block language="python">{`class ConvAutoencoder(nn.Module):
    def __init__(self, channels=3, latent_dim=128):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 32, 4, 2, 1),  # 32x32 -> 16x16
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),       # 16x16 -> 8x8
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),      # 8x8 -> 4x4
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),     # 4x4 -> 2x2
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256 * 2 * 2, latent_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256 * 2 * 2),
            nn.ReLU(),
            nn.Unflatten(1, (256, 2, 2)),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 2x2 -> 4x4
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 4x4 -> 8x8
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),    # 8x8 -> 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(32, channels, 4, 2, 1), # 16x16 -> 32x32
            nn.Tanh()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded`}</Code>
              </Paper>
            </Grid.Col>
          </Grid>
        </div>

        {/* Slide 3: Variational Autoencoders */}
        <div data-slide className="min-h-[500px]" id="variational-autoencoders">
          <Title order={2} mb="xl">Variational Autoencoders (VAEs)</Title>
          
          <Paper className="p-6 bg-purple-50 mb-6">
            <Text size="lg">
              VAEs extend autoencoders by learning a probabilistic latent space. They enable generation
              of new samples by sampling from the learned latent distribution and decoding.
            </Text>
          </Paper>
          
          <Grid gutter="lg">
            <Grid.Col span={12}>
              <Paper className="p-4 bg-yellow-50">
                <Title order={4} mb="sm">VAE Implementation</Title>
                <Code block language="python">{`class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Latent space parameters
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick for backpropagation through stochastic node"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar
    
    def generate(self, num_samples, device):
        """Generate new samples"""
        z = torch.randn(num_samples, self.fc_mu.out_features, device=device)
        return self.decode(z)

def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    """VAE loss: reconstruction + KL divergence"""
    # Reconstruction loss (BCE for binary data)
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    
    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + beta * kl_loss`}</Code>
              </Paper>
            </Grid.Col>
          </Grid>
        </div>

        {/* Slide 4: Generative Adversarial Networks */}
        <div data-slide className="min-h-[500px]" id="gans">
          <Title order={2} mb="xl">Generative Adversarial Networks (GANs)</Title>
          
          <Grid gutter="lg">
            <Grid.Col span={6}>
              <Paper className="p-4 bg-red-50">
                <Title order={4} mb="sm">Basic GAN</Title>
                <Code block language="python">{`class Generator(nn.Module):
    def __init__(self, noise_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()  # Output in [-1, 1]
        )
    
    def forward(self, noise):
        return self.net(noise)

class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)

# GAN training loop
def train_gan(generator, discriminator, dataloader, num_epochs):
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    criterion = nn.BCELoss()
    
    for epoch in range(num_epochs):
        for batch_idx, (real_data, _) in enumerate(dataloader):
            batch_size = real_data.size(0)
            
            # Train Discriminator
            d_optimizer.zero_grad()
            
            # Real data
            real_labels = torch.ones(batch_size, 1)
            real_output = discriminator(real_data)
            d_loss_real = criterion(real_output, real_labels)
            
            # Fake data
            noise = torch.randn(batch_size, noise_dim)
            fake_data = generator(noise).detach()
            fake_labels = torch.zeros(batch_size, 1)
            fake_output = discriminator(fake_data)
            d_loss_fake = criterion(fake_output, fake_labels)
            
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()
            
            # Train Generator
            g_optimizer.zero_grad()
            noise = torch.randn(batch_size, noise_dim)
            fake_data = generator(noise)
            output = discriminator(fake_data)
            g_loss = criterion(output, real_labels)  # Fool discriminator
            g_loss.backward()
            g_optimizer.step()`}</Code>
              </Paper>
            </Grid.Col>
            
            <Grid.Col span={6}>
              <Paper className="p-4 bg-orange-50">
                <Title order={4} mb="sm">DCGAN (Deep Convolutional GAN)</Title>
                <Code block language="python">{`class DCGANGenerator(nn.Module):
    def __init__(self, noise_dim, num_channels=3, feature_maps=64):
        super().__init__()
        self.main = nn.Sequential(
            # Input: noise_dim x 1 x 1
            nn.ConvTranspose2d(noise_dim, feature_maps * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.ReLU(True),
            # State: (feature_maps*8) x 4 x 4
            
            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),
            # State: (feature_maps*4) x 8 x 8
            
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),
            # State: (feature_maps*2) x 16 x 16
            
            nn.ConvTranspose2d(feature_maps * 2, feature_maps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),
            # State: (feature_maps) x 32 x 32
            
            nn.ConvTranspose2d(feature_maps, num_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # Output: num_channels x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

class DCGANDiscriminator(nn.Module):
    def __init__(self, num_channels=3, feature_maps=64):
        super().__init__()
        self.main = nn.Sequential(
            # Input: num_channels x 64 x 64
            nn.Conv2d(num_channels, feature_maps, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # State: feature_maps x 32 x 32
            
            nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (feature_maps*2) x 16 x 16
            
            nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (feature_maps*4) x 8 x 8
            
            nn.Conv2d(feature_maps * 4, feature_maps * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (feature_maps*8) x 4 x 4
            
            nn.Conv2d(feature_maps * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # Output: 1 x 1 x 1
        )

    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)`}</Code>
              </Paper>
            </Grid.Col>
          </Grid>
        </div>

        {/* Slide 5: Advanced GAN Techniques */}
        <div data-slide className="min-h-[500px]" id="advanced-gans">
          <Title order={2} mb="xl">Advanced GAN Techniques</Title>
          
          <Grid gutter="lg">
            <Grid.Col span={6}>
              <Paper className="p-4 bg-indigo-50">
                <Title order={4} mb="sm">Wasserstein GAN</Title>
                <Code block language="python">{`class WGANCritic(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1)
            # No sigmoid! WGAN uses Wasserstein distance
        )
    
    def forward(self, x):
        return self.net(x)

def gradient_penalty(critic, real_samples, fake_samples, device):
    """Compute gradient penalty for WGAN-GP"""
    batch_size = real_samples.size(0)
    
    # Random interpolation
    alpha = torch.rand(batch_size, 1, device=device)
    alpha = alpha.expand_as(real_samples)
    
    interpolated = alpha * real_samples + (1 - alpha) * fake_samples
    interpolated.requires_grad_(True)
    
    # Critic score for interpolated samples
    interpolated_score = critic(interpolated)
    
    # Compute gradients
    gradients = torch.autograd.grad(
        outputs=interpolated_score,
        inputs=interpolated,
        grad_outputs=torch.ones_like(interpolated_score),
        create_graph=True,
        retain_graph=True
    )[0]
    
    # Gradient penalty
    gradients_norm = gradients.view(batch_size, -1).norm(2, dim=1)
    penalty = ((gradients_norm - 1) ** 2).mean()
    
    return penalty

def train_wgan_gp(generator, critic, dataloader, lambda_gp=10):
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.9))
    c_optimizer = torch.optim.Adam(critic.parameters(), lr=0.0001, betas=(0.5, 0.9))
    
    for epoch in range(num_epochs):
        for batch_idx, (real_data, _) in enumerate(dataloader):
            
            # Train Critic
            for _ in range(5):  # Train critic more frequently
                c_optimizer.zero_grad()
                
                noise = torch.randn(batch_size, noise_dim)
                fake_data = generator(noise).detach()
                
                real_score = critic(real_data)
                fake_score = critic(fake_data)
                
                gp = gradient_penalty(critic, real_data, fake_data, device)
                c_loss = fake_score.mean() - real_score.mean() + lambda_gp * gp
                
                c_loss.backward()
                c_optimizer.step()
            
            # Train Generator
            g_optimizer.zero_grad()
            noise = torch.randn(batch_size, noise_dim)
            fake_data = generator(noise)
            g_loss = -critic(fake_data).mean()
            g_loss.backward()
            g_optimizer.step()`}</Code>
              </Paper>
            </Grid.Col>
            
            <Grid.Col span={6}>
              <Paper className="p-4 bg-green-50">
                <Title order={4} mb="sm">Conditional GAN</Title>
                <Code block language="python">{`class ConditionalGenerator(nn.Module):
    def __init__(self, noise_dim, num_classes, hidden_dim, output_dim):
        super().__init__()
        self.label_embedding = nn.Embedding(num_classes, num_classes)
        
        self.net = nn.Sequential(
            nn.Linear(noise_dim + num_classes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()
        )
    
    def forward(self, noise, labels):
        # Embed labels and concatenate with noise
        label_embed = self.label_embedding(labels)
        input_vector = torch.cat([noise, label_embed], dim=1)
        return self.net(input_vector)

class ConditionalDiscriminator(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim):
        super().__init__()
        self.label_embedding = nn.Embedding(num_classes, num_classes)
        
        self.net = nn.Sequential(
            nn.Linear(input_dim + num_classes, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, labels):
        label_embed = self.label_embedding(labels)
        input_vector = torch.cat([x, label_embed], dim=1)
        return self.net(input_vector)

# Progressive GAN concept
class ProgressiveGenerator(nn.Module):
    def __init__(self, noise_dim, max_resolution=256):
        super().__init__()
        self.max_resolution = max_resolution
        self.current_resolution = 4
        
        # Build progressive blocks
        self.blocks = nn.ModuleDict()
        self.to_rgb = nn.ModuleDict()
        
        resolution = 4
        in_channels = 512
        
        while resolution <= max_resolution:
            self.blocks[f'{resolution}x{resolution}'] = self._make_block(in_channels, in_channels // 2)
            self.to_rgb[f'{resolution}x{resolution}'] = nn.Conv2d(in_channels // 2, 3, 1)
            
            in_channels = in_channels // 2
            resolution *= 2
    
    def _make_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
    
    def forward(self, noise, alpha=1.0):
        # Start with base resolution
        x = noise.view(noise.size(0), -1, 1, 1)
        
        # Progressive upsampling
        for resolution in [4, 8, 16, 32, 64]:
            if resolution <= self.current_resolution:
                x = self.blocks[f'{resolution}x{resolution}'](x)
        
        return self.to_rgb[f'{self.current_resolution}x{self.current_resolution}'](x)`}</Code>
              </Paper>
            </Grid.Col>
          </Grid>
        </div>

        {/* Slide 6: Diffusion Models */}
        <div data-slide className="min-h-[500px]" id="diffusion-models">
          <Title order={2} mb="xl">Diffusion Models</Title>
          
          <Paper className="p-6 bg-purple-50 mb-6">
            <Text size="lg">
              Diffusion models generate data by learning to reverse a noise process.
              They gradually denoise random noise to produce high-quality samples.
            </Text>
          </Paper>
          
          <Grid gutter="lg">
            <Grid.Col span={12}>
              <Paper className="p-4 bg-blue-50">
                <Title order={4} mb="sm">Simple Diffusion Model</Title>
                <Code block language="python">{`class SimpleDiffusion(nn.Module):
    def __init__(self, timesteps=1000):
        super().__init__()
        self.timesteps = timesteps
        
        # Noise schedule
        self.register_buffer('betas', torch.linspace(0.0001, 0.02, timesteps))
        self.register_buffer('alphas', 1 - self.betas)
        self.register_buffer('alpha_bars', torch.cumprod(self.alphas, dim=0))
        
        # U-Net for denoising
        self.unet = UNet(in_channels=3, out_channels=3, time_dim=32)
    
    def forward_process(self, x0, t):
        """Add noise to clean images"""
        noise = torch.randn_like(x0)
        alpha_bar = self.alpha_bars[t].view(-1, 1, 1, 1)
        
        # Noise formula: x_t = sqrt(alpha_bar) * x0 + sqrt(1 - alpha_bar) * noise
        x_t = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * noise
        return x_t, noise
    
    def reverse_process(self, x_t, t):
        """Predict noise to remove"""
        return self.unet(x_t, t)
    
    def sample(self, shape, device):
        """Generate samples by denoising"""
        x = torch.randn(shape, device=device)
        
        for t in reversed(range(self.timesteps)):
            t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.long)
            
            with torch.no_grad():
                predicted_noise = self.unet(x, t_tensor)
                
                alpha = self.alphas[t]
                alpha_bar = self.alpha_bars[t]
                beta = self.betas[t]
                
                # Denoising step
                x = (1 / torch.sqrt(alpha)) * (
                    x - (beta / torch.sqrt(1 - alpha_bar)) * predicted_noise
                )
                
                if t > 0:
                    noise = torch.randn_like(x)
                    x += torch.sqrt(beta) * noise
        
        return x

class UNet(nn.Module):
    """Simplified U-Net for diffusion"""
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim)
        )
        
        # Downsampling
        self.down1 = DoubleConv(in_channels, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        
        # Bottleneck
        self.bottleneck = DoubleConv(256, 512)
        
        # Upsampling
        self.up3 = DoubleConv(512 + 256, 256)
        self.up2 = DoubleConv(256 + 128, 128)
        self.up1 = DoubleConv(128 + 64, 64)
        
        self.final_conv = nn.Conv2d(64, out_channels, 1)
    
    def forward(self, x, t):
        # Time embedding
        t_emb = self.time_mlp(self.time_embedding(t))
        
        # Encoder
        d1 = self.down1(x)
        d2 = self.down2(F.max_pool2d(d1, 2))
        d3 = self.down3(F.max_pool2d(d2, 2))
        
        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(d3, 2))
        
        # Decoder with skip connections
        u3 = self.up3(torch.cat([F.interpolate(bottleneck, scale_factor=2), d3], 1))
        u2 = self.up2(torch.cat([F.interpolate(u3, scale_factor=2), d2], 1))
        u1 = self.up1(torch.cat([F.interpolate(u2, scale_factor=2), d1], 1))
        
        return self.final_conv(u1)
    
    def time_embedding(self, t):
        """Sinusoidal time embedding"""
        half_dim = 16
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        return embeddings

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.double_conv(x)`}</Code>
              </Paper>
            </Grid.Col>
          </Grid>
        </div>

        {/* Slide 7: Evaluation and Applications */}
        <div data-slide className="min-h-[500px]" id="evaluation-applications">
          <Title order={2} mb="xl">Evaluation and Applications</Title>
          
          <Grid gutter="lg">
            <Grid.Col span={6}>
              <Paper className="p-4 bg-yellow-50">
                <Title order={4} mb="sm">Evaluation Metrics</Title>
                <Code block language="python">{`import numpy as np
from scipy import linalg
from sklearn.metrics.pairwise import polynomial_kernel

def calculate_fid(real_features, generated_features):
    """Fréchet Inception Distance"""
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = generated_features.mean(axis=0), np.cov(generated_features, rowvar=False)
    
    ssdiff = np.sum((mu1 - mu2)**2.0)
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

def calculate_inception_score(generated_images, model, splits=10):
    """Inception Score"""
    N = len(generated_images)
    
    with torch.no_grad():
        preds = model(generated_images)
        preds = F.softmax(preds, dim=1)
    
    scores = []
    for i in range(splits):
        part = preds[i * N // splits: (i + 1) * N // splits]
        kl = part * (torch.log(part) - torch.log(part.mean(dim=0, keepdim=True)))
        kl = kl.sum(dim=1).mean()
        scores.append(torch.exp(kl))
    
    return torch.stack(scores).mean(), torch.stack(scores).std()

def perceptual_path_length(generator, num_samples=1000, epsilon=1e-4):
    """Perceptual Path Length for GANs"""
    total_distance = 0
    
    for _ in range(num_samples):
        z1 = torch.randn(1, generator.noise_dim)
        z2 = torch.randn(1, generator.noise_dim)
        
        # Interpolate
        t = torch.rand(1)
        z_interp = (1 - t) * z1 + t * z2
        
        # Generate images
        with torch.no_grad():
            img1 = generator(z_interp)
            img2 = generator(z_interp + epsilon * (z2 - z1))
        
        # Compute perceptual distance (using VGG features)
        distance = torch.norm(img1 - img2).item()
        total_distance += distance / epsilon
    
    return total_distance / num_samples`}</Code>
              </Paper>
            </Grid.Col>
            
            <Grid.Col span={6}>
              <Paper className="p-4 bg-red-50">
                <Title order={4} mb="sm">Applications</Title>
                <List spacing="sm">
                  <List.Item><strong>Image Generation:</strong> Creating realistic photos, artwork, and synthetic datasets</List.Item>
                  <List.Item><strong>Data Augmentation:</strong> Generating additional training samples to improve model performance</List.Item>
                  <List.Item><strong>Style Transfer:</strong> Converting images from one style to another (e.g., photos to paintings)</List.Item>
                  <List.Item><strong>Super Resolution:</strong> Enhancing low-resolution images to high resolution</List.Item>
                  <List.Item><strong>Inpainting:</strong> Filling in missing or corrupted parts of images</List.Item>
                  <List.Item><strong>Text-to-Image:</strong> Generating images from textual descriptions</List.Item>
                  <List.Item><strong>Drug Discovery:</strong> Generating new molecular structures</List.Item>
                  <List.Item><strong>Music Generation:</strong> Creating new musical compositions</List.Item>
                  <List.Item><strong>Video Generation:</strong> Producing realistic video sequences</List.Item>
                  <List.Item><strong>Privacy Protection:</strong> Generating synthetic data to protect sensitive information</List.Item>
                </List>
                
                <Paper className="p-3 bg-white mt-4">
                  <Title order={5} className="mb-2">Training Tips</Title>
                  <List size="sm">
                    <List.Item>Use spectral normalization for stable training</List.Item>
                    <List.Item>Apply progressive growing for high-resolution images</List.Item>
                    <List.Item>Monitor mode collapse and diversity</List.Item>
                    <List.Item>Use appropriate learning rates and optimizers</List.Item>
                    <List.Item>Implement gradient penalties for WGAN variants</List.Item>
                  </List>
                </Paper>
              </Paper>
            </Grid.Col>
          </Grid>
        </div>

      </Stack>
    </Container>
  );
};

export default GenerativeModels;