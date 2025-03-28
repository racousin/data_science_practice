# Deep Learning for Time Series

## Table of Contents
1. [Introduction](#introduction)
2. [Markovian approach](#markovian-approach)
3. [Non-Markovian Methods](#non-markovian-methods)
4. [WaveNet](#wavenet)
5. [Recurrent Neural Networks](#recurrent-neural-networks)
6. [Latent Neural ODEs](#latent-neural-odes)
7. [Transformers](#transformers) 
8. [State Space Models](#state-space-models)
9. [Evaluation Metrics](#evaluation-metrics)



Time series analysis is a critical area of research and application in machine learning, where the goal is to understand, model, and predict temporal data. Deep learning offers a robust toolkit for tackling these problems, enabling the modeling of complex patterns and dependencies over time.

This course provides a comprehensive introduction to deep learning methods for time series, focusing on two major paradigms: **Markovian methods** and **Non-Markovian methods**. Each paradigm is suited to different scenarios depending on the nature of the underlying dynamics and data observability.

---

## Introduction

Time series data consists of observations collected sequentially over time, where the ordering is crucial to the analysis. Examples include stock prices, weather data, sensor measurements, and biological signals. A key challenge in time series modeling lies in capturing dependencies across time to make accurate predictions or understand underlying dynamics.

Time series problems can be approached with two primary frameworks:

1. **Markovian Methods**: Assumes that the future state depends only on the current state, not on the full history.
2. **Non-Markovian Methods**: Accounts for long-term dependencies or partial observability when the Markovian assumption breaks down.

---

## Markovian Approach

Markovian methods assume that the system can be fully described by its current state, such that:

\[
P(s_{t+1} \mid s_t, s_{t-1}, \ldots, s_0) = P(s_{t+1} \mid s_t),
\]

where \( s_t \) represents the state at time \( t \).

### When is a Markovian Approach Applicable?

1. **Complete Observation of the State**:
   - The Markovian assumption holds when all relevant information about the system is captured in the observed state.
   - Example: Modeling the dynamics of a physical system with full state information, such as position and velocity.

2. **Partial Observability**:
   - If the system is only partially observed, the observables must meet specific criteria to enable Markovian modeling. This often involves ensuring that the **Gramian of Observability** is non-singular, indicating that the current observable contains sufficient information about the system's dynamics.

### Challenges with Markovian Assumptions

- When the state is not fully observable, or the dynamics involve long-term dependencies, the Markovian framework becomes insufficient.
- This limitation motivates the need for **Non-Markovian methods**, which can account for hidden states or memory effects in the data.

---

## Non-Markovian Methods

Non-Markovian methods address scenarios where the Markovian assumption fails, either due to partial observability or the presence of long-term dependencies. These methods draw from theories like **Takens's Theorem** and the **Mori-Zwanzig Formalism**, which provide tools to reconstruct missing information in partially observed systems.

### Theoretical Foundations

1. **Takens's Theorem**:
   - Provides a framework for reconstructing the dynamics of a system using time-delayed embeddings of observed variables.
   - If \( x(t) \) is an observable, a state space can be reconstructed as:
     \[
     \mathbf{X}_t = [x(t), x(t-\tau), x(t-2\tau), \ldots],
     \]
     where \( \tau \) is a time delay.

![taken](/images/taken.png)

2. **The Mori-Zwanzig Formalism**

    The Mori-Zwanzig formalism provides a theoretical framework to describe the dynamics of partially observed systems. It is particularly useful when the system's full state \( \mathbf{s}(t) \) evolves in a high-dimensional space, but only a subset of the variables, the **observables** \( \mathbf{y}(t) \), is available.

    The key idea is to project the full dynamics onto the space of observables, separating the contributions of the observed and unobserved components. This separation introduces **memory effects** and noise-like terms that capture the influence of the unobserved dynamics.

    **Formalism Definition**

    Consider the full state of the system evolving according to:

    \[
    \frac{d\mathbf{s}(t)}{dt} = \mathcal{L}\mathbf{s}(t),
    \]

    where:
    - \( \mathbf{s}(t) \) is the state vector of the full system,
    - \( \mathcal{L} \) is the generator of the dynamics (e.g., a differential operator or a matrix).

    Using a projection operator \( \mathcal{P} \) that maps the full state space onto the space of observables \( \mathbf{y}(t) = \mathcal{P} \mathbf{s}(t) \), the dynamics can be decomposed as:

    \[\frac{d\mathbf{y}(t)}{dt} = \underbrace{\mathcal{PL}\mathbf{y}(t)}_{\text{Markovian term}} + \underbrace{\int_0^t K(t - \tau) \mathbf{y}(\tau) \, d\tau}_{\text{Non-Markovian memory term}} + \underbrace{\mathcal{Q} \mathbf{L} e^{t \mathcal{QL}} \mathcal{Q}\mathbf{s}(0)}_{\text{Source term}}.\]

    This equation consists of three terms:

    **1. Markovian Term**
    \[
    \mathcal{PL}\mathbf{y}(t)
    \]

    - **Description**:
    - This term depends only on the observable \( \mathbf{y}(t) \) and describes the Markovian dynamics in the observable space.
    - It represents the part of the dynamics that can be directly explained by the current state of the observed variables, assuming no influence from hidden states or memory.

    **2. Non-Markovian Memory Term**
    \[
    \int_0^t K(t - \tau) \mathbf{y}(\tau) \, d\tau
    \]

    - **Description**:
    - This term describes how the history of the observable \( \mathbf{y}(\tau) \) affects its current dynamics. 
    - The **memory kernel** \( K(t - \tau) \) encapsulates the influence of the unobserved dynamics projected onto the observable space.

    **3. Source Term**
    \[
    \mathcal{Q} \mathbf{L} e^{t \mathcal{QL}} \mathcal{Q}\mathbf{s}(0)
    \]

    - **Description**:
    - This term represents the residual dynamics within the unobserved space (orthogonal to the observable space), introduced by the projection operator \( \mathcal{Q} = \mathcal{I} - \mathcal{P} \).
    - It captures noise-like contributions and stochastic effects resulting from the unobserved dynamics of the initial condition.

**Take home message** : do not use markovian model with time series if you do not know the governing equation generating the time series.

### Non-Markovian Approaches: Capturing Past Information

In a **non-Markovian approach**, the goal is to recover the necessary information to predict the solution at time \( t \) by looking at the past. This is critical when the system dynamics depend not only on the current state but also on previous states. 

The missing past information can be incorporated in several ways:
1. **Explicitly Adding Past Inputs**:
   - Include \( N \) past inputs (\( x(t), x(t - \Delta t), \ldots, x(t - N \cdot \Delta t) \)) as part of the input to predict the solution at \( t + \Delta t \).

2. **Learning a Convolution Filter**:
   - Use a convolutional filter to aggregate past information over a defined time window, enabling the model to automatically extract relevant temporal features.

3. **Defining a Memory**:
   - Introduce an internal memory structure (e.g., hidden states in RNNs or cell states in LSTMs) that is updated at each time step to store information about the sequence history.

#### Examples of Delayed Dynamics

1. **Discrete Delay**:
   - The system dynamics depend explicitly on discrete delays:
     \[
     \frac{dx}{dt} = f(x(t), x(t-\tau), x(t-2\tau), \ldots),
     \]
     where \( \tau \) is the delay. This type of model is useful for systems where the influence of the past occurs at specific intervals.

2. **Convolutional Delay**:
   - The system dynamics depend on a continuous convolution of past states:
     \[
     \frac{dx}{dt} = f\left(x(t), \int_{-\infty}^0 x(t + \tau) e^{\lambda \tau} \, d\tau\right),
     \]
     where the term \( e^{\lambda \tau} \) is a weighting function that determines how past states contribute to the current dynamics.

3. **Augmented ODE**:
   - The dynamics are augmented by introducing an auxiliary state \( y \) to store delayed information:
     \[
     \frac{dx}{dt} = f(x, y), \quad \frac{dy}{dt} = x - \lambda y.
     \]
     Here, \( y \) serves as a memory term that smooths and integrates past information.

These approaches illustrate how non-Markovian models aim to reconstruct and leverage the hidden structure of past information, enabling them to handle systems with memory effects, delays, or long-term dependencies effectively.


### Non-Markovian Deep Learning Models

1. **WaveNet** (convolution based): 
   - A convolutional neural network (CNN) designed for time series with temporal convolution layers that capture long-range dependencies.
   - Effective for audio and sequential data modeling.

2. **Recurrent Neural Networks (RNNs)** (memory based):
   - Classical models for sequential data that maintain hidden states to capture temporal dependencies.
   - **LSTMs (Long Short-Term Memory)**:
     - A special type of RNN that addresses vanishing gradient issues by introducing memory cells and gates (input, forget, output) to manage long-term dependencies.

3. **Latent NeuralODE** (memory based):
   - Combines Neural Ordinary Differential Equations (NeuralODE) with a latent representation to model continuous-time dynamics.
   - Captures both observed and unobserved components in time series.

4. **Transformers** (convolution based):
   - State-of-the-art models originally designed for natural language processing but highly effective for time series.
   - Uses self-attention mechanisms to capture dependencies across arbitrary time points, enabling parallel processing and scalability.

5. **State Space Models** (convolution and memory based):
   - Represent systems as a combination of state and observation equations:
     \[
     s_{t+1} = f(s_t, u_t), \quad y_t = g(s_t) + \epsilon,
     \]
     where \( s_t \) is the state, \( u_t \) is the input, and \( y_t \) is the observation.
   - Neural extensions of state space models integrate deep learning to enhance flexibility and scalability.

The models discussed earlier represent some of the most popular and effective architectures for time series analysis. However, these are not the only models available. Many other models exist, often tailored to specific applications such as rare event prediction, long-horizon forecasting, or probabilistic predictions. For instance:

- **Rare Event Prediction**: Techniques like anomaly detection using **Autoencoders** or **Neural Hawkes Processes**.
- **Long-Horizon Forecasting**: Models such as **N-BEATS** or **Temporal Fusion Transformers**.
- **Probabilistic Predictions**: Method like **DeepAR**

These specialized models can provide significant advantages in their respective domains, and choosing the right approach depends on the specific requirements of the application.

## WaveNet

WaveNet is a deep learning architecture initially designed for audio generation, but it has proven highly effective for sequential data, including time series. Unlike recurrent neural networks (RNNs), which rely on sequential processing, WaveNet uses **causal convolutions** to capture temporal dependencies in a parallelizable manner. This section explains the core ideas and motivations behind WaveNet, focusing on the concept of multi-scale convolutions.

### Core Idea

The primary goal of WaveNet is to model temporal dependencies across different scales by employing a stack of **causal dilated convolutions**. These convolutions:
- Ensure that predictions for a given time step \( t \) depend only on past inputs (\( t-k, t-k-1, \dots \)), preserving the causality of the time series.
- Expand the receptive field of the model exponentially with depth, allowing it to capture both short-term and long-term dependencies efficiently.

The model can be summarized as follows:
1. **Input Representation**:
   - The time series is passed through causal convolution layers where each layer is responsible for capturing dependencies over a specific scale.
2. **Dilations**:
   - Each convolution layer uses a dilation factor \( d \), which determines the spacing between the input elements the kernel operates on. 
3. **Stacking**:
   - By stacking dilated convolutions with exponentially increasing dilation factors, WaveNet achieves a large receptive field without a significant increase in computational cost.

![wavenet](/images/wavenet.jpg)

### Motivation for Multi-Scale Convolutions

1. **Causality**:
   - In time series, future predictions should not depend on future inputs. Causal convolutions enforce this constraint naturally by ensuring that the output at time \( t \) depends only on inputs up to time \( t \).

2. **Efficient Long-Term Dependency Modeling**:
   - Sequential models like RNNs often struggle with capturing long-term dependencies due to vanishing gradients and sequential processing.
   - WaveNet's dilated convolutions allow the model to capture dependencies across multiple scales (short-term to long-term) in a computationally efficient manner.

3. **Exponential Growth of Receptive Field**:
   - Stacking \( L \) convolutional layers with dilation factors doubling at each layer results in a receptive field of size:
     \[
     \text{Receptive Field} = 2^L - 1.
     \]
     For example, with \( L=10 \) layers, the receptive field covers 1023 time steps.

4. **Parallelizability**:
   - Unlike RNNs, which process data sequentially, WaveNet's convolutional structure allows parallel processing across all time steps, leading to significant computational speedups during training.

### Mathematical Formulation

Let \( x_t \) denote the input at time \( t \), and \( h^{(l)}_t \) be the output of the \( l \)-th convolutional layer at time \( t \). The dilated convolution operation can be expressed as:

\[
h^{(l)}_t = \sum_{i=0}^{k-1} w^{(l)}_i \cdot h^{(l-1)}_{t - i \cdot d^{(l)}},
\]

where:
- \( w^{(l)}_i \): Filter weights for the \( l \)-th layer.
- \( k \): Kernel size.
- \( d^{(l)} \): Dilation factor for the \( l \)-th layer.
- \( h^{(l-1)}_{t - i \cdot d^{(l)}} \): Input to the \( l \)-th layer from the previous layer at time \( t - i \cdot d^{(l)} \).

By varying \( d^{(l)} \), the model captures dependencies across different temporal scales.

### Wavenet code snippets 

```python 
import torch
import torch.nn as nn

class WaveNet(nn.Module):
    def __init__(self, input_channels=1, residual_channels=32, dilation_channels=32, skip_channels=64, kernel_size=2, num_layers=10):
        super(WaveNet, self).__init__()
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        
        # Causal Convolution
        self.causal_conv = nn.Conv1d(input_channels, residual_channels, kernel_size=kernel_size, padding=kernel_size - 1, bias=False)
        
        # Dilated Convolution Blocks
        self.dilated_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        
        for i in range(num_layers):
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation
            self.dilated_convs.append(
                nn.Conv1d(residual_channels, dilation_channels, kernel_size=kernel_size, dilation=dilation, padding=padding, bias=False)
            )
            self.residual_convs.append(
                nn.Conv1d(dilation_channels, residual_channels, kernel_size=1, bias=False)
            )
            self.skip_convs.append(
                nn.Conv1d(dilation_channels, skip_channels, kernel_size=1, bias=False)
            )
        
        # Output layers
        self.output_conv1 = nn.Conv1d(skip_channels, skip_channels, kernel_size=1, bias=False)
        self.output_conv2 = nn.Conv1d(skip_channels, input_channels, kernel_size=1, bias=False)
    
    def forward(self, x):
        # Initial causal convolution
        x = self.causal_conv(x)
        skip_connections = []
        
        # Dilated convolution layers
        for dilated_conv, residual_conv, skip_conv in zip(self.dilated_convs, self.residual_convs, self.skip_convs):
            residual = x
            x = dilated_conv(x)
            
            # Gated activation unit
            tanh_out = torch.tanh(x)
            sigm_out = torch.sigmoid(x)
            x = tanh_out * sigm_out
            
            # Skip connection
            skip = skip_conv(x)
            skip_connections.append(skip)
            
            # Residual connection
            x = residual_conv(x)
            x += residual[:, :, -x.size(2):]  # Match dimensions
        
        # Aggregate skip connections
        x = sum(skip_connections)
        
        # Output layers
        x = torch.relu(self.output_conv1(x))
        x = self.output_conv2(x)
        return x

# Example usage
if __name__ == "__main__":
    # Define the model
    model = WaveNet(input_channels=1, residual_channels=32, dilation_channels=32, skip_channels=64, kernel_size=2, num_layers=10)
    
    # Generate dummy time series data (batch_size, channels, sequence_length)
    dummy_input = torch.randn(16, 1, 100)
    
    # Forward pass
    output = model(dummy_input)

```

Diagram of the wavenet layer

```
Input --> Causal Conv --> Dilated Conv --> Gated Activation --> Residual Conv --> [Residual Connection]
                                 |--> Skip Conv --> [Skip Connection]
```

#### Advantages of WaveNet

1. **Scalability**:
   - The use of convolutions makes WaveNet more scalable than RNNs for large datasets or long sequences.
2. **Flexible Receptive Field**:
   - The receptive field size can be adjusted by modifying the number of layers and dilation factors, enabling customization for specific time series tasks.
3. **Causal Structure**:
   - Ensures that predictions respect the temporal order of the data.


## Recurrent Neural Networks

Recurrent Neural Networks (RNNs) are a foundational deep learning architecture for modeling sequential data, including time series. They are designed to process data step-by-step while maintaining an internal state that evolves over time, making them suitable for capturing temporal dependencies.

#### Core Idea

The central idea of RNNs is the concept of **memory**, which allows the network to retain information about previous inputs. At each time step \( t \), the network takes an input \( x_t \), updates its hidden state \( h_t \), and produces an output \( y_t \). The hidden state \( h_t \) serves as a form of memory, summarizing the information from all prior time steps. This process can be expressed as:

\[
h_t = f(W_h h_{t-1} + W_x x_t + b_h),
\]
\[
y_t = g(W_y h_t + b_y),
\]

where:
- \( W_h, W_x, W_y \): Weight matrices,
- \( b_h, b_y \): Bias terms,
- \( f \): Non-linear activation function (e.g., \( \tanh \) or \( \text{ReLU} \)),
- \( g \): Output activation function (e.g., softmax for classification).

![rnn](/images/rnn.png)

#### Memory in RNNs

The hidden state \( h_t \) acts as a memory that is updated at each time step. This enables RNNs to model dependencies in sequences, but the quality of memory depends on the ability to propagate gradients through time during training.

---

### Backpropagation Through Time (BPTT)

Backpropagation Through Time (BPTT) is an extension of the backpropagation algorithm, specifically designed for training recurrent neural networks (RNNs). RNNs process sequential data by maintaining a hidden state that evolves over time, which introduces dependencies between parameters at different time steps. BPTT handles this temporal dependency by unrolling the network across time steps and applying backpropagation to compute gradients.


#### How BPTT Works

1. **Unrolling the RNN**:
   - The RNN is unrolled across a sequence of \( T \) time steps, creating a computational graph where each layer corresponds to the RNN's operations at a specific time step.
   - The parameters \( W_h, W_x, W_y \) are shared across all time steps, reflecting the recurrent nature of the model.

2. **Forward Pass**:
   - Compute the outputs and hidden states for each time step, starting from the initial state \( h_0 \):
     \[
     h_t = f(W_h h_{t-1} + W_x x_t + b_h), \quad y_t = g(W_y h_t + b_y).
     \]

3. **Backward Pass**:
   - Compute the gradients of the loss with respect to the parameters by backpropagating errors through the unrolled network. The gradient of the loss \( \mathcal{L} \) with respect to the hidden state at time \( t \) is:
     \[
     \frac{\partial \mathcal{L}}{\partial h_t} = \frac{\partial \mathcal{L}}{\partial y_t} \frac{\partial y_t}{\partial h_t} + \frac{\partial \mathcal{L}}{\partial h_{t+1}} \frac{\partial h_{t+1}}{\partial h_t}.
     \]

4. **Gradient Accumulation**:
   - The temporal dependencies result in gradients that accumulate over time. This makes the training process computationally intensive, especially for long sequences.

![bptt](/images/BPTT.png)
---

#### Challenges with BPTT

1. **Vanishing Gradients**:
   - Gradients diminish exponentially as they are propagated backward through many time steps, making it difficult to update parameters effectively for long-term dependencies.

2. **Exploding Gradients**:
   - Gradients can grow uncontrollably during backpropagation, leading to instability. Techniques like gradient clipping are often used to address this.

3. **Computational Cost**:
   - Unrolling the network over many time steps increases memory and computational requirements.

#### Truncated BPTT

To address the computational inefficiency of full BPTT, **truncated BPTT** is commonly used:
- The sequence is divided into smaller chunks, and backpropagation is applied within each chunk.
- This reduces memory usage and computational cost, but it may limit the ability to capture long-term dependencies.

---

#### Challenges with Vanilla RNNs

1. **Vanishing Gradients**:
   - During backpropagation through time (BPTT), gradients become smaller as they are propagated backward through many time steps. This makes it difficult for RNNs to learn long-term dependencies, as the updates to weights for distant time steps become negligible.

2. **Exploding Gradients**:
   - In some cases, gradients can grow exponentially, leading to instability during training. Techniques like gradient clipping are used to mitigate this issue.

3. **Short-Term Memory**:
   - Due to vanishing gradients, vanilla RNNs are primarily limited to capturing short-term dependencies.


### Long Short-Term Memory (LSTM)

To address the limitations of vanilla RNNs, **Long Short-Term Memory (LSTM)** networks were introduced. LSTMs enhance the memory capability of RNNs by introducing a more sophisticated structure for controlling information flow. The key innovation is the **cell state**, which acts as a long-term memory and is explicitly designed to mitigate vanishing gradients.


##### LSTM Architecture

At each time step \( t \), an LSTM has three gates (input, forget, output) and a cell state \( C_t \), which is updated as follows:

1. **Forget Gate**:
   - Determines what information to discard from the previous cell state:
     \[
     f_t = \sigma(W_f [h_{t-1}, x_t] + b_f),
     \]
     where \( \sigma \) is the sigmoid activation.

2. **Input Gate**:
   - Determines what new information to add to the cell state:
     \[
     i_t = \sigma(W_i [h_{t-1}, x_t] + b_i),
     \]
     \[
     \tilde{C}_t = \tanh(W_C [h_{t-1}, x_t] + b_C),
     \]
     \[
     C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t,
     \]
     where \( \odot \) denotes element-wise multiplication.

3. **Output Gate**:
   - Determines what information to output from the cell state:
     \[
     o_t = \sigma(W_o [h_{t-1}, x_t] + b_o),
     \]
     \[
     h_t = o_t \odot \tanh(C_t).
     \]

By separating the cell state \( C_t \) from the hidden state \( h_t \), LSTMs enable gradients to flow more effectively through time, thereby addressing the vanishing gradient problem.

![lstm](/images/lstm.png)

### LSTM snippet code

```python 
import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Define the LSTM layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        
        # Define the output layer
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers , x.size(0), self.hidden_dim).to(x.device)
        
        # Pass through LSTM layer
        out, _ = self.lstm(x, (h0, c0))  # out: (batch_size, seq_length, hidden_dim)
        
        # Pass the last hidden state to the output layer
        out = self.fc(out[:, -1, :])  # out: (batch_size, output_dim)
        return out

# Example usage
if __name__ == "__main__":
    # Define model parameters
    input_dim = 10    # Number of input features
    hidden_dim = 50   # Number of LSTM units
    output_dim = 1    # Number of output features
    seq_length = 20   # Length of the input sequence
    batch_size = 16   # Batch size

    # Create an instance of the model
    model = LSTMModel(input_dim, hidden_dim, output_dim, num_layers=2)
    
    # Generate random input tensor (batch_size, seq_length, input_dim)
    x = torch.randn(batch_size, seq_length, input_dim)
    
    # Forward pass
    output = model(x)

```


### Gated Recurrent Units (GRU)

**Gated Recurrent Units (GRUs)** are a simplified variant of LSTMs that retain much of their effectiveness while reducing computational complexity. GRUs merge the forget and input gates into a single **update gate** and simplify the cell state management.

1. **Update Gate**:
   \[
   z_t = \sigma(W_z [h_{t-1}, x_t] + b_z).
   \]

2. **Reset Gate**:
   \[
   r_t = \sigma(W_r [h_{t-1}, x_t] + b_r).
   \]

3. **Hidden State Update**:
   \[
   \tilde{h}_t = \tanh(W_h [r_t \odot h_{t-1}, x_t] + b_h),
   \]
   \[
   h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t.
   \]

GRUs are computationally more efficient than LSTMs, making them a popular choice for time series tasks with limited resources.

![gru](/images/gru.png)

### Summary

1. **RNNs**:
   - Introduced the concept of memory through hidden states.
   - Struggle with long-term dependencies due to vanishing gradients.

2. **LSTMs**:
   - Introduced cell states and gating mechanisms to address vanishing gradients and improve long-term memory.

3. **GRUs**:
   - Simplified LSTM architecture with fewer parameters, retaining much of the power of LSTMs.

Both LSTMs and GRUs have become standard tools in deep learning for time series and sequential data, with the choice between them often dictated by the specific task and computational constraints.

---

## Latent Neural ODEs

Latent Neural Ordinary Differential Equations (Latent Neural ODEs) extend the concept of Neural ODEs to model continuous-time dynamics in latent spaces. This allows for powerful representations of sequential data, particularly when the observed dynamics are complex or partially observable.


### Neural ODEs and the Augmented Lagrangian

In Neural ODEs, the dynamics of the state \( h(t) \) are described by:

\[
\frac{dh(t)}{dt} = f_\theta(h(t), t),
\]

where \( f_\theta \) is a neural network parameterized by \( \theta \). To compute gradients efficiently during training, we use the **augmented Lagrangian** formulation to derive the equations governing the forward and adjoint dynamics.

#### Augmented Lagrangian

The augmented Lagrangian \( \mathcal{L} \) is defined as:

\[
\mathcal{L} = J(h(T)) + \int_0^T a^T \left( \frac{dh}{dt} - f_\theta(h(t), t) \right) dt,
\]

where:
- \( J(h(T)) \): Loss function evaluated at the final time \( T \),
- \( a(t) \): Adjoint variable (Lagrange multiplier) enforcing the ODE constraint \( \frac{dh}{dt} = f_\theta(h(t), t) \).

#### Derivation of the Forward Equation

To derive the **forward dynamics**, we take the variation of \( \mathcal{L} \) with respect to the adjoint variable \( a(t) \):

\[
\frac{\delta \mathcal{L}}{\delta a} = \frac{dh}{dt} - f_\theta(h(t), t).
\]

Setting this to zero yields the **forward equation**:

\[
\frac{dh}{dt} = f_\theta(h(t), t).
\]

This is the original Neural ODE governing the evolution of \( h(t) \).


#### Derivation of the Adjoint Equation

Next, we derive the **adjoint dynamics** by taking the variation of \( \mathcal{L} \) with respect to \( h(t) \). Using the augmented Lagrangian:

\[
\frac{\delta \mathcal{L}}{\delta h} = \int_0^T \frac{\partial}{\partial h} \left[ a^T \left( \frac{dh}{dt} - f_\theta(h(t), t) \right) \right] dt.
\]

Breaking this down step-by-step:

1. **Separate terms**:
   \[
   \frac{\delta \mathcal{L}}{\delta h} = \int_0^T \frac{\partial}{\partial h} \left[ a^T \frac{dh}{dt} \right] dt - \int_0^T a^T \frac{\partial f_\theta(h(t), t)}{\partial h} dt.
   \]

2. **First term integration by parts**:
   \[
   \int_0^T \frac{\partial}{\partial h} \left[ a^T \frac{dh}{dt} \right] dt = \int_0^T -\frac{d a^T}{dt} \, dt + \frac{\partial}{\partial h(t)}\left[ a^T h \right]_0^T.
   \]

   The boundary term \( [a^T h]_0^T \) adds a compatibility condition to the final state, and the remaining term contributes to the adjoint equation.

3. **Combine terms**:
   Substituting back:
   \[
   \frac{\delta \mathcal{L}}{\delta h} = \int_0^T \left[ -\frac{d a^T}{dt} h \right] dt - \int_0^T a^T \frac{\partial f_\theta(h(t), t)}{\partial h} dt.
   \]

From this, we derive the **adjoint equation**:

\[
\frac{d a(t)}{dt} = -a^T \frac{\partial f_\theta(h(t), t)}{\partial h}.
\]

#### Boundary Conditions and Compatibility

To solve the adjoint equation, we also need the **boundary condition** for \( a(T) \), which is derived from the terminal loss \( J(h(T)) \):

\[
a(T) = \frac{\partial J(h(T))}{\partial h}.
\]

This boundary condition ensures that the adjoint equation integrates backward in time correctly.

#### Summary of Dynamics

1. **Forward Equation** (integrate forward in time):
   \[
   \frac{dh}{dt} = f_\theta(h(t), t).
   \]

2. **Adjoint Equation** (integrate backward in time):
   \[
   \frac{d a(t)}{dt} = -a^T \frac{\partial f_\theta(h(t), t)}{\partial h}.
   \]

3. **Boundary Condition for Adjoint**:
   \[
   a(T) = \frac{\partial J(h(T))}{\partial h}.
   \]

4. **Gradient with Respect to Parameters**:
   Finally, the gradient of the loss with respect to the parameters \( \theta \) is given by:
   \[
   \frac{d\mathcal{L}}{d\theta} = \int_0^T a(t) \frac{\partial f_\theta(h(t), t)}{\partial \theta} dt.
   \]

#### Latent Neural ODE (Latent ODE)

Latent Neural ODEs (Latent ODEs) extend Neural ODEs by evolving a **latent state** instead of the observed state. This is particularly useful for partially observed systems, irregularly sampled data, or systems with hidden dynamics. The latent state \( z(t) \) evolves continuously in time, and the observations \( x(t) \) are reconstructed from the latent dynamics.

##### Workflow of a Latent ODE

1. **Latent State Initialization**:
   - The initial latent state \( z_0 \) must be inferred from the data points \( \{x_i, t_i\}_{i=1}^N \) and their timestamps.
   - An **ODE-RNN Encoder** is used to estimate \( z_0 \) in a way that captures temporal dependencies and irregular sampling.

2. **Latent State Evolution**:
   - Starting from \( z_0 \), the latent state evolves continuously over time using an ODE defined by a neural network \( f_\theta \):
     \[
     \frac{dz(t)}{dt} = f_\theta(z(t), t).
     \]

3. **Reconstruction**:
   - At each timestamp \( t_i \), the latent state \( z(t_i) \) is decoded into the observation space via a neural network \( \text{DecoderNN} \):
     \[
     x(t_i) \approx \text{DecoderNN}(z(t_i)).
     \]

#### ODE-RNN Encoder

The ODE-RNN Encoder is designed to handle irregularly sampled data points and their timestamps. It combines an RNN-like update mechanism with the continuous-time evolution of Neural ODEs.

##### Structure of ODE-RNN Encoder:
1. **Input**:
   - Observed data points \( \{x_i\}_{i=1}^N \) and their timestamps \( \{t_i\}_{i=1}^N \).

2. **Initialization**:
   - Start with an initial hidden state \( h_0 = 0 \).

3. **Iterative Updates**:
   - For each data point \( (x_i, t_i) \), perform the following steps:
     1. **Continuous Evolution**:
        - Use an ODE solver to evolve the hidden state from \( t_{i-1} \) to \( t_i \):
          \[
          h_i' = \text{ODESolve}(f_\theta, h_{i-1}, (t_{i-1}, t_i)).
          \]
     2. **Discrete Update**:
        - Update the hidden state using the current observation \( x_i \):
          \[
          h_i = \text{RNNCell}(h_i', x_i).
          \]
     3. **Output**:
        - Optionally, compute an output \( o_i = \text{OutputNN}(h_i) \).

4. **Return**:
   - The final hidden state \( h_N \) is used to estimate a posterior Gaussian distribution \( p(z_0 \mid x_1, x_2, \ldots, x_N) \) for the initial latent state \( z_0 \).


#### Variational Posterior for \( z_0 \)

The ODE-RNN Encoder runs **backward in time** from \( t_N \) to \( t_0 \), constructing a posterior Gaussian distribution \( p(z_0 \mid x_1, x_2, \ldots, x_N) \), parameterized by:
- Mean \( \mu(z_0) \),
- Variance \( \sigma^2(z_0) \).

Using the **reparameterization trick**, a sample of \( z_0 \) is drawn as:
\[
z_0 = \mu(z_0) + \sigma(z_0) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I).
\]


#### Latent Dynamics with Neural ODE

Once \( z_0 \) is sampled, the latent state evolves forward in time using a Neural ODE:

\[
z(t) = z_0 + \int_{0}^{t} f_\theta(z(\tau), \tau) \, d\tau.
\]

At each timestamp \( t_i \), the latent state \( z(t_i) \) is passed through a decoder network to produce the corresponding observation \( x_i \):
\[
x_i \approx \text{DecoderNN}(z(t_i)).
\]

![latentODE](/images/LatentODE.png)

#### NeuralODE snippet code 

```python
import torch
import torch.nn as nn
from torchdyn.core import NeuralODE
from torchdyn.datasets import ToyDataset

# Define the vector field (dynamics) for the ODE
class SimpleDynamics(nn.Module):
    def __init__(self):
        super(SimpleDynamics, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 2)
        )

    def forward(self, t, x):
        return self.net(x)

# Create a Neural ODE instance with the dynamics
vector_field = SimpleDynamics()
neural_ode = NeuralODE(vector_field, solver='rk4', sensitivity='autograd')

# Generate toy data
dataset = ToyDataset()
X, y = dataset.generate(n_samples=500, noise=0.05)
X, y = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# Training the Neural ODE
optimizer = torch.optim.Adam(neural_ode.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# Training loop
epochs = 100
for epoch in range(epochs):
    optimizer.zero_grad()
    
    # Solve the ODE for the input data
    t_span = torch.linspace(0, 1, 10)  # Time span for solving the ODE
    X_pred = neural_ode(X, t_span)
    
    # Compute loss
    loss = criterion(X_pred[-1], y)  # Compare the final state to the labels
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
```
---

## Transformers

Transformers were initially introduced in the field of natural language processing (NLP) for tasks such as translation and text generation. Their architecture, based on the **attention mechanism**, eliminates the need for sequential processing, making them highly parallelizable and effective for capturing long-range dependencies.

A transformer is composed of two main components:
- **Encoder**: Encodes the input sequence into a latent representation.
- **Decoder**: Translates the latent representation into the target sequence, processing the output sequentially.

For example, in translation tasks:
1. The encoder processes the input sentence (e.g., in English) to produce a latent representation.
2. The decoder uses this representation to conditionally generate the output sentence (e.g., in French), one word at a time.

![transformer](/images/transformer.jpg)

### Unboxing the Transformer Architecture

The transformer consists of several key components, which are explained below.


#### 1. **Positional Encoding**

Unlike RNNs or CNNs, transformers do not have a natural notion of order in sequences. To encode the order of elements in a sequence, a **positional encoding** is added to the input embeddings.

The positional encoding is defined as:
\[
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right), \quad PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right),
\]
where:
- \( pos \): Position in the sequence.
- \( i \): Dimension index.
- \( d \): Embedding dimension.

This adds unique positional information to each input token, enabling the transformer to distinguish their order.

![pe](/images/positional_encoding.png)

#### 2. **Self-Attention Mechanism**

The self-attention mechanism is the core innovation of transformers. It computes the relevance of each token in the input sequence with respect to all other tokens, enabling the model to focus on important relationships.

##### Key, Query, and Value

Each input token is mapped to three vectors:
- **Query (Q)**: Represents the token making the query.
- **Key (K)**: Represents the token being queried.
- **Value (V)**: The information content of the token.

These are computed using learned linear transformations:
\[
Q = XW_Q, \quad K = XW_K, \quad V = XW_V,
\]
where \( X \) is the input sequence and \( W_Q, W_K, W_V \) are learnable weight matrices.

##### Attention Scores

The relevance between tokens is computed as:
\[
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V,
\]
where:
- \( QK^T \): Computes the similarity between queries and keys.
- \( \sqrt{d_k} \): Rescales the dot product to prevent large gradients as \( d_k \) grows.
- **Softmax**: Converts raw scores into a probability distribution, emphasizing the most relevant tokens.

##### Effect of Softmax and Rescaling

- The **softmax** ensures that attention weights are normalized, focusing the model's attention on the most relevant tokens.
- The rescaling by \( \sqrt{d_k} \) prevents overly large dot products when the embedding dimension \( d_k \) is large, stabilizing training.

![sdp](/images/scaled-dot-product.png)

#### 3. **Multi-Head Attention**

Instead of computing attention once, transformers use **multi-head attention** to capture different aspects of relationships between tokens.

Each head independently computes:
\[
\text{head}_i = \text{Attention}(QW_Q^i, KW_K^i, VW_V^i),
\]
where \( W_Q^i, W_K^i, W_V^i \) are projection matrices for the \( i \)-th head.

The outputs of all heads are concatenated and passed through a linear transformation:
\[
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W_O,
\]
where \( W_O \) is a learnable weight matrix.

![mha](/images/multi-head-attention.png)


#### 4. **Feedforward Network**

Each transformer layer contains a position-wise feedforward network, applied independently to each token:
\[
FFN(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2,
\]
where \( W_1, W_2 \) are learnable weight matrices.

This adds non-linearity and further transforms the token embeddings.


#### 5. **Decoder and Cross-Attention**

The decoder generates the output sequence by attending to both:
- The output of the encoder (cross-attention).
- Its own previous outputs (self-attention).

In **cross-attention**, the queries \( Q \) come from the decoder, while the keys \( K \) and values \( V \) come from the encoder:
\[
\text{CrossAttention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V.
\]

This allows the decoder to condition its generation on the encoded representation of the input sequence.


### Transformers for Time Series

Transformers have been successfully adapted for time series applications due to their ability to capture long-range dependencies and handle irregularly sampled data. In time series, the main challenge lies in incorporating the sequential nature of the data while ensuring that the model respects causality. To address this, **causal masking** is introduced, along with other adaptations to tailor transformers for time series tasks.


#### Key Adaptations for Time Series

1. **Causal Masking**:
   - In time series, predictions at time \( t \) should depend only on observations from \( t \) and earlier. To enforce this causality in the transformer, a **causal mask** is applied to the self-attention mechanism.
   - The causal mask ensures that when computing the attention for a token at time \( t \), only tokens at times \( \leq t \) are considered.

   Mathematically, let \( A \) denote the raw attention scores \( QK^T \). The causal mask is implemented as:
   \[
   A_{ij} =
   \begin{cases} 
      A_{ij} & \text{if } i \geq j \\
      -\infty & \text{if } i < j
   \end{cases}
   \]
   - The attention scores for future tokens (\( i < j \)) are set to \( -\infty \), effectively masking them out after applying the softmax:
     \[
     \text{softmax}(A)_{ij} = \frac{\exp(A_{ij})}{\sum_{k \leq j} \exp(A_{ik})}.
     \]

2. **Position Encoding**:
   - As in NLP, transformers for time series require a mechanism to encode the temporal order of data points. **Positional encoding** or learnable embeddings are added to the input embeddings to provide a sense of time.

3. **Handling Missing Data**:
   - Time series data often contain missing values or irregular sampling. Transformers handle this by incorporating explicit masks for missing data or using imputation techniques.

4. **Windowing**:
   - For long time series, attention computation can become prohibitively expensive. Transformers often process data in fixed-size windows to reduce computational overhead, using specialized mechanisms like sliding windows or memory modules to capture long-term dependencies.

#### Workflow of a Transformer for Time Series

1. **Input Embedding**:
   - The raw time series \( x(t) \) is mapped to embeddings using a combination of learnable embeddings and positional encodings.

2. **Causal Masking in Self-Attention**:
   - The self-attention layer applies the causal mask to ensure predictions at time \( t \) depend only on past observations.

3. **Multi-Head Self-Attention**:
   - Captures relationships across different time points and features, enabling the model to focus on relevant parts of the sequence.

4. **Feedforward Layers**:
   - Position-wise feedforward networks transform the embeddings independently for each time step.

5. **Output Layer**:
   - For forecasting tasks, the output layer predicts future values based on the embeddings from the transformer.

#### Advantages of Transformers for Time Series

1. **Long-Range Dependencies**:
   - Unlike RNNs, transformers capture dependencies across arbitrarily long time ranges without suffering from vanishing gradients.

2. **Parallel Processing**:
   - Transformers process entire sequences simultaneously, significantly speeding up training compared to sequential models.

### Transformer snippet code

```python
import torch
import torch.nn as nn

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_encoder_layers, num_decoder_layers, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim, dropout)
        
        self.transformer = nn.Transformer(
            d_model=embed_dim,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout
        )
        
        self.fc_out = nn.Linear(embed_dim, input_dim)
    
    def forward(self, src, tgt):
        # Apply embedding and positional encoding
        src = self.embedding(src) * torch.sqrt(torch.tensor(src.size(-1), dtype=torch.float32)).to(src.device)
        src = self.positional_encoding(src)
        
        tgt = self.embedding(tgt) * torch.sqrt(torch.tensor(tgt.size(-1), dtype=torch.float32)).to(tgt.device)
        tgt = self.positional_encoding(tgt)
        
        # Pass through the transformer
        output = self.transformer(src, tgt)
        
        # Map to the original input dimension
        output = self.fc_out(output)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # Add batch dimension
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# Example usage
if __name__ == "__main__":
    batch_size, seq_len, input_dim, embed_dim = 32, 10, 1, 64
    num_heads = 4
    num_encoder_layers = 3
    num_decoder_layers = 3
    
    # Create model
    model = TimeSeriesTransformer(input_dim, embed_dim, num_heads, num_encoder_layers, num_decoder_layers)

    # Input data
    src = torch.rand(seq_len, batch_size, input_dim)  # (seq_len, batch_size, input_dim)
    tgt = torch.rand(seq_len, batch_size, input_dim)  # (seq_len, batch_size, input_dim)

    # Forward pass
    output = model(src, tgt)
```
---

## State Space Models

State Space Models (SSMs) provide a mathematical framework to describe dynamical systems using a set of internal states. These states evolve over time and are influenced by inputs and noise. SSMs are widely used in fields such as control theory, signal processing, and time series modeling due to their ability to represent both the system's internal dynamics and its relationship with external inputs.


### Mathematical Formulation

An SSM represents a dynamical system using two key equations:

1. **State Dynamics**:
   \[
   x'(t) = \mathbf{A}x(t) + \mathbf{B}u(t),
   \]
   where:
   - \( x(t) \): The state vector at time \( t \),
   - \( \mathbf{A} \): State transition matrix, governing how the state evolves,
   - \( \mathbf{B} \): Input matrix, describing how inputs affect the state,
   - \( u(t) \): Input vector at time \( t \).

2. **Output Equation**:
   \[
   y(t) = \mathbf{C}x(t) + \mathbf{D}u(t),
   \]
   where:
   - \( y(t) \): Output vector at time \( t \),
   - \( \mathbf{C} \): Output matrix, mapping the state to the output,
   - \( \mathbf{D} \): Feedthrough matrix, directly relating inputs to the output.

In practice, the feedthrough matrix \( \mathbf{D} \) is often omitted because its behavior can be easily replicated in deep learning models through a skip connection. This simplification helps focus on the dynamics governed by \( \mathbf{A}, \mathbf{B}, \) and \( \mathbf{C} \).

![ssm](/images/SSM.png)

### Control System Perspective

In control systems, the components of an SSM have intuitive interpretations:
- **State (\( x \))**: Represents the system's memory, storing information about past inputs and outputs.
- **Input (\( u \))**: External signals or commands that influence the system's behavior.
- **Output (\( y \))**: Observed quantities generated by the system.

Key concepts in control theory align with SSMs:
- **Control Law**: Determines the inputs (\( u(t) \)) needed to drive the system to a desired state.
- **Observation**: Uses the output (\( y(t) \)) to infer the internal state (\( x(t) \)).

### Convolutional Solution

The output \( y_k \) at a given time \( k \) can be expressed as a convolution of the system's input \( u_k \) with a kernel \( \bar{K} \):

\[
y_k = K * u,
\]

where the kernel \( \bar{K} \) is defined as:
\[
\mathbf{\bar{K}}_k = (\mathbf{\bar{C}} \mathbf{\bar{B}}, \mathbf{\bar{C}} \mathbf{\bar{A}} \mathbf{\bar{B}}, \dots, \mathbf{\bar{C}} \mathbf{\bar{A}}^k \mathbf{\bar{B}}),
\]
with \( \mathbf{\bar{A}}, \mathbf{\bar{B}}, \mathbf{\bar{C}} \) being discretized versions of \( \mathbf{A}, \mathbf{B}, \mathbf{C} \).

![sketch](/images/sketch-SSM.png)

### Stabilizing the Convolution Kernel

To ensure the system remains stable during training, the matrix \( \mathbf{A} \) is designed such that its eigenvalues lie within a stable region. A notable example is the **HiPPO matrix**, which enforces stability while maintaining the capacity to represent long-term dependencies. Stability guarantees that the convolution kernel has a **compact support**, preventing gradients from exploding or vanishing.

### Advantages of SSMs

1. **Efficient Training with Convolutions**:
   - During training, the convolutional formulation of SSMs is used. Convolutions allow efficient computation of gradients, making the optimization process stable and scalable.
   - Easy to parallelize

2. **Fast Inference with Temporal Iteration**:
   - In inference, the linear recurrence formulation is used:
     \[
     x_{k+1} = \mathbf{A}x_k + \mathbf{B}u_k,
     \]
     \[
     y_k = \mathbf{C}x_k + \mathbf{D}u_k.
     \]
     This approach is computationally cheaper and requires minimal memory, as only the current state \( x_k \) needs to be stored.

![train-vs-inference](/images/train-inference-SSM.png)

State Space Models bridge the gap between traditional dynamical systems and modern machine learning. By leveraging their dual convolutional and iterative formulations, SSMs enable efficient and robust training while maintaining scalability for inference in time series and control applications.

### SSM Snippets Code

Currently, there is no direct implementation of a state-space model (SSM) layer in the standard PyTorch distribution. However, various research groups have developed implementations of popular SSM models, particularly for deep learning applications. Two notable implementations are:

1. **[S4](https://github.com/TariqAHassan/S4Torch) (Structured State Space)**:
   - S4 introduces a structured approach to SSMs with long-range memory and computational efficiency.
   - It is specifically designed to handle long-range dependencies in sequences, making it suitable for tasks such as time series forecasting, NLP, and vision.

2. **[MAMBA](https://github.com/state-spaces/mamba) (Matrix Algebra Method for Band Attention)**:
   - MAMBA is an advanced SSM-based method that efficiently computes convolutions with structured matrices while maintaining stability and scalability.
   - It leverages algebraic properties to optimize the computation of the state-space model kernel.

A more user-friendly implementation of mamba is in the transformers lib of Huggingface: 

```python 
from transformers.models.mamba import MambaConfig, MambaModel
import torch

# Step 1: Define the configuration for the MAMBA model
config = MambaConfig(
    d_model=128,               # Dimensionality of model
    num_layers=4,              # Number of SSM layers
    hidden_size=256,           # Size of the feedforward layers
    dropout=0.1,               # Dropout rate
    layer_norm_eps=1e-5,       # Layer normalization epsilon
    initializer_range=0.02     # Range for initializing weights
)

# Step 2: Initialize the MAMBA model
model = MambaModel(config)

# Step 3: Create some dummy time series data
batch_size, seq_len, feature_dim = 16, 50, 128
input_data = torch.randn(batch_size, seq_len, feature_dim)  # Shape: (batch_size, seq_len, d_model)

# Step 4: Pass the data through the MAMBA model
output = model(input_data)

```

Both S4 and MAMBA are excellent examples of modern adaptations of SSMs in deep learning and are available in open-source repositories for experimentation and integration into custom pipelines.

## Evaluation Metrics

Evaluating time series models requires metrics that align with the nature of the task, whether it's forecasting, classification, or similarity measurement. Here, we discuss two common metrics: **Mean Squared Error (MSE)** for prediction and **Dynamic Time Warping (DTW)** for aligning time series.


#### **1. Mean Squared Error (MSE)**

The **Mean Squared Error** measures the average squared difference between predicted and actual values. It is a widely used metric for time series forecasting tasks, providing a sense of how closely the model's predictions match the ground truth.

\[
\text{MSE} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2,
\]

where:
- \( y_i \): Actual value at time \( i \),
- \( \hat{y}_i \): Predicted value at time \( i \),
- \( n \): Number of time steps.

**Key Features**:
- Penalizes large errors more than small ones, making it sensitive to outliers.
- Works well when the time series data is smooth and outliers are not dominant.

#### **2. Dynamic Time Warping (DTW)**

**Dynamic Time Warping** is a distance metric used to measure the similarity between two time series. Unlike point-wise metrics like MSE, DTW aligns sequences by warping their time dimensions to minimize the distance between them.

**Alignment Objective**:
DTW finds an optimal alignment path \( P \) that minimizes the cumulative distance between two sequences \( A \) and \( B \):

\[
\text{DTW}(A, B) = \min_P \sum_{(i,j) \in P} d(A_i, B_j),
\]

where:
- \( d(A_i, B_j) \): Distance between points \( A_i \) and \( B_j \),
- \( P \): Alignment path that allows for non-linear time warping.

**Key Features**:
- Handles sequences of different lengths or those with temporal shifts.
- Commonly used for clustering, classification, or anomaly detection in time series.

**Applications**:
- Aligning two time series for comparison.
- Measuring similarity in pattern recognition tasks.

These metrics address distinct aspects of time series evaluation:
- MSE focuses on precise point-wise prediction.
- DTW emphasizes pattern and temporal alignment, making it suitable for tasks where shape and timing matter more than exact values.
