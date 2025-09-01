import React from 'react';
import { Container, Title, Text, Stack, Grid, Paper, List } from '@mantine/core';
import { InlineMath, BlockMath } from 'react-katex';
import 'katex/dist/katex.min.css';
import CodeBlock from 'components/CodeBlock';

const OptimizationAlgorithms = () => {
  return (
    <Container size="xl">
      <Stack spacing="xl">
        
        <div id="modern-optimizers">
          <Title order={1} mb="xl">
            Optimization Algorithms
          </Title>
          <Text size="xl" className="mb-6">
            Mathematical Foundations of Modern Optimizers
          </Text>
          
          <Paper className="p-6 bg-blue-50 mb-6">
            <Title order={3} className="mb-4">Optimization Theory</Title>
            <Text className="mb-4">
              Modern optimizers adapt learning rates and incorporate momentum to improve convergence.
              Understanding their mathematical foundations is crucial for effective training.
            </Text>
            
            <Text className="mb-4">
              <strong>General Optimization Problem:</strong> Find <InlineMath>{`\\theta^* = \\arg\\min_{\\theta} \\mathcal{L}(\\theta)`}</InlineMath>
            </Text>
            <BlockMath>{`\\theta_{t+1} = \\theta_t + \\Delta\\theta_t`}</BlockMath>
            
            <Grid gutter="lg">
              <Grid.Col span={6}>
                <Paper className="p-4 bg-white">
                  <Title order={4} mb="sm">Momentum-based Methods</Title>
                  <Text className="mb-3">
                    <strong>SGD with Momentum:</strong> Accelerates convergence by accumulating gradients
                  </Text>
                  <BlockMath>{`
                    \\\\begin{aligned}
                    \\\\mathbf{v}_t &= \\\\beta \\\\mathbf{v}_{t-1} + \\\\nabla \\\\mathcal{L}(\\\\theta_t) \\\\\\\\
                    \\\\theta_{t+1} &= \\\\theta_t - \\\\eta \\\\mathbf{v}_t
                    \\\\end{aligned}
                  `}</BlockMath>
                  <CodeBlock language="python" code={`import torch
import torch.optim as optim

# SGD with Momentum
# Mathematical formulation:
# v_t = β*v_{t-1} + ∇L(θ_t)  
# θ_{t+1} = θ_t - η*v_t
# where β controls momentum strength

class SGDWithMomentum:
    def __init__(self, params, lr=0.01, momentum=0.9):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.velocity = [torch.zeros_like(p) for p in self.params]
    
    def step(self):
        for i, param in enumerate(self.params):
            if param.grad is not None:
                # Update velocity
                self.velocity[i] = (self.momentum * self.velocity[i] + 
                                  param.grad.data)
                # Update parameters
                param.data -= self.lr * self.velocity[i]
    
    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()

# Test on simple quadratic function
def quadratic_loss(x):
    return (x[0] - 1)**2 + (x[1] - 2)**2

x = torch.tensor([0.0, 0.0], requires_grad=True)
optimizer = SGDWithMomentum([x], lr=0.1, momentum=0.9)

print("SGD with Momentum optimization:")
for step in range(10):
    optimizer.zero_grad()
    loss = quadratic_loss(x)
    loss.backward()
    
    print(f"Step {step}: x = {x.data}, loss = {loss:.4f}")
    optimizer.step()`} />
                </Paper>
              </Grid.Col>
              
              <Grid.Col span={6}>
                <Paper className="p-4 bg-white">
                  <Title order={4} mb="sm">Adaptive Learning Rates</Title>
                  <CodeBlock language="python" code={`# Adagrad: Adaptive learning rates
# Gₜ = Gₜ₋₁ + ∇f(θₜ)²
# θₜ₊₁ = θₜ - η/√(Gₜ + ε) · ∇f(θₜ)

class AdaGrad:
    def __init__(self, params, lr=0.01, eps=1e-8):
        self.params = list(params)
        self.lr = lr
        self.eps = eps
        self.sum_squared_grads = [torch.zeros_like(p) for p in self.params]
    
    def step(self):
        for i, param in enumerate(self.params):
            if param.grad is not None:
                # Accumulate squared gradients
                self.sum_squared_grads[i] += param.grad.data ** 2
                
                # Adaptive learning rate
                adapted_lr = self.lr / (torch.sqrt(
                    self.sum_squared_grads[i]) + self.eps)
                
                # Update parameters
                param.data -= adapted_lr * param.grad.data
    
    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()

# Compare with built-in optimizer
x1 = torch.tensor([0.0, 0.0], requires_grad=True)
x2 = torch.tensor([0.0, 0.0], requires_grad=True)

custom_optimizer = AdaGrad([x1], lr=0.1)
builtin_optimizer = optim.Adagrad([x2], lr=0.1)

print("\\nAdaGrad comparison:")
for step in range(5):
    # Custom implementation
    custom_optimizer.zero_grad()
    loss1 = quadratic_loss(x1)
    loss1.backward()
    custom_optimizer.step()
    
    # Built-in implementation
    builtin_optimizer.zero_grad()
    loss2 = quadratic_loss(x2)
    loss2.backward()
    builtin_optimizer.step()
    
    print(f"Step {step}: Custom x = {x1.data}, Built-in x = {x2.data}")
    print(f"  Difference: {torch.norm(x1.data - x2.data):.6f}")`} />
                </Paper>
              </Grid.Col>
            </Grid>
          </Paper>
        </div>

        <div id="adam-rmsprop-adagrad">
          <Title order={2} className="mb-6">Adam, RMSprop, AdaGrad Derivations</Title>
          
          <Paper className="p-6 bg-gray-50 mb-6">
            <Title order={3} className="mb-4">Adam Optimizer Implementation</Title>
            <Text className="mb-4">
              Adam combines momentum with adaptive learning rates using bias-corrected moving averages.
            </Text>
            
            <CodeBlock language="python" code={`# Adam Optimizer from scratch
# mₜ = β₁mₜ₋₁ + (1-β₁)∇f(θₜ)
# vₜ = β₂vₜ₋₁ + (1-β₂)∇f(θₜ)²
# m̂ₜ = mₜ/(1-β₁ᵗ), v̂ₜ = vₜ/(1-β₂ᵗ)
# θₜ₊₁ = θₜ - η·m̂ₜ/(√v̂ₜ + ε)

class Adam:
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = list(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.step_count = 0
        
        # Initialize momentum and velocity
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
    
    def step(self):
        self.step_count += 1
        
        for i, param in enumerate(self.params):
            if param.grad is not None:
                grad = param.grad.data
                
                # Update biased first moment estimate
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
                
                # Update biased second raw moment estimate  
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad**2
                
                # Compute bias-corrected first moment estimate
                m_hat = self.m[i] / (1 - self.beta1**self.step_count)
                
                # Compute bias-corrected second raw moment estimate
                v_hat = self.v[i] / (1 - self.beta2**self.step_count)
                
                # Update parameters
                param.data -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)
    
    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()

# Test Adam implementation
def rosenbrock(x):
    """Rosenbrock function: challenging optimization landscape"""
    return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

x_custom = torch.tensor([-1.0, 1.0], requires_grad=True)
x_builtin = torch.tensor([-1.0, 1.0], requires_grad=True)

custom_adam = Adam([x_custom], lr=0.01)
builtin_adam = optim.Adam([x_builtin], lr=0.01)

print("Adam optimizer comparison on Rosenbrock function:")
for step in range(100):
    # Custom Adam
    custom_adam.zero_grad()
    loss1 = rosenbrock(x_custom)
    loss1.backward()
    custom_adam.step()
    
    # Built-in Adam
    builtin_adam.zero_grad()
    loss2 = rosenbrock(x_builtin)
    loss2.backward()
    builtin_adam.step()
    
    if step % 20 == 0:
        print(f"Step {step}:")
        print(f"  Custom: x = {x_custom.data}, loss = {loss1:.6f}")
        print(f"  Built-in: x = {x_builtin.data}, loss = {loss2:.6f}")
        print(f"  Difference: {torch.norm(x_custom.data - x_builtin.data):.8f}")

print(f"\\nFinal difference: {torch.norm(x_custom.data - x_builtin.data):.8f}")`} />
          </Paper>

          <Paper className="p-4 bg-yellow-50">
            <Title order={4} mb="sm">RMSprop Implementation</Title>
            <CodeBlock language="python" code={`# RMSprop: Root Mean Square Propagation
# E[g²]ₜ = γE[g²]ₜ₋₁ + (1-γ)g²ₜ
# θₜ₊₁ = θₜ - η/√(E[g²]ₜ + ε) · gₜ

class RMSprop:
    def __init__(self, params, lr=0.01, alpha=0.99, eps=1e-8):
        self.params = list(params)
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.square_avg = [torch.zeros_like(p) for p in self.params]
    
    def step(self):
        for i, param in enumerate(self.params):
            if param.grad is not None:
                grad = param.grad.data
                
                # Update exponential moving average of squared gradients
                self.square_avg[i] = (self.alpha * self.square_avg[i] + 
                                    (1 - self.alpha) * grad**2)
                
                # Update parameters
                param.data -= (self.lr * grad / 
                             (torch.sqrt(self.square_avg[i]) + self.eps))
    
    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()

# Performance comparison
def compare_optimizers():
    optimizers = {
        'SGD': lambda params: optim.SGD(params, lr=0.01),
        'Momentum': lambda params: optim.SGD(params, lr=0.01, momentum=0.9),
        'AdaGrad': lambda params: optim.Adagrad(params, lr=0.1),
        'RMSprop': lambda params: optim.RMSprop(params, lr=0.01),
        'Adam': lambda params: optim.Adam(params, lr=0.01),
    }
    
    results = {}
    
    for name, opt_fn in optimizers.items():
        x = torch.tensor([0.0, 0.0], requires_grad=True)
        optimizer = opt_fn([x])
        
        losses = []
        for step in range(50):
            optimizer.zero_grad()
            loss = rosenbrock(x)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        results[name] = {
            'final_x': x.data.clone(),
            'final_loss': losses[-1],
            'convergence_speed': sum(1 for l in losses if l < 1.0)
        }
    
    print("Optimizer comparison results:")
    for name, result in results.items():
        print(f"{name:>10}: final_x = {result['final_x']}, "
              f"final_loss = {result['final_loss']:.6f}, "
              f"steps_to_converge = {result['convergence_speed']}")

compare_optimizers()`} />
          </Paper>
        </div>

        <div id="learning-rate-scheduling">
          <Title order={2} className="mb-6">Learning Rate Scheduling Strategies</Title>
          
          <Paper className="p-4 bg-purple-50">
            <Title order={4} mb="sm">Learning Rate Schedules</Title>
            <CodeBlock language="python" code={`# Various learning rate scheduling strategies
import torch.optim.lr_scheduler as lr_scheduler
import math

class CustomScheduler:
    def __init__(self, optimizer, schedule_type='cosine', **kwargs):
        self.optimizer = optimizer
        self.schedule_type = schedule_type
        self.initial_lr = optimizer.param_groups[0]['lr']
        self.step_count = 0
        self.kwargs = kwargs
    
    def step(self):
        self.step_count += 1
        
        if self.schedule_type == 'exponential':
            gamma = self.kwargs.get('gamma', 0.95)
            new_lr = self.initial_lr * (gamma ** self.step_count)
            
        elif self.schedule_type == 'cosine':
            T_max = self.kwargs.get('T_max', 100)
            new_lr = (self.initial_lr * 
                     (1 + math.cos(math.pi * self.step_count / T_max)) / 2)
            
        elif self.schedule_type == 'polynomial':
            power = self.kwargs.get('power', 2)
            max_steps = self.kwargs.get('max_steps', 100)
            new_lr = (self.initial_lr * 
                     (1 - self.step_count / max_steps) ** power)
            
        elif self.schedule_type == 'warmup_cosine':
            warmup_steps = self.kwargs.get('warmup_steps', 10)
            T_max = self.kwargs.get('T_max', 100)
            
            if self.step_count <= warmup_steps:
                new_lr = self.initial_lr * self.step_count / warmup_steps
            else:
                cos_steps = self.step_count - warmup_steps
                cos_max = T_max - warmup_steps
                new_lr = (self.initial_lr * 
                         (1 + math.cos(math.pi * cos_steps / cos_max)) / 2)
        
        # Update optimizer learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        
        return new_lr

# Test different schedules
def test_lr_schedules():
    x = torch.tensor([0.0], requires_grad=True)
    
    schedules = ['exponential', 'cosine', 'polynomial', 'warmup_cosine']
    
    for schedule in schedules:
        print(f"\\nTesting {schedule} schedule:")
        
        optimizer = optim.SGD([x], lr=0.1)
        scheduler = CustomScheduler(
            optimizer, 
            schedule_type=schedule,
            gamma=0.9,
            T_max=50,
            power=2,
            max_steps=50,
            warmup_steps=5
        )
        
        lrs = []
        for step in range(20):
            current_lr = scheduler.step()
            lrs.append(current_lr)
            
            if step % 5 == 0:
                print(f"  Step {step}: lr = {current_lr:.6f}")

test_lr_schedules()`} />
          </Paper>
        </div>

        <div id="second-order-optimization">
          <Title order={2} className="mb-6">Second-order Optimization Methods</Title>
          
          <Paper className="p-4 bg-green-50">
            <Title order={4} mb="sm">Newton's Method and BFGS</Title>
            <CodeBlock language="python" code={`# Second-order optimization methods
def newton_method(func, x0, max_iter=10, tol=1e-6):
    """Newton's method for optimization"""
    x = x0.clone().detach().requires_grad_(True)
    
    for i in range(max_iter):
        # Compute function value and gradient
        f = func(x)
        grad = torch.autograd.grad(f, x, create_graph=True)[0]
        
        # Compute Hessian
        hessian = torch.zeros(x.size(0), x.size(0))
        for j in range(x.size(0)):
            grad2 = torch.autograd.grad(grad[j], x, retain_graph=True)[0]
            hessian[j] = grad2
        
        # Newton update: x = x - H⁻¹∇f
        try:
            delta = torch.linalg.solve(hessian, grad)
            x_new = x - delta
        except RuntimeError:
            print(f"Hessian not invertible at iteration {i}")
            break
        
        # Check convergence
        if torch.norm(x_new - x) < tol:
            print(f"Converged in {i+1} iterations")
            break
        
        x = x_new.detach().requires_grad_(True)
        print(f"Iteration {i+1}: x = {x.data}, f(x) = {func(x):.6f}")
    
    return x

# L-BFGS approximation
class LBFGS:
    def __init__(self, params, lr=1.0, history_size=10):
        self.params = list(params)
        self.lr = lr
        self.history_size = history_size
        self.s_history = []  # Parameter changes
        self.y_history = []  # Gradient changes
        self.prev_grad = None
    
    def step(self):
        # Flatten all gradients
        grad = torch.cat([p.grad.flatten() for p in self.params])
        
        if self.prev_grad is not None:
            # Compute s and y for BFGS update
            s = self.prev_params - torch.cat([p.data.flatten() for p in self.params])
            y = grad - self.prev_grad
            
            # Store in history
            if len(self.s_history) >= self.history_size:
                self.s_history.pop(0)
                self.y_history.pop(0)
            
            self.s_history.append(s)
            self.y_history.append(y)
        
        # Compute search direction using L-BFGS two-loop recursion
        q = grad.clone()
        alpha = []
        
        # First loop (backward)
        for i in reversed(range(len(self.s_history))):
            s_i, y_i = self.s_history[i], self.y_history[i]
            rho_i = 1.0 / torch.dot(y_i, s_i)
            alpha_i = rho_i * torch.dot(s_i, q)
            q = q - alpha_i * y_i
            alpha.append(alpha_i)
        
        # Scale initial Hessian approximation
        if len(self.y_history) > 0:
            y_k = self.y_history[-1]
            s_k = self.s_history[-1]
            gamma = torch.dot(y_k, s_k) / torch.dot(y_k, y_k)
            r = gamma * q
        else:
            r = q
        
        # Second loop (forward)
        alpha.reverse()
        for i in range(len(self.s_history)):
            s_i, y_i = self.s_history[i], self.y_history[i]
            rho_i = 1.0 / torch.dot(y_i, s_i)
            beta = rho_i * torch.dot(y_i, r)
            r = r + (alpha[i] - beta) * s_i
        
        # Update parameters
        self.prev_params = torch.cat([p.data.flatten() for p in self.params])
        self.prev_grad = grad.clone()
        
        # Apply update
        idx = 0
        for param in self.params:
            numel = param.numel()
            param.data -= self.lr * r[idx:idx+numel].view(param.shape)
            idx += numel

# Test Newton's method
print("Newton's method on quadratic function:")
def simple_quadratic(x):
    return 0.5 * torch.sum((x - torch.tensor([1.0, 2.0]))**2)

x0 = torch.tensor([0.0, 0.0])
x_opt = newton_method(simple_quadratic, x0)
print(f"Optimum found: {x_opt.data}")
print(f"Expected optimum: [1.0, 2.0]")`} />
          </Paper>
        </div>

        <div>
          <Title order={2} className="mb-8">Summary: Optimization Algorithms</Title>
          
          <Grid gutter="lg">
            <Grid.Col span={6}>
              <Paper className="p-6 bg-gradient-to-br from-blue-50 to-blue-100 h-full">
                <Title order={3} className="mb-4">First-order Methods</Title>
                <List spacing="md">
                  <List.Item>SGD with momentum improves convergence</List.Item>
                  <List.Item>Adaptive methods (Adam, RMSprop) handle different scales</List.Item>
                  <List.Item>Learning rate scheduling prevents stagnation</List.Item>
                  <List.Item>Each optimizer has specific use cases and trade-offs</List.Item>
                </List>
              </Paper>
            </Grid.Col>
            
            <Grid.Col span={6}>
              <Paper className="p-6 bg-gradient-to-br from-green-50 to-green-100 h-full">
                <Title order={3} className="mb-4">Advanced Techniques</Title>
                <List spacing="md">
                  <List.Item>Second-order methods use curvature information</List.Item>
                  <List.Item>L-BFGS approximates Newton's method efficiently</List.Item>
                  <List.Item>Warm-up and cosine schedules improve training</List.Item>
                  <List.Item>Custom optimizers enable research innovations</List.Item>
                </List>
              </Paper>
            </Grid.Col>
          </Grid>
        </div>

      </Stack>
    </Container>
  );
};

export default OptimizationAlgorithms;