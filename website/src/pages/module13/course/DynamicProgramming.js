import React from "react";
import { Container, Row, Col, Image } from "react-bootstrap";
import { BlockMath, InlineMath } from "react-katex";
import "katex/dist/katex.min.css";
import CodeBlock from "components/CodeBlock";
import { Paper, Alert, Text, Title, Code, Grid } from '@mantine/core';
import { AlertTriangle } from 'lucide-react';

const DynamicProgrammingSection = () => {
  const codeString = `import numpy as np

def policy_evaluation(env, policy, gamma=1.0, theta=1e-8):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        env: Gymnasium env. env.P represents the transition probabilities of the environment
        policy: [S, A] shaped matrix representing the policy
        gamma: discount factor
        theta: we stop evaluation once our value function change is less than theta for all states
    """
    V = np.zeros(env.observation_space.n)
    
    while True:
        delta = 0
        # For each state, perform a "full backup"
        for s in range(env.observation_space.n):
            v = 0
            # Look at the possible next actions
            for a, action_prob in enumerate(policy[s]):
                # For each action, look at the possible next states
                for prob, next_state, reward, done in env.P[s][a]:
                    # Calculate the expected value
                    v += action_prob * prob * (reward + gamma * V[next_state])
            # How much our value function changed (across all states)
            delta = max(delta, np.abs(V[s] - v))
            V[s] = v
        # Stop evaluating once our value function change is below a threshold
        if delta < theta:
            break
    return V

def q_from_v(env, V, s, gamma=1.0):
    """
    Calculate Q-values for all actions in a given state.
    
    Args:
        env: Gymnasium env
        V: value function array
        s: state to consider
        gamma: discount factor
    """
    q = np.zeros(env.action_space.n)
    for a in range(env.action_space.n):
        for prob, next_state, reward, done in env.P[s][a]:
            q[a] += prob * (reward + gamma * V[next_state])
    return q

def policy_improvement(env, V, gamma=1.0):
    """
    Given a value function, calculate a new policy using one-step look-ahead.
    
    Args:
        env: Gymnasium env
        V: value function array
        gamma: discount factor
    """
    policy = np.zeros([env.observation_space.n, env.action_space.n])
    
    for s in range(env.observation_space.n):
        # Find the best action by one-step look-ahead
        q = q_from_v(env, V, s, gamma)
        best_actions = np.argwhere(q == np.max(q)).flatten()
        # Assign equal probability to all best actions
        policy[s] = np.sum([np.eye(env.action_space.n)[i] for i in best_actions], axis=0) / len(best_actions)
    
    return policy

def policy_iteration(env, gamma=1.0, theta=1e-8):
    """
    Policy Iteration Algorithm.
    
    Args:
        env: Gymnasium env
        gamma: discount factor
        theta: threshold for value function convergence
    """
    # Start with a random policy
    policy = np.ones([env.observation_space.n, env.action_space.n]) / env.action_space.n
    
    while True:
        # Evaluate the current policy
        V = policy_evaluation(env, policy, gamma, theta)
        
        # Generate a new policy based on the updated value function
        new_policy = policy_improvement(env, V, gamma)
        
        # Check if we've converged to an optimal policy
        if np.all(policy == new_policy):
            break
            
        policy = new_policy
    
    return policy, V

def value_iteration(env, gamma=1.0, theta=1e-8):
    """
    Value Iteration Algorithm.
    
    Args:
        env: Gymnasium env
        gamma: discount factor
        theta: threshold for value function convergence
    """
    V = np.zeros(env.observation_space.n)
    
    while True:
        delta = 0
        # Update each state
        for s in range(env.observation_space.n):
            v = V[s]
            # Look at all possible actions for this state
            q_values = q_from_v(env, V, s, gamma)
            V[s] = np.max(q_values)
            # Track maximum change in value function
            delta = max(delta, np.abs(v - V[s]))
        
        # Stop when value function change is below threshold
        if delta < theta:
            break
    
    # Create a deterministic policy using the optimal value function
    policy = policy_improvement(env, V, gamma)
    
    return policy, V`;

  return (
    <div className="space-y-6">
      <Title order={2} id="dynamic-programming" className="text-2xl font-bold mb-4">
        Dynamic Programming Implementation
      </Title>
      
      <Alert icon={<AlertTriangle className="h-4 w-4" />} className="mb-4">
        <Text className="font-semibold">Implementation Notes:</Text>
        <Text>This implementation assumes a discrete state and action space using an environment structure with access to the transition model P.</Text>
      </Alert>

      <div className="space-y-4">
        <Paper className="p-4">
          <Title order={3} className="text-xl font-semibold mb-2">Key Components:</Title>
          <div className="space-y-2">
            <Text>1. <span className="font-semibold">Policy Evaluation:</span> Iteratively computes the state-value function for a given policy.</Text>
            <Text>2. <span className="font-semibold">Policy Improvement:</span> Generates a better policy using one-step look-ahead with current value estimates.</Text>
            <Text>3. <span className="font-semibold">Policy Iteration:</span> Alternates between evaluation and improvement until convergence.</Text>
            <Text>4. <span className="font-semibold">Value Iteration:</span> Combines policy evaluation and improvement into a single update step.</Text>
          </div>
        </Paper>

        <Paper className="p-4 bg-gray-50">
          <CodeBlock code={codeString} language="python"/>
        </Paper>

        <Paper className="p-4">
          <Title order={3} className="text-xl font-semibold mb-2">Usage Example:</Title>
          <CodeBlock code={`# Example usage with a simple environment
import gymnasium as gym
env = gym.make('FrozenLake-v1')

# Run policy iteration
optimal_policy, value_func = policy_iteration(env, gamma=0.99)

# Or run value iteration
optimal_policy, value_func = value_iteration(env, gamma=0.99)`}
language="python"/>
        </Paper>
      </div>
    </div>
  );
};

const DynamicProgramming = () => {
  return (
    <Container fluid>
      <h2>Dynamic Programming in MDPs</h2>
      <Grid.Col>
      <Grid className="mt-4">
          <h3 id="bellman-equations">Bellman Equations Development</h3>
          <BlockMath
            math={`
      \\begin{aligned}
      V_{\\pi}(s) &= \\sum_{a \\in \\mathcal{A}} \\pi(a \\vert s) Q_{\\pi}(s, a) \\\\
      Q_{\\pi}(s, a) &= R(s, a) + \\gamma \\sum_{s' \\in \\mathcal{S}} P_{ss'}^a V_{\\pi} (s') \\\\
      V_{\\pi}(s) &= \\sum_{a \\in \\mathcal{A}} \\pi(a \\vert s) \\big( R(s, a) + \\gamma \\sum_{s' \\in \\mathcal{S}} P_{ss'}^a V_{\\pi} (s') \\big) \\\\
      Q_{\\pi}(s, a) &= R(s, a) + \\gamma \\sum_{s' \\in \\mathcal{S}} P_{ss'}^a \\sum_{a' \\in \\mathcal{A}} \\pi(a' \\vert s') Q_{\\pi} (s', a')
      \\end{aligned}
    `}
          />
        </Grid>
      </Grid.Col>
      <Grid className="mt-4">
  <Grid.Col>
    <h3 id="optimal-policy" >Equivalence Between Optimal Policy, Maximum Value, and Q-Functions</h3>
    <p>
      The best policy: {" "} 
      <InlineMath math="\pi^* = \arg \max_{\pi} E_{\tau \sim \pi}[G(\tau)]" />,

    </p>
    <p>
    verify
    </p>
    <BlockMath
      math={`
      V_{\\pi^*}(s) \\geq V_{\\pi}(s), \\quad \\forall \\pi, \\forall s \\in \\mathcal{S}.
      `}
    />
    <p>
      as well as:
    </p>
    <BlockMath
      math={`
      Q_{\\pi^*}(s, a) \\geq Q_{\\pi}(s, a), \\quad \\forall \\pi, \\forall s \\in \\mathcal{S}, \\forall a \\in \\mathcal{A}.
      `}
    />
    <p>
    Reciprocally, the optimal policy can be derived by choosing the action that maximizes the optimal action-value function <InlineMath math="Q^*(s, a)" /> for each state \( s \):
    </p>
    <BlockMath
      math={`
      \\pi^*(s) = \\arg\\max_{a \\in \\mathcal{A}} Q^*(s, a).
      `}
    />
    <p>
      Thus, solving for <InlineMath math="V^*(s)" /> or <InlineMath math="Q^*(s, a)" /> fully characterizes the optimal policy <InlineMath math={`\\pi^*`}/>, which achieves the maximum expected reward.
    </p>
  </Grid.Col>
</Grid>





      <Grid className="mt-4">
        <Grid.Col>
          <h3 id="mdp-solution" >The MDP Solution</h3>
          <p>
            Dynamic Programming allows to resolve the MDP optimization problem{" "}
            <InlineMath math="\pi^* = \arg \max_{\pi} E_{\tau\sim \pi}[{G(\tau)}]" />
            . It is an iterative process:
          </p>
          <ul>
            <li>Policy initialization</li>
            <li>Policy evaluation</li>
            <li>Policy improvement</li>
          </ul>
        </Grid.Col>
      </Grid>

      <Grid className="mt-4">
        <Grid.Col>
          <h3 id="policy-evaluation">Policy Evaluation</h3>
          <p>
            Policy Evaluation: compute the state-value{" "}
            <InlineMath math="V_\pi" /> for a given policy{" "}
            <InlineMath math="\pi" />. We initialize <InlineMath math="V_0" />{" "}
            arbitrarily. And we update it using:
          </p>
          <BlockMath
            math={`
            \\begin{aligned}
            V_{k+1}(s) &= \\mathbb{E}_\\pi [r + \\gamma V_k(s_{t+1}) | S_t = s]\\\\
            &= \\sum_a \\pi(a | s) \\sum_{s'} P(s' | s, a) (R(s,a) + \\gamma V_k(s'))
            \\end{aligned}
          `}
          />
          <p>
            <InlineMath math="V_\pi(s)" /> is a fixed point for this equation,
            so if <InlineMath math="(V_k)_{k\in \mathbb{N}}" /> converges, it
            converges to <InlineMath math="V_\pi" />.
          </p>
        </Grid.Col>
      </Grid>

      <Grid className="mt-4">
        <Grid.Col>
          <h3 id="policy-improvement" >Policy Improvement</h3>
          <p>
            Policy Improvement generates a better policy{" "}
            <InlineMath math="\pi' \geq \pi" /> by acting greedily. Compute{" "}
            <InlineMath math="Q" /> from <InlineMath math="V" /> (
            <InlineMath math="\forall a,s" />
            ):
          </p>
          <BlockMath
            math={`
            \\begin{aligned}
            Q_\\pi(s, a) &= \\mathbb{E} [R_{t+1} + \\gamma V_\\pi(S_{t+1}) | S_t=s, A_t=a]\\\\
            &= \\sum_{s'} P(s' | s, a) (R(s,a) + \\gamma V_\\pi(s'))
            \\end{aligned}
          `}
          />
          <p>
            Update greedily:{" "}
            <InlineMath math="\pi'(s) = \arg\max_{a \in \mathcal{A}} Q_\pi(s, a)" />{" "}
            (<InlineMath math="\forall s" />)
          </p>
        </Grid.Col>
      </Grid>

      <Grid className="mt-4">
        <Grid.Col>
          <h3>Policy Improvement Visualization</h3>
          <p>
            <InlineMath math="\pi' (s) = \arg\max_{a \in A} Q_{\pi}(s, a)" />
          </p>
          <Grid>
            <Grid.Col span={{ md: 4 }}>
              <Image
                src="/assets/module13/tikz_picture_16.png"
                alt="Initial Policy"
                fluid
              />
              <p className="text-center">Initial Policy (π)</p>
            </Grid.Col>
            <Grid.Col span={{ md: 4 }}>
              <Image
                src="/assets/module13/tikz_picture_17.png"
                alt="Q-values"
                fluid
              />
              <p className="text-center">
                Q-values (Q<sub>π</sub>)
              </p>
            </Grid.Col>
            <Grid.Col span={{ md: 4 }}>
              <Image
                src="/assets/module13/tikz_picture_18.png"
                alt="Improved Policy"
                fluid
              />
              <p className="text-center">Improved Policy (π')</p>
            </Grid.Col>
          </Grid>
          <p className="text-center mt-3">
            Policy Improvement Process: Initial Policy → Q-values → Improved
            Policy
          </p>
        </Grid.Col>
      </Grid>

      <Grid className="mt-4">
        <Grid.Col>
          <h3 id="implementation">Dynamic Programming</h3>
          <p>
            Policy Iteration: iterative procedure to improve the policy when
            combining policy evaluation and improvement.
          </p>
          <BlockMath
            math={`
            \\pi_0 \\xrightarrow[]{\\text{evaluation}} V_{\\pi_0} \\xrightarrow[]{\\text{improve}}
            \\pi_1 \\xrightarrow[]{\\text{evaluation}}\\dots \\xrightarrow[]{\\text{improve}}
            \\pi_* \\xrightarrow[]{\\text{evaluation}} V_*
          `}
          />
        </Grid.Col>
      </Grid>


      <Grid className="mt-4">
        <Grid.Col>
          <h3>Take Home Message</h3>
          <p>
            Initialize <InlineMath math="\pi(s), \forall s" />
          </p>
          <ol>
            <li>
              Evaluate <InlineMath math="V_\pi (s), \forall s" /> (using{" "}
              <InlineMath math="\mathbb{P}^a_{ss'}" />)
            </li>
            <li>
              Compute <InlineMath math="Q_\pi(s,a), \forall s,a" /> (using{" "}
              <InlineMath math="\mathbb{P}^a_{ss'}" />)
            </li>
            <li>
              Update{" "}
              <InlineMath math="\pi'(s) = \max_a Q_\pi (s,a), \forall s" />
            </li>
            <li>
              While <InlineMath math="\pi'(s) \neq \pi(s)" /> do{" "}
              <InlineMath math="\pi(s) = \pi'(s)" /> and iterate
            </li>
          </ol>
          <p>
            Result: <InlineMath math="\pi = \arg \max_{\pi} E[G]" />
          </p>
        </Grid.Col>
      </Grid>

      <DynamicProgrammingSection/>
    </Container>
  );
};

export default DynamicProgramming;
