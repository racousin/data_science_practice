import React from "react";
import { BlockMath, InlineMath } from "react-katex";
import "katex/dist/katex.min.css";
import CodeBlock from "components/CodeBlock";
import { Text, Title, Grid, Container, Image, List, Center } from '@mantine/core';
import { IconAlertCircle } from '@tabler/icons-react';

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
    <div data-slide>
      <Title order={3} id="dynamic-programming" mb="md">
        Dynamic Programming Implementation
      </Title>

      <Title order={4} mb="md">Key Components:</Title>
      <List mb="md">
        <List.Item><strong>Policy Evaluation:</strong> Iteratively computes the state-value function for a given policy.</List.Item>
        <List.Item><strong>Policy Improvement:</strong> Generates a better policy using one-step look-ahead with current value estimates.</List.Item>
        <List.Item><strong>Policy Iteration:</strong> Alternates between evaluation and improvement until convergence.</List.Item>
        <List.Item><strong>Value Iteration:</strong> Combines policy evaluation and improvement into a single update step.</List.Item>
      </List>

      <CodeBlock code={codeString} language="python"/>

      <Title order={4} mt="xl" mb="md">Usage Example:</Title>
      <CodeBlock code={`# Example usage with a simple environment
import gymnasium as gym
env = gym.make('FrozenLake-v1')

# Run policy iteration
optimal_policy, value_func = policy_iteration(env, gamma=0.99)

# Or run value iteration
optimal_policy, value_func = value_iteration(env, gamma=0.99)`}
language="python"/>
    </div>
  );
};

const DynamicProgramming = () => {
  return (
    <Container fluid>
      <div data-slide>
        <Title order={2} mb="md">Dynamic Programming in MDPs</Title>

        <Title order={3} id="bellman-equations" mb="md">Bellman Equations Development</Title>
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
      </div>
      <div data-slide>
        <Title order={3} id="optimal-policy" mb="md">
          Equivalence Between Optimal Policy, Maximum Value, and Q-Functions
        </Title>
        <Text mb="md">
          The best policy:{" "}
          <InlineMath math="\pi^* = \arg \max_{\pi} E_{\tau \sim \pi}[G(\tau)]" />,
        </Text>
        <Text mb="md">verify</Text>
        <BlockMath
          math={`
      V_{\\pi^*}(s) \\geq V_{\\pi}(s), \\quad \\forall \\pi, \\forall s \\in \\mathcal{S}.
      `}
        />
        <Text mb="md">as well as:</Text>
        <BlockMath
          math={`
      Q_{\\pi^*}(s, a) \\geq Q_{\\pi}(s, a), \\quad \\forall \\pi, \\forall s \\in \\mathcal{S}, \\forall a \\in \\mathcal{A}.
      `}
        />
        <Text mb="md">
          Reciprocally, the optimal policy can be derived by choosing the action that maximizes the optimal action-value function <InlineMath math="Q^*(s, a)" /> for each state <InlineMath math="s" />:
        </Text>
        <BlockMath
          math={`
      \\pi^*(s) = \\arg\\max_{a \\in \\mathcal{A}} Q^*(s, a).
      `}
        />
        <Text mb="md">
          Thus, solving for <InlineMath math="V^*(s)" /> or <InlineMath math="Q^*(s, a)" /> fully characterizes the optimal policy <InlineMath math={`\\pi^*`}/>, which achieves the maximum expected reward.
        </Text>
      </div>

      <div data-slide>
        <Title order={3} id="mdp-solution" mb="md">The MDP Solution</Title>
        <Text mb="md">
          Dynamic Programming allows to resolve the MDP optimization problem{" "}
          <InlineMath math="\pi^* = \arg \max_{\pi} E_{\tau\sim \pi}[{G(\tau)}]" />
          . It is an iterative process:
        </Text>
        <List>
          <List.Item>Policy initialization</List.Item>
          <List.Item>Policy evaluation</List.Item>
          <List.Item>Policy improvement</List.Item>
        </List>
      </div>

      <div data-slide>
        <Title order={3} id="policy-evaluation" mb="md">Policy Evaluation</Title>
        <Text mb="md">
          Policy Evaluation: compute the state-value{" "}
          <InlineMath math="V_\pi" /> for a given policy{" "}
          <InlineMath math="\pi" />. We initialize <InlineMath math="V_0" />{" "}
          arbitrarily. And we update it using:
        </Text>
        <BlockMath
          math={`
            \\begin{aligned}
            V_{k+1}(s) &= \\mathbb{E}_\\pi [r + \\gamma V_k(s_{t+1}) | S_t = s]\\\\
            &= \\sum_a \\pi(a | s) \\sum_{s'} P(s' | s, a) (R(s,a) + \\gamma V_k(s'))
            \\end{aligned}
          `}
        />
        <Text mb="md">
          <InlineMath math="V_\pi(s)" /> is a fixed point for this equation,
          so if <InlineMath math="(V_k)_{k\in \mathbb{N}}" /> converges, it
          converges to <InlineMath math="V_\pi" />.
        </Text>
      </div>

      <div data-slide>
        <Title order={3} id="policy-improvement" mb="md">Policy Improvement</Title>
        <Text mb="md">
          Policy Improvement generates a better policy{" "}
          <InlineMath math="\pi' \geq \pi" /> by acting greedily. Compute{" "}
          <InlineMath math="Q" /> from <InlineMath math="V" /> (
          <InlineMath math="\forall a,s" />
          ):
        </Text>
        <BlockMath
          math={`
            \\begin{aligned}
            Q_\\pi(s, a) &= \\mathbb{E} [R_{t+1} + \\gamma V_\\pi(S_{t+1}) | S_t=s, A_t=a]\\\\
            &= \\sum_{s'} P(s' | s, a) (R(s,a) + \\gamma V_\\pi(s'))
            \\end{aligned}
          `}
        />
        <Text mb="md">
          Update greedily:{" "}
          <InlineMath math="\pi'(s) = \arg\max_{a \in \mathcal{A}} Q_\pi(s, a)" />{" "}
          (<InlineMath math="\forall s" />)
        </Text>
      </div>

      <div data-slide>
        <Title order={3} mb="md">Policy Improvement Visualization</Title>
        <Text mb="md">
          <InlineMath math="\pi' (s) = \arg\max_{a \in A} Q_{\pi}(s, a)" />
        </Text>
        <Grid>
          <Grid.Col span={{ md: 4 }}>
            <Image
              src="/assets/data-science-practice/module9/tikz_picture_16.png"
              alt="Initial Policy"
            />
            <Center>
              <Text size="sm">Initial Policy (π)</Text>
            </Center>
          </Grid.Col>
          <Grid.Col span={{ md: 4 }}>
            <Image
              src="/assets/data-science-practice/module9/tikz_picture_17.png"
              alt="Q-values"
            />
            <Center>
              <Text size="sm">Q-values (Q<sub>π</sub>)</Text>
            </Center>
          </Grid.Col>
          <Grid.Col span={{ md: 4 }}>
            <Image
              src="/assets/data-science-practice/module9/tikz_picture_18.png"
              alt="Improved Policy"
            />
            <Center>
              <Text size="sm">Improved Policy (π')</Text>
            </Center>
          </Grid.Col>
        </Grid>
        <Center mt="md">
          <Text size="sm">
            Policy Improvement Process: Initial Policy → Q-values → Improved
            Policy
          </Text>
        </Center>
      </div>

      <div data-slide>
        <Title order={3} id="implementation" mb="md">Dynamic Programming</Title>
        <Text mb="md">
          Policy Iteration: iterative procedure to improve the policy when
          combining policy evaluation and improvement.
        </Text>
        <BlockMath
          math={`
            \\pi_0 \\xrightarrow[]{\\text{evaluation}} V_{\\pi_0} \\xrightarrow[]{\\text{improve}}
            \\pi_1 \\xrightarrow[]{\\text{evaluation}}\\dots \\xrightarrow[]{\\text{improve}}
            \\pi_* \\xrightarrow[]{\\text{evaluation}} V_*
          `}
        />
      </div>

      <div data-slide>
        <Title order={3} mb="md">Take Home Message</Title>
        <Text mb="md">
          Initialize <InlineMath math="\pi(s), \forall s" />
        </Text>
        <List type="ordered">
          <List.Item>
            Evaluate <InlineMath math="V_\pi (s), \forall s" /> (using{" "}
            <InlineMath math="\mathbb{P}^a_{ss'}" />)
          </List.Item>
          <List.Item>
            Compute <InlineMath math="Q_\pi(s,a), \forall s,a" /> (using{" "}
            <InlineMath math="\mathbb{P}^a_{ss'}" />)
          </List.Item>
          <List.Item>
            Update{" "}
            <InlineMath math="\pi'(s) = \max_a Q_\pi (s,a), \forall s" />
          </List.Item>
          <List.Item>
            While <InlineMath math="\pi'(s) \neq \pi(s)" /> do{" "}
            <InlineMath math="\pi(s) = \pi'(s)" /> and iterate
          </List.Item>
        </List>
        <Text mt="md">
          Result: <InlineMath math="\pi = \arg \max_{\pi} E[G]" />
        </Text>
      </div>

      <DynamicProgrammingSection/>
    </Container>
  );
};

export default DynamicProgramming;
