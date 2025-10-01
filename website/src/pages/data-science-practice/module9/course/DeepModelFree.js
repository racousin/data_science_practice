import React from "react";
import { Container, Title, Text, List, Anchor } from '@mantine/core';
import { BlockMath, InlineMath } from "react-katex";
import "katex/dist/katex.min.css";
const DeepRL = () => {
  return (
    <Container fluid>
      <div data-slide>
        <Title order={2} mb="md">Deep Model Free</Title>

        <Title order={3} mb="md">Glossary</Title>
        <List>
          <List.Item>Reinforce/VPG</List.Item>
          <List.Item>Deep Q-learning</List.Item>
          <List.Item>Actor-Critic</List.Item>
        </List>
      </div>

      <div data-slide>
        <Title order={3} id="deep-q-learning" mb="md">Limitations of Traditional Q Learning</Title>
        <Text mb="md">Q Learning faces challenges when scaling to complex problems:</Text>
        <List>
          <List.Item>High-dimensional state spaces lead to slow convergence.</List.Item>
          <List.Item>Inapplicable to environments with continuous action spaces.</List.Item>
        </List>
      </div>
      <div data-slide>
        <Title order={3} mb="md">Deep Q Learning Overview</Title>
        <Text mb="md">Deep Q Learning extends Q Learning by using neural networks:</Text>
        <List>
          <List.Item>
            Parametrize <InlineMath math="Q" /> function with{" "}
            <InlineMath math="\theta" />,{" "}
            <InlineMath math="Q_\theta : S \times A \rightarrow \mathbb{R}" />
            .
          </List.Item>
          <List.Item>
            Objective: Find <InlineMath math="\theta^*" /> that approximates
            the optimal <InlineMath math="Q" /> function.
          </List.Item>
          <List.Item>
            Define Q target as:{" "}
            <InlineMath math="y = R_{t+1} + \gamma \max_{a'} Q_\theta(S_{t+1}, a')" />
            .
          </List.Item>
          <List.Item>
            Minimize loss (e.g., MSE):{" "}
            <InlineMath math="L(\theta) = \mathbb{E}_{s,a \sim Q} [(y - Q(s,a,\theta))^2]" />
            .
          </List.Item>
        </List>
      </div>
      <div data-slide>
        <Title order={3} mb="md">Executing the Deep Q Learning Algorithm</Title>
        <Text mb="md">Steps to implement Deep Q Learning:</Text>
        <List type="ordered">
          <List.Item>
            For current state <InlineMath math="S_t" />, compute{" "}
            <InlineMath math="Q_\theta(S_t, a)" /> for all actions.
          </List.Item>
          <List.Item>
            Take action <InlineMath math="A_t" /> with highest{" "}
            <InlineMath math="Q" /> value, observe reward and next state.
          </List.Item>
          <List.Item>
            Compute target <InlineMath math="y" /> for{" "}
            <InlineMath math="S_{t+1}" /> and minimize loss{" "}
            <InlineMath math="L(\theta)" />.
          </List.Item>
          <List.Item>
            Iterate to refine <InlineMath math="\theta" /> towards optimal.
          </List.Item>
        </List>
      </div>

      <div data-slide>
        <Title order={3} mb="md">Improving Deep Q Learning Stability</Title>
        <Text mb="md">Key techniques for enhancing DQL:</Text>
        <List>
          <List.Item>
            <strong>Experience Replay:</strong> Store transitions{" "}
            <InlineMath math="(S_t, A_t, R_{t+1}, S_{t+1})" /> and sample
            randomly to break correlation in sequences.
          </List.Item>
          <List.Item>
            <strong>Target Network:</strong> Use a separate, slowly updated
            network to stabilize targets.
          </List.Item>
          <List.Item>
            <strong>Additional Improvements:</strong> Epsilon decay for
            exploration, reward clipping, Double Q Learning to reduce
            overestimation.
          </List.Item>
        </List>
      </div>
      <div data-slide>
        <Title order={3} id="policy-gradient" mb="md">Policy Optimization</Title>
        <List mb="md">
          <List.Item>
            Parametrization of policy, <InlineMath math="\pi_{\theta}" />.
          </List.Item>
          <List.Item>
            We aim to maximize the expected return{" "}
            <InlineMath math="J(\pi_{\theta}) = E_{\tau \sim \pi_{\theta}}[{G(\tau)}]" />
            .
          </List.Item>
          <List.Item>
            Gradient ascent:{" "}
            <InlineMath math="\theta_{k+1} = \theta_k + \alpha \left. \nabla_{\theta} J(\pi_{\theta}) \right|_{\theta_k}" />
            .
          </List.Item>
          <List.Item>
            We can prove that:{" "}
            <BlockMath math="\nabla_{\theta} J(\pi_{\theta}) = E_{\tau \sim \pi_{\theta}}[{\sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t |s_t) G(\tau)}]" />
          </List.Item>
        </List>
        <Text mb="md">
          For more details, see:{" "}
          <Anchor
            href="https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html"
            target="_blank"
            rel="noopener noreferrer"
          >
            Policy Gradient Algorithms
          </Anchor>
        </Text>
      </div>
      <div data-slide>
        <Title order={3} mb="md">Reinforce/VPG algorithm</Title>
        <Text mb="md">
          Initialize policy <InlineMath math="\pi_{\theta}" />
        </Text>
        <List type="ordered">
          <List.Item>
            Generate episodes{" "}
            <InlineMath math="\mathcal{D} = \{\tau_i\}_{i=1,...,N}" /> with
            the policy <InlineMath math="\pi_\theta" />
          </List.Item>
          <List.Item>
            Compute gradient approximation{" "}
            <BlockMath math="\hat{\nabla} = \frac{1}{|\mathcal{D}|} \sum_{\tau \in \mathcal{D}} \sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t |s_t) G_t" />
          </List.Item>
          <List.Item>
            Update policy (apply gradient ascent){" "}
            <InlineMath math="\theta \leftarrow \theta + \alpha \hat{\nabla}" />
          </List.Item>
          <List.Item>Iterate</List.Item>
        </List>
      </div>
      <div data-slide>
        <Title order={3} id="actor-critic" mb="md">Introduction to Actor-Critic Models</Title>
        <Text mb="md">
          Actor-Critic models combine the benefits of policy-based and
          value-based approaches:
        </Text>
        <List>
          <List.Item>
            The <strong>Actor</strong> updates the policy distribution in the
            direction suggested by the <strong>Critic</strong>.
          </List.Item>
          <List.Item>
            The <strong>Critic</strong> estimates the value function (
            <InlineMath math="V" /> or <InlineMath math="Q" />) to critique
            the actions taken by the Actor.
          </List.Item>
          <List.Item>
            This interaction enhances learning by using the Critic's value
            function to reduce the variance in policy gradient estimates.
          </List.Item>
        </List>
      </div>

      <div data-slide>
        <Title order={3} mb="md">Policy Gradient in Actor-Critic</Title>
        <Text mb="md">The policy gradient in Actor-Critic models can be written as:</Text>
        <BlockMath math="\nabla_{\theta} J(\pi_{\theta}) = E_{\tau \sim \pi_{\theta}}\left[\sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t |s_t) \Phi_t\right]" />
        <Text mb="md">
          Where <InlineMath math="\Phi_t" /> represents:
        </Text>
        <List mb="md">
          <List.Item>
            Total return <InlineMath math="G_t" />.
          </List.Item>
          <List.Item>
            Advantage function: <InlineMath math="R_{t+1} - V(s_t)" /> or{" "}
            <InlineMath math="R_{t+1} - Q(s_t, a_t)" />.
          </List.Item>
        </List>
        <Text mb="md">
          Using <InlineMath math="\Phi_t" /> improves policy updates by
          evaluating actions more effectively.
        </Text>
      </div>
      <div data-slide>
        <Title order={3} mb="md">Actor-Critic Algorithm Steps</Title>
        <Text mb="md">Implementing the Actor-Critic algorithm involves:</Text>
        <List type="ordered">
          <List.Item>
            Initializing parameters for both the Actor (
            <InlineMath math="\theta" />) and the Critic (
            <InlineMath math="\phi" />
            ).
          </List.Item>
          <List.Item>
            For each episode:
            <List type="ordered" withPadding>
              <List.Item>
                Generate an action <InlineMath math="A_t" /> using the current
                policy <InlineMath math="\pi_{\theta_t}" />.
              </List.Item>
              <List.Item>
                Update the Actor by applying gradient ascent using the
                Critic's feedback.
              </List.Item>
              <List.Item>
                Update the Critic by minimizing the difference between
                estimated and actual returns.
              </List.Item>
            </List>
          </List.Item>
          <List.Item>Repeat the process to refine both Actor and Critic.</List.Item>
        </List>
      </div>
    </Container>
  );
};
export default DeepRL;
