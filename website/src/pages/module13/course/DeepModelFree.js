import React from "react";
import { Row, Col } from 'react-bootstrap';
import { Container, Grid } from '@mantine/core';
import { BlockMath, InlineMath } from "react-katex";
import "katex/dist/katex.min.css";
import CodeBlock from "components/CodeBlock";
const DeepRL = () => {
  return (
    <Container fluid>
      <h2>Deep Model Free</h2>
      <Grid className="mt-4">
        <Grid.Col>
          <h3>Glossary</h3>
          <ul>
            <li>Reinforce/VPG</li>
            <li>Deep Q-learning</li>
            <li>Actor-Critic</li>
          </ul>
        </Grid.Col>
      </Grid>
      <Grid className="mt-4">
        <Grid.Col>
          <h3 id="deep-q-learning">Limitations of Traditional Q Learning</h3>
          <p>Q Learning faces challenges when scaling to complex problems:</p>
          <ul>
            <li>High-dimensional state spaces lead to slow convergence.</li>
            <li>Inapplicable to environments with continuous action spaces.</li>
          </ul>
        </Grid.Col>
      </Grid>
      <Grid className="mt-4">
        <Grid.Col>
          <h3>Deep Q Learning Overview</h3>
          <p>Deep Q Learning extends Q Learning by using neural networks:</p>
          <ul>
            <li>
              Parametrize <InlineMath math="Q" /> function with{" "}
              <InlineMath math="\theta" />,{" "}
              <InlineMath math="Q_\theta : S \times A \rightarrow \mathbb{R}" />
              .
            </li>
            <li>
              Objective: Find <InlineMath math="\theta^*" /> that approximates
              the optimal <InlineMath math="Q" /> function.
            </li>
            <li>
              Define Q target as:{" "}
              <InlineMath math="y = R_{t+1} + \gamma \max_{a'} Q_\theta(S_{t+1}, a')" />
              .
            </li>
            <li>
              Minimize loss (e.g., MSE):{" "}
              <InlineMath math="L(\theta) = \mathbb{E}_{s,a \sim Q} [(y - Q(s,a,\theta))^2]" />
              .
            </li>
          </ul>
        </Grid.Col>
      </Grid>
      <Grid className="mt-4">
        <Grid.Col>
          <h3>Executing the Deep Q Learning Algorithm</h3>
          <p>Steps to implement Deep Q Learning:</p>
          <ol>
            <li>
              For current state <InlineMath math="S_t" />, compute{" "}
              <InlineMath math="Q_\theta(S_t, a)" /> for all actions.
            </li>
            <li>
              Take action <InlineMath math="A_t" /> with highest{" "}
              <InlineMath math="Q" /> value, observe reward and next state.
            </li>
            <li>
              Compute target <InlineMath math="y" /> for{" "}
              <InlineMath math="S_{t+1}" /> and minimize loss{" "}
              <InlineMath math="L(\theta)" />.
            </li>
            <li>
              Iterate to refine <InlineMath math="\theta" /> towards optimal.
            </li>
          </ol>
        </Grid.Col>
      </Grid>
      <Grid className="mt-4">
        <Grid.Col>
          <h3>Improving Deep Q Learning Stability</h3>
          <p>Key techniques for enhancing DQL:</p>
          <ul>
            <li>
              <strong>Experience Replay:</strong> Store transitions{" "}
              <InlineMath math="(S_t, A_t, R_{t+1}, S_{t+1})" /> and sample
              randomly to break correlation in sequences.
            </li>
            <li>
              <strong>Target Network:</strong> Use a separate, slowly updated
              network to stabilize targets.
            </li>
            <li>
              <strong>Additional Improvements:</strong> Epsilon decay for
              exploration, reward clipping, Double Q Learning to reduce
              overestimation.
            </li>
          </ul>
        </Grid.Col>
      </Grid>
      <Grid className="mt-4">
        <Grid.Col>
          <h3 id="policy-gradient">Policy Optimization</h3>
          <ul>
            <li>
              Parametrization of policy, <InlineMath math="\pi_{\theta}" />.
            </li>
            <li>
              We aim to maximize the expected return{" "}
              <InlineMath math="J(\pi_{\theta}) = E_{\tau \sim \pi_{\theta}}[{G(\tau)}]" />
              .
            </li>
            <li>
              Gradient ascent:{" "}
              <InlineMath math="\theta_{k+1} = \theta_k + \alpha \left. \nabla_{\theta} J(\pi_{\theta}) \right|_{\theta_k}" />
              .
            </li>
            <li>
              We can prove that:{" "}
              <BlockMath math="\nabla_{\theta} J(\pi_{\theta}) = E_{\tau \sim \pi_{\theta}}[{\sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t |s_t) G(\tau)}]" />
            </li>
          </ul>
          <p>
            For more details, see:{" "}
            <a
              href="https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html"
              target="_blank"
              rel="noopener noreferrer"
            >
              Policy Gradient Algorithms
            </a>
          </p>
        </Grid.Col>
      </Grid>
      <Grid className="mt-4">
        <Grid.Col>
          <h3>Reinforce/VPG algorithm</h3>
          <p>
            Initialize policy <InlineMath math="\pi_{\theta}" />
          </p>
          <ol>
            <li>
              Generate episodes{" "}
              <InlineMath math="\mathcal{D} = \{\tau_i\}_{i=1,...,N}" /> with
              the policy <InlineMath math="\pi_\theta" />
            </li>
            <li>
              Compute gradient approximation{" "}
              <BlockMath math="\hat{\nabla} = \frac{1}{|\mathcal{D}|} \sum_{\tau \in \mathcal{D}} \sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t |s_t) G_t" />
            </li>
            <li>
              Update policy (apply gradient ascent){" "}
              <InlineMath math="\theta \leftarrow \theta + \alpha \hat{\nabla}" />
            </li>
            <li>Iterate</li>
          </ol>
        </Grid.Col>
      </Grid>
      <Grid className="mt-4">
        <Grid.Col>
          <h3 id="actor-critic">Introduction to Actor-Critic Models</h3>
          <p>
            Actor-Critic models combine the benefits of policy-based and
            value-based approaches:
          </p>
          <ul>
            <li>
              The <strong>Actor</strong> updates the policy distribution in the
              direction suggested by the <strong>Critic</strong>.
            </li>
            <li>
              The <strong>Critic</strong> estimates the value function (
              <InlineMath math="V" /> or <InlineMath math="Q" />) to critique
              the actions taken by the Actor.
            </li>
            <li>
              This interaction enhances learning by using the Critic's value
              function to reduce the variance in policy gradient estimates.
            </li>
          </ul>
        </Grid.Col>
      </Grid>
      <Grid className="mt-4">
        <Grid.Col>
          <h3>Policy Gradient in Actor-Critic</h3>
          <p>The policy gradient in Actor-Critic models can be written as:</p>
          <BlockMath math="\nabla_{\theta} J(\pi_{\theta}) = E_{\tau \sim \pi_{\theta}}\left[\sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t |s_t) \Phi_t\right]" />
          <p>
            Where <InlineMath math="\Phi_t" /> represents:
          </p>
          <ul>
            <li>
              Total return <InlineMath math="G_t" />.
            </li>
            <li>
              Advantage function: <InlineMath math="R_{t+1} - V(s_t)" /> or{" "}
              <InlineMath math="R_{t+1} - Q(s_t, a_t)" />.
            </li>
          </ul>
          <p>
            Using <InlineMath math="\Phi_t" /> improves policy updates by
            evaluating actions more effectively.
          </p>
        </Grid.Col>
      </Grid>
      <Grid className="mt-4">
        <Grid.Col>
          <h3>Actor-Critic Algorithm Steps</h3>
          <p>Implementing the Actor-Critic algorithm involves:</p>
          <ol>
            <li>
              Initializing parameters for both the Actor (
              <InlineMath math="\theta" />) and the Critic (
              <InlineMath math="\phi" />
              ).
            </li>
            <li>
              For each episode:
              <ol type="a">
                <li>
                  Generate an action <InlineMath math="A_t" /> using the current
                  policy <InlineMath math="\pi_{\theta_t}" />.
                </li>
                <li>
                  Update the Actor by applying gradient ascent using the
                  Critic's feedback.
                </li>
                <li>
                  Update the Critic by minimizing the difference between
                  estimated and actual returns.
                </li>
              </ol>
            </li>
            <li>Repeat the process to refine both Actor and Critic.</li>
          </ol>
        </Grid.Col>
      </Grid>
    </Container>
  );
};
export default DeepRL;
