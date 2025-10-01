import React from "react";
import { Title, Text, Stack, Container, Accordion, Grid, Image, List, Center } from '@mantine/core';
import { BlockMath, InlineMath } from "react-katex";
import "katex/dist/katex.min.css";
import CodeBlock from "components/CodeBlock";
import { Link } from 'react-router-dom';
import QLearningConvergence from './QLearningConvergence';

const QLearning = () => {
  return (
    <>
      <Title order={3} mb="md">Q-learning Estimation</Title>

      <Text mb="md">Remember the optimal Q-function:</Text>
      <BlockMath
        math="Q_{\pi^*}(S_t,A_t) = \mathbb{E} [R_{t+1} + \gamma \max_{a'} Q_*(S_{t+1}, a') \vert S_t = s, A_t = a]"
      />

      <Text mb="md">
        So <InlineMath math="R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a')" /> is an unbiased estimate for the optimal Q function policy <InlineMath math="Q_{\pi^*}(S_t, A_t)" />
      </Text>

      <Text mb="md">
        <InlineMath math="R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a')" /> is called the Q target.
      </Text>

      <Text mb="md">
        <InlineMath math="\alpha" /> improvement:
      </Text>
      <BlockMath
        math="Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha (R_{t+1} + \gamma \max_{a \in \mathcal{A}} Q(S_{t+1}, a) - Q(S_t, A_t))"
      />

      <Text mb="md">
        This is called an off-policy update, because we update the Q function not from real environment feedback, but from theoretical policy information
        (i.e., instead of using <InlineMath math="Q(S_{t+1}, A_{t+1})" /> like in SARSA, we use <InlineMath math="\max_{a} Q(S_{t+1}, a)" />).
      </Text>
    </>
  );
};

const SarsaAlgorithm = () => {
  return (
    <>
      <Title order={3} id="sarsa" mb="md">SARSA Algorithm</Title>

      <Text mb="md">
        Initialize <InlineMath math="Q(s,a)" /> for all states and actions <InlineMath math="s,a" />
      </Text>

      <Text mb="md">
        <InlineMath math="S_t = \text{initial state}" />, act with <InlineMath math="\pi" />
        (extract from <InlineMath math="Q" /> <InlineMath math="\epsilon" />-greedy)
        to get <InlineMath math="A_t,R_{t+1},S_{t+1}" />
      </Text>

      <List type="ordered" mb="md">
        <List.Item>
          Act with <InlineMath math="\pi" /> (extract from <InlineMath math="Q" /> {" "}
          <InlineMath math="\epsilon" />-greedy) to get <InlineMath math="A_{t+1}, R_{t+2},S_{t+2}" />
        </List.Item>

        <List.Item>
          Update Q using the observation step:{" "}
          <BlockMath math="Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha (R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t))" />
        </List.Item>

        <List.Item>
          Iterate
        </List.Item>
      </List>
    </>
  );
};

const RLObjectiveSection = () => {
  return (
    <>
      <Title order={3} mb="md">Reinforcement Learning Objective</Title>
      <Text mb="md">
        RL aims to optimize decision-making in environments without a known
        transition model <InlineMath math="P" />.
      </Text>
      <BlockMath
        math={`
          \\begin{align*}
          \\pi^* &= \\arg\\max_{\\pi} J(\\pi)\\\\
          J(\\pi) &= \\mathbb{E}_{\\tau\\sim\\pi}[G(\\tau)] = \\int_{\\tau} \\mathbb{P}(\\tau|\\pi) G(\\tau)
          \\end{align*}
        `}
      />

      <Text mb="md">
        As shown in <Link to="/module9/course/dynamic-programming">
        dynamic programming</Link>, we can start with a random policy and evaluate its V/Q function
        to iteratively build better policies until we reach the optimal policy.
      </Text>

      <Text mb="md">
        However, in reinforcement learning, we don't have access to the transition model,
        so we cannot compute V and Q exactly.
      </Text>

      <Text mb="md">
        Instead, we can estimate these values, which is what we'll explore next.
      </Text>
    </>
  );
};

const ModelFreeMethods = () => {
  const agentCode = `
import numpy as np
import random

class Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = 0.1  # Exploration rate
        self.Q = np.zeros((state_size, action_size))  # Q-table initialization
        
    def choose_action(self, state):
        """Choose action based on epsilon-greedy strategy"""
        if random.random() > self.epsilon:  # With probability 1-ε, choose best action
            return np.argmax(self.Q[state, :])
        else:  # With probability ε, choose random action
            return random.randint(0, self.action_size - 1)`;

  return (
    <Container fluid>
      <div data-slide>
        <Title order={2} mb="md">Model-Free Reinforcement Learning Methods</Title>

        <RLObjectiveSection />
      </div>

      <div data-slide>
        <Title order={3} mb="md">Monte-Carlo Method</Title>
        <Text mb="md">
          To evaluate{" "}
          <InlineMath math="V_\pi(s) = E_{\tau \sim \pi}[{G_t\left| s_t = s\right.}]" />
          :
        </Text>
        <List type="ordered" mb="md">
          <List.Item>
            Generate an episode (<InlineMath math={"s_1,a_1,r_2,…,s_T"}/>)  with the policy <InlineMath math="\pi" />
          </List.Item>
          <List.Item>
            Compute{" "}
            <InlineMath math="G_t = \sum_{k=0}^{T-t-1} \gamma^k R_{t+k+1}" />
          </List.Item>
          <List.Item>
            Evaluate the value function:
            <BlockMath math="V_\pi(s) = \frac{\sum_{t=1}^T \mathbb{1}[S_t = s] G_t}{\sum_{t=1}^T \mathbb{1}[S_t = s]}" />
          </List.Item>
        </List>
        <Text mb="md">Similarly, for the action-value function:</Text>
        <BlockMath math="Q_\pi(s, a) = \frac{\sum_{t=1}^T \mathbb{1}[S_t = s, A_t = a] G_t}{\sum_{t=1}^T \mathbb{1}[S_t = s, A_t = a]}" />
      </div>

      <div data-slide>
        <Title order={3} id="monte-carlo-algorithm" mb="md">Monte-Carlo Algorithm</Title>
        <Text mb="md">Initialise <InlineMath math="Q(s,a) \forall s, a" /></Text>
        <List type="ordered" mb="md">
          <List.Item>
            Generate an episode with the policy <InlineMath math="\pi" />{" "}
            (extract from <InlineMath math="Q" />{" "}
            <InlineMath math="\epsilon" />
            -greedy, i.e: <InlineMath math="a = \arg\max Q(s, a)"/> or random action)
          </List.Item>
          <List.Item>
            Evaluate Q using the episode:
            <BlockMath math="Q_\pi(s, a) = \frac{\sum_{t=1}^T \big( \mathbb{1}[S_t = s, A_t = a] \sum_{k=0}^{T-t-1} \gamma^k R_{t+k+1} \big)}{\sum_{t=1}^T \mathbb{1}[S_t = s, A_t = a]}" />
          </List.Item>
          <List.Item>Iterate</List.Item>
        </List>

        <Title order={4} mt="xl" mb="md">Visual Steps in Monte Carlo</Title>
        <Grid>
          <Grid.Col span={{ md: 4 }}>
            <Image
              src="/assets/data-science-practice/module9/tikz_images_2/tikz_picture_1.png"
              alt="Generate Episode"
            />
            <Center>
              <Text size="sm">
                1. Generate episode following{" "}
                <InlineMath math="\arg\max Q(s, a)" />
              </Text>
            </Center>
          </Grid.Col>
          <Grid.Col span={{ md: 4 }}>
            <Image
              src="/assets/data-science-practice/module9/tikz_images_2/tikz_picture_2.png"
              alt="Evaluate Q"
            />
            <Center>
              <Text size="sm">2. Evaluate Q</Text>
            </Center>
          </Grid.Col>
          <Grid.Col span={{ md: 4 }}>
            <Image
              src="/assets/data-science-practice/module9/tikz_images_2/tikz_picture_3.png"
              alt="Iterate"
            />
            <Center>
              <Text size="sm">3. Iterate</Text>
            </Center>
          </Grid.Col>
        </Grid>
      </div>
      <div data-slide>
        <Title order={3} id="td-learning" mb="md">Temporal Difference (TD) Learning</Title>
        <Text mb="md">
          TD Learning combines Monte Carlo and dynamic programming ideas,
          using bootstrapping for value updates.
        </Text>
        <Title order={4} mb="md">Bellman equations:</Title>
        <BlockMath math="V(S_t) = \mathbb{E}[R_{t+1} + \gamma V(S_{t+1}) | S_t = s]" />
        <BlockMath math="Q(s, a) = \mathbb{E} [R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) \mid S_t = s, A_t = a]" />
        <Title order={4} mt="md" mb="md">TD Target (unbiased estimate):</Title>
        <List mb="md">
          <List.Item>
            For <InlineMath math="V(S_t)" />:{" "}
            <InlineMath math="R_{t+1} + \gamma V(S_{t+1})" />
          </List.Item>
          <List.Item>
            For <InlineMath math="Q(S_t, A_t)" />:{" "}
            <InlineMath math="R_{t+1} + \gamma Q(S_{t+1}, A_{t+1})" />
          </List.Item>
        </List>
      </div>

      <div data-slide>
        <Title order={3} mb="md">TD Learning - V/Q Estimation</Title>
        <Title order={4} mb="md">
          TD Error (<InlineMath math="\delta_t" />):
        </Title>
        <BlockMath math="\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)" />
        <Text mb="md">Or</Text>
        <BlockMath math="\delta_t = R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - \gamma Q(S_{t}, A_{t})" />
        <Title order={4} mt="md" mb="md">Update Rule:</Title>
        <BlockMath math="V(S_t) \leftarrow V(S_t) + \alpha \delta_t" />
        <Text mb="md">Or</Text>
        <BlockMath math="Q(S_{t}, A_{t}) \leftarrow Q(S_{t}, A_{t}) + \alpha \delta_t" />
        <Text mb="md">
          Where <InlineMath math="\alpha" /> is the learning rate.
        </Text>
      </div>

      <div data-slide>
        <SarsaAlgorithm/>
      </div>
      <div data-slide>
        <Title order={3} mb="md">Visual Steps in SARSA</Title>
        <Grid>
          <Grid.Col span={{ md: 3 }}>
            <Image
              src="/assets/data-science-practice/module9/tikz_images_2/tikz_picture_4.png"
              alt="SARSA Step 1"
            />
            <Center>
              <Text size="sm">
                1. Choose action <InlineMath math="a_3=\arg \max Q(s_3,a)" />
              </Text>
            </Center>
          </Grid.Col>
          <Grid.Col span={{ md: 3 }}>
            <Image
              src="/assets/data-science-practice/module9/tikz_images_2/tikz_picture_5.png"
              alt="SARSA Step 2"
            />
            <Center>
              <Text size="sm">
                2. Update <InlineMath math="Q(s_2, a_2)" /> with{" "}
                <InlineMath math="r_3 + \gamma Q(s_3, a_3)" />
              </Text>
            </Center>
          </Grid.Col>
          <Grid.Col span={{ md: 3 }}>
            <Image
              src="/assets/data-science-practice/module9/tikz_images_2/tikz_picture_6.png"
              alt="SARSA Step 3"
            />
            <Center>
              <Text size="sm">
                3. Choose action <InlineMath math="a_4=\arg \max Q(s_4,a)" />
              </Text>
            </Center>
          </Grid.Col>
          <Grid.Col span={{ md: 3 }}>
            <Image
              src="/assets/data-science-practice/module9/tikz_images_2/tikz_picture_7.png"
              alt="SARSA Step 4"
            />
            <Center>
              <Text size="sm">
                4. Update <InlineMath math="Q(s_3, a_3)" /> with{" "}
                <InlineMath math="r_4 + \gamma Q(s_4, a_4)" />
              </Text>
            </Center>
          </Grid.Col>
        </Grid>
      </div>

      <div data-slide>
        <QLearning/>
      </div>
      <div data-slide>
        <Title order={3} id="q-learning" mb="md">Q-learning Algorithm</Title>

        <Text mb="md">
          Initialize <InlineMath math="Q(s,a)" /> for all states and actions <InlineMath math="s,a" />
        </Text>
        <Text mb="md">
          <InlineMath math="S_t = \text{initial state}" />
        </Text>

        <List type="ordered" mb="md">
          <List.Item>
            Act with <InlineMath math="\pi" /> (extract from <InlineMath math="Q" /> {' '}
            <InlineMath math="\epsilon" />-greedy) to get <InlineMath math="A_t, R_{t+1}, S_{t+1}" />
          </List.Item>
          <List.Item>
            Update <InlineMath math="Q" /> using the observation step:{' '}
            <BlockMath
              math="Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha (R_{t+1} + \gamma \max Q(S_{t+1}, a) - Q(S_t, A_t))"
            />
          </List.Item>
          <List.Item>
            Iterate
          </List.Item>
        </List>
      </div>

      <div data-slide>
        <Title order={3} mb="md">Visual Steps in Q-Learning</Title>
        <Grid>
          <Grid.Col span={{ md: 3 }}>
            <Image
              src="/assets/data-science-practice/module9/tikz_images_2/tikz_picture_8.png"
              alt="Q-Learning Step 1"
            />
            <Center>
              <Text size="sm">
                1. Choose action <InlineMath math="a_3=\arg \max Q(s_3,a)" />
              </Text>
            </Center>
          </Grid.Col>
          <Grid.Col span={{ md: 3 }}>
            <Image
              src="/assets/data-science-practice/module9/tikz_images_2/tikz_picture_9.png"
              alt="Q-Learning Step 2"
            />
            <Center>
              <Text size="sm">
                2. Update <InlineMath math="Q(s_3, a_3)" /> with{" "}
                <InlineMath math="r_4 + \gamma \max Q(s_4, a)" />
              </Text>
            </Center>
          </Grid.Col>
          <Grid.Col span={{ md: 3 }}>
            <Image
              src="/assets/data-science-practice/module9/tikz_images_2/tikz_picture_10.png"
              alt="Q-Learning Step 3"
            />
            <Center>
              <Text size="sm">
                3. Choose action <InlineMath math="a_4=\arg \max Q(s_4,a)" />
              </Text>
            </Center>
          </Grid.Col>
          <Grid.Col span={{ md: 3 }}>
            <Image
              src="/assets/data-science-practice/module9/tikz_images_2/tikz_picture_11.png"
              alt="Q-Learning Step 4"
            />
            <Center>
              <Text size="sm">
                4. Update <InlineMath math="Q(s_4, a_4)" /> with{" "}
                <InlineMath math="r_5 + \gamma \max Q(s_4, a)" />
              </Text>
            </Center>
          </Grid.Col>
        </Grid>
      </div>

      <div data-slide>
        <Accordion variant="separated">
          <Accordion.Item value="weight-gradient">
            <Accordion.Control>
              <Title order={4}>Q Learning Convergence Proof</Title>
            </Accordion.Control>
            <Accordion.Panel>
              <QLearningConvergence />
            </Accordion.Panel>
          </Accordion.Item>
        </Accordion>
      </div>
    </Container>
  );
};

export default ModelFreeMethods;
