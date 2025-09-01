import React from "react";
import { Title, Text, Stack, Container, Accordion, Grid, Image } from '@mantine/core';
import { BlockMath, InlineMath } from "react-katex";
import "katex/dist/katex.min.css";
import CodeBlock from "components/CodeBlock";
import { Link } from 'react-router-dom';
import QLearningConvergence from './QLearningConvergence';

const QLearning = () => {
  return (
    <Stack className="mt-4 space-y-4">
      <Title order={3}>Q-learning Estimation</Title>
      
      <Stack className="space-y-3">
        <Text>
          Remember the optimal Q-function:
        </Text>
        <BlockMath
          math="Q_{\pi^*}(S_t,A_t) = \mathbb{E} [R_{t+1} + \gamma \max_{a'} Q_*(S_{t+1}, a') \vert S_t = s, A_t = a]"
        />
        
        <Text>
          So <InlineMath math="R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a')" /> is an unbiased estimate for the optimal Q function policy <InlineMath math="Q_{\pi^*}(S_t, A_t)" />
        </Text>
        
        <Text>
          <InlineMath math="R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a')" /> is called the Q target.
        </Text>

        <Text>
          <InlineMath math="\alpha" /> improvement:
        </Text>
        <BlockMath
          math="Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha (R_{t+1} + \gamma \max_{a \in \mathcal{A}} Q(S_{t+1}, a) - Q(S_t, A_t))"
        />
        
        <Text className="mt-2 text-gray-700">
          This is called an off-policy update, because we update the Q function not from real environment feedback, but from theoretical policy information 
          (i.e., instead of using <InlineMath math="Q(S_{t+1}, A_{t+1})" /> like in SARSA, we use <InlineMath math="\max_{a} Q(S_{t+1}, a)" />).
        </Text>
      </Stack>
    </Stack>
  );
};

const SarsaAlgorithm = () => {
  return (
    <Stack className="p-4">
      <Title order={3} id="sarsa">SARSA Algorithm</Title>
      
      <Text>
        Initialize <InlineMath math="Q(s,a)" /> for all states and actions <InlineMath math="s,a" />
      </Text>
      
      <Text>
        <InlineMath math="S_t = \text{initial state}" />, act with <InlineMath math="\pi" /> 
        (extract from <InlineMath math="Q" /> <InlineMath math="\epsilon" />-greedy) 
        to get <InlineMath math="A_t,R_{t+1},S_{t+1}" />
      </Text>

      <ol className="list-decimal pl-6 space-y-2">
        <li>
          Act with <InlineMath math="\pi" /> (extract from <InlineMath math="Q" /> {" "}
          <InlineMath math="\epsilon" />-greedy) to get <InlineMath math="A_{t+1}, R_{t+2},S_{t+2}" />
        </li>
        
        <li>
          Update Q using the observation step:{" "}
          <BlockMath math="Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha (R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t))" />
        </li>
        
        <li>
          Iterate
        </li>
      </ol>
    </Stack>
  );
};

const RLObjectiveSection = () => {

  return (

      <Stack fluid>
        <div>
          <Title order={3} mb="md">Reinforcement Learning Objective</Title>
          <Text className="mb-4">
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
        </div>

        <Stack fluid>
          <Text>
            As shown in <Link to="/module13/course/dynamic-programming" className="text-blue-600 hover:text-blue-800">
            dynamic programming</Link>, we can start with a random policy and evaluate its V/Q function 
            to iteratively build better policies until we reach the optimal policy.
          </Text>
          
          <Text>
            However, in reinforcement learning, we don't have access to the transition model, 
            so we cannot compute V and Q exactly.
          </Text>
          
          <Text>
            Instead, we can estimate these values, which is what we'll explore next.
          </Text>
        </Stack>
      </Stack>
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
      <h2>Model-Free Reinforcement Learning Methods</h2>

      <RLObjectiveSection />

      <Grid className="mt-4">
        <Grid.Col>
          <h3>Monte-Carlo Method</h3>
          <p>
            To evaluate{" "}
            <InlineMath math="V_\pi(s) = E_{\tau \sim \pi}[{G_t\left| s_t = s\right.}]" />
            :
          </p>
          <ol>
            <li>
              Generate an episode (<InlineMath math={"s_1,a_1,r_2,…,s_T"}/>)  with the policy <InlineMath math="\pi" />
            </li>
            <li>
              Compute{" "}
              <InlineMath math="G_t = \sum_{k=0}^{T-t-1} \gamma^k R_{t+k+1}" />
            </li>
            <li>
              Evaluate the value function:
              <BlockMath math="V_\pi(s) = \frac{\sum_{t=1}^T \mathbb{1}[S_t = s] G_t}{\sum_{t=1}^T \mathbb{1}[S_t = s]}" />
            </li>
          </ol>
          <p>Similarly, for the action-value function:</p>
          <BlockMath math="Q_\pi(s, a) = \frac{\sum_{t=1}^T \mathbb{1}[S_t = s, A_t = a] G_t}{\sum_{t=1}^T \mathbb{1}[S_t = s, A_t = a]}" />
        </Grid.Col>
      </Grid>

      <Grid className="mt-4">
        <Grid.Col>
          <h3 id="monte-carlo-algorithm">Monte-Carlo Algorithm</h3>
          <ol>
            Initialise <InlineMath math="Q(s,a) \forall s, a" />
            <li>
              Generate an episode with the policy <InlineMath math="\pi" />{" "}
              (extract from <InlineMath math="Q" />{" "}
              <InlineMath math="\epsilon" />
              -greedy, i.e: <InlineMath math="a = \arg\max Q(s, a)"/> or random action)
            </li>
            <li>
              Evaluate Q using the episode:
              <BlockMath math="Q_\pi(s, a) = \frac{\sum_{t=1}^T \big( \mathbb{1}[S_t = s, A_t = a] \sum_{k=0}^{T-t-1} \gamma^k R_{t+k+1} \big)}{\sum_{t=1}^T \mathbb{1}[S_t = s, A_t = a]}" />
            </li>
            <li>Iterate</li>
          </ol>
        </Grid.Col>
      </Grid>
      <Grid className="mt-4">
        <Grid.Col>
          <h3>Visual Steps in Monte Carlo</h3>
          <Grid>
            <Grid.Col span={{ md: 4 }}>
              <Image
                src="/assets/data-science-practice/module13/tikz_images_2/tikz_picture_1.png"
                alt="Generate Episode"
                fluid
              />
              <p className="text-center">
                1. Generate episode following{" "}
                <InlineMath math="\arg\max Q(s, a)" />
              </p>
            </Grid.Col>
            <Grid.Col span={{ md: 4 }}>
              <Image
                src="/assets/data-science-practice/module13/tikz_images_2/tikz_picture_2.png"
                alt="Evaluate Q"
                fluid
              />
              <p className="text-center">2. Evaluate Q</p>
            </Grid.Col>
            <Grid.Col span={{ md: 4 }}>
              <Image
                src="/assets/data-science-practice/module13/tikz_images_2/tikz_picture_3.png"
                alt="Iterate"
                fluid
              />
              <p className="text-center">3. Iterate</p>
            </Grid.Col>
          </Grid>
        </Grid.Col>
      </Grid>
      <Grid className="mt-4">
        <Grid.Col>
          <h3 id="td-learning">Temporal Difference (TD) Learning</h3>
          <p>
            TD Learning combines Monte Carlo and dynamic programming ideas,
            using bootstrapping for value updates.
          </p>
          <h4>Bellman equations:</h4>
          <BlockMath math="V(S_t) = \mathbb{E}[R_{t+1} + \gamma V(S_{t+1}) | S_t = s]" />
          <BlockMath math="Q(s, a) = \mathbb{E} [R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) \mid S_t = s, A_t = a]" />
          <h4>TD Target (unbiased estimate):</h4>
          <ul>
            <li>
              For <InlineMath math="V(S_t)" />:{" "}
              <InlineMath math="R_{t+1} + \gamma V(S_{t+1})" />
            </li>
            <li>
              For <InlineMath math="Q(S_t, A_t)" />:{" "}
              <InlineMath math="R_{t+1} + \gamma Q(S_{t+1}, A_{t+1})" />
            </li>
          </ul>
        </Grid.Col>
      </Grid>

      <Grid className="mt-4">
        <Grid.Col>
          <h3>TD Learning - V/Q Estimation</h3>
          <h4>
            TD Error (<InlineMath math="\delta_t" />
            ):
          </h4>
          <BlockMath math="\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)" />
          Or
          <BlockMath math="\delta_t = R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - \gamma Q(S_{t}, A_{t})" />
          <h4>Update Rule:</h4>
          <BlockMath math="V(S_t) \leftarrow V(S_t) + \alpha \delta_t" />
          Or
          <BlockMath math="Q(S_{t}, A_{t}) \leftarrow Q(S_{t}, A_{t}) + \alpha \delta_t" />
          <p>
            Where <InlineMath math="\alpha" /> is the learning rate.
          </p>
        </Grid.Col>
      </Grid>
      <SarsaAlgorithm/>
      <Grid className="mt-4">
        <Grid.Col>
          <h3>Visual Steps in SARSA</h3>
          <Grid>
            <Grid.Col span={{ md: 3 }}>
              <Image
                src="/assets/data-science-practice/module13/tikz_images_2/tikz_picture_4.png"
                alt="SARSA Step 1"
                fluid
              />
              <p className="text-center">
                1. Choose action <InlineMath math="a_3=\arg \max Q(s_3,a)" />
              </p>
            </Grid.Col>
            <Grid.Col span={{ md: 3 }}>
              <Image
                src="/assets/data-science-practice/module13/tikz_images_2/tikz_picture_5.png"
                alt="SARSA Step 2"
                fluid
              />
              <p className="text-center">
                2. Update <InlineMath math="Q(s_2, a_2)" /> with{" "}
                <InlineMath math="r_3 + \gamma Q(s_3, a_3)" />
              </p>
            </Grid.Col>
            <Grid.Col span={{ md: 3 }}>
              <Image
                src="/assets/data-science-practice/module13/tikz_images_2/tikz_picture_6.png"
                alt="SARSA Step 3"
                fluid
              />
              <p className="text-center">
                3. Choose action <InlineMath math="a_4=\arg \max Q(s_4,a)" />
              </p>
            </Grid.Col>
            <Grid.Col span={{ md: 3 }}>
              <Image
                src="/assets/data-science-practice/module13/tikz_images_2/tikz_picture_7.png"
                alt="SARSA Step 4"
                fluid
              />
              <p className="text-center">
                4. Update <InlineMath math="Q(s_3, a_3)" /> with{" "}
                <InlineMath math="r_4 + \gamma Q(s_4, a_4)" />
              </p>
            </Grid.Col>
          </Grid>
        </Grid.Col>
      </Grid>
      <QLearning/>
      <Stack className="space-y-4">
        <Title order={3} id="q-learning">Q-learning Algorithm</Title>
        
        <Text>
          Initialize <InlineMath math="Q(s,a)" /> for all states and actions <InlineMath math="s,a" />
        </Text>
        <Text>
          <InlineMath math="S_t = \text{initial state}" />
        </Text>
        
        <ol className="list-decimal pl-6 space-y-3">
          <li>
            Act with <InlineMath math="\pi" /> (extract from <InlineMath math="Q" /> {' '}
            <InlineMath math="\epsilon" />-greedy) to get <InlineMath math="A_t, R_{t+1}, S_{t+1}" />
          </li>
          <li>
            Update <InlineMath math="Q" /> using the observation step:{' '}
            <BlockMath
              math="Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha (R_{t+1} + \gamma \max Q(S_{t+1}, a) - Q(S_t, A_t))"
            />
          </li>
          <li>
            Iterate
          </li>
        </ol>
      </Stack>
      <Grid className="mt-4">
        <Grid.Col>
          <h3>Visual Steps in Q-Learning</h3>
          <Grid>
            <Grid.Col span={{ md: 3 }}>
              <Image
                src="/assets/data-science-practice/module13/tikz_images_2/tikz_picture_8.png"
                alt="Q-Learning Step 1"
                fluid
              />
              <p className="text-center">
                1. Choose action <InlineMath math="a_3=\arg \max Q(s_3,a)" />
              </p>
            </Grid.Col>
            <Grid.Col span={{ md: 3 }}>
              <Image
                src="/assets/data-science-practice/module13/tikz_images_2/tikz_picture_9.png"
                alt="Q-Learning Step 2"
                fluid
              />
              <p className="text-center">
                2. Update <InlineMath math="Q(s_3, a_3)" /> with{" "}
                <InlineMath math="r_4 + \gamma \max Q(s_4, a)" />
              </p>
            </Grid.Col>
            <Grid.Col span={{ md: 3 }}>
              <Image
                src="/assets/data-science-practice/module13/tikz_images_2/tikz_picture_10.png"
                alt="Q-Learning Step 3"
                fluid
              />
              <p className="text-center">
                3. Choose action <InlineMath math="a_4=\arg \max Q(s_4,a)" />
              </p>
            </Grid.Col>
            <Grid.Col span={{ md: 3 }}>
              <Image
                src="/assets/data-science-practice/module13/tikz_images_2/tikz_picture_11.png"
                alt="Q-Learning Step 4"
                fluid
              />
              <p className="text-center">
                4. Update <InlineMath math="Q(s_4, a_4)" /> with{" "}
                <InlineMath math="r_5 + \gamma \max Q(s_4, a)" />
              </p>
            </Grid.Col>
          </Grid>
        </Grid.Col>
      </Grid>
      <div className="mt-8 mb-12">
        <Accordion 
          variant="separated"
          styles={{
            item: {
              marginBottom: '2rem',
              border: '1px solid var(--mantine-color-gray-3)',
              borderRadius: 'var(--mantine-radius-md)',
            },
            control: {
              padding: '1rem',
              '&:hover': {
                backgroundColor: 'var(--mantine-color-gray-0)',
              },
            },
            content: {
              padding: '1.5rem',
            },
          }}
        >
          <Accordion.Item value="weight-gradient">
            <Accordion.Control>
              <div className="text-lg font-medium">Q Learning Convergence Proof</div>
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
