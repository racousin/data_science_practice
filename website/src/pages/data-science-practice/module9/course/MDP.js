import React from "react";
import { Container, Grid, Image, Stack, Text, Title, Code } from '@mantine/core';
import { BlockMath, InlineMath } from "react-katex";
import "katex/dist/katex.min.css";
import CodeBlock from "components/CodeBlock";



const CompleteRLExample = () => {
  const environmentCode = `
class Environment:
    def __init__(self, size=10):
        self.size = size        # Grid size
        self.state = 0         # Start position
        self.goal = size - 1   # Goal position
    
    def reset(self):
        """Reset environment to initial state"""
        self.state = 0
        return self.state
    
    def step(self, action):
        """Execute action and return new state, reward, done flag"""
        if action == 1:    # Move right
            self.state = min(self.state + 1, self.size - 1)
        else:             # Move left
            self.state = max(self.state - 1, 0)
        
        done = (self.state == self.goal)
        reward = 1.0 if done else 0.0
        
        return self.state, reward, done`;

  const agentCode = `
class Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = 0.1  # Exploration rate
        
    def choose_action(self, state):
        """Choose action using knowledge"""
        return random.randint(0, self.action_size - 1)
    
    def learn(self, state, action, reward, next_state):
        """Update agent knowledge and strategy"""
        pass`;

  const experimentCode = `
# Training Loop
env = Environment(size=10)
agent = Agent(state_size=10, action_size=2)

for episode in range(100):
    state = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        # Agent selects action
        action = agent.choose_action(state)
        
        # Environment step
        next_state, reward, done = env.step(action)
        
        # Agent learns
        agent.learn(state, action, reward, next_state)
        
        total_reward += reward
        state = next_state
    
    print(f"Episode {episode}: total_reward={total_reward}")`;

  return (
    <Container fluid>
      <Title order={1} mb="xl">Environment, Agent Components</Title>

      <Title order={2} className="mb-4">Environment</Title>
      <Text className="mb-4">
        The environment represents the world in which the agent operates. Key methods and variables include:
      </Text>
      <ul className="list-disc pl-6 mb-4">
        <li>
          <Code>reset()</Code>: Initializes the environment to its starting state. Returns the initial state.
        </li>
        <li>
          <Code>step(action)</Code>: Executes the given action and returns a tuple of (next_state, reward, done).
        </li>
        <li>
          <Code>done</Code>: Boolean flag indicating if the episode has ended (goal reached or failure).
        </li>
        <li>
          <Code>state</Code> or <Code>observation</Code>: Current position/configuration of the environment.
        </li>
      </ul>
      <pre className="bg-gray-100 p-4 rounded-lg overflow-x-auto mb-6">
        <CodeBlock code={environmentCode} language="python"/>
      </pre>

      <Title order={2} className="mb-4">Agent</Title>
      <Text className="mb-4">
        The agent makes decisions and learns from experience. Essential components include:
      </Text>
      <ul className="list-disc pl-6 mb-4">
        <li>
          <Code>choose_action(state)</Code>: Chooses an action based on the current state using an exploration strategy.
        </li>
        <li>
          <Code>learn(state, action, reward, next_state)</Code>: Updates the agent's knowledge based on experience.
        </li>
      </ul>
      <pre className="bg-gray-100 p-4 rounded-lg overflow-x-auto mb-6">
      <CodeBlock code={agentCode} language="python"/>
      </pre>

      <Title order={2} className="mb-4">Experiment</Title>
      <Text className="mb-4">
        The training loop that connects the environment and agent. Key components:
      </Text>
      <ul className="list-disc pl-6 mb-4">
        <li>
          <Code>episode</Code>: One complete sequence of interaction from start to terminal state.
        </li>
        <li>
          <Code>total_reward</Code>: Cumulative reward obtained during an episode.
        </li>
        <li>
          <Code>done</Code>: Signal for episode termination.
        </li>
        <li>
          <Code>state/next_state</Code>: Current and resulting states from actions.
        </li>
      </ul>
      <pre className="bg-gray-100 p-4 rounded-lg overflow-x-auto">
      <CodeBlock code={experimentCode} language="python"/>
      </pre>
    </Container>
  );
};


const MDP = () => {
  return (
    <Container fluid>
      <h2>Understanding Markov Decision Processes (MDPs)</h2>

      <Grid className="justify-content-center">
        <Grid.Col xs={12} md={10} lg={8}>
          <div align="center">
            <Image
              src="/assets/data-science-practice/module9/mdp.jpg"
              alt="MDP Illustration"
                          w="75%"
            h="auto"
          />
            <p>
              Illustrative example of an MDP, showcasing state transitions,
              actions, and rewards.
            </p>
          </div>
        </Grid.Col>
      </Grid>

      <Grid>
        <Grid.Col>
          <h3>Glossary</h3>
          <ul>
            <li>
              State Space (<InlineMath math="S" />)
            </li>
            <li>
              Action Space (<InlineMath math="A" />)
            </li>
            <li>
              Transition Model (<InlineMath math="P" />)
            </li>
            <li>
              Reward function (<InlineMath math="R" />)
            </li>
            <li>
              Policy (<InlineMath math="\pi" />)
            </li>
            <li>
              Trajectory (<InlineMath math="\tau" />)
            </li>
            <li>
              Return (<InlineMath math="G" />)
            </li>
          </ul>
        </Grid.Col>
      </Grid>

      <Grid mt="xl">
        <Grid.Col>
          <h3>Example of Simple Grid World Problem</h3>
          <p>
            Our environment is a 4x4 grid where an agent aims to reach a goal.
          </p>
          <div align="center">
            <Image
              src="/assets/data-science-practice/module9/tikz_picture_1.png"
              alt="Grid World"
                          w="45%"
            h="auto"
          />
            <p>A: Agent, G: Goal</p>
          </div>
        </Grid.Col>
      </Grid>

      <Grid mt="xl" justify="center"> {/* justify="center" replaces justify-content-center */}
        <Grid.Col span={6}> {/* span={6} instead of md={6} for same-row layout */}
          <h3 id="state-space">
            State Space (<InlineMath math="S" />)
          </h3>
          <p>16 discrete states.</p>
          <Image
            src="/assets/data-science-practice/module9/tikz_picture_2.png"
            alt="State Space"
            w="100%"
            h="auto"
          />
        </Grid.Col>
        <Grid.Col span={6}>
          <h3 id="action-space">
            Action Space (<InlineMath math="A" />)
          </h3>
          <p>4 discrete actions (Up, Down, Left, Right).</p>
          <Image
            src="/assets/data-science-practice/module9/tikz_picture_3.png"
            alt="Action Space"
            w="100%"
            h="auto"
          />
        </Grid.Col>
      </Grid>

      <Grid mt="xl">
        <Grid.Col>
          <h3 id="transition-model">
            Transition Model:{" "}
            <InlineMath math="P_{ss'}^a = \mathbb{P} [S_{t+1} = s' \vert S_t = s, A_t = a]" />
          </h3>
          <Grid justify="center">
            <Grid.Col span={6}>
              <p>Deterministic environment.</p>
              <Image
                src="/assets/data-science-practice/module9/tikz_picture_4.png"
                alt="Deterministic Transition"
                w="100%"
                h="auto"
              />
            </Grid.Col>
            <Grid.Col span={6}>
              <p>Stochastic environment.</p>
              <Image
                src="/assets/data-science-practice/module9/tikz_picture_5.png"
                alt="Stochastic Transition"
                w="100%"
                h="auto"
              />
            </Grid.Col>
          </Grid>
        </Grid.Col>
      </Grid>

      <Grid mt="xl">
        <Grid.Col>
          <h3 id="reward-function">
            Reward function: <InlineMath math="r = R(s, a) = r(s')" />
          </h3>
          <Grid justify="center">
            <Grid.Col span={6}>
              <p>Simple goal reward.</p>
              <Image
                src="/assets/data-science-practice/module9/tikz_picture_6.png"
                alt="Simple Reward"
                w="100%"
                h="auto"
              />
            </Grid.Col>
            <Grid.Col span={6}>
              <p>Other example of environment reward function.</p>
              <Image
                src="/assets/data-science-practice/module9/tikz_picture_7.png"
                alt="Complex Reward"
                w="100%"
                h="auto"
              />
            </Grid.Col>
          </Grid>
        </Grid.Col>
      </Grid>

      <Grid mt="xl">
        <Grid.Col>
          <h3 id="policy">
            Policy: (<InlineMath math="\pi: S \rightarrow A" />)
          </h3>
          <p>
            Agent action in a state defined by its policy
            deterministic/stochastic
          </p>
          <div align="center">
          <Image src="/assets/data-science-practice/module9/tikz_picture_8.png" alt="Policy"             w="45%"
            h="auto" />
            </div>
        </Grid.Col>
      </Grid>

      <Grid mt="xl">
        <Grid.Col>
        <h3>
  Trajectory:{" "}
  <InlineMath math={"\\small (s_{0,0}, \\rightarrow, 0, s_{1,0}, \\rightarrow, 0, s_{2,0}, \\uparrow, 0, s_{2,1}, \\uparrow, 0, s_{2,2}, \\leftarrow, 0, s_{1,2}, \\uparrow, 0, s_{1,3}, \\rightarrow, 0, s_{2,3}, \\rightarrow, 1)"} />
</h3>
<div align="center">
          <Image
            src="/assets/data-science-practice/module9/tikz_picture_9.png"
            alt="Trajectory"
            w="45%"
            h="auto"
          />
          </div>
        </Grid.Col>
      </Grid>

      <Grid mt="xl">
        <Grid.Col>
          <h3>
            Return: <InlineMath math="G_t=\sum_{k=1}^T \gamma^k r_{t+k}" />
          </h3>
        </Grid.Col>
        <Grid justify="center">
          <Grid.Col span={6}>
            <p>Cumulative rewards</p>
            <Image
              src="/assets/data-science-practice/module9/tikz_picture_10.png"
              alt="Return"
              w="100%"
              h="auto"
            />
          </Grid.Col>
          <Grid.Col span={6}>
            <p>Discounted rewards ( <InlineMath math="\gamma = 0.95"/>)</p>
            <Image
              src="/assets/data-science-practice/module9/tikz_picture_11.png"
              alt="Optimal Policy"
              w="100%"
              h="auto"
            />
          </Grid.Col>
        </Grid>
      </Grid>

      <Grid mt="xl">
        <Grid.Col>
          <h3>Objective: Find best Policy</h3>
          <BlockMath math="\pi^* = \arg \max_{\pi} E_{\tau\sim \pi}[{G(\tau)}]" />
          <p>Optimal policy in the grid world environment.</p>
          
        <div align="center">
          <Image
            src="/assets/data-science-practice/module9/tikz_picture_12.png"
            alt="Optimal Policy"
            w="45%"
            h="auto"
          />
          </div>
        </Grid.Col>
      </Grid>

<CompleteRLExample/>

      <Grid mt="xl">
        <Grid.Col>
          <h3>Glossary</h3>
          <ul>
            <li>
                Value Function (<InlineMath math="V" />)
            </li>
            <li>
                Action Value Function (<InlineMath math="Q" />)
            </li>
            <li>
            Bellman Equations
            </li>
          </ul>
        </Grid.Col>
      </Grid>

      <Grid mt="xl">
        <Grid.Col>
          <h3 id="value-function">
            Value Function:{" "}
            <InlineMath math="V^{\pi}(s) = E_{\tau \sim \pi}[{G_t\left| S_t = s\right.}]" />
          </h3>
          <p>
            Expected Return for State following <InlineMath math="\pi" />
          </p>
          <div align="center">
          <Image
            src="/assets/data-science-practice/module9/tikz_picture_13.png"
            alt="Value Function"
            w="45%"
            h="auto"
          />
          </div>
        </Grid.Col>
      </Grid>

      <Grid mt="xl">
        <Grid.Col>
          <h3>
            Action Value Function:{" "}
            <InlineMath math="Q^{\pi}(s,a) = E_{\tau \sim \pi}[{G_t\left| S_t = s, A_t = a\right.}]" />
          </h3>
          <p>
            Expected Return for State-Action following <InlineMath math="\pi" />
          </p>
          <div align="center">
          <Image
            src="/assets/data-science-practice/module9/tikz_picture_14.png"
            alt="Action Value Function"
            w="45%"
            h="auto"
          />
          </div>
        </Grid.Col>
      </Grid>

      <Grid mt="xl">
        <Grid.Col>
          <h3 id="bellman-equations">Bellman Equations</h3>
          <p>
            <strong>Idea:</strong> The value of your starting point is the
            reward you expect to get from being there, plus the value of
            wherever you land next.
          </p>
          <BlockMath
            math={`
      \\begin{aligned}
      V(s) &= \\mathbb{E}[G_t \\vert S_t = s] \\\\
      &= \\mathbb{E} [R_{t+1} + \\gamma G_{t+1} \\vert S_t = s] \\\\
      &= \\mathbb{E} [R_{t+1} + \\gamma V(S_{t+1}) \\vert S_t = s]
      \\end{aligned}
    `}
          />
          <BlockMath math="Q(s, a) = \mathbb{E} [R_{t+1} + \gamma \mathbb{E}_{a\sim\pi} Q(S_{t+1}, a) \mid S_t = s, A_t = a]" />
        </Grid.Col>
      </Grid>

      <Grid mt="xl">
        <Grid.Col>
          <h3>
            Value Function Decomposition: <InlineMath math="V^{\pi}(s)" />
          </h3>
          <p align="center">
            <strong>Value Function:</strong>{" "}
            <InlineMath math="V^{\pi}(s) = E[R_{t+1} + \gamma V^{\pi}(S_{t+1})|S_t = s]" />
          </p>
          <div align="center">
            <Image
              src="/assets/data-science-practice/module9/tikz_picture_15.png"
              alt="Value Function Decomposition"
                          w="45%"
            h="auto"
          />
          </div>
        </Grid.Col>
      </Grid>


    </Container>
  );
};

export default MDP;
