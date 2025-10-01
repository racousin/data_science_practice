import React from "react";
import { Container, Grid, Image, Stack, Text, Title, Code, List, Center } from '@mantine/core';
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
    <div data-slide>
      <Title order={2} mb="md">Environment</Title>
      <Text mb="md">
        The environment represents the world in which the agent operates. Key methods and variables include:
      </Text>
      <List mb="md">
        <List.Item>
          <Code>reset()</Code>: Initializes the environment to its starting state. Returns the initial state.
        </List.Item>
        <List.Item>
          <Code>step(action)</Code>: Executes the given action and returns a tuple of (next_state, reward, done).
        </List.Item>
        <List.Item>
          <Code>done</Code>: Boolean flag indicating if the episode has ended (goal reached or failure).
        </List.Item>
        <List.Item>
          <Code>state</Code> or <Code>observation</Code>: Current position/configuration of the environment.
        </List.Item>
      </List>
      <CodeBlock code={environmentCode} language="python"/>

      <Title order={2} mt="xl" mb="md">Agent</Title>
      <Text mb="md">
        The agent makes decisions and learns from experience. Essential components include:
      </Text>
      <List mb="md">
        <List.Item>
          <Code>choose_action(state)</Code>: Chooses an action based on the current state using an exploration strategy.
        </List.Item>
        <List.Item>
          <Code>learn(state, action, reward, next_state)</Code>: Updates the agent's knowledge based on experience.
        </List.Item>
      </List>
      <CodeBlock code={agentCode} language="python"/>

      <Title order={2} mt="xl" mb="md">Experiment</Title>
      <Text mb="md">
        The training loop that connects the environment and agent. Key components:
      </Text>
      <List mb="md">
        <List.Item>
          <Code>episode</Code>: One complete sequence of interaction from start to terminal state.
        </List.Item>
        <List.Item>
          <Code>total_reward</Code>: Cumulative reward obtained during an episode.
        </List.Item>
        <List.Item>
          <Code>done</Code>: Signal for episode termination.
        </List.Item>
        <List.Item>
          <Code>state/next_state</Code>: Current and resulting states from actions.
        </List.Item>
      </List>
      <CodeBlock code={experimentCode} language="python"/>
    </div>
  );
};


const MDP = () => {
  return (
    <Container fluid>
      <div data-slide>
        <Title order={2} mb="md">Understanding Markov Decision Processes (MDPs)</Title>

        <Center>
          <Stack align="center">
            <Image
              src="/assets/data-science-practice/module9/mdp.jpg"
              alt="MDP Illustration"
              w="75%"
              h="auto"
            />
            <Text size="sm">
              Illustrative example of an MDP, showcasing state transitions,
              actions, and rewards.
            </Text>
          </Stack>
        </Center>

        <Title order={3} mt="xl" mb="md">Glossary</Title>
        <List>
          <List.Item>
            State Space (<InlineMath math="S" />)
          </List.Item>
          <List.Item>
            Action Space (<InlineMath math="A" />)
          </List.Item>
          <List.Item>
            Transition Model (<InlineMath math="P" />)
          </List.Item>
          <List.Item>
            Reward function (<InlineMath math="R" />)
          </List.Item>
          <List.Item>
            Policy (<InlineMath math="\pi" />)
          </List.Item>
          <List.Item>
            Trajectory (<InlineMath math="\tau" />)
          </List.Item>
          <List.Item>
            Return (<InlineMath math="G" />)
          </List.Item>
        </List>
      </div>

      <div data-slide>
        <Title order={3} mb="md">Example of Simple Grid World Problem</Title>
        <Text mb="md">
          Our environment is a 4x4 grid where an agent aims to reach a goal.
        </Text>
        <Center>
          <Stack align="center">
            <Image
              src="/assets/data-science-practice/module9/tikz_picture_1.png"
              alt="Grid World"
              w="45%"
              h="auto"
            />
            <Text size="sm">A: Agent, G: Goal</Text>
          </Stack>
        </Center>
      </div>

      <div data-slide>
        <Grid justify="center">
          <Grid.Col span={6}>
            <Title order={3} id="state-space" mb="md">
              State Space (<InlineMath math="S" />)
            </Title>
            <Text mb="md">16 discrete states.</Text>
            <Image
              src="/assets/data-science-practice/module9/tikz_picture_2.png"
              alt="State Space"
              w="100%"
              h="auto"
            />
          </Grid.Col>
          <Grid.Col span={6}>
            <Title order={3} id="action-space" mb="md">
              Action Space (<InlineMath math="A" />)
            </Title>
            <Text mb="md">4 discrete actions (Up, Down, Left, Right).</Text>
            <Image
              src="/assets/data-science-practice/module9/tikz_picture_3.png"
              alt="Action Space"
              w="100%"
              h="auto"
            />
          </Grid.Col>
        </Grid>
      </div>

      <div data-slide>
        <Title order={3} id="transition-model" mb="md">
          Transition Model:{" "}
          <InlineMath math="P_{ss'}^a = \mathbb{P} [S_{t+1} = s' \vert S_t = s, A_t = a]" />
        </Title>
        <Grid justify="center">
          <Grid.Col span={6}>
            <Text mb="md">Deterministic environment.</Text>
            <Image
              src="/assets/data-science-practice/module9/tikz_picture_4.png"
              alt="Deterministic Transition"
              w="100%"
              h="auto"
            />
          </Grid.Col>
          <Grid.Col span={6}>
            <Text mb="md">Stochastic environment.</Text>
            <Image
              src="/assets/data-science-practice/module9/tikz_picture_5.png"
              alt="Stochastic Transition"
              w="100%"
              h="auto"
            />
          </Grid.Col>
        </Grid>
      </div>

      <div data-slide>
        <Title order={3} id="reward-function" mb="md">
          Reward function: <InlineMath math="r = R(s, a) = r(s')" />
        </Title>
        <Grid justify="center">
          <Grid.Col span={6}>
            <Text mb="md">Simple goal reward.</Text>
            <Image
              src="/assets/data-science-practice/module9/tikz_picture_6.png"
              alt="Simple Reward"
              w="100%"
              h="auto"
            />
          </Grid.Col>
          <Grid.Col span={6}>
            <Text mb="md">Other example of environment reward function.</Text>
            <Image
              src="/assets/data-science-practice/module9/tikz_picture_7.png"
              alt="Complex Reward"
              w="100%"
              h="auto"
            />
          </Grid.Col>
        </Grid>
      </div>

      <div data-slide>
        <Title order={3} id="policy" mb="md">
          Policy: (<InlineMath math="\pi: S \rightarrow A" />)
        </Title>
        <Text mb="md">
          Agent action in a state defined by its policy
          deterministic/stochastic
        </Text>
        <Center>
          <Image
            src="/assets/data-science-practice/module9/tikz_picture_8.png"
            alt="Policy"
            w="45%"
            h="auto"
          />
        </Center>
      </div>

      <div data-slide>
        <Title order={3} mb="md">
          Trajectory:{" "}
          <InlineMath math={"\\small (s_{0,0}, \\rightarrow, 0, s_{1,0}, \\rightarrow, 0, s_{2,0}, \\uparrow, 0, s_{2,1}, \\uparrow, 0, s_{2,2}, \\leftarrow, 0, s_{1,2}, \\uparrow, 0, s_{1,3}, \\rightarrow, 0, s_{2,3}, \\rightarrow, 1)"} />
        </Title>
        <Center>
          <Image
            src="/assets/data-science-practice/module9/tikz_picture_9.png"
            alt="Trajectory"
            w="45%"
            h="auto"
          />
        </Center>
      </div>

      <div data-slide>
        <Title order={3} mb="md">
          Return: <InlineMath math="G_t=\sum_{k=1}^T \gamma^k r_{t+k}" />
        </Title>
        <Grid justify="center">
          <Grid.Col span={6}>
            <Text mb="md">Cumulative rewards</Text>
            <Image
              src="/assets/data-science-practice/module9/tikz_picture_10.png"
              alt="Return"
              w="100%"
              h="auto"
            />
          </Grid.Col>
          <Grid.Col span={6}>
            <Text mb="md">Discounted rewards ( <InlineMath math="\gamma = 0.95"/>)</Text>
            <Image
              src="/assets/data-science-practice/module9/tikz_picture_11.png"
              alt="Optimal Policy"
              w="100%"
              h="auto"
            />
          </Grid.Col>
        </Grid>
      </div>

      <div data-slide>
        <Title order={3} mb="md">Objective: Find best Policy</Title>
        <BlockMath math="\pi^* = \arg \max_{\pi} E_{\tau\sim \pi}[{G(\tau)}]" />
        <Text mb="md">Optimal policy in the grid world environment.</Text>

        <Center>
          <Image
            src="/assets/data-science-practice/module9/tikz_picture_12.png"
            alt="Optimal Policy"
            w="45%"
            h="auto"
          />
        </Center>
      </div>

<CompleteRLExample/>

      <div data-slide>
        <Title order={3} mb="md">Glossary</Title>
        <List>
          <List.Item>
            Value Function (<InlineMath math="V" />)
          </List.Item>
          <List.Item>
            Action Value Function (<InlineMath math="Q" />)
          </List.Item>
          <List.Item>
            Bellman Equations
          </List.Item>
        </List>
      </div>

      <div data-slide>
        <Title order={3} id="value-function" mb="md">
          Value Function:{" "}
          <InlineMath math="V^{\pi}(s) = E_{\tau \sim \pi}[{G_t\left| S_t = s\right.}]" />
        </Title>
        <Text mb="md">
          Expected Return for State following <InlineMath math="\pi" />
        </Text>
        <Center>
          <Image
            src="/assets/data-science-practice/module9/tikz_picture_13.png"
            alt="Value Function"
            w="45%"
            h="auto"
          />
        </Center>
      </div>

      <div data-slide>
        <Title order={3} mb="md">
          Action Value Function:{" "}
          <InlineMath math="Q^{\pi}(s,a) = E_{\tau \sim \pi}[{G_t\left| S_t = s, A_t = a\right.}]" />
        </Title>
        <Text mb="md">
          Expected Return for State-Action following <InlineMath math="\pi" />
        </Text>
        <Center>
          <Image
            src="/assets/data-science-practice/module9/tikz_picture_14.png"
            alt="Action Value Function"
            w="45%"
            h="auto"
          />
        </Center>
      </div>

      <div data-slide>
        <Title order={3} id="bellman-equations" mb="md">Bellman Equations</Title>
        <Text mb="md">
          <strong>Idea:</strong> The value of your starting point is the
          reward you expect to get from being there, plus the value of
          wherever you land next.
        </Text>
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
      </div>

      <div data-slide>
        <Title order={3} mb="md">
          Value Function Decomposition: <InlineMath math="V^{\pi}(s)" />
        </Title>
        <Center>
          <Text mb="md">
            <strong>Value Function:</strong>{" "}
            <InlineMath math="V^{\pi}(s) = E[R_{t+1} + \gamma V^{\pi}(S_{t+1})|S_t = s]" />
          </Text>
        </Center>
        <Center>
          <Image
            src="/assets/data-science-practice/module9/tikz_picture_15.png"
            alt="Value Function Decomposition"
            w="45%"
            h="auto"
          />
        </Center>
      </div>

    </Container>
  );
};

export default MDP;
