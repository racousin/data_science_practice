import React from 'react';
import { Title, Text, Container, Image, Stack, List, Center } from '@mantine/core';
import 'katex/dist/katex.min.css';
import { BlockMath, InlineMath } from 'react-katex';
import CodeBlock from "components/CodeBlock";



const BanditParadigms = () => {
  return (
    <>
      <Text mb="md">
        The multi-armed bandit problem represents the simplest form of the exploration-exploitation
        dilemma.
      </Text>

      <Title order={3} mb="md">1. Multi-Armed Bandit Paradigm</Title>
      <Text mb="md">At each time step t:</Text>
      <BlockMath>
        {`\\begin{aligned}
        a_t &\\in \\mathcal{A} \\text{ (select action)} \\\\
        r_t &= R(a_t) \\text{ (receive reward)}
        \\end{aligned}`}
      </BlockMath>
      <Text mb="md">
        States don't exist in this framework - only actions and their immediate rewards matter.
      </Text>

      <Title order={3} mt="xl" mb="md">2. Contextual Bandit Paradigm</Title>
      <Text mb="md">At each time step t:</Text>
      <BlockMath>
        {`\\begin{aligned}
        s_t &\\sim P(s) \\text{ (observe state/context)} \\\\
        a_t &\\in \\mathcal{A} \\text{ (select action)} \\\\
        r_t &= R(s_t, a_t) \\text{ (receive reward)}
        \\end{aligned}`}
      </BlockMath>
      <Text mb="md">
        Each state is independently sampled - future states don't depend on current actions.
      </Text>

      <Title order={3} mt="xl" mb="md">3. Full RL Paradigm</Title>
      <Text mb="md">At each time step t:</Text>
      <BlockMath>
        {`\\begin{aligned}
        s_t &\\text{ (observe current state)} \\\\
        a_t &\\in \\mathcal{A} \\text{ (select action)} \\\\
        r_t &= R(s_t, a_t) \\text{ (receive reward)} \\\\
        s_{t+1} &= f(s_t, a_t) \\text{ (state transitions based on action)}
        \\end{aligned}`}
      </BlockMath>
      <Text mb="md">
        Key distinction: The next state depends on the current state and action - creating a sequential decision process.
      </Text>
    </>
  );
};



const RLParadigms = () => {
  const agentCode = `
class Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = 0.1  # Exploration rate
        
    def choose_action(self, state):
        """Choose action based on epsilon-greedy strategy"""
        rand = random.random()  # Generate a random float in [0, 1)
        if rand < self.epsilon:
            # Choose action using knowledge
            return self.knowledge_based_action(state)
        else:
            # Play randomly
            return random.randint(0, self.action_size - 1)
    
    def learn(self, state, action, reward, next_state):
        """Update agent knowledge and strategy"""
        pass`;

  return (
    <Container fluid>
      <div data-slide>
        <Title order={2} id="rl-paradigms" mb="md">
          Reinforcement Learning Paradigms
        </Title>

        <Title order={3} id="mdp-connection" mb="md">
          From MDP to RL
        </Title>

        <Text mb="md">
          In previous sections, we explored MDPs where we had complete knowledge of the environment's
          dynamics through the transition probability function P. In many scenarios,
          we don't have access to this model. Our objective remains the same:
          finding an optimal policy that maximizes expected returns.
        </Text>

        <BlockMath>
          {`\\begin{align*}
          \\pi^* &= \\arg\\max_{\\pi} J(\\pi)\\\\
          J(\\pi) &= \\mathbb{E}_{\\tau\\sim\\pi}[G(\\tau)] = \\int_{\\tau} \\mathbb{P}(\\tau|\\pi) G(\\tau)
          \\end{align*}`}
        </BlockMath>
      </div>

      <div data-slide>
        <Title order={3} id="exploration-exploitation" mb="md">
          The Exploration vs. Exploitation Dilemma
        </Title>

        <Text mb="md">
          Without a known model, agents must learn through interaction, leading to a fundamental
          challenge: the exploration-exploitation dilemma. Should the agent exploit its current
          knowledge to maximize immediate rewards, or explore to potentially discover better strategies?
        </Text>
        <Center>
          <Image
            src="/assets/data-science-practice/module9/tikz_images_2/explore_vs_exploit.jpeg"
            alt="Exploration vs Exploitation trade-off"
            h={400}
            w="auto"
          />
        </Center>
      </div>

      <div data-slide>
        <Title order={3} id="bandit-vs-rl" mb="md">
          From Bandits to RL
        </Title>

        <BanditParadigms/>
      </div>

      <div data-slide>
        <Title order={3} id="model-based-vs-free" mb="md">
          Model-Based vs Model-Free Approaches
        </Title>

        <Text mb="md">
          Two main paradigms have emerged for solving RL problems: model-based and model-free approaches.
          Each has its own advantages and trade-offs.
        </Text>
        <Center>
          <Image
            src="/assets/data-science-practice/module9/model-based-free.jpg"
            alt="Model-based vs Model-free RL"
            h={400}
            w="auto"
          />
        </Center>
        <List mb="md">
          <List.Item>
            Model-Based: Learn or use an explicit model of the environment
          </List.Item>
          <List.Item>
            Model-Free: Learn directly from experience without building a model
          </List.Item>
        </List>
        <Center>
          <Image
            src="/assets/data-science-practice/module9/tikz_images_2/tax.png"
            alt="Taxonomy of RL algorithms"
            h={400}
            w="auto"
          />
        </Center>
        <Text mb="md">
          It exists a big variety of algorithms.
        </Text>
      </div>

      <div data-slide>
        <Title order={3} id="epsilon-greedy" mb="md">
          Simple Exploration Strategy: ε-Greedy
        </Title>

        <Text mb="md">
          One of the simplest yet effective approaches to balance exploration and exploitation
          is the ε-greedy strategy. With probability ε, the agent explores randomly; otherwise,
          it exploits its current knowledge.
        </Text>

        <CodeBlock code={agentCode} language="python" />
      </div>
    </Container>
  );
};

export default RLParadigms;