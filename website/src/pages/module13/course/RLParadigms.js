import React from 'react';
import { Title, Text, Container, Image, Stack, Code } from '@mantine/core';
import 'katex/dist/katex.min.css';
import { BlockMath, InlineMath } from 'react-katex';
import CodeBlock from "components/CodeBlock";



const BanditParadigms = () => {
  return (
    <section className="space-y-6">
      <Text className="mb-4">
        The multi-armed bandit problem represents the simplest form of the exploration-exploitation 
        dilemma.
      </Text>

      <div className="mb-8">
        <Text component="h3" className="text-xl font-bold mb-4">
          1. Multi-Armed Bandit Paradigm
        </Text>
        <Text className="mb-2">
          At each time step t:
        </Text>
        <BlockMath>
          {`\\begin{aligned}
          a_t &\\in \\mathcal{A} \\text{ (select action)} \\\\
          r_t &= R(a_t) \\text{ (receive reward)}
          \\end{aligned}`}
        </BlockMath>
        <Text className="mb-2">
          States don't exist in this framework - only actions and their immediate rewards matter.
        </Text>
      </div>

      <div className="mb-8">
        <Text component="h3" className="text-xl font-bold mb-4">
          2. Contextual Bandit Paradigm
        </Text>
        <Text className="mb-2">
          At each time step t:
        </Text>
        <BlockMath>
          {`\\begin{aligned}
          s_t &\\sim P(s) \\text{ (observe state/context)} \\\\
          a_t &\\in \\mathcal{A} \\text{ (select action)} \\\\
          r_t &= R(s_t, a_t) \\text{ (receive reward)}
          \\end{aligned}`}
        </BlockMath>
        <Text className="mb-2">
          Each state is independently sampled - future states don't depend on current actions.
        </Text>
      </div>

      <div className="mb-8">
        <Text component="h3" className="text-xl font-bold mb-4">
          3. Full RL Paradigm
        </Text>
        <Text className="mb-2">
          At each time step t:
        </Text>
        <BlockMath>
          {`\\begin{aligned}
          s_t &\\text{ (observe current state)} \\\\
          a_t &\\in \\mathcal{A} \\text{ (select action)} \\\\
          r_t &= R(s_t, a_t) \\text{ (receive reward)} \\\\
          s_{t+1} &= f(s_t, a_t) \\text{ (state transitions based on action)}
          \\end{aligned}`}
        </BlockMath>
        <Text className="mb-2">
          Key distinction: The next state depends on the current state and action - creating a sequential decision process.
        </Text>
      </div>
    </section>
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
    <Container size="lg" className="py-8">
      <Title order={1} className="mb-6" id="rl-paradigms">
        Reinforcement Learning Paradigms
      </Title>

      <Stack spacing="xl">
        <section>
          <Title order={2} className="mb-4" id="mdp-connection">
          From MDP to RL 
          </Title>
          
          <Text className="mb-4">
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
        </section>

        <section>
          <Title order={2} className="mb-4" id="exploration-exploitation">
            The Exploration vs. Exploitation Dilemma
          </Title>

          <Text className="mb-4">
            Without a known model, agents must learn through interaction, leading to a fundamental 
            challenge: the exploration-exploitation dilemma. Should the agent exploit its current 
            knowledge to maximize immediate rewards, or explore to potentially discover better strategies?
          </Text>
          <div align="center">
          <Image
  src="/assets/module13/tikz_images_2/explore_vs_exploit.jpeg"
  alt="Exploration vs Exploitation trade-off"
      h={400}
      w="auto"
/></div>
        </section>

          <Title order={2} className="mb-4" id="bandit-vs-rl">
            From Bandits to RL
          </Title>

<BanditParadigms/>

        <section>
          <Title order={2} className="mb-4" id="model-based-vs-free">
            Model-Based vs Model-Free Approaches
          </Title>

          <Text className="mb-4">
            Two main paradigms have emerged for solving RL problems: model-based and model-free approaches.
            Each has its own advantages and trade-offs.
          </Text>
          <div align="center">
          <Image
            src="/assets/module13/model-based-free.jpg"
            alt="Model-based vs Model-free RL"
            className="my-4"
                  h={400}
      w="auto"
          />
          </div>
          <Text className="mb-4">
          <ul>
            <li>
            Model-Based: Learn or use an explicit model of the environment
            </li>
            <li>
            Model-Free: Learn directly from experience without building a model
            </li>
          </ul>
          </Text>
          <div align="center">
          <Image
            src="/assets/module13/tikz_images_2/tax.png"
            alt="Taxonomy of RL algorithms"
            className="my-4"
                  h={400}
      w="auto"
          />
          </div>
          <Text className="mb-4">
            It exists a big variety of algorithms.
          </Text>

        </section>

        <section>
          <Title order={2} className="mb-4" id="epsilon-greedy">
            Simple Exploration Strategy: ε-Greedy
          </Title>

          <Text className="mb-4">
            One of the simplest yet effective approaches to balance exploration and exploitation 
            is the ε-greedy strategy. With probability ε, the agent explores randomly; otherwise, 
            it exploits its current knowledge.
          </Text>

          <CodeBlock code={agentCode} language="python" />

        </section>
        {/* <section>
        <Title order={2} className="mb-4" id="epsilon-greedy">
          Implementation Q ε-Greedy Policy
        </Title>


<CodeBlock code={agentCode} language="python" />

</section> */}
      </Stack>
    </Container>
  );
};

export default RLParadigms;