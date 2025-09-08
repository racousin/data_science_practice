import React from 'react';
import { Title, Text, Stack, Anchor } from '@mantine/core';
import CodeBlock from 'components/CodeBlock';
import AdvancedPettingZooTraining from './AdvancedPettingZooTraining'

const PettingZooGuide = () => {
  return (
    <Stack spacing="md">
      <Title order={2} id="pettingzoo-environments" className="mb-2">
        Multi-Agent Training with PettingZoo
      </Title>

      <Text>
        <Anchor href="https://pettingzoo.farama.org/" target="_blank" rel="noopener noreferrer">
          PettingZoo
        </Anchor> extends the Gymnasium API for multi-agent environments. This guide focuses on the
        Agent-Environment-Cycle (AEC) API, which provides precise control over agent interactions.
      </Text>

      <Title order={3} className="mb-2">Installation</Title>
      <CodeBlock language="bash" code={`
# Base installation
pip install pettingzoo

# Install specific environment dependencies
pip install pettingzoo[atari]     # For Atari multi-agent environments
pip install pettingzoo[butterfly] # For Butterfly environments
pip install pettingzoo[mpe]       # For Multi-particle environments`} />

      <Title order={3} className="mb-2">AEC API Basics</Title>
      <Text className="mb-4">
        The AEC API enforces a strict turn-based execution where agents act in a cyclic order:
      </Text>

      <CodeBlock language="python" code={`
from pettingzoo.butterfly import knights_archers_zombies_v10

# Create the environment
env = knights_archers_zombies_v10.env()

# Reset the environment
env.reset()

# AEC environment loop
for agent in env.agent_iter():
    # Get observation for current agent
    observation, reward, termination, truncation, info = env.last()
    
    # Check if the agent is done
    if termination or truncation:
        action = None
    else:
        # Get action for current agent (using observation)
        action = env.action_space(agent).sample()
    
    # Environment step for current agent
    env.step(action)

env.close()`} />
 <AdvancedPettingZooTraining/>

    </Stack>
   );
};

export default PettingZooGuide;