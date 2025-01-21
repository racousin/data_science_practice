import React from 'react';
import { Title, Text, Container, Alert, Divider }  from '@mantine/core';
import { Info, AlertTriangle } from 'lucide-react';
import CodeBlock from 'components/CodeBlock';
const AdvancedPettingZooTraining = () => {
  return (
<>
      <Title order={1} className="mb-6">Advanced Multi-Opponent Training in PettingZoo</Title>
      
      <Text className="mb-6 text-lg">
        This guide covers advanced techniques for training agents against multiple opponents
        in parallel using PettingZoo. We'll explore environment wrapping, parallel execution,
        and curriculum learning strategies.
      </Text>

      <Title order={2} id="environment-wrapper" className="mb-4 mt-8">MultiOpponentEnv Wrapper</Title>
      
      <Text className="mb-4">
        The MultiOpponentEnv wrapper manages multiple opponents and their interactions within
        a single environment instance. It handles policy selection, execution order, and
        state management.
      </Text>

      <CodeBlock language="python" code={`
from pettingzoo import AECEnv
from typing import Dict, List, Callable
import numpy as np

class MultiOpponentEnv:
    """Environment wrapper for managing multiple opponents"""
    def __init__(self, 
                 env_creator: Callable[[], AECEnv],
                 opponent_policies: Dict[str, Callable],
                 our_agent_id: str = "player_0"):
        self.env = env_creator()
        self.opponent_policies = opponent_policies
        self.our_agent_id = our_agent_id
        self.opponent_rewards = {opponent: 0 for opponent in opponent_policies.keys()}
        
    def reset(self):
        """Reset environment and all opponent states"""
        observations = self.env.reset()
        self.opponent_rewards = {opponent: 0 for opponent in self.opponent_policies.keys()}
        return observations[self.our_agent_id]
        
    def step(self, our_action):
        """Execute one step of the environment"""
        current_rewards = {}
        
        # Complete one full cycle of the environment
        for agent in self.env.agent_iter():
            if agent == self.our_agent_id:
                # Execute our agent's action
                obs, rew, term, trunc, info = self.env.step(our_action)
                current_rewards[agent] = rew
            else:
                # Get observation and execute opponent policy
                obs, rew, term, trunc, info = self.env.last()
                
                if term or trunc:
                    action = None
                else:
                    # Get appropriate opponent policy and execute
                    policy = self.opponent_policies[agent]
                    action = policy(obs)
                    
                obs, rew, term, trunc, info = self.env.step(action)
                current_rewards[agent] = rew
                self.opponent_rewards[agent] += rew
        
        # Return state from our agent's perspective
        final_obs = self.env.observe(self.our_agent_id)
        return final_obs, current_rewards[self.our_agent_id], term, trunc, info`} />

      <Alert className="my-6 bg-blue-50">
        <div className="flex items-start">
          <Info className="w-5 h-5 mt-1 mr-2" />
          <div>
            <Text className="font-medium mb-2">Key Features</Text>
            <Text>
              - Maintains separate reward tracking for each opponent
              - Handles proper turn ordering in the AEC environment
              - Provides clean interface for policy execution
            </Text>
          </div>
        </div>
      </Alert>

      <Title order={2} id="parallel-setup" className="mb-4 mt-8">Parallel Environment Setup</Title>
      
      <CodeBlock language="python" code={`
from gymnasium.vector import AsyncVectorEnv
import torch

def create_parallel_environments(
    num_envs: int,
    env_creator: Callable,
    opponent_policies: Dict[str, List[Callable]]
) -> AsyncVectorEnv:
    """Create multiple environments with different opponent combinations"""
    
    def make_env(idx: int):
        def _init():
            # Select subset of opponents for this environment
            env_opponents = {
                f"opponent_{i}": opponent_policies[i % len(opponent_policies)]
                for i in range(idx, idx + 3)  # Each env has 3 opponents
            }
            
            return MultiOpponentEnv(
                env_creator=env_creator,
                opponent_policies=env_opponents
            )
        return _init
    
    # Create vector of environments
    return AsyncVectorEnv([make_env(i) for i in range(num_envs)])

# Example usage with different opponent types
def create_diverse_opponent_set():
    return {
        "random": lambda obs: np.random.randint(4),
        "heuristic": lambda obs: simple_heuristic_policy(obs),
        "defensive": lambda obs: defensive_policy(obs),
        "aggressive": lambda obs: aggressive_policy(obs),
        "champion": lambda obs: pretrained_champion_policy(obs)
    }`} />

      <Title order={2} id="curriculum-learning" className="mb-4 mt-8">Advanced Training Features</Title>

      <CodeBlock language="python" code={`
class CurriculumTrainer:
    """Manages curriculum-based training against multiple opponents"""
    def __init__(self, 
                 agent,
                 env_creator,
                 opponent_policies,
                 num_envs=4):
        self.agent = agent
        self.env_creator = env_creator
        self.base_policies = opponent_policies
        self.num_envs = num_envs
        self.current_stage = 0
        self.stages = self._create_curriculum_stages()
        
    def _create_curriculum_stages(self):
        """Create progressively harder training stages"""
        return [
            {  # Stage 1: Basic opponents
                "policies": {
                    "random": self.base_policies["random"],
                    "simple": self.base_policies["heuristic"]
                },
                "threshold": 0.6  # Win rate to progress
            },
            {  # Stage 2: Intermediate opponents
                "policies": {
                    "defensive": self.base_policies["defensive"],
                    "aggressive": self.base_policies["aggressive"]
                },
                "threshold": 0.5
            },
            {  # Stage 3: Advanced opponents
                "policies": {
                    "champion": self.base_policies["champion"],
                    "mixed": self._create_mixed_policy()
                },
                "threshold": 0.4
            }
        ]
    
    def _create_mixed_policy(self):
        """Create a policy that randomly selects from other policies"""
        policies = list(self.base_policies.values())
        return lambda obs: np.random.choice(policies)(obs)
        
    def train_epoch(self, num_steps=1000):
        """Train for one epoch at current curriculum stage"""
        stage = self.stages[self.current_stage]
        envs = create_parallel_environments(
            self.num_envs,
            self.env_creator,
            stage["policies"]
        )
        
        observations = envs.reset()
        episode_rewards = np.zeros(self.num_envs)
        
        for step in range(num_steps):
            actions = self.agent.get_actions(observations)
            next_obs, rewards, terms, truncs, infos = envs.step(actions)
            
            # Update episodes
            episode_rewards += rewards
            
            # Handle episode termination
            for i, (term, trunc) in enumerate(zip(terms, truncs)):
                if term or trunc:
                    self.log_episode(episode_rewards[i])
                    episode_rewards[i] = 0
            
            # Update agent
            self.agent.update(observations, actions, rewards, next_obs, terms)
            observations = next_obs
        
        # Check for curriculum progression
        self.check_progression()
    
    def check_progression(self):
        """Check if agent is ready to progress to next stage"""
        if self.get_win_rate() > self.stages[self.current_stage]["threshold"]:
            self.current_stage = min(self.current_stage + 1, len(self.stages) - 1)
            print(f"Progressing to curriculum stage {self.current_stage + 1}")
`} />

      <Alert className="my-6 bg-yellow-50">
        <div className="flex items-start">
          <AlertTriangle className="w-5 h-5 mt-1 mr-2" />
          <div>
            <Text className="font-medium mb-2">Training Considerations</Text>
            <Text>
              - Monitor win rates against each opponent type separately
              - Adjust curriculum thresholds based on task difficulty
              - Consider implementing early stopping for efficient training
            </Text>
          </div>
        </div>
      </Alert>

      <Title order={2} id="best-practices" className="mb-4 mt-8">Best Practices and Configuration</Title>

      <CodeBlock language="python" code={`
class TrainingConfig:
    """Configuration for multi-opponent training"""
    def __init__(self):
        self.num_envs = 4
        self.batch_size = 256
        self.learning_rate = 3e-4
        self.gamma = 0.99
        self.update_interval = 100
        self.eval_interval = 1000
        
        # Curriculum settings
        self.curriculum_stages = [
            {"opponent_count": 2, "min_win_rate": 0.6},
            {"opponent_count": 4, "min_win_rate": 0.5},
            {"opponent_count": 6, "min_win_rate": 0.4}
        ]
        
        # Opponent mixing probabilities
        self.opponent_probs = {
            "random": 0.2,
            "heuristic": 0.3,
            "defensive": 0.2,
            "aggressive": 0.2,
            "champion": 0.1
        }

def create_training_session(config: TrainingConfig):
    """Set up a complete training session with best practices"""
    # Initialize environments with proper error handling
    try:
        trainer = CurriculumTrainer(
            agent=create_agent(config),
            env_creator=create_env_with_monitoring,
            opponent_policies=create_diverse_opponent_set(),
            num_envs=config.num_envs
        )
    except Exception as e:
        print(f"Error setting up training: {e}")
        return None
        
    # Set up logging and monitoring
    setup_tensorboard_logging(trainer)
    setup_opponent_tracking(trainer)
    
    return trainer`} />

      <Text className="mt-6 mb-4">
        Following these practices ensures robust training while maintaining code clarity and
        debugging capabilities. The modular structure allows for easy experimentation with
        different opponent combinations and training strategies.
      </Text>
      </>
  );
};

export default AdvancedPettingZooTraining;