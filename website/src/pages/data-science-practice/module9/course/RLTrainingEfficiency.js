import React from 'react';
import { Title, Text, Container } from '@mantine/core';
import GymnasiumGuide from './GymnasiumGuide';
import PettingZooGuide from './PettingZooGuide';
import RLFrameworks from './RLFrameworks';

const RLTrainingEfficiency = () => {
  return (
    <Container fluid>
      <div data-slide>
        <Title order={2} id="rl-training-efficiency" mb="md">
          Reinforcement Learning Training Guide
        </Title>

        <Text mb="md">
          This comprehensive guide covers the fundamentals of training reinforcement learning
          agents using modern environments and frameworks. We'll explore standard environments,
          multi-agent setups, and popular RL frameworks to help you implement efficient
          training pipelines.
        </Text>
      </div>

      <GymnasiumGuide />
      <PettingZooGuide />
      <RLFrameworks />
    </Container>
  );
};

export default RLTrainingEfficiency;