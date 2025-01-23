import React from 'react';
import { Title, Text, Container, Alert, Stack } from '@mantine/core';
import { Info } from 'lucide-react';
import GymnasiumGuide from './GymnasiumGuide';
import PettingZooGuide from './PettingZooGuide';
import RLFrameworks from './RLFrameworks';

const RLTrainingEfficiency = () => {

  // import scipy.signal
  // def discount_cumsum(x, discount):
  //     return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

  return (
    <Container size="xl" className="py-6">
      <Stack spacing="xl">
        <div>
          <Title order={1} id="rl-training-efficiency" className="mb-4">
            Reinforcement Learning Training Guide
          </Title>
          
          <Text className="mb-6">
            This comprehensive guide covers the fundamentals of training reinforcement learning
            agents using modern environments and frameworks. We'll explore standard environments,
            multi-agent setups, and popular RL frameworks to help you implement efficient
            training pipelines.
          </Text>

        </div>

        {/* Individual Guide Components */}
        <GymnasiumGuide />
        <PettingZooGuide />
        <RLFrameworks />
      </Stack>
    </Container>
  );
};

export default RLTrainingEfficiency;