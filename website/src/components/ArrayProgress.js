import React from "react";
import { Progress, Text, Box } from '@mantine/core';

const ArrayProgress = ({ progressPercent }) => {
  const getProgressColor = () => {
    if (progressPercent < 33) return "red";
    if (progressPercent < 66) return "yellow";
    return "green";
  };

  return (
    <Box style={{ width: 200 }}>
      <Progress
        value={progressPercent}
        color={getProgressColor()}
        size="xl"
        radius="xl"
      />
      <Text align="center" size="sm" mt={5}>
        {progressPercent.toFixed(0)}%
      </Text>
    </Box>
  );
};

export default ArrayProgress;