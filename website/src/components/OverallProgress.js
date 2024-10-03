import React from "react";
import { Card, Text, Progress, Group, Box, Tooltip } from "@mantine/core";

const OverallProgress = ({ progress, errors }) => {
  return (
    <Card shadow="sm" p="lg" radius="md" withBorder mb="xl">
      <Card.Section withBorder inheritPadding py="xs">
        <Text fw={500}>Overall Progress</Text>
      </Card.Section>
      <Group justify="space-between" mt="md" mb="xs">
        <Text size="xl" fw={700}>{progress.toFixed(1)}% Complete</Text>
      </Group>
      <Tooltip
        label={`Progress: ${progress.toFixed(1)}%, Errors: ${errors.toFixed(1)}%`}
        position="top"
        withArrow
      >
        <Box>
          <Progress.Root size="xl">
            <Progress.Section value={progress} color="green"/>
            <Progress.Section value={errors} color="gray"/>
          </Progress.Root>
        </Box>
      </Tooltip>
      <Text c="dimmed" size="sm" mt="md">
        This is an overview of your current completion rate and errors across all modules. Keep up the good work!
      </Text>
    </Card>
  );
};

export default OverallProgress;