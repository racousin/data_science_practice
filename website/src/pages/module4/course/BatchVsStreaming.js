import React from 'react';
import { Text, Title, Paper, Grid, List, ThemeIcon } from '@mantine/core';
import { BarChart, Activity, Database, Clock } from 'lucide-react';
import CodeBlock from "components/CodeBlock";

const BatchVsStreaming = () => {
  return (
    <div>
      <Title order={1}>Batch vs. Streaming Data Collection</Title>
      
      <Text mt="md">
        In data science and engineering, there are two primary approaches to collecting and processing data: batch processing and stream processing. Each method has its own advantages and use cases.
      </Text>

      <Grid mt="xl">
        <Grid.Col span={6}>
          <Paper withBorder p="md">
            <Title order={2} mb="sm">
              <ThemeIcon size={30} radius="md" variant="light" color="blue" mr="xs">
                <Database size={20} />
              </ThemeIcon>
              Batch Collection
            </Title>
            <Text>
              Batch processing involves collecting and processing data in large, scheduled batches.
            </Text>
            <List mt="sm">
              <List.Item>Processed at regular intervals (e.g., daily, hourly)</List.Item>
              <List.Item>Suitable for historical analysis</List.Item>
              <List.Item>Typically involves larger datasets</List.Item>
            </List>
          </Paper>
        </Grid.Col>
        
        <Grid.Col span={6}>
          <Paper withBorder p="md">
            <Title order={2} mb="sm">
              <ThemeIcon size={30} radius="md" variant="light" color="green" mr="xs">
                <Activity size={20} />
              </ThemeIcon>
              Streaming/Event-based Collection
            </Title>
            <Text>
              Stream processing involves collecting and processing data in real-time or near real-time as it becomes available.
            </Text>
            <List mt="sm">
              <List.Item>Processed continuously as data arrives</List.Item>
              <List.Item>Enables immediate analysis and action</List.Item>
              <List.Item>Ideal for real-time applications</List.Item>
            </List>
          </Paper>
        </Grid.Col>
      </Grid>

      <Title order={2} mt="xl">Queues in Event-based Systems</Title>
      <Text mt="md">
        In streaming architectures, message queues play a crucial role in managing data flow. Tools like Apache Kafka or RabbitMQ are commonly used to store and deliver messages between services.
      </Text>
      <CodeBlock
        language="python"
        code=
{`# Example: Using Kafka consumer in Python
from kafka import KafkaConsumer

consumer = KafkaConsumer('my_topic',
                         bootstrap_servers=['localhost:9092'],
                         auto_offset_reset='earliest',
                         enable_auto_commit=True,
                         group_id='my-group')

for message in consumer:
    print(f"Received: {message.value.decode('utf-8')}")
`}
      />

      <Title order={2} mt="xl">Choosing Between Batch and Streaming</Title>
      <Paper withBorder p="md" mt="md">
        <List spacing="sm">
          <List.Item icon={
            <ThemeIcon color="blue" size={24} radius="xl">
              <Clock size={16} />
            </ThemeIcon>
          }>
            <Text weight={700}>Data Latency Requirements:</Text> If you need real-time insights, streaming is preferred. For historical analysis, batch processing works well.
          </List.Item>
          <List.Item icon={
            <ThemeIcon color="green" size={24} radius="xl">
              <Database size={16} />
            </ThemeIcon>
          }>
            <Text weight={700}>Data Volume:</Text> Batch processing can handle larger volumes of data more efficiently, while streaming is better for continuous, smaller chunks of data.
          </List.Item>
          <List.Item icon={
            <ThemeIcon color="orange" size={24} radius="xl">
              <BarChart size={16} />
            </ThemeIcon>
          }>
            <Text weight={700}>Processing Complexity:</Text> Complex computations might be more suited to batch processing, while simpler, incremental computations work well with streaming.
          </List.Item>
        </List>
      </Paper>
    </div>
  );
};

export default BatchVsStreaming;