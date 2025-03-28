import React from 'react';
import { Text, Title, Paper, Grid, List, ThemeIcon } from '@mantine/core';
import { Database, Activity, Clock } from 'lucide-react';
import CodeBlock from "components/CodeBlock";

const BatchVsStreaming = () => {
  return (
    <div>
      <Title order={1}>Batch vs. Streaming Data Collection</Title>
      
      <Text mt="md">
        Data collection in data science follows two main patterns: batch processing and stream processing. 
        Understanding when to use each approach is crucial for effective data collection strategies.
      </Text>

      <Grid mt="xl" gutter="lg">
        <Grid.Col span={6}>
          <Paper p="lg" radius="md" className="bg-slate-50">
            <Title order={2} mb="md" className="flex items-center gap-2">
              <Database size={24} className="text-blue-600" />
              Batch Collection
            </Title>
            <List spacing="sm">
              <List.Item><strong>When:</strong> Data is processed at scheduled intervals</List.Item>
              <List.Item><strong>Volume:</strong> Handles large amounts of data efficiently</List.Item>
              <List.Item><strong>Use Case:</strong> Historical analysis, daily reports</List.Item>
              <List.Item><strong>Example:</strong> End-of-day financial data processing</List.Item>
            </List>
          </Paper>
        </Grid.Col>
        
        <Grid.Col span={6}>
          <Paper p="lg" radius="md" className="bg-slate-50">
            <Title order={2} mb="md" className="flex items-center gap-2">
              <Activity size={24} className="text-green-600" />
              Stream Collection
            </Title>
            <List spacing="sm">
              <List.Item><strong>When:</strong> Data is processed as it arrives</List.Item>
              <List.Item><strong>Volume:</strong> Handles continuous flow of small updates</List.Item>
              <List.Item><strong>Use Case:</strong> Real-time monitoring, live updates</List.Item>
              <List.Item><strong>Example:</strong> Social media sentiment analysis</List.Item>
            </List>
          </Paper>
        </Grid.Col>
      </Grid>

      <Paper p="lg" mt="xl" radius="md" className="bg-slate-50">
        <Title order={2} className="flex items-center gap-2 mb-md">
          <Clock size={24} className="text-violet-600" />
          When to Use Each
        </Title>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <Title order={3} size="h4" mb="xs">Choose Batch When:</Title>
            <List>
              <List.Item>Processing large volumes of historical data</List.Item>
              <List.Item>Running complex aggregations</List.Item>
              <List.Item>Resource efficiency is priority</List.Item>
              <List.Item>Real-time updates aren't required</List.Item>
            </List>
          </div>
          <div>
            <Title order={3} size="h4" mb="xs">Choose Streaming When:</Title>
            <List>
              <List.Item>Immediate data processing is needed</List.Item>
              <List.Item>Monitoring real-time metrics</List.Item>
              <List.Item>Detecting patterns as they occur</List.Item>
              <List.Item>Building interactive dashboards</List.Item>
            </List>
          </div>
        </div>
      </Paper>
    </div>
  );
};

export default BatchVsStreaming;