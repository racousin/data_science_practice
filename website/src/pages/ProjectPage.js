import React from 'react';
import { Title, Text, Timeline, List, ThemeIcon, Paper, Grid, Alert } from '@mantine/core';
import { Calendar, Github, Award, Presentation } from 'lucide-react';

const ProjectPage = () => {
  return (
    <div className="max-w-4xl mx-auto p-6">
      <Title order={1} className="text-3xl font-bold mb-6">
        Reinforcement Learning Project: Multiplayer Pong Competition
      </Title>

      <Text size="lg" className="mb-8">
        Develop an intelligent agent for playing multiplayer Pong using reinforcement learning techniques. 
        This project combines practical implementation, documentation, and presentation skills in a 
        competitive environment.
      </Text>

      <Paper className="p-6 mb-8 bg-blue-50">
        <Title order={2} className="text-xl mb-4">Quick Overview</Title>
        <Grid>
          <Grid.Col span={{ base: 12, md: 4 }}>
            <Text fw={600}>Team Size</Text>
            <Text>2-3 students</Text>
          </Grid.Col>
          <Grid.Col span={{ base: 12, md: 4 }}>
            <Text fw={600}>Platform</Text>
            <Text>ML-Arena.com (PettingZoo 2-player)</Text>
          </Grid.Col>
          <Grid.Col span={{ base: 12, md: 4 }}>
            <Text fw={600}>Competition Period</Text>
            <Text>Jan 22 - Mar 31, 2024</Text>
          </Grid.Col>
        </Grid>
      </Paper>

      <div className="mb-8">
        <Title order={2} className="text-2xl mb-4">Project Timeline</Title>
        <Timeline active={1}>
          <Timeline.Item bullet={<ThemeIcon size="sm"><Calendar className="w-4 h-4" /></ThemeIcon>} title="Project Start">
            <Text fw={600}>January 22, 2024</Text>
            <Text>Competition opens, team formation begins</Text>
          </Timeline.Item>
          
          <Timeline.Item bullet={<ThemeIcon size="sm"><Github className="w-4 h-4" /></ThemeIcon>} title="Development Phase">
            <Text fw={600}>January - March 2024</Text>
            <Text>Agent development and iterative improvement</Text>
          </Timeline.Item>
          
          <Timeline.Item bullet={<ThemeIcon size="sm"><Award className="w-4 h-4" /></ThemeIcon>} title="Competition End">
            <Text fw={600}>March 31, 2024</Text>
            <Text>Final agent submissions and performance evaluation</Text>
          </Timeline.Item>
          
          <Timeline.Item bullet={<ThemeIcon size="sm"><Presentation className="w-4 h-4" /></ThemeIcon>} title="Presentations">
            <Text fw={600}>April 2024</Text>
            <Text>Project presentations and final evaluation</Text>
          </Timeline.Item>
        </Timeline>
      </div>

      <div className="mb-8">
        <Title order={2} className="text-2xl mb-4">Evaluation Criteria (100%)</Title>
        
        <div className="space-y-6">
          <Paper className="p-4 bg-green-50">
            <Title order={3} className="text-xl mb-2">Performance (33%)</Title>
            <Text>
              Leaderboard ranking determines score:
            </Text>
            <List className="mt-2">
              <List.Item>Top performer: 100%</List.Item>
              <List.Item>Linear scale down to 50% for last place above benchmark</List.Item>
              <List.Item>Below benchmark: 0%</List.Item>
            </List>
          </Paper>

          <Paper className="p-4 bg-yellow-50">
            <Title order={3} className="text-xl mb-2">GitHub Repository (33%)</Title>
            <List>
              <List.Item>Contributor participation (10%)</List.Item>
              <List.Item>Pull request & review process (10%)</List.Item>
              <List.Item>Repository structure (10%)</List.Item>
              <List.Item>Code organization (10%)</List.Item>
              <List.Item>Documentation quality (10%)</List.Item>
              <List.Item>Installation & deployment (10%)</List.Item>
            </List>
          </Paper>

          <Paper className="p-4 bg-purple-50">
            <Title order={3} className="text-xl mb-2">Presentation (33%)</Title>
            <Text className="mb-2">
              15-minute presentation + 5-minute Q&A session
            </Text>
            <List>
              <List.Item>Presentation clarity (20%)</List.Item>
              <List.Item>Creative approach (20%)</List.Item>
              <List.Item>Technical depth (20%)</List.Item>
              <List.Item>Teamwork reflection (20%)</List.Item>
              <List.Item>Future improvements (20%)</List.Item>
            </List>
          </Paper>
        </div>
      </div>

      <div className="mb-8">
        <Title order={2} className="text-2xl mb-4">Getting Started</Title>
        
        <div className="space-y-4">
          <Paper className="p-4">
            <Title order={3} className="text-xl mb-2">ML-Arena Setup</Title>
            <List>
              <List.Item>Create an account on ml-arena.com</List.Item>
              <List.Item>Form a team (2-3 members)</List.Item>
              <List.Item>Submit your agent through the platform</List.Item>
            </List>
            <Alert variant="light" className="mt-2">
              Platform is in development. Contact raphael.cousin.education@gmail.com for support.
            </Alert>
          </Paper>

          <Paper className="p-4">
            <Title order={3} className="text-xl mb-2">GitHub Setup</Title>
            <List>
              <List.Item>Create a private repository</List.Item>
              <List.Item>Add team members as collaborators</List.Item>
              <List.Item>Begin agent development</List.Item>
            </List>
          </Paper>
        </div>
      </div>

      <Alert variant="light" className="mt-4">
        Feel free to use any tools and approaches in your implementation. Creativity is encouraged!
      </Alert>
    </div>
  );
};

export default ProjectPage;