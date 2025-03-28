import React from 'react';
import { Title, Text, Paper, Grid, List, Alert, Table, Badge, Divider, Anchor } from '@mantine/core';
import { Trophy, GitBranch, FileText, AlertCircle, Info } from 'lucide-react';
import { BlockMath, InlineMath } from "react-katex";
import "katex/dist/katex.min.css";

const ProjectPage = () => {
  return (
    <div className="max-w-6xl mx-auto p-6">
      <Title order={1} className="text-4xl font-bold mb-6" id="main-title">
        Project 2024 DeepRL Competition
      </Title>

      {/* Overview Section */}
      <Paper className="p-6 mb-8 bg-blue-50">
        <Text className="mb-6">
          Welcome to the Pong2024 Competition! This project challenges you to apply what your learned during the data science course. Working in teams of 1-3 members, you'll develop an AI agent 
          capable of playing Pong against other agents. This project emphasizes both theoretical understanding 
          and practical implementation of Machine Learning.
        </Text>
        <Alert 
      icon={<Info />}
      title="Note About Competition Choice"
      color="blue"
      radius="md"
    >
      <Text size="sm" mb="xs">
        If you found{' '}
        <Anchor href="https://www.raphaelcousin.com/module13/exercise/exercise4">
          Exercise 4 of Module 13
        </Anchor>{' '}
        particularly challenging, we encourage you to consider participating in the{' '}
        <Anchor href="https://ml-arena.com/viewcompetition/1">
          LunarLander competition
        </Anchor>{' '}
        instead of the Pong competition for your final project. LunarLander offers an excellent opportunity to demonstrate your reinforcement learning skills while being more approachable.
      </Text>
      <Text size="sm">
        The grading criteria and 30% performance metric will be adjusted proportionally based on the participation in each competition track.      </Text>
    </Alert>
        <Grid className="mb-4">
          <Grid.Col span={{ base: 12, md: 6 }}>
            <Paper className="p-4 h-full">
              <Title order={3} className="text-xl mb-2">Timeline</Title>
              <List>
                <List.Item><strong>February 4th:</strong> Competition opens - Begin team registration and environment setup</List.Item>
                <List.Item><strong>February 4th - March 31st:</strong> Development phase - Submit and iterate on your agents, with unlimited submissions for testing and improvement</List.Item>
                <List.Item><strong>March 31st, 23:59 (Paris Time):</strong> Final submission deadline - Performance evaluation and repository freeze</List.Item>
                <List.Item><strong>Early April:</strong> Project presentations - Showcase your methodology and results</List.Item>
              </List>
            </Paper>
          </Grid.Col>
          <Grid.Col span={{ base: 12, md: 6 }}>
            <Paper className="p-4 h-full">
              <Title order={3} className="text-xl mb-2">Grade Distribution</Title>
              <Text className="mb-4">The project comprises 70% of your Data Science practice mark, with the remaining 30% based on weekly session results. The project evaluation is equally divided into three key areas:</Text>
              <List>
                <List.Item><strong>Performance (33%):</strong> How well your agent performs against others and the benchmark</List.Item>
                <List.Item><strong>Code Quality & Reproducibility (33%):</strong> Clean, well-documented, and maintainable implementation</List.Item>
                <List.Item><strong>Presentation (33%):</strong> Clear communication of your approach and findings</List.Item>
              </List>
            </Paper>
          </Grid.Col>
        </Grid>
        
      </Paper>

      {/* Performance Section */}
      <Paper className="p-6 mb-8" withBorder>
        <Title order={2} className="text-2xl mb-4" id="performance">
          <Trophy className="inline-block mr-2" size={24} /> 1. Performance (10 points)
        </Title>
        
        <Text className="mb-4">
          Your agent's performance will be evaluated through the ML Arena platform, where it will compete against 
          other teams' agents. The evaluation system ensures fair comparison and rewards consistent performance 
          against various opponents.
        </Text>

        <List className="mb-6">
          <List.Item><strong>Platform:</strong> <a href="https://ml-arena.com" target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:text-blue-800">ML Arena</a> - Create an account and join the competition</List.Item>
          <List.Item><strong>Competition Link:</strong> Access the <a href="https://ml-arena.com/viewcompetition/3" target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:text-blue-800">Pong2024 Competition page</a> for submissions and rankings</List.Item>
          <List.Item><strong>Development Process:</strong> You can submit multiple agents during the development phase to test different strategies and improvements</List.Item>
        </List>

        <Paper className="p-4 bg-gray-50 mb-4">
          <Text className="mb-2">Your performance mark is calculated using the following formula:</Text>
          <BlockMath>
            {"\\text{mark} = 0.5 * \\frac{\\text{nb\\_users\\_above\\_MarkBench} - (\\text{your\\_rank} - 1)}{\\text{nb\\_users\\_above\\_MarkBench}}"}
          </BlockMath>
               </Paper>

        <List>
          <List.Item><strong>Below benchmark:</strong> 0 points - Failing to surpass the baseline agent "Mark Bench"</List.Item>
          <List.Item><strong>Above benchmark:</strong> Score between 5 and 10</List.Item>
          <List.Item><strong>Special Bonus:</strong> Additional 2 points for finishing top-ranked</List.Item>
        </List>
      </Paper>

      {/* Code Quality Section */}
      <Paper className="p-6 mb-8" withBorder>
        <Title order={2} className="text-2xl mb-4" id="code-quality">
          <GitBranch className="inline-block mr-2" size={24} /> 2. Code Quality & Reproducibility (10 points)
        </Title>

        <Text className="mb-4">
          Your code will be evaluated based on its quality, organization, and reproducibility. This emphasizes 
          the importance of software engineering practices in machine learning projects.
        </Text>

        <Grid className="mb-4">
          <Grid.Col span={{ base: 12, md: 6 }}>
            <Paper className="p-4 h-full">
              <Title order={3} className="text-xl mb-2">Repository Setup</Title>
              <List>
                <List.Item><strong>Repository Name:</strong> Create a private GitHub repository named "ML-Arena-Pong2024"</List.Item>
                <List.Item><strong>Collaboration:</strong> Add teammates and instructor (racousin) as collaborators</List.Item>
                <List.Item><strong>Documentation:</strong> Include comprehensive README with:
                  <List withPadding className="mt-2">
                    <List.Item>Environment setup instructions</List.Item>
                    <List.Item>Dependencies and requirements</List.Item>
                    <List.Item>Training and evaluation procedures</List.Item>
                  </List>
                </List.Item>
              </List>
            </Paper>
          </Grid.Col>
          <Grid.Col span={{ base: 12, md: 6 }}>
            <Paper className="p-4 h-full">
              <Title order={3} className="text-xl mb-2">Evaluation Criteria</Title>
              <List>
                <List.Item><strong>Reproducibility (0 to 4):</strong> Clear documentation, easy setup process, and reliable execution</List.Item>
                <List.Item><strong>Implementation of robust technics(0 to 6):</strong> example RL algorithm (PPO, SAC), RL parallele training, transfer learning</List.Item>
                <List.Item><strong>Git Workflow Bonus (+2):</strong> Demonstrate proper version control:
                  <List withPadding className="mt-2">
                    <List.Item>Minimum 2 feature branches</List.Item>
                    <List.Item>At least 2 reviewed pull requests</List.Item>
                    <List.Item>Meaningful commit messages</List.Item>
                  </List>
                </List.Item>
              </List>
            </Paper>
          </Grid.Col>
        </Grid>
      </Paper>

            {/* Presentation Section */}
      <Paper className="p-6 mb-8" withBorder>
        <Title order={2} className="text-2xl mb-4" id="presentation">
          <FileText className="inline-block mr-2" size={24} /> 3. Presentation (10 points)
        </Title>

        <Text className="mb-4">
          The presentation is your opportunity to showcase your team's journey, methodology, and insights. 
          Focus on clear communication of your technical approach and lessons learned.
        </Text>

        <Grid className="mb-4">
          <Grid.Col span={{ base: 12, md: 6 }}>
            <Paper className="p-4 h-full">
              <Title order={3} className="text-xl mb-2">Format & Requirements</Title>
              <List>
                <List.Item><strong>Duration:</strong> 15-minute presentation + 5-minute Q&A session</List.Item>
                <List.Item><strong>Format Freedom:</strong> Choose your preferred style:
                  <List withPadding className="mt-2">
                  <List.Item>PowerPoint/Slides presentation</List.Item>
                <List.Item>Live coding demonstration</List.Item>
                <List.Item>Interactive visualization</List.Item>
                <List.Item>Blackboard demonstration</List.Item>
                <List.Item>Hybrid approach</List.Item>
                  </List>
                </List.Item>
                <List.Item><strong>Content Coverage:</strong> Include:
                  <List withPadding className="mt-2">
                    <List.Item>Problem approach and methodology</List.Item>
                    <List.Item>Theorical support</List.Item>
                    <List.Item>Key challenges</List.Item>
                    <List.Item>Performance analysis</List.Item>
                    <List.Item>Future improvements</List.Item>
                  </List>
                </List.Item>
              </List>
            </Paper>
          </Grid.Col>
          <Grid.Col span={{ base: 12, md: 6 }}>
            <Paper className="p-4 h-full">
              <Title order={3} className="text-xl mb-2">Evaluation Criteria</Title>
              <List>
                <List.Item><strong>Clarity of Explanation (0 to 5):</strong>
                  <List withPadding className="mt-2">
                    <List.Item>Well-structured presentation flow</List.Item>
                    <List.Item>Clear technical explanations</List.Item>
                    <List.Item>Pedagogic Content</List.Item>
                  </List>
                </List.Item>
                <List.Item><strong>Creativity & Theoretical Approach (0 to 5):</strong>
                  <List withPadding className="mt-2">
                    <List.Item>Innovation in solution approach</List.Item>
                    <List.Item>Depth of technical and theoretical understanding</List.Item>
                    <List.Item>Quality of analysis and insights</List.Item>
                  </List>
                </List.Item>
                <List.Item><strong>Bonus: "Wow" Factor (+2):</strong>
                  <List withPadding className="mt-2">
                    <List.Item>Show a wow live Demo (that works!)</List.Item>
                  </List>
                </List.Item>
              </List>
            </Paper>
          </Grid.Col>
        </Grid>
      </Paper>

      {/* Summary Table */}
      <Paper className="p-6 mb-8" withBorder>
        <Title order={2} className="text-2xl mb-4">Summary of Evaluation</Title>
        <Table>
          <Table.Thead>
            <Table.Tr>
              <Table.Th>Category</Table.Th>
              <Table.Th>Criteria</Table.Th>
              <Table.Th>Points</Table.Th>
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            <Table.Tr>
              <Table.Td rowSpan={3}>Performance</Table.Td>
              <Table.Td>Below benchmark</Table.Td>
              <Table.Td>0</Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td>Above benchmark</Table.Td>
              <Table.Td>5 - 10</Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td>Bonus: finish top-ranked</Table.Td>
              <Table.Td>+2</Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td rowSpan={3}>Code</Table.Td>
              <Table.Td>Reproducibility</Table.Td>
              <Table.Td>0 - 5</Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td>Implementation Quality</Table.Td>
              <Table.Td>0 - 5</Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td>Bonus: Git workflow</Table.Td>
              <Table.Td>+2</Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td rowSpan={3}>Presentation</Table.Td>
              <Table.Td>Clarity of Explanation</Table.Td>
              <Table.Td>0 - 5</Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td>Creativity & Theory</Table.Td>
              <Table.Td>0 - 5</Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td>Bonus: "Wow" effect</Table.Td>
              <Table.Td>+2</Table.Td>
            </Table.Tr>
          </Table.Tbody>
        </Table>
      </Paper>

      {/* Next Steps */}
      <Paper className="p-6" withBorder>
        <Title order={2} className="text-2xl mb-4">Next Steps</Title>
        <Text className="mb-4">
          To get started with your project, follow these key steps in order:
        </Text>
        <List>
          <List.Item><strong>Team Formation:</strong> Create a team of 2-3 members and register on ML Arena:
            <List withPadding className="mt-2">
            <List.Item>Visit <a href="https://ml-arena.com/viewcompetition/3" target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:text-blue-800">ml-arena.com/viewcompetition/3</a></List.Item>
              <List.Item>Create your team account</List.Item>
              <List.Item>Invite team members</List.Item>
            </List>
          </List.Item>
          <List.Item><strong>Repository Setup:</strong> Initialize your GitHub workspace:
            <List withPadding className="mt-2">
              <List.Item>Create private repo: ML-Arena-Pong2024</List.Item>
              <List.Item>Add collaborators (teammates + racousin)</List.Item>
              <List.Item>Set up initial project structure</List.Item>
            </List>
          </List.Item>
          <List.Item><strong>Development:</strong> Begin your agent development:
            <List withPadding className="mt-2">
              <List.Item>Grab some info from <a href="https://github.com/ml-arena/pong2024" target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:text-blue-800">Pong2024 repository</a></List.Item>
              <List.Item>Follow the <a href="https://colab.research.google.com/github/ml-arena/pong2024/blob/main/notebook/getting_started.ipynb" target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:text-blue-800">Getting Started colab guide</a></List.Item>
              <List.Item>Implement initial agent version</List.Item>
              <List.Item>Test and iterate based on performance</List.Item>
            </List>
          </List.Item>
          <List.Item><strong>Documentation & Presentation:</strong> Prepare for final evaluation:
            <List withPadding className="mt-2">
              <List.Item>Document your methodology and findings</List.Item>
              <List.Item>Create presentation materials</List.Item>
              <List.Item>Practice your presentation</List.Item>
            </List>
          </List.Item>
        </List>
        
        <Alert variant="light" color="blue" className="mt-6">
          <Title order={4} className="mb-2">Need Help?</Title>
          <Text>For technical support or questions about the competition, contact raphaelcousin.teaching@gmail.com</Text>
          <Text className="mt-2">Good luck with your project! 🚀</Text>
        </Alert>
      </Paper>
    </div>
  );
};

export default ProjectPage;