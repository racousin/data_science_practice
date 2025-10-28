import React from 'react';
import { Paper, Title, Text, Group, Alert, Table, Stack } from '@mantine/core';
import { IconTrophy, IconAlertCircle } from "@tabler/icons-react";

const ProjectEvaluation = () => {
  return (
    <Paper p="xl" mt="xl" withBorder>
      <Group mb="xl">
        <IconTrophy size={28} />
        <Title order={2}>Project Evaluation</Title>
      </Group>
      <Text size="sm" mb="md">
        The project grade constitutes <strong>50% of the total course grade</strong>.
        It is evaluated on two main components: ML-Arena performance and GitHub package quality.
      </Text>

      <Alert icon={<IconAlertCircle />} color="blue" mb="xl">
        <Text weight={600} mb="xs">Note on AI-Assisted Development</Text>
        <Text size="sm">
          Using chatbots and AI coding assistants is allowed and encouraged. However,
          <strong> you must read and understand all generated code</strong>. Unnecessary boilerplate,
          over-engineered patterns, verbose documentation, or unused functions typical of AI-generated
          content will be counted as <strong>negative value</strong>. Quality over quantity—ensure
          every line serves a purpose.
        </Text>
      </Alert>
      <Stack spacing="xl">
        {/* Component 1: ML-Arena Performance */}
        <div>
          <Group mb="md">
            <Title order={3}>1. ML-Arena Competition Performance</Title>
          </Group>

          <Table withBorder withColumnBorders mb="md">
            <thead>
              <tr>
                <th>Criterion</th>
                <th style={{width: '120px', textAlign: 'center'}}>Points</th>
                <th style={{width: '120px', textAlign: 'center'}}>Score</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>
                  <Text weight={500}>Leaderboard Ranking (by user in class + Mark Bench)</Text>
                  <Text size="sm" c="dimmed">
                    Top: 30pts | Top 10%: 25pts | Top 25%: 20pts | Top 50%: 15pts | Top 75%: 10pts
                  </Text>
                </td>
                <td style={{textAlign: 'center'}}>30</td>
                <td style={{textAlign: 'center'}}>
                  <Text c="dimmed">TBD</Text>
                </td>
              </tr>
              <tr>
                <td>
                  <Text weight={500}>Performance Score</Text>
                  <Text size="sm" c="dimmed">
                    Absolute performance: Above 99% accuracy / 300 rewards: 20pts | Beats baseline algorithm (named Mark Bench): 15pts | Between benchmark and 98% accuracy (300-200 rewards): 10pts | Between 98% and 97% accuracy (200-100 rewards): 5pts
                  </Text>
                </td>
                <td style={{textAlign: 'center'}}>20</td>
                <td style={{textAlign: 'center'}}>
                  <Text c="dimmed">TBD</Text>
                </td>
              </tr>
            </tbody>
            <tfoot>
              <tr>
                <td><strong>Subtotal</strong></td>
                <td style={{textAlign: 'center'}}><strong>50</strong></td>
                <td style={{textAlign: 'center'}}>
                  <Text weight={700} c="dimmed">TBD</Text>
                </td>
              </tr>
            </tfoot>
          </Table>
        </div>

        {/* Component 2: GitHub Package */}
        <div>
          <Group mb="md">
            <Title order={3}>2. GitHub Package Quality</Title>
          </Group>

          <Table withBorder withColumnBorders mb="md">
            <thead>
              <tr>
                <th>Criterion</th>
                <th style={{width: '120px', textAlign: 'center'}}>Points</th>
                <th style={{width: '120px', textAlign: 'center'}}>Score</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>
                  <Text weight={500}>Code Quality & Structure</Text>
                  <Text size="sm" c="dimmed">
                    Package installable via pip/project.toml (3pts) | Clear functions &lt;50 lines, no duplication (2pts) |
                    Clear separation: agents/evaluation/utils/... modules (2pts) | No unused imports/functions (2pt) | Syntax and variables are meaningful (1pt)
                  </Text>
                </td>
                <td style={{textAlign: 'center'}}>10</td>
                <td style={{textAlign: 'center'}}>
                  <Text c="dimmed">TBD</Text>
                </td>
              </tr>
              <tr>
                <td>
                  <Text weight={500}>Evaluation Package</Text>
                  <Text size="sm" c="dimmed">
                    Evaluation and metrics implementations (3pts) | Performance comparison table/plot (2pts) |
                    Resources: Memory/time tracking system (2pts) | Easy Reproducibility (2pts) |
                    Save/load trained agents/model (1pt)
                  </Text>
                </td>
                <td style={{textAlign: 'center'}}>10</td>
                <td style={{textAlign: 'center'}}>
                  <Text c="dimmed">TBD</Text>
                </td>
              </tr>
              <tr>
                <td>
                  <Text weight={500}>Benchmark Algorithms</Text>
                  <Text size="sm" c="dimmed">
                    Distinct algorithms implemented (5pts) | Documented hyperparameter choices (3pts) | Failed experiments documented (2pts)
                  </Text>
                </td>
                <td style={{textAlign: 'center'}}>10</td>
                <td style={{textAlign: 'center'}}>
                  <Text c="dimmed">TBD</Text>
                </td>
              </tr>
              <tr>
                <td>
                  <Text weight={500}>Resume Notebook (report.ipynb)</Text>
                  <Text size="sm" c="dimmed">
                    Problem statement and methodology (2pts) | Results table (3pts) |
                    Performance evolution plots (2pts) | Reproduces best submission (3pts) |
                    Best agent name clearly stated (2pts) | Failure analysis and next steps (3pts)
                  </Text>
                </td>
                <td style={{textAlign: 'center'}}>15</td>
                <td style={{textAlign: 'center'}}>
                  <Text c="dimmed">TBD</Text>
                </td>
              </tr>
              <tr>
                <td>
                  <Text weight={500}>README & Documentation</Text>
                  <Text size="sm" c="dimmed">
                    Installation works in one command (2pts) | Training/evaluation example commands (1pt) |
                    Repository structure diagram (1pt) | Dependencies list with versions (1pt)
                  </Text>
                </td>
                <td style={{textAlign: 'center'}}>5</td>
                <td style={{textAlign: 'center'}}>
                  <Text c="dimmed">TBD</Text>
                </td>
              </tr>
            </tbody>
            <tfoot>
              <tr>
                <td><strong>Subtotal</strong></td>
                <td style={{textAlign: 'center'}}><strong>50</strong></td>
                <td style={{textAlign: 'center'}}>
                  <Text weight={700} c="dimmed">TBD</Text>
                </td>
              </tr>
            </tfoot>
          </Table>
        </div>

        {/* Bonus Features */}
        <div>
          <Group mb="md">
            <Title order={3}>3. Bonus Features</Title>
          </Group>

          <Table withBorder withColumnBorders mb="md">
            <thead>
              <tr>
                <th>Feature</th>
                <th style={{width: '120px', textAlign: 'center'}}>Points</th>
                <th style={{width: '120px', textAlign: 'center'}}>Score</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>
                  <Text weight={500}>Research Paper Implementation</Text>
                  <Text size="sm" c="dimmed">
                    Implement algorithm from recent paper or interesting package found and used, cite source, compare to baseline
                  </Text>
                </td>
                <td style={{textAlign: 'center'}}>+5</td>
                <td style={{textAlign: 'center'}}>
                  <Text c="dimmed">TBD</Text>
                </td>
              </tr>
              <tr>
                <td>
                  <Text weight={500}>CI/CD Pipeline</Text>
                  <Text size="sm" c="dimmed">
                    GitHub Actions with: code linting, unit tests, automated evaluation runs
                  </Text>
                </td>
                <td style={{textAlign: 'center'}}>+5</td>
                <td style={{textAlign: 'center'}}>
                  <Text c="dimmed">TBD</Text>
                </td>
              </tr>
            </tbody>
            <tfoot>
              <tr>
                <td><strong>Bonus Total</strong></td>
                <td style={{textAlign: 'center'}}><strong>+10</strong></td>
                <td style={{textAlign: 'center'}}>
                  <Text weight={700} c="dimmed">TBD</Text>
                </td>
              </tr>
            </tfoot>
          </Table>
        </div>

        {/* Final Grade Summary */}
        <Paper p="md" withBorder>
          <Title order={4} mb="md">Final Project Grade</Title>
          <Table>
            <tbody>
              <tr>
                <td><Text weight={500}>ML-Arena Performance</Text></td>
                <td style={{width: '120px', textAlign: 'center'}}>
                  <Text c="dimmed">TBD / 50</Text>
                </td>
              </tr>
              <tr>
                <td><Text weight={500}>GitHub Package Quality</Text></td>
                <td style={{width: '120px', textAlign: 'center'}}>
                  <Text c="dimmed">TBD / 50</Text>
                </td>
              </tr>
              <tr>
                <td><Text weight={500}>Bonus Features</Text></td>
                <td style={{width: '120px', textAlign: 'center'}}>
                  <Text c="dimmed">TBD / 10</Text>
                </td>
              </tr>
              <tr style={{borderTop: '2px solid #dee2e6'}}>
                <td><Text weight={700} size="lg">Total Project Score</Text></td>
                <td style={{width: '120px', textAlign: 'center'}}>
                  <Text weight={700} size="lg" c="dimmed">TBD / 100</Text>
                </td>
              </tr>
            </tbody>
          </Table>
          <Text size="xs" c="dimmed" mt="md" ta="center">
            Note: Maximum possible score is 110/100 with all bonuses
          </Text>
        </Paper>
      </Stack>
    </Paper>
  );
};

export default ProjectEvaluation;
