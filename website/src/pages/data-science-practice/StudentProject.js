import React, { useState, useEffect } from "react";
import { useParams, useNavigate, Link } from "react-router-dom";
import { Container, Title, Text, Group, Button, Badge, Alert, Table, Paper, Stack, Progress } from '@mantine/core';
import { IconRefresh, IconCheck, IconX, IconExternalLink, IconAlertCircle, IconChartBar, IconArrowLeft, IconTrophy } from "@tabler/icons-react";
import styles from './StudentProject.module.css';

const BackButton = () => {
  const navigate = useNavigate();
  return (
    <Button
      onClick={() => navigate(-1)}
      variant="default"
      leftSection={<IconArrowLeft size={14} />}
    >
      Back
    </Button>
  );
};

const StudentProject = () => {
  const { repositoryId, studentId } = useParams();
  const [studentData, setStudentData] = useState(null);
  const [partnerData, setPartnerData] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const fetchData = () => {
    const cacheBuster = `?t=${new Date().getTime()}`;
    setLoading(true);

    fetch(`/repositories/${repositoryId}/students/config/students.json${cacheBuster}`)
      .then((response) => {
        if (!response.ok) {
          throw new Error(`Failed to load the data - Status: ${response.status}`);
        }

        const contentType = response.headers.get("content-type");
        if (contentType && contentType.includes("text/html")) {
          throw new Error("Received HTML instead of JSON - file not found");
        }

        return response.json();
      })
      .then((data) => {
        let currentStudent = null;
        for (const [key, value] of Object.entries(data)) {
          if (value.github_username === studentId) {
            currentStudent = value;
            break;
          }
        }

        if (currentStudent) {
          setStudentData(currentStudent);

          if (currentStudent.project?.partner) {
            let partner = null;
            for (const [key, value] of Object.entries(data)) {
              if (value.github_username === currentStudent.project.partner) {
                partner = value;
                break;
              }
            }
            setPartnerData(partner);
          }
        }

        setError("");
        setLoading(false);
      })
      .catch((error) => {
        console.error("Error fetching data:", error);
        setError(`Failed to fetch student data: ${error.message}`);
        setLoading(false);
      });
  };

  useEffect(() => {
    fetchData();
  }, [repositoryId, studentId]);

  if (error) {
    return (
      <Container size="lg">
        <Group justify="space-between" mb="md">
          <BackButton />
        </Group>
        <Alert color="red" title="Error">{error}</Alert>
      </Container>
    );
  }

  if (!studentData) {
    return (
      <Container size="lg">
        <Group justify="space-between" mb="md">
          <BackButton />
        </Group>
        <Text>Loading...</Text>
      </Container>
    );
  }

  const project = studentData.project || {};
  const projectType = project.projet?.toLowerCase();
  const hasProject = projectType === 'a' || projectType === 'b';

  const getCompetitionUrl = (type) => {
    if (type === 'a') return 'https://ml-arena.com/viewcompetition/8';
    if (type === 'b') return 'https://ml-arena.com/viewcompetition/10';
    return null;
  };

  const getProjectPageUrl = (type) => {
    if (type === 'a') return '/data-science-practice/project-pages/permuted-mnist';
    if (type === 'b') return '/data-science-practice/project-pages/bipedal-walker';
    return null;
  };

  return (
    <Container size="lg" py="md">
      <Group justify="space-between" mb="md">
        <BackButton />
        <Group gap="sm">
          <Button
            component={Link}
            to={`/courses/data-science-practice/student/${repositoryId}/${studentId}`}
            leftSection={<IconChartBar size={14} />}
            variant="default"
          >
            View Progress
          </Button>
          <Button
            leftSection={<IconRefresh size={14} />}
            onClick={fetchData}
            loading={loading}
            variant="default"
          >
            Refresh
          </Button>
        </Group>
      </Group>

      <div className={styles.header}>
        <Group justify="space-between" mb="md">
          <Title order={2}>{studentData.firstname} {studentData.lastname}</Title>
          {hasProject && (
            <Badge size="lg" color={projectType === 'a' ? 'blue' : 'green'}>
              {projectType === 'a' ? 'Permuted MNIST' : 'Bipedal Walker'}
            </Badge>
          )}
        </Group>

        {!hasProject ? (
          <Alert icon={<IconAlertCircle />} color="yellow">
            No project selected. Please send an email to{' '}
            <a href="mailto:raphaelcousin.education@gmail.com" style={{color: 'inherit', fontWeight: 600}}>
              raphaelcousin.education@gmail.com
            </a>{' '}
            with your project choice and repository information.
          </Alert>
        ) : (
          <>
            {/* Missing Information Alert */}
            {(!project.repo || project.racousin_invited !== "1") && (
              <Alert icon={<IconAlertCircle />} color="red" mb="md">
                <Text weight={600} mb="xs">Missing Project Information</Text>
                <Text size="sm">
                  {!project.repo && !project.racousin_invited && "Your repository URL and evaluator invitation are not configured. "}
                  {!project.repo && project.racousin_invited === "1" && "Your repository URL is not configured. "}
                  {project.repo && project.racousin_invited !== "1" && "Evaluator has not been invited to your repository. "}
                  Please send an email to{' '}
                  <a href="mailto:raphaelcousin.education@gmail.com" style={{color: 'inherit', fontWeight: 600}}>
                    raphaelcousin.education@gmail.com
                  </a>{' '}
                  with:
                </Text>
                <ul style={{marginTop: '8px', marginBottom: 0, paddingLeft: '20px'}}>
                  {!project.repo && <li>Your GitHub repository URL</li>}
                  {project.racousin_invited !== "1" && <li>Confirmation that you've invited <code>racousin</code> as collaborator</li>}
                </ul>
              </Alert>
            )}

            <div className={styles.infoGrid}>
              {/* Project Resources */}
              <div className={styles.infoItem}>
                <div className={styles.infoLabel}>Resources</div>
                <div className={styles.infoContent}>
                  <Button
                    component="a"
                    href={getCompetitionUrl(projectType)}
                    target="_blank"
                    rel="noopener noreferrer"
                    rightSection={<IconExternalLink size={12} />}
                    className={styles.exerciseButton}
                    size="xs"
                  >
                    Competition
                  </Button>
                  <Button
                    component={Link}
                    to={getProjectPageUrl(projectType)}
                    className={styles.projectButton}
                    size="xs"
                  >
                    Description
                  </Button>
                </div>
              </div>

              {/* Repository */}
              <div className={styles.infoItem}>
                <div className={styles.infoLabel}>Repository</div>
                <div className={styles.infoContent}>
                  {project.repo ? (
                    <Button
                      component="a"
                      href={project.repo}
                      target="_blank"
                      rel="noopener noreferrer"
                      rightSection={<IconExternalLink size={12} />}
                      className={styles.exerciseButton}
                      size="xs"
                    >
                      View
                    </Button>
                  ) : (
                    <Badge color="red" size="sm" leftSection={<IconX size={12} />}>
                      Not configured
                    </Badge>
                  )}
                </div>
              </div>

              {/* Partner */}
              {project.partner && (
                <div className={styles.infoItem}>
                  <div className={styles.infoLabel}>Partner</div>
                  <div className={styles.infoContent}>
                    {partnerData ? (
                      <>
                        <Text size="sm">{partnerData.firstname} {partnerData.lastname}</Text>
                        <Button
                          component="a"
                          href={`https://github.com/${partnerData.github_username}`}
                          target="_blank"
                          rel="noopener noreferrer"
                          rightSection={<IconExternalLink size={12} />}
                          className={styles.exerciseButton}
                          size="xs"
                        >
                          GitHub
                        </Button>
                      </>
                    ) : (
                      <Text size="sm" c="dimmed">@{project.partner}</Text>
                    )}
                  </div>
                </div>
              )}

              {/* Corrector Status */}
              <div className={styles.infoItem}>
                <div className={styles.infoLabel}>Evaluator Status</div>
                <div className={styles.infoContent}>
                  {project.racousin_invited === "1" ? (
                    <Badge color="green" size="sm" leftSection={<IconCheck size={12} />}>
                      Invited
                    </Badge>
                  ) : (
                    <Badge color="red" size="sm" leftSection={<IconX size={12} />}>
                      Not Invited
                    </Badge>
                  )}
                </div>
              </div>
            </div>
          </>
        )}
      </div>

      {/* Evaluation section */}
      {hasProject && (
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
                        Absolute performance: Above 99% accuracy / 300 rewards: 20pts | Beats baseline algorithm (named Mark Bench): 15pts | Between benchmark and 98% accuracy (200 rewards): 10pts | Between 98% and 97% accuracy (100 rewards): 5pts
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
                      <Text weight={500}>Resume Notebook (resume.ipynb)</Text>
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
                    <td style={{textAlign: 'center'}}><strong>+15</strong></td>
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
      )}
    </Container>
  );
};

export default StudentProject;
