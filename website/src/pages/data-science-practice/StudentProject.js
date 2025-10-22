import React, { useState, useEffect } from "react";
import { useParams, useNavigate, Link } from "react-router-dom";
import { Container, Title, Text, Group, Button, Badge, Alert } from '@mantine/core';
import { IconRefresh, IconCheck, IconX, IconExternalLink, IconAlertCircle } from "@tabler/icons-react";
import styles from './StudentProject.module.css';

const BackButton = () => {
  const navigate = useNavigate();
  return <Button onClick={() => navigate(-1)}>Back</Button>;
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
        <Button
          leftSection={<IconRefresh size={14} />}
          onClick={fetchData}
          loading={loading}
          variant="subtle"
          size="xs"
        >
          Refresh
        </Button>
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
            No project selected
          </Alert>
        ) : (
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
                  <Text size="sm" c="red">Not configured</Text>
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
                  <Badge color="gray" size="sm" leftSection={<IconX size={12} />}>
                    Not Invited
                  </Badge>
                )}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Evaluation section - placeholder */}
      {hasProject && (
        <div>
          <Title order={3} mb="md">Evaluation</Title>
          <Text c="dimmed" size="sm">Evaluation criteria will be displayed here</Text>
        </div>
      )}
    </Container>
  );
};

export default StudentProject;
