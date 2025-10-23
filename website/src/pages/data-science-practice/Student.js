import React, { useState, useEffect } from "react";
import { useParams, useNavigate, Link } from "react-router-dom";
import { Container, Title, Accordion, Badge, Text, Group, Button, Stack, Code, Alert, Box, Paper, Divider } from '@mantine/core';
import { IconRefresh, IconCircleCheckFilled, IconCircleXFilled, IconCircleDashed, IconArrowLeft, IconFileText } from "@tabler/icons-react";
import OverallProgress from "components/OverallProgress";

const BackButton = () => {
  const navigate = useNavigate();
  return (
    <Button
      onClick={() => navigate(-1)}
      variant="default"
      leftSection={<IconArrowLeft size={16} />}
    >
      Back
    </Button>
  );
};

const StatusIndicator = ({ progressPercent, hasUpdates }) => {
  let color, label, icon;

  if (progressPercent === 100) {
    color = "teal";
    label = "Passed";
    icon = <IconCircleCheckFilled size={14} />;
  } else if (hasUpdates) {
    color = "red";
    label = "Failed";
    icon = <IconCircleXFilled size={14} />;
  } else {
    color = "gray";
    label = "Not Published";
    icon = <IconCircleDashed size={14} />;
  }

  return (
    <Badge
      color={color}
      variant="light"
      size="lg"
      leftSection={icon}
      styles={(theme) => ({
        root: {
          paddingLeft: theme.spacing.xs,
          fontWeight: 500,
        },
      })}
    >
      {label}
    </Badge>
  );
};

const Student = () => {
  const { repositoryId, studentId } = useParams();
  const [modulesResults, setModulesResults] = useState({});
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [overallProgress, setOverallProgress] = useState({ progress: 0, errors: 0 });
  const [studentName, setStudentName] = useState("");

  const fetchData = () => {
    const cacheBuster = `?t=${new Date().getTime()}`;
    setLoading(true);
    Promise.all([
      fetch(`/repositories/${repositoryId}/students/config/students.json${cacheBuster}`),
      fetch(`/repositories/${repositoryId}/students/${studentId}.json${cacheBuster}`)
    ])
      .then(([overallResponse, detailsResponse]) => {
        if (!overallResponse.ok || !detailsResponse.ok) {
          throw new Error(`Failed to load the data - Status: ${overallResponse.status}, ${detailsResponse.status}`);
        }

        // Check if responses are JSON
        const overallContentType = overallResponse.headers.get("content-type");
        const detailsContentType = detailsResponse.headers.get("content-type");

        if ((overallContentType && overallContentType.includes("text/html")) ||
            (detailsContentType && detailsContentType.includes("text/html"))) {
          throw new Error("Received HTML instead of JSON - file not found");
        }

        return Promise.all([overallResponse.json(), detailsResponse.json()]);
      })
      .then(([overallData, detailsData]) => {
        // Find student data by matching github_username
        let studentOverall = null;
        for (const [key, value] of Object.entries(overallData)) {
          if (value.github_username === studentId) {
            studentOverall = value;
            break;
          }
        }

        if (studentOverall) {
          setOverallProgress({
            progress: parseFloat(studentOverall.progress_percentage || 0) * 100,
            errors: parseFloat(studentOverall.error_percentage || 0) * 100
          });
          setStudentName(`${studentOverall.firstname} ${studentOverall.lastname}`);
        }
        setModulesResults(detailsData);
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

  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleString("en-US", {
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      hour12: false,
    });
  };

  return (
    <Container size="lg" py="xl">
      <Stack gap="xl">
        {/* Header Section */}
        <Box>
          <Group justify="space-between" mb="xl">
            <BackButton />
            <Group gap="sm">
              <Button
                component={Link}
                to={`/courses/data-science-practice/student-project/${repositoryId}/${studentId}`}
                leftSection={<IconFileText size={16} />}
                variant="default"
              >
                View Project
              </Button>
              <Button
                leftSection={<IconRefresh size={16} />}
                onClick={fetchData}
                loading={loading}
                variant="default"
              >
                Refresh
              </Button>
            </Group>
          </Group>

          <Title order={1} mb="xs" size="h1">
            {studentName || studentId}
          </Title>
          <Text size="md" c="dimmed" mb="xl">
            Student Progress Overview
          </Text>

          <OverallProgress progress={overallProgress.progress} errors={overallProgress.errors} />
        </Box>

        {/* Error Alert */}
        {error && (
          <Alert color="red" title="Error" variant="light">
            {error}
          </Alert>
        )}

        {/* Modules Accordion */}
        <Accordion
          variant="separated"
          radius="md"
          styles={(theme) => ({
            item: {
              border: `1px solid ${theme.colorScheme === 'dark' ? theme.colors.dark[4] : theme.colors.gray[2]}`,
              backgroundColor: theme.colorScheme === 'dark' ? theme.colors.dark[6] : theme.white,
              '&[data-active]': {
                backgroundColor: theme.colorScheme === 'dark' ? theme.colors.dark[5] : theme.colors.gray[0],
              },
            },
            control: {
              padding: `${theme.spacing.md} ${theme.spacing.lg}`,
              '&:hover': {
                backgroundColor: theme.colorScheme === 'dark' ? theme.colors.dark[5] : theme.colors.gray[0],
              },
            },
          })}
        >
          {Object.entries(modulesResults).map(([moduleName, exercises]) => {
            const exerciseDetails = Object.values(exercises)[0];
            const progressPercent = exerciseDetails?.is_passed_test ? 100 : 0;
            const hasUpdates = exerciseDetails?.updated_time_utc;

            return (
              <Accordion.Item key={moduleName} value={moduleName}>
                <Accordion.Control>
                  <Box style={{ flex: 1 }}>
                    <Group justify="space-between" wrap="nowrap" mb="xs">
                      <Text fw={600} size="md" tt="uppercase" c="dark">
                        {moduleName}
                      </Text>
                      <StatusIndicator
                        progressPercent={progressPercent}
                        hasUpdates={hasUpdates}
                      />
                    </Group>
                    <Group gap="md" mt="xs">
                      <Group gap={4}>
                        <Text size="xs" c="dimmed">Score:</Text>
                        <Text size="xs" fw={600}>{exerciseDetails?.score || "N/A"}</Text>
                      </Group>
                      <Group gap={4}>
                        <Text size="xs" c="dimmed">Updated:</Text>
                        <Text size="xs" fw={500}>
                          {exerciseDetails?.updated_time_utc
                            ? formatDate(exerciseDetails.updated_time_utc)
                            : "Not updated"}
                        </Text>
                      </Group>
                    </Group>
                  </Box>
                </Accordion.Control>
                <Accordion.Panel>
                  <Box>
                    <Text size="sm" fw={500} mb="xs" c="dimmed">
                      Execution Logs
                    </Text>
                    <Code
                      block
                      styles={(theme) => ({
                        root: {
                          borderRadius: theme.radius.sm,
                          fontSize: theme.fontSizes.xs,
                          padding: theme.spacing.md,
                          backgroundColor: theme.colorScheme === 'dark' ? theme.colors.dark[8] : theme.colors.gray[0],
                          border: `1px solid ${theme.colorScheme === 'dark' ? theme.colors.dark[4] : theme.colors.gray[2]}`,
                          color: theme.colorScheme === 'dark' ? theme.colors.gray[3] : 'inherit',
                        },
                      })}
                      style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}
                    >
                      {exerciseDetails?.logs || "No logs available"}
                    </Code>
                  </Box>
                </Accordion.Panel>
              </Accordion.Item>
            );
          })}
        </Accordion>
      </Stack>
    </Container>
  );
};

export default Student;