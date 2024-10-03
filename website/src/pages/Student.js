import React, { useState, useEffect } from "react";
import { useParams, useNavigate } from "react-router-dom";
import {
  Container,
  Title,
  Accordion,
  Badge,
  Text,
  Group,
  Button,
  Stack,
  Paper,
  Code,
  Alert,
} from "@mantine/core";
import { IconRefresh, IconAlertTriangle } from "@tabler/icons-react";
import OverallProgress from "components/OverallProgress";

const BackButton = () => {
  const navigate = useNavigate();
  return <Button onClick={() => navigate(-1)}>Back</Button>;
};

const StatusIndicator = ({ progressPercent, hasUpdates }) => {
  let color, label;

  if (progressPercent === 100) {
    color = "teal";
    label = "Passed";
  } else if (hasUpdates) {
    color = "red";
    label = "Failed";
  } else {
    color = "gray";
    label = "Not Published";
  }

  return (
    <Badge color={color} variant="light" size="lg">
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

  const fetchData = () => {
    const cacheBuster = `?t=${new Date().getTime()}`;
    setLoading(true);
    Promise.all([
      fetch(`/repositories/${repositoryId}/students/config/students.json${cacheBuster}`),
      fetch(`/repositories/${repositoryId}/students/${studentId}.json${cacheBuster}`)
    ])
      .then(([overallResponse, detailsResponse]) => {
        if (!overallResponse.ok || !detailsResponse.ok) {
          throw new Error("Failed to load the data");
        }
        return Promise.all([overallResponse.json(), detailsResponse.json()]);
      })
      .then(([overallData, detailsData]) => {
        const studentOverall = overallData[studentId];
        if (studentOverall) {
          setOverallProgress({
            progress: parseFloat(studentOverall.progress_percentage) * 100,
            errors: parseFloat(studentOverall.error_percentage) * 100
          });
        }
        setModulesResults(detailsData);
        setLoading(false);
      })
      .catch((error) => {
        console.error("Error fetching data:", error);
        setError("Failed to fetch student data.");
        setLoading(false);
      });
  };

  useEffect(() => {
    fetchData();
  }, [repositoryId]);

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
    <Container size="lg">
      <Group justify="space-between" mb="md">
        <BackButton />
        <Button
          leftSection={<IconRefresh size={14} />}
          onClick={fetchData}
          loading={loading}
          variant="light"
        >
          Refresh Data
        </Button>
      </Group>
      <Title order={1} mb="md">Results for {studentId}</Title>
      <OverallProgress progress={overallProgress.progress} errors={overallProgress.errors} />
      {error && <Alert color="red" mb="md" title="Error">{error}</Alert>}
      <Accordion variant="contained">
        {Object.entries(modulesResults).map(([moduleName, exercises]) => {
          const totalExercises = Object.values(exercises).length;
          const passedExercises = Object.values(exercises).filter((ex) => ex.is_passed_test).length;
          const progressPercent = totalExercises > 0 ? (passedExercises / totalExercises) * 100 : 0;
          const hasUpdates = Object.values(exercises).some(ex => ex.updated_time_utc);

          return (
            <Accordion.Item key={moduleName} value={moduleName}>
              <Accordion.Control>
                <Group justify="space-between">
                  <Text fw={500}>{moduleName.toUpperCase()} - Progress: {passedExercises}/{totalExercises}</Text>
                  <StatusIndicator progressPercent={progressPercent} hasUpdates={hasUpdates} />
                </Group>
              </Accordion.Control>
              <Accordion.Panel>
                <Stack gap="md">
                  {Object.entries(exercises).map(([exerciseName, exerciseDetails], index) => (
                    <Paper key={index} p="md" withBorder shadow="sm">
                      <Group justify="space-between" mb="xs">
                        <Text fw={500}>{exerciseName}</Text>
                        <Badge color={exerciseDetails.is_passed_test ? "teal" : "red"} variant="light" size="lg">
                          {exerciseDetails.is_passed_test ? "Passed" : "Failed"}
                        </Badge>
                      </Group>
                      <Text size="sm" mb="xs">Score: {exerciseDetails.score}</Text>
                      {/* {!exerciseDetails.is_passed_test && exerciseDetails.updated_time_utc && (
                        <Alert icon={<IconAlertTriangle size="1rem" />} title="Issue Detected" color="yellow" mb="xs">
                          This exercise failed but has been updated. It may require attention.
                        </Alert>
                      )} */}
                      <Text size="sm" mb="xs">Logs:</Text>
                      <Code block mb="xs">{exerciseDetails.logs || "No logs available"}</Code>
                      <Text size="sm" c="dimmed">
                        Updated: {exerciseDetails.updated_time_utc ? formatDate(exerciseDetails.updated_time_utc) : "Not updated"}
                      </Text>
                    </Paper>
                  ))}
                </Stack>
              </Accordion.Panel>
            </Accordion.Item>
          );
        })}
      </Accordion>
    </Container>
  );
};

export default Student;