import React, { useState, useEffect, useMemo } from "react";
import { Link, useParams } from "react-router-dom";
import { Table, Container, Alert, TextInput, Button, Progress, Box, Title, Group, Tooltip, Text, ActionIcon, Stack, Badge, Anchor } from '@mantine/core';
import { IconRefresh, IconChevronUp, IconChevronDown, IconArrowLeft, IconSearch, IconFileText, IconBriefcase, IconChartBar, IconUser } from "@tabler/icons-react";
import { useNavigate } from "react-router-dom";

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

const ProgressBar = ({ progressPercent, errorPercent }) => {
  return (
    <Box>
      <Tooltip
        label={`Progress: ${progressPercent.toFixed(1)}%, Errors: ${errorPercent.toFixed(1)}%`}
        position="top"
        withArrow
      >
        <Progress.Root size="lg" radius="md">
          <Progress.Section value={progressPercent} color="teal">
            <Progress.Label>{progressPercent > 10 ? `${progressPercent.toFixed(0)}%` : ''}</Progress.Label>
          </Progress.Section>
          <Progress.Section value={errorPercent} color="gray" />
        </Progress.Root>
      </Tooltip>
      <Group gap="xs" mt={4}>
        <Badge size="xs" variant="light" color="teal">
          {progressPercent.toFixed(0)}% Complete
        </Badge>
        {errorPercent > 0 && (
          <Badge size="xs" variant="light" color="gray">
            {errorPercent.toFixed(0)}% Errors
          </Badge>
        )}
      </Group>
    </Box>
  );
};



const StudentsList = () => {
  const { repoName } = useParams();
  const [students, setStudents] = useState([]);
  const [error, setError] = useState("");
  const [filter, setFilter] = useState("");
  const [sortConfig, setSortConfig] = useState({ key: "fullName", direction: "ascending" });

  const fetchStudents = () => {
    const cacheBuster = `?t=${new Date().getTime()}`;
    const url = `/repositories/${repoName}/students/config/students.json${cacheBuster}`;

    fetch(url)
      .then((response) => {
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        // Check if response is JSON
        const contentType = response.headers.get("content-type");
        if (contentType && contentType.includes("text/html")) {
          throw new Error("Received HTML instead of JSON - file not found");
        }

        return response.text().then(text => {
          try {
            return JSON.parse(text);
          } catch (e) {
            throw new Error("Invalid JSON response");
          }
        });
      })
      .then((data) => {
        const formattedData = Object.entries(data).map(([username, details]) => {
          const progress = parseFloat(details.progress_percentage || 0) * 100;
          const errors = parseFloat(details.error_percentage || 0) * 100;
          return {
            username,
            fullName: `${details.firstname} ${details.lastname}`,
            githubUsername: details.github_username,
            nbReview: details.nb_review || 0,
            ...details,
            progress_percentage: progress,
            error_percentage: errors,
          };
        });
        setStudents(formattedData);
        setError("");
      })
      .catch((error) => {
        setError(`Failed to fetch student data: ${error.message}`);
        setStudents([]);
      });
  };

  useEffect(() => {
    fetchStudents();
  }, [repoName]);

  const sortedStudents = useMemo(() => {
    let sortableItems = [...students];
    if (sortConfig !== null) {
      sortableItems.sort((a, b) => {
        if (a[sortConfig.key] < b[sortConfig.key]) {
          return sortConfig.direction === "ascending" ? -1 : 1;
        }
        if (a[sortConfig.key] > b[sortConfig.key]) {
          return sortConfig.direction === "ascending" ? 1 : -1;
        }
        return 0;
      });
    }
    return sortableItems;
  }, [students, sortConfig]);

  const filteredStudents = useMemo(() => {
    return sortedStudents.filter((student) =>
      student.fullName.toLowerCase().includes(filter.toLowerCase()) ||
      student.githubUsername.toLowerCase().includes(filter.toLowerCase())
    );
  }, [sortedStudents, filter]);

  const requestSort = (key) => {
    let direction = "ascending";
    if (sortConfig.key === key && sortConfig.direction === "ascending") {
      direction = "descending";
    }
    setSortConfig({ key, direction });
  };

  const getSortIcon = (key) => {
    if (sortConfig.key === key) {
      return sortConfig.direction === "ascending" ? (
        <IconChevronUp size={14} />
      ) : (
        <IconChevronDown size={14} />
      );
    }
    return null;
  };


  return (
    <Container size="xl" py="xl">
      <Stack gap="xl">
        {/* Header Section */}
        <Box>
          <Group justify="space-between" mb="xl">
            <BackButton />
            <Button
              leftSection={<IconRefresh size={16} />}
              onClick={fetchStudents}
              variant="default"
            >
              Refresh
            </Button>
          </Group>

          <Group align="center" gap="sm" mb="xs">
            <IconUser size={32} stroke={1.5} />
            <Title order={1} size="h1">
              {repoName}
            </Title>
          </Group>

          {/* Search Bar */}
          <TextInput
            leftSection={<IconSearch size={16} />}
            placeholder="Search by name or username..."
            size="md"
            value={filter}
            onChange={(event) => setFilter(event.currentTarget.value)}
            styles={(theme) => ({
              input: {
                borderRadius: theme.radius.md,
              },
            })}
          />
        </Box>

        {/* Error Alert */}
        {error && (
          <Alert color="red" title="Error" variant="light">
            {error}
          </Alert>
        )}

        {/* Students Table */}
        <Box
          style={(theme) => ({
            border: `1px solid ${theme.colorScheme === 'dark' ? theme.colors.dark[4] : theme.colors.gray[2]}`,
            borderRadius: theme.radius.md,
            overflow: 'hidden',
          })}
        >
          <Table
            highlightOnHover
            verticalSpacing="md"
            styles={(theme) => ({
              thead: {
                backgroundColor: theme.colorScheme === 'dark' ? theme.colors.dark[6] : theme.colors.gray[0],
              },
              th: {
                fontWeight: 600,
                fontSize: theme.fontSizes.sm,
                color: theme.colorScheme === 'dark' ? theme.colors.gray[5] : theme.colors.gray[7],
                textTransform: 'uppercase',
                letterSpacing: '0.5px',
              },
            })}
          >
            <Table.Thead>
              <Table.Tr>
                <Table.Th>
                  <Group
                    gap="xs"
                    style={{ cursor: 'pointer', userSelect: 'none' }}
                    onClick={() => requestSort('fullName')}
                  >
                    <Text size="sm">Name</Text>
                    {getSortIcon('fullName')}
                  </Group>
                </Table.Th>
                <Table.Th>
                  <Group
                    gap="xs"
                    style={{ cursor: 'pointer', userSelect: 'none' }}
                    onClick={() => requestSort('githubUsername')}
                  >
                    <Text size="sm">GitHub</Text>
                    {getSortIcon('githubUsername')}
                  </Group>
                </Table.Th>
                <Table.Th>
                  <Group
                    gap="xs"
                    style={{ cursor: 'pointer', userSelect: 'none' }}
                    onClick={() => requestSort('nbReview')}
                  >
                    <Text size="sm">Reviews</Text>
                    {getSortIcon('nbReview')}
                  </Group>
                </Table.Th>
                <Table.Th>
                  <Group
                    gap="xs"
                    style={{ cursor: 'pointer', userSelect: 'none' }}
                    onClick={() => requestSort('progress_percentage')}
                  >
                    <Text size="sm">Progress</Text>
                    {getSortIcon('progress_percentage')}
                  </Group>
                </Table.Th>
                <Table.Th>
                  <Text size="sm">Details</Text>
                </Table.Th>
              </Table.Tr>
            </Table.Thead>
            <Table.Tbody>
              {filteredStudents.map((student, index) => (
                <Table.Tr key={index}>
                  <Table.Td>
                    <Text fw={500}>{student.fullName}</Text>
                  </Table.Td>
                  <Table.Td>
                    <Anchor
                      href={`https://github.com/${student.githubUsername}`}
                      target="_blank"
                      rel="noopener noreferrer"
                      size="sm"
                      fw={500}
                    >
                      {student.githubUsername}
                    </Anchor>
                  </Table.Td>
                  <Table.Td>
                    <Badge variant="light" size="md">
                      {student.nbReview}
                    </Badge>
                  </Table.Td>
                  <Table.Td style={{ width: '25%' }}>
                    <ProgressBar
                      progressPercent={student.progress_percentage}
                      errorPercent={student.error_percentage}
                    />
                  </Table.Td>
                  <Table.Td>
                    <Group gap="xs">
                      <Button
                        component={Link}
                        to={`/courses/data-science-practice/student/${repoName}/${student.githubUsername}`}
                        variant="default"
                        size="xs"
                        leftSection={<IconFileText size={14} />}
                      >
                        Exercises
                      </Button>
                      <Button
                        component={Link}
                        to={`/courses/data-science-practice/student-project/${repoName}/${student.githubUsername}`}
                        variant="default"
                        size="xs"
                        leftSection={<IconBriefcase size={14} />}
                      >
                        Project
                      </Button>
                    </Group>
                  </Table.Td>
                </Table.Tr>
              ))}
            </Table.Tbody>
          </Table>
        </Box>

        {/* Results Counter */}
        <Text size="sm" c="dimmed" ta="center">
          Showing {filteredStudents.length} of {students.length} students
        </Text>
      </Stack>
    </Container>
  );
};

export default StudentsList;