import React, { useState, useEffect, useMemo } from "react";
import { Link, useParams } from "react-router-dom";
import { Table, Container, Alert, TextInput, Button, Progress, Box, Title, Group, Tooltip, Text, ActionIcon } from '@mantine/core';
import { IconRefresh, IconChevronUp, IconChevronDown, IconArrowLeft, IconSearch  } from "@tabler/icons-react";
import { useNavigate } from "react-router-dom";

const BackButton = () => {
  const navigate = useNavigate();
  return <Button onClick={() => navigate(-1)}>Back</Button>;
};

const ProgressBar = ({ progressPercent, errorPercent }) => {
  return (
    <Tooltip
      label={`Progress: ${progressPercent.toFixed(1)}%, Errors: ${errorPercent.toFixed(1)}%`}
      position="top"
      withArrow
    >
      <Box>
        <Progress.Root size="xl">
          <Progress.Section value={progressPercent} color="green" />
          <Progress.Section value={errorPercent} color="gray" />
        </Progress.Root>
      </Box>
    </Tooltip>
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
      <Group justify="space-between" mb="lg">
        <Group>
          <BackButton />
        </Group>
        <Button
          leftSection={<IconRefresh size={14} />}
          onClick={fetchStudents}
          variant="light"
        >
          Refresh Data
        </Button>
      </Group>
      <Title order={1}>Student List: {repoName}</Title>
      {error && <Alert color="red" mb="md">{error}</Alert>}
      
      <TextInput
        leftSection={<IconSearch size={14} />}
        placeholder="Filter by name..."
        mb="md"
        value={filter}
        onChange={(event) => setFilter(event.currentTarget.value)}
      />
      
      <Table striped highlightOnHover>
        <Table.Thead>
          <Table.Tr>
            <Table.Th>
              <Group gap="xs" style={{cursor: 'pointer'}} onClick={() => requestSort('fullName')}>
                <Text>Name</Text>
                {getSortIcon('fullName')}
              </Group>
            </Table.Th>
            <Table.Th>
              <Group gap="xs" style={{cursor: 'pointer'}} onClick={() => requestSort('githubUsername')}>
                <Text>Username</Text>
                {getSortIcon('githubUsername')}
              </Group>
            </Table.Th>
            <Table.Th>
              <Group gap="xs" style={{cursor: 'pointer'}} onClick={() => requestSort('nbReview')}>
                <Text>Reviews</Text>
                {getSortIcon('nbReview')}
              </Group>
            </Table.Th>
            <Table.Th>
              <Group gap="xs" style={{cursor: 'pointer'}} onClick={() => requestSort('progress_percentage')}>
                <Text>Progress/Errors</Text>
                {getSortIcon('progress_percentage')}
              </Group>
            </Table.Th>
            <Table.Th>Details</Table.Th>
          </Table.Tr>
        </Table.Thead>
        <Table.Tbody>
          {filteredStudents.map((student, index) => (
            <Table.Tr key={index}>
              <Table.Td>{student.fullName}</Table.Td>
              <Table.Td>
                <a
                  href={`https://github.com/${student.githubUsername}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  style={{ color: '#0969da', textDecoration: 'none' }}
                >
                  {student.githubUsername}
                </a>
              </Table.Td>
              <Table.Td>{student.nbReview}</Table.Td>
              <Table.Td style={{ width: '30%' }}>
                <ProgressBar
                  progressPercent={student.progress_percentage}
                  errorPercent={student.error_percentage}
                />
              </Table.Td>
              <Table.Td>
                <Button
                  component={Link}
                  to={`/courses/data-science-practice/student/${repoName}/${student.githubUsername}`}
                  variant="light"
                  size="sm"
                >
                  View Details
                </Button>
              </Table.Td>
            </Table.Tr>
          ))}
        </Table.Tbody>
      </Table>
    </Container>
  );
};

export default StudentsList;