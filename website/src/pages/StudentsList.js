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
  const { repositoryId } = useParams();
  const [students, setStudents] = useState([]);
  const [error, setError] = useState("");
  const [filter, setFilter] = useState("");
  const [sortConfig, setSortConfig] = useState({ key: "name", direction: "ascending" });

  const fetchStudents = () => {
    const cacheBuster = `?t=${new Date().getTime()}`;
    fetch(`/repositories/${repositoryId}/students/config/students.json${cacheBuster}`)
      .then((response) => {
        if (!response.ok) {
          throw new Error("Network response was not ok");
        }
        return response.json();
      })
      .then((data) => {
        const formattedData = Object.entries(data).map(([name, details]) => {
          const progress = parseFloat(details.progress_percentage) * 100;
          const errors = parseFloat(details.error_percentage) * 100;
          return {
            name,
            ...details,
            progress_percentage: progress,
            error_percentage: errors,
          };
        });
        setStudents(formattedData);
      })
      .catch((error) => {
        console.error("Error fetching student list:", error);
        setError("Failed to fetch student data.");
      });
  };

  useEffect(() => {
    fetchStudents();
  }, [repositoryId]);

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
      student.name.toLowerCase().includes(filter.toLowerCase())
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

  const handleHardRefresh = () => {
    fetchStudents();
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
      <Title order={1}>Student List: {repositoryId}</Title>
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
              <Group gap="xs" style={{cursor: 'pointer'}} onClick={() => requestSort('name')}>
                <Text>Name</Text>
                {getSortIcon('name')}
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
              <Table.Td>{student.name}</Table.Td>
              <Table.Td style={{ width: '40%' }}>
                <ProgressBar
                  progressPercent={student.progress_percentage}
                  errorPercent={student.error_percentage}
                />
              </Table.Td>
              <Table.Td>
                <Button
                  component={Link}
                  to={`/student/${repositoryId}/${student.name}`}
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