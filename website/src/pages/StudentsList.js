import React, { useState, useEffect, useMemo } from "react";
import { Link, useParams } from "react-router-dom";
import {
  Table,
  Container,
  Alert,
  TextInput,
  Button,
  Progress,
  Box,
  Title,
  Group,
} from "@mantine/core";
import { IconRefresh, IconChevronUp, IconChevronDown } from "@tabler/icons-react";
import { useNavigate } from "react-router-dom";

const BackButton = () => {
  const navigate = useNavigate();
  return <Button onClick={() => navigate(-1)}>Back</Button>;
};

const ArrayProgress = ({ progressPercent }) => (
  <Progress value={progressPercent} size="xl" radius="xl" />
);

const StudentsList = () => {
  const { repositoryId } = useParams();
  const [students, setStudents] = useState([]);
  const [error, setError] = useState("");
  const [filter, setFilter] = useState("");
  const [sortConfig, setSortConfig] = useState({ key: "name", direction: "ascending" });

  const fetchStudents = () => {
    fetch(`/repositories/${repositoryId}/students/config/students.json`)
      .then((response) => {
        if (!response.ok) {
          throw new Error("Network response was not ok");
        }
        return response.json();
      })
      .then((data) => {
        const formattedData = Object.entries(data).map(([name, details]) => ({
          name,
          ...details,
        }));
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
    <Container size="xl">
      <Group position="apart" mb="md">
        <BackButton />
        <Button
          leftIcon={<IconRefresh size={14} />}
          onClick={handleHardRefresh}
        >
          Refresh Data
        </Button>
      </Group>
      <Title order={1} mb="md">
        Student List for {repositoryId}
      </Title>
      {error && <Alert color="red">{error}</Alert>}
      <TextInput
        placeholder="Filter by name..."
        mb="md"
        value={filter}
        onChange={(event) => setFilter(event.currentTarget.value)}
      />
      <Table striped highlightOnHover>
        <thead>
          <tr>
            <th onClick={() => requestSort("name")}>
              Name {getSortIcon("name")}
            </th>
            <th onClick={() => requestSort("progress_percentage")}>
              Progress {getSortIcon("progress_percentage")}
            </th>
            <th>Details</th>
          </tr>
        </thead>
        <tbody>
          {filteredStudents.map((student, index) => (
            <tr key={index}>
              <td>{student.name}</td>
              <td>
                <Box sx={{ width: 200 }}>
                  <ArrayProgress
                    progressPercent={student.progress_percentage * 100}
                  />
                </Box>
              </td>
              <td>
                <Button
                  component={Link}
                  to={`/student/${repositoryId}/${student.name}`}
                  variant="light"
                >
                  View Details
                </Button>
              </td>
            </tr>
          ))}
        </tbody>
      </Table>
    </Container>
  );
};

export default StudentsList;