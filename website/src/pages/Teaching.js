import React, { useState } from 'react';
import { Container, Grid, Card, Button, Select, Title, Text } from '@mantine/core';


const modules = [
  {
    id: 1,
    title: "Module 1: Git",
    linkCourse: "/module1/course",
    linkExercise: "/module1/exercise",
    tags: ["Version Control"],
  },
  {
    id: 2,
    title: "Module 2: Python environment and package",
    linkCourse: "/module2/course",
    linkExercise: "/module2/exercise",
    tags: ["Programming"],
  },
  {
    id: 3,
    title: "Module 3: Data Science landscape",
    linkCourse: "/module3/course",
    linkExercise: "/module3/exercise",
    tags: ["Machine Learning", "Analysis"],
  },
  {
    id: 4,
    title: "Module 4: Data Collection",
    linkCourse: "/module4/course",
    linkExercise: "/module4/exercise",
    tags: ["Analysis"],
  },
  {
    id: 5,
    title: "Module 5: Data Cleaning and Preparation",
    linkCourse: "/module5/course",
    linkExercise: "/module5/exercise",
    tags: ["Analysis"],
  },
  {
    id: 6,
    title: "Module 6: Tabular Models",
    linkCourse: "/module6/course",
    linkExercise: "/module6/exercise",
    tags: ["Machine Learning"],
  },
  {
    id: 7,
    title: "Module 7: Deep Learning Fundamentals",
    linkCourse: "/module7/course",
    linkExercise: "/module7/exercise",
    tags: ["Deep Learning"],
  },

  {
    id: 8,
    title: "Module 8: Image Processing",
    linkCourse: "/module8/course",
    linkExercise: "/module8/exercise",
    tags: ["Machine Learning", "Deep Learning"],
  },
  {
    id: 9,
    title: "Module 9: TimeSeries Processing",
    linkCourse: "/module9/course",
    linkExercise: "/module9/exercise",
    tags: ["TimeSeries", "Deep Learning"],
  },
  {
    id: 10,
    title: "Module 10: Text Processing",
    linkCourse: "/module10/course",
    linkExercise: "/module10/exercise",
    tags: ["NLP", "Machine Learning"],
  },
  {
    id: 11,
    title: "Module 11: Generative Models",
    linkCourse: "/module11/course",
    linkExercise: "/module11/exercise",
    tags: ["Deep Learning"],
  },
  {
    id: 12,
    title: "Module 12: Recommendation Systems",
    linkCourse: "/module12/course",
    linkExercise: "/module12/exercise",
    tags: ["Machine Learning", "Deep Learning"],
  },
  {
    id: 13,
    title: "Module 13: Reinforcement Learning",
    linkCourse: "/module13/course",
    linkExercise: "/module13/exercise",
    tags: ["Machine Learning", "Deep Learning"],
  },
  {
    id: 14,
    title: "Module 14: Docker",
    linkCourse: "/module14/course",
    linkExercise: "/module14/exercise",
    tags: ["DevOps", "Cloud"],
  },
  {
    id: 15,
    title: "Module 15: Cloud Integration",
    linkCourse: "/module15/course",
    linkExercise: "/module15/exercise",
    tags: ["Cloud"],
  },
];
const tags = ["All", "Version Control", "Programming", "Machine Learning", "Cloud", "Deep Learning"];

export default function Teaching() {
  const [selectedTag, setSelectedTag] = useState("All");

  const filteredModules = selectedTag === "All"
    ? modules
    : modules.filter(module => module.tags.includes(selectedTag));

  return (
    <Container size="xl" py="xl">
      <Title order={1} align="center" mb="xl">Teaching Portal</Title>
      
      <Select
        label="Select a tag to filter modules:"
        placeholder="Choose a tag"
        data={tags}
        value={selectedTag}
        onChange={setSelectedTag}
        mb="xl"
      />

      <Grid>
        {filteredModules.map((module) => (
          <Grid.Col key={module.id} span={{ base: 12, sm: 6, md: 4 }}>
            <Card shadow="sm" padding="lg" radius="md" withBorder>
              <Text fw={500} size="lg" mb="md">{`Module ${module.id}: ${module.title}`}</Text>
              <Button
                variant="light"
                color="blue"
                fullWidth
                mb="sm"
                component="a"
                href={module.linkCourse}
              >
                {module.id === 0 ? "Getting Started" : "Go to Course"}
              </Button>
              {module.id !== 0 && (
                <Button
                  variant="outline"
                  color="gray"
                  fullWidth
                  component="a"
                  href={module.linkExercise}
                >
                  Go to Exercise
                </Button>
              )}
            </Card>
          </Grid.Col>
        ))}
      </Grid>
    </Container>
  );
}