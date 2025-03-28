import React, { useState } from 'react';
import { Container, Grid, Card, Button, Select, Title, Text } from '@mantine/core';


const ModuleGrid = ({ filteredModules }) => {
  const getButtonConfig = (moduleId) => {
    // Handle main button text
    const mainButtonText = (() => {
      if (moduleId === 0) return "Getting Started";
      if (moduleId === 16) return "Go to Project";
      return "Go to Course";
    })();

    // Handle whether to show exercise button
    const showExerciseButton = moduleId !== 0 && moduleId !== 16;

    return {
      mainButtonText,
      showExerciseButton
    };
  };

  return (
    <Grid>
      {filteredModules.map((module) => {
        const { mainButtonText, showExerciseButton } = getButtonConfig(module.id);
        
        return (
          <Grid.Col key={module.id} span={{ base: 12, sm: 6, md: 4 }}>
            <Card shadow="sm" padding="lg" radius="md" withBorder>
              <Text fw={500} size="lg" mb="md">
                {`Module ${module.id}: ${module.title}`}
              </Text>
              
              <Button
                variant="light"
                color="blue"
                fullWidth
                mb="sm"
                component="a"
                href={module.linkCourse}
              >
                {mainButtonText}
              </Button>
              
              {showExerciseButton && (
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
        );
      })}
    </Grid>
  );
};


const modules = [
  {
    id: 0,
    title: "Prerequisite And Methodology",
    linkCourse: "/module0/course",
    tags: [],
  },
  {
    id: 1,
    title: "Git and Github",
    linkCourse: "/module1/course",
    linkExercise: "/module1/exercise",
    tags: ["Version Control"],
  },
  {
    id: 2,
    title: "Python environment and package",
    linkCourse: "/module2/course",
    linkExercise: "/module2/exercise",
    tags: ["Programming"],
  },
  {
    id: 3,
    title: "Data Science landscape",
    linkCourse: "/module3/course",
    linkExercise: "/module3/exercise",
    tags: ["Machine Learning", "Analysis"],
  },
  {
    id: 4,
    title: "Data Collection",
    linkCourse: "/module4/course",
    linkExercise: "/module4/exercise",
    tags: ["Analysis"],
  },
  {
    id: 5,
    title: "Data Preprocessing",
    linkCourse: "/module5/course",
    linkExercise: "/module5/exercise",
    tags: ["Analysis"],
  },
  {
    id: 6,
    title: "Tabular Models",
    linkCourse: "/module6/course",
    linkExercise: "/module6/exercise",
    tags: ["Machine Learning"],
  },
  {
    id: 7,
    title: "Deep Learning Fundamentals",
    linkCourse: "/module7/course",
    linkExercise: "/module7/exercise",
    tags: ["Deep Learning"],
  },

  {
    id: 8,
    title: "Image Processing",
    linkCourse: "/module8/course",
    linkExercise: "/module8/exercise",
    tags: ["Machine Learning", "Deep Learning"],
  },
  {
    id: 9,
    title: "TimeSeries Processing",
    linkCourse: "/module9/course",
    linkExercise: "/module9/exercise",
    tags: ["TimeSeries", "Deep Learning"],
  },
  {
    id: 10,
    title: "Natural Language Processing",
    linkCourse: "/module10/course",
    linkExercise: "/module10/exercise",
    tags: ["NLP", "Machine Learning"],
  },
  {
    id: 11,
    title: "Generative Models",
    linkCourse: "/module11/course",
    linkExercise: "/module11/exercise",
    tags: ["Deep Learning"],
  },
  {
    id: 12,
    title: "Recommendation Systems",
    linkCourse: "/module12/course",
    linkExercise: "/module12/exercise",
    tags: ["Machine Learning", "Deep Learning"],
  },
  {
    id: 13,
    title: "Reinforcement Learning",
    linkCourse: "/module13/course",
    linkExercise: "/module13/exercise",
    tags: ["Machine Learning", "Deep Learning"],
  },
  {
    id: 14,
    title: "Docker",
    linkCourse: "/module14/course",
    linkExercise: "/module14/exercise",
    tags: ["DevOps", "Cloud"],
  },
  {
    id: 15,
    title: "Cloud Integration",
    linkCourse: "/module15/course",
    linkExercise: "/module15/exercise",
    tags: ["Cloud"],
  },
  {
    id: 16,
    title: "Project",
    linkCourse: "/project-page",
    tags: ["Machine Learning", "Deep Learning"],
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

      <ModuleGrid filteredModules={filteredModules} />
    </Container>
  );
}