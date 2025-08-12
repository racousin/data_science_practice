import React from 'react';
import { Container, Title, Text, Stack, List, Paper, Group, ThemeIcon } from '@mantine/core';
import { Link, useLocation, useParams } from 'react-router-dom';
import { IconBook, IconClipboardList } from '@tabler/icons-react';
import { getCourseContentLinks, getExerciseContentLinks } from './SideNavigation';

const ModuleTableOfContents = ({ type = 'course' }) => {
  const location = useLocation();
  const params = useParams();
  
  // Extract courseId and moduleId from the pathname
  const pathParts = location.pathname.split('/').filter(Boolean);
  const courseId = pathParts[1]; // courses/[courseId]/...
  const moduleId = pathParts[2]; // courses/courseId/[moduleId]/...
  
  const isExercise = type === 'exercise';
  
  // Get the appropriate content links
  const links = isExercise 
    ? getExerciseContentLinks(moduleId, courseId)
    : getCourseContentLinks(moduleId, courseId);
  
  if (!links || links.length === 0) {
    return (
      <Container size="lg" py="xl">
        <Text color="dimmed" align="center">
          No {isExercise ? 'exercises' : 'content'} available for this module yet.
        </Text>
      </Container>
    );
  }
  
  return (
    <Container size="lg" py="xl">
      <Stack spacing="lg">
        <Group>
          <ThemeIcon size="lg" radius="md" variant="light">
            {isExercise ? <IconClipboardList size={20} /> : <IconBook size={20} />}
          </ThemeIcon>
          <div>
            <Title order={2}>
              {isExercise ? 'Exercise' : 'Course'} Overview
            </Title>
            <Text size="sm" color="dimmed">
              Module {moduleId.replace('module', '')} {isExercise ? 'Exercises' : 'Content'}
            </Text>
          </div>
        </Group>
        
        <Paper shadow="xs" p="md" radius="md" withBorder>
          <Title order={4} mb="md">
            Available {isExercise ? 'Exercises' : 'Topics'}
          </Title>
          <List spacing="sm">
            {links.map((link, index) => (
              <List.Item key={index}>
                <Text 
                  component={Link} 
                  to={`/courses/${courseId}/${moduleId}/${isExercise ? 'exercise' : 'course'}${link.to}`}
                  style={{ textDecoration: 'none' }}
                  color="blue"
                  fw={500}
                >
                  {link.label}
                </Text>
                {link.subLinks && link.subLinks.length > 0 && (
                  <List withPadding mt="xs">
                    {link.subLinks.map((subLink, subIndex) => (
                      <List.Item key={subIndex}>
                        <Text 
                          component="a" 
                          href={`/courses/${courseId}/${moduleId}/${isExercise ? 'exercise' : 'course'}${link.to}#${subLink.id}`}
                          size="sm"
                          color="dimmed"
                          style={{ textDecoration: 'none' }}
                        >
                          {subLink.label}
                        </Text>
                      </List.Item>
                    ))}
                  </List>
                )}
              </List.Item>
            ))}
          </List>
        </Paper>
      </Stack>
    </Container>
  );
};

export default ModuleTableOfContents;