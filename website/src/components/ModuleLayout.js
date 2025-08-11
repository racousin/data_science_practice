import React from 'react';
import { Box, Paper, Group, Title, Button, Anchor } from '@mantine/core';
import { useNavigate, Link } from 'react-router-dom';
import { IconArrowLeft, IconArrowRight, IconBook, IconClipboardList } from '@tabler/icons-react';

const ModuleLayout = ({ 
  module, 
  isCourse = null, 
  title, 
  children, 
  courseLinks,
  courseId = "data-science-practice"
}) => {
  const navigate = useNavigate();
  
  // Helper to get prev/next module number for navigation
  const getAdjacentModules = () => {
    if (typeof module === 'number') {
      return {
        prev: module > 1 ? module - 1 : null,
        next: module < 15 ? module + 1 : null
      };
    }
    return { prev: null, next: null };
  };
  
  const { prev, next } = getAdjacentModules();
  
  return (
    <Box>
      {/* Module navigation header */}
      <Paper 
        shadow="xs" 
        p="md" 
        mb="md"
        style={{
          position: 'sticky',
          top: 60,
          zIndex: 10,
          backgroundColor: 'white'
        }}
      >
        <Group position="apart">
          <Title order={2}>{title}</Title>
          <Group>
            {/* Previous module button */}
            {prev && (
              <Button 
                variant="outline" 
                color="blue"
                leftSection={<IconArrowLeft size={16} />}
                component={Link}
                to={`/courses/${courseId}/module${prev}/course`}
              >
                Previous Module
              </Button>
            )}
            
            {/* Toggle course/exercise button */}
            <Button 
              variant="outline" 
              color="gray"
              leftSection={isCourse ? <IconClipboardList size={16} /> : <IconBook size={16} />}
              component={Link}
              to={`/courses/${courseId}/${
                typeof module === 'number' ? `module${module}` : module
              }/${isCourse ? "exercise" : "course"}`}
            >
              {isCourse ? "Exercises" : "Course"}
            </Button>
            
            {/* Next module button */}
            {next && (
              <Button 
                variant="outline" 
                color="green"
                rightSection={<IconArrowRight size={16} />}
                component={Link}
                to={`/courses/${courseId}/module${next}/course`}
              >
                Next Module
              </Button>
            )}
          </Group>
        </Group>
      </Paper>
      
      {/* Content */}
      <Box>
        {children}
      </Box>
    </Box>
  );
};

export default ModuleLayout;