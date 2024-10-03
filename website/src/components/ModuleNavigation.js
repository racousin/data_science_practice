import React, { useEffect, useState } from 'react';
import { Group, Button, Title, Box } from '@mantine/core';
import { useNavigate } from 'react-router-dom';
import EvaluationModal from "components/EvaluationModal";

const ModuleNavigation = ({ module, isCourse, title = "" }) => {
  const [navBarHeight, setNavBarHeight] = useState(0);
  const navigate = useNavigate();

  useEffect(() => {
    const updateNavBarHeight = () => {
      const navbar = document.querySelector('.mantine-AppShell-header');
      if (navbar) {
        setNavBarHeight(navbar.offsetHeight);
      }
    };

    updateNavBarHeight();
    window.addEventListener('resize', updateNavBarHeight);
    return () => window.removeEventListener('resize', updateNavBarHeight);
  }, []);

  const navigateTo = (path) => {
    navigate(path);
  };

  return (
    <Box 
      className="module-navigation"
      style={{ 
        position: 'sticky', 
        top: navBarHeight, 
        zIndex: 10, 
        backgroundColor: 'white', 
        padding: '15px', 
        borderBottom: '1px solid #e0e0e0',
        marginBottom: '20px'
      }}
    >
      <Group justify="space-between" position="apart">
        <Title order={2}>{title}</Title>
        <Group>
          {module > 0 && (
            <>
            <Button 
              variant="outline" 
              color="blue"
              onClick={() => navigateTo(`/module${module - 1}/course`)}
            >
              Previous Module
            </Button>
          
          <Button 
            variant="outline" 
              color="gray"
              onClick={() => navigateTo(`/module${module}/${isCourse ? "exercise" : "course"}`)}
            >
              {isCourse ? "Exercises" : "Course"}
          </Button>
          
          {!isCourse && <EvaluationModal module={module} />}
          </>
        )}
          {module < 14 && (
            <Button 
              variant="outline" 
              color="green"
              onClick={() => navigateTo(`/module${module + 1}/course`)}
            >
              Next Module
            </Button>
          )}
        </Group>
      </Group>
    </Box>
  );
};

export default ModuleNavigation;
