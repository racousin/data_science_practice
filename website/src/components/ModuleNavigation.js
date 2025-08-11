import React, { useEffect, useState } from 'react';
import { Group, Title, Box } from '@mantine/core';
import EvaluationModal from "components/EvaluationModal";

const ModuleNavigation = ({ module, isCourse, title = "" }) => {
  const [navBarHeight, setNavBarHeight] = useState(0);

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
        {!isCourse && module > 0 && (
          <EvaluationModal module={module} />
        )}
      </Group>
    </Box>
  );
};

export default ModuleNavigation;