import React from "react";
import { Container, Box } from '@mantine/core';

const ModuleFrame = ({ module, isCourse, title, children }) => {
  return (
    <Container size="xl">
      <Box className="module-content">
        {children}
      </Box>
    </Container>
  );
};

export default ModuleFrame;
