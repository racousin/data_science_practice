import React from "react";
import { Container, Box } from '@mantine/core';
import SlideView from './SlideView';

const ModuleFrame = ({ module, isCourse, title, children, enableSlides = false }) => {
  return (
    <Container size="xl">
      <Box className="module-content">
        <SlideView enabled={enableSlides}>
          {children}
        </SlideView>
      </Box>
    </Container>
  );
};

export default ModuleFrame;
