import React from 'react';
import { Group, Text, Button, Container } from '@mantine/core';
import { Link } from 'react-router-dom';
import { IconHome, IconSchool } from '@tabler/icons-react';

const MainHeader = () => {
  
  return (
    <Container fluid h="100%" style={{ backgroundColor: '#1a1b1e' }}>
      <Group justify="space-between" align="center" h="100%" px="md">
        <Text 
          component={Link} 
          to="/" 
          size="xl" 
          fw={700} 
          c="white"
          style={{ textDecoration: 'none' }}
        >
          Raphaël Cousin
        </Text>
        
        <Group>
          <Button 
            component={Link} 
            to="/" 
            variant="subtle"
            c="white"
            leftSection={<IconHome size={18} />}
          >
            Home
          </Button>
          
          <Button 
            component={Link} 
            to="/courses" 
            variant="subtle"
            c="white"
            leftSection={<IconSchool size={18} />}
          >
            Teaching
          </Button>
        </Group>
      </Group>
    </Container>
  );
};

export default MainHeader;