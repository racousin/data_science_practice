import React from 'react';
import { Group, Text, Button, Container, ActionIcon, useMantineColorScheme } from '@mantine/core';
import { Link } from 'react-router-dom';
import { IconHome, IconSchool, IconSun, IconMoon } from '@tabler/icons-react';

const MainHeader = () => {
  const { colorScheme, toggleColorScheme } = useMantineColorScheme();
  
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
          Raphael Cousin
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
          
          <ActionIcon
            variant="subtle"
            color="white"
            onClick={() => toggleColorScheme()}
            size="lg"
          >
            {colorScheme === 'dark' ? <IconSun size={18} /> : <IconMoon size={18} />}
          </ActionIcon>
        </Group>
      </Group>
    </Container>
  );
};

export default MainHeader;