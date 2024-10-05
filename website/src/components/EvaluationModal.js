import React from 'react';
import { useDisclosure } from '@mantine/hooks';
import { Modal, Button, Text, Alert, Accordion, Container, Title, List, Anchor } from '@mantine/core';
import { IconAlertCircle, IconGitBranch, IconCodeDots, IconGitPullRequest } from '@tabler/icons-react';
import CodeBlock from "components/CodeBlock";

const EvaluationModal = ({ module }) => {
  const [opened, { open, close }] = useDisclosure(false);

  return (
    <>
      <Button onClick={open} variant="outline" color="blue">
        Submit your exercises
      </Button>

      <Modal opened={opened} onClose={close} size="lg" title="Exercise Submission Guidelines">
        <Container>
          <Alert icon={<IconAlertCircle size="1rem" />} title="Prerequisites" color="blue" mb="md">
            Make sure to complete the initial setup exercises in{' '}
            <Anchor href="/module0/course" target="_blank">Module 0</Anchor> before proceeding.
          </Alert>

          <Title order={2} mb="md">Steps to Submit Your Exercise</Title>

          <Accordion variant="separated">
            <Accordion.Item value="initial-setup">
              <Accordion.Control icon={<IconGitBranch size="1.2rem" />}>
                Initial Setup
              </Accordion.Control>
              <Accordion.Panel>
                <Text mb="xs">Navigate to your project directory:</Text>
                <CodeBlock code="cd data_science_practice_2024" />
                <Text mb="xs" mt="sm">Ensure you are on the main branch and your repository is up to date:</Text>
                <CodeBlock code={`git checkout main
git pull origin main`} />
              </Accordion.Panel>
            </Accordion.Item>

            <Accordion.Item value="prepare-work">
              <Accordion.Control icon={<IconCodeDots size="1.2rem" />}>
                Prepare Your Work
              </Accordion.Control>
              <Accordion.Panel>
                <Text mb="xs">Create and switch to a new branch for your exercise:</Text>
                <CodeBlock code={`git checkout -b exercise_branch/$username/module${module}`} />
                <Text mb="xs" mt="sm">Create a directory for your module (if it doesn't already exist):</Text>
                <CodeBlock code={`mkdir -p $username/module${module}`} />
                <Text mt="sm">Perform your work in this directory.</Text>
              </Accordion.Panel>
            </Accordion.Item>

            <Accordion.Item value="submit-exercise">
              <Accordion.Control icon={<IconGitPullRequest size="1.2rem" />}>
                Submit Your Exercise
              </Accordion.Control>
              <Accordion.Panel>
                <Text mb="xs">Stage your changes for commit:</Text>
                <CodeBlock code={`git add $username/module${module}/your_files`} />
                <Text mb="xs" mt="sm">Commit your changes:</Text>
                <CodeBlock code="git commit -m 'Update exercise files'" />
                <Text mb="xs" mt="sm">Push your branch to the repository:</Text>
                <CodeBlock code={`git push origin exercise_branch/$username/module${module}`} />
              </Accordion.Panel>
            </Accordion.Item>

            <Accordion.Item value="create-pr">
              <Accordion.Control icon={<IconGitPullRequest size="1.2rem" />}>
                Create a Pull Request
              </Accordion.Control>
              <Accordion.Panel>
                <Text mb="xs">After pushing your work to your remote branch:</Text>
                <List>
                  <List.Item>Visit <Anchor href="https://github.com/racousin/data_science_practice_2024/pulls" target="_blank">GitHub Pull Requests</Anchor>.</List.Item>
                  <List.Item>Create a new pull request from your exercise branch to the main branch.</List.Item>
                  <List.Item>Request a review and make necessary changes based on feedback.</List.Item>
                  <List.Item>Merge the pull request once approved.</List.Item>
                </List>
                <Text mt="sm">
                  After merging, you can consult your results in the{' '}
                  <Anchor href="/repositories" target="_blank">repository results page</Anchor>.
                </Text>
              </Accordion.Panel>
            </Accordion.Item>
          </Accordion>
        </Container>
      </Modal>
    </>
  );
};

export default EvaluationModal;