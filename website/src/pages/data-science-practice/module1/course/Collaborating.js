import React from "react";
import { Container, Grid, Image, Title, Text, List } from '@mantine/core';

const Collaborating = () => {
  return (
    <Container fluid>
      <div data-slide>
        <Title order={2} mb="md">Collaborating</Title>
        <Text size="md" mb="md">
          Collaboration is essential in software development for scaling projects and improving code quality.
          Git, along with hosting services like GitHub, provides powerful tools to enhance collaboration among developers.
          This guide will walk you through the process of creating, reviewing, and merging pull requests on GitHub.
        </Text>
      </div>

      <div data-slide>
        <Grid>
          <Grid.Col>
            <Title order={3} id="creating-pull-request" mb="md">Creating a Pull Request</Title>
            <Text size="md" mb="md">
              After you've made changes in your feature branch and pushed it to GitHub, follow these steps to create a pull request:
            </Text>
            <List type="ordered" spacing="sm">
              <List.Item>Go to your repository on GitHub.</List.Item>
              <List.Item>Click on the "Pull requests" tab.</List.Item>
              <List.Item>Click the "New pull request" button.</List.Item>
              <List.Item>Select your feature branch as the compare branch and the main branch as the base branch.</List.Item>
              <List.Item>Review your changes and click "Create pull request".</List.Item>
              <List.Item>Add a title and description for your pull request, explaining the changes you've made.</List.Item>
              <List.Item>Click "Create pull request" to submit it.</List.Item>
            </List>
          </Grid.Col>
        </Grid>
      </div>

      <div data-slide>
        <Image src="/assets/data-science-practice/module1/createPR.png" alt="Creating a Pull Request on GitHub" fluid />
      </div>

      <div data-slide>
        <Grid>
          <Grid.Col>
            <Title order={3} id="review-process" mb="md">The Review Process</Title>
            <Text size="md" mb="md">
              Once you've created a pull request, it's time for others to review your code:
            </Text>
            <List type="ordered" spacing="sm">
              <List.Item>
                <strong>Request reviewers:</strong> Click on the gear icon in the "Reviewers" section of your pull request and select team members to review your code.
              </List.Item>
            </List>
          </Grid.Col>
        </Grid>
      </div>

      <div data-slide>
        <Image src="/assets/data-science-practice/module1/askReview.png" alt="Reviewing a Pull Request on GitHub" fluid />
      </div>

      <div data-slide>
        <List type="ordered" spacing="sm" start={2}>
          <List.Item>
            <strong>Reviewers examine the code:</strong> They will look at your changes, leave comments, and suggest improvements.
          </List.Item>
        </List>
        <Image src="/assets/data-science-practice/module1/review.png" alt="Reviewing a Pull Request on GitHub" fluid />
      </div>

      <div data-slide>
        <List type="ordered" spacing="sm" start={3}>
          <List.Item>
            <strong>Address feedback:</strong> Make any necessary changes based on the reviews and push new commits to your branch.
          </List.Item>
          <List.Item>
            <strong>Re-request review:</strong> After making changes, you can re-request a review.
          </List.Item>
        </List>
      </div>

      <div data-slide>
        <Grid>
          <Grid.Col>
            <Title order={3} id="merging-pull-request" mb="md">Merging a Pull Request</Title>
            <Text size="md" mb="md">
              Once your pull request has been approved, you can merge it into the main branch:
            </Text>
            <List type="ordered" spacing="sm">
              <List.Item>Go to your pull request page on GitHub.</List.Item>
              <List.Item>If all checks have passed and you have the necessary approvals, the "Merge pull request" button will be green.</List.Item>
              <List.Item>Click "Merge pull request".</List.Item>
              <List.Item>Confirm the merge by clicking "Confirm merge".</List.Item>
            </List>
          </Grid.Col>
        </Grid>
      </div>

      <div data-slide>
        <Image src="/assets/data-science-practice/module1/mergePR.png" alt="Merging a Pull Request on GitHub" fluid />
      </div>

      <div data-slide>
        <Grid>
          <Grid.Col>
            <Title order={3} id="best-practices" mb="md">Best Practices</Title>
            <List spacing="sm">
              <List.Item>Keep your pull requests small and focused on a single feature or bug fix.</List.Item>
              <List.Item>Write clear and descriptive titles and descriptions for your pull requests.</List.Item>
              <List.Item>Respond promptly to review comments and be open to feedback.</List.Item>
              <List.Item>If you're a reviewer, be constructive and specific in your feedback.</List.Item>
            </List>
          </Grid.Col>
        </Grid>
      </div>
    </Container>
  );
};

export default Collaborating;