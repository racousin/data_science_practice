import React from "react";
import { Container, Grid, Image, Alert, Card, Badge, List, Anchor, Text, Title, Space, Divider, Group, ThemeIcon } from '@mantine/core';
import { IconAlertCircle, IconChecks, IconGitPullRequest, IconBrandGithub, IconFileCheck, IconUsers, IconRocket } from '@tabler/icons-react';
import CodeBlock from "components/CodeBlock";
import EvaluationModal from "components/EvaluationModal";

const Exercise1 = () => {
  return (
    <Container fluid>
      <h1 className="my-4">
        Exercise 1<span style={{color: 'red', fontWeight: 'bold'}}>*</span>: Your First Pull Request - Creating and Submitting a User File
      </h1>

      <Alert icon={<IconAlertCircle size="1rem" />} title="Exercise Overview" color="blue" mb="lg">
        This exercise will guide you through the complete process of contributing to a collaborative repository.
        You will create a simple user file, submit it through a pull request, participate in the review process,
        and understand the three-stage validation system that ensures code quality.
      </Alert>

      <Title order={2} mb="md">What You Will Learn</Title>
        <List spacing="sm" size="md">
          <List.Item>How to create and format a simple information file correctly</List.Item>
          <List.Item>The complete pull request workflow from creation to merge</List.Item>
          <List.Item>How to participate in code reviews (both as author and reviewer)</List.Item>
          <List.Item>Understanding the three-stage validation system</List.Item>
          <List.Item>How to fix issues if validation fails</List.Item>
        </List>

      <Title order={2} mb="md" mt="xl">The Three-Stage Validation System</Title>
        <Text size="md" mb="md">
          Your submission will go through three validation tests to ensure quality and correctness:
        </Text>

        <Grid>
          <Grid.Col span={{ md: 4 }}>
            <Card shadow="sm" padding="lg" radius="md" withBorder>
              <ThemeIcon size="xl" radius="md" variant="light" color="blue" mb="md">
                <IconFileCheck size={28} />
              </ThemeIcon>
              <Title order={4} mb="xs">Test 1: Pre-Push Validation</Title>
              <Badge color="blue" variant="light" mb="sm">Automatic - Before Push</Badge>
              <Text size="sm">
                When you try to push your branch, an automatic test checks if your file is in the correct location
                (<code>your_username/module1/user</code>). If the file is in the wrong place, the push will be blocked.
              </Text>
            </Card>
          </Grid.Col>

          <Grid.Col span={{ md: 4 }}>
            <Card shadow="sm" padding="lg" radius="md" withBorder>
              <ThemeIcon size="xl" radius="md" variant="light" color="green" mb="md">
                <IconUsers size={28} />
              </ThemeIcon>
              <Title order={4} mb="xs">Test 2: Peer Review</Title>
              <Badge color="green" variant="light" mb="sm">Manual - Pull Request</Badge>
              <Text size="sm">
                Your pull request must be reviewed by peers. The best reviewers will receive points for their
                thorough and helpful reviews. This ensures code quality through human validation.
              </Text>
            </Card>
          </Grid.Col>

          <Grid.Col span={{ md: 4 }}>
            <Card shadow="sm" padding="lg" radius="md" withBorder>
              <ThemeIcon size="xl" radius="md" variant="light" color="orange" mb="md">
                <IconChecks size={28} />
              </ThemeIcon>
              <Title order={4} mb="xs">Test 3: Content Validation</Title>
              <Badge color="orange" variant="light" mb="sm">Automatic - After Merge</Badge>
              <Text size="sm">
                After merging to main, a final test validates the content format (username,firstname,surname).
                If incorrect, you'll need to create a new pull request to fix it.
              </Text>
            </Card>
          </Grid.Col>
        </Grid>

      <Title order={2} mb="md" mt="xl">Step 1: Prepare Your Repository</Title>

        <Text size="md" mb="md">
          Before starting, ensure your local repository is up to date with the latest changes from the main branch.
        </Text>

        <List spacing="sm" size="md" mb="md">
          <List.Item>Navigate to your project directory</List.Item>
          <List.Item>Switch to the main branch</List.Item>
          <List.Item>Pull the latest changes</List.Item>
        </List>

        <CodeBlock
          code={`cd data_science_practice_2025
git checkout main
git pull origin main`}
          language="bash"
        />

        <Alert icon={<IconAlertCircle size="1rem" />} color="yellow" mt="md">
          Always start from an updated main branch to avoid merge conflicts later!
        </Alert>

      <Title order={2} mb="md" mt="xl">Step 2: Create Your Feature Branch</Title>

        <Text size="md" mb="md">
          Create a new branch specifically for this exercise. Use a descriptive name that includes your username and the module number.
        </Text>

        <CodeBlock
          code={`git checkout -b exercise_branch/your_username/module1`}
          language="bash"
        />

        <Text size="sm" color="dimmed" mt="md">
          Replace <code>your_username</code> with your actual GitHub username. This naming convention helps identify
          your work and keeps the repository organized.
        </Text>

      <Title order={2} mb="md" mt="xl">Step 3: Create Your User File</Title>

        <Grid>
          <Grid.Col span={{ md: 6 }}>
            <Title order={3} mb="sm">File Structure</Title>
            <Text size="md" mb="md">
              Create the required directory structure and file:
            </Text>

            <CodeBlock
              code={`# Create your username directory and module1 subdirectory
mkdir -p your_username/module1

# Create the user file
touch your_username/module1/user

# Open the file in your preferred editor
# For example, using nano:
nano your_username/module1/user`}
              language="bash"
            />
          </Grid.Col>

          <Grid.Col span={{ md: 6 }}>
            <Title order={3} mb="sm">File Content Format</Title>
            <Text size="md" mb="md">
              The file must contain exactly one line with your information in this format:
            </Text>

            <CodeBlock
              code={`username,firstname,surname`}
              language="text"
            />

            <Alert icon={<IconAlertCircle size="1rem" />} color="red" mt="md">
              <strong>Important:</strong> No spaces between commas! The format must be exact.
            </Alert>

            <Title order={4} mt="md" mb="sm">Example:</Title>
            <CodeBlock
              code={`johndoe,John,Doe`}
              language="text"
            />
          </Grid.Col>
        </Grid>

      <Title order={2} mb="md" mt="xl">Step 4: Commit Your Changes</Title>

        <Text size="md" mb="md">
          After creating your file, you need to stage and commit it to your local repository.
        </Text>

        <List spacing="sm" size="md" mb="md">
          <List.Item>Check the status to see your new file</List.Item>
          <List.Item>Stage the file for commit</List.Item>
          <List.Item>Create a descriptive commit message</List.Item>
        </List>

        <CodeBlock
          code={`# Check what files have been created/modified
git status

# Stage your user file
git add your_username/module1/user

# Commit with a clear message
git commit -m "Add user file for module 1 exercise"`}
          language="bash"
        />

        <Text size="sm" color="dimmed" mt="md">
          Good commit messages help reviewers understand what changes you've made.
        </Text>

      <Title order={2} mb="md" mt="xl">Step 5: Push to GitHub (First Validation Test)</Title>

        <Text size="md" mb="md">
          When you push your branch, the first automatic validation will run:
        </Text>

        <CodeBlock
          code={`git push origin exercise_branch/your_username/module1`}
          language="bash"
        />

        <Alert icon={<IconFileCheck size="1rem" />} color="blue" mt="md" mb="md">
          <strong>Test 1 - File Location Check:</strong> The system will verify that your file is in the correct location
          (<code>your_username/module1/user</code>). If not, the push will be rejected.
        </Alert>

      <Title order={2} mb="md" mt="xl">Step 6: Create a Pull Request</Title>

        <Text size="md" mb="md">
          After successfully pushing your branch, navigate to GitHub to create a pull request:
        </Text>

        <List spacing="md" size="md" mb="lg">
          <List.Item>
            Go to the repository: <Anchor href="https://github.com/racousin/data_science_practice_2025" target="_blank">
              data_science_practice_2025 <IconBrandGithub size={16} style={{verticalAlign: 'middle'}} />
            </Anchor>
          </List.Item>
          <List.Item>Click on the "Pull requests" tab</List.Item>
          <List.Item>Click "New pull request"</List.Item>
          <List.Item>Select your branch as the compare branch</List.Item>
          <List.Item>Ensure main is the base branch</List.Item>
        </List>

        <Image
          src="/assets/data-science-practice/module1/createPR.png"
          alt="Creating a Pull Request on GitHub"
          radius="md"
          mb="md"
        />

        <Alert icon={<IconGitPullRequest size="1rem" />} color="green" mt="md">
          Add a clear title and description explaining what you've done. This helps reviewers understand your changes.
        </Alert>

      <Title order={2} mb="md" mt="xl">Step 7: The Review Process (Second Validation Test)</Title>

        <Grid>
          <Grid.Col span={{ md: 6 }}>
            <Title order={3} mb="sm">As the Author</Title>

            <List spacing="sm" size="md" mb="md">
              <List.Item>Request reviews from your peers</List.Item>
              <List.Item>Respond to feedback constructively</List.Item>
              <List.Item>Make requested changes if needed</List.Item>
              <List.Item>Push updates to the same branch</List.Item>
            </List>

            <Image
              src="/assets/data-science-practice/module1/askReview.png"
              alt="Requesting a review"
              radius="md"
            />
          </Grid.Col>

          <Grid.Col span={{ md: 6 }}>
            <Title order={3} mb="sm">As a Reviewer</Title>

            <Alert icon={<IconUsers size="1rem" />} color="green" mb="md">
              <strong>Earn Points!</strong> The best reviewers receive points for helpful, thorough reviews.
            </Alert>

            <List spacing="sm" size="md" mb="md">
              <List.Item>Check file location and naming</List.Item>
              <List.Item>Verify content format</List.Item>
              <List.Item>Provide constructive feedback</List.Item>
              <List.Item>Approve or request changes</List.Item>
            </List>

            <Image
              src="/assets/data-science-practice/module1/review.png"
              alt="Reviewing code"
              radius="md"
            />
          </Grid.Col>
        </Grid>

        <Text size="sm" mt="md">
          Link to pull requests: <Anchor href="https://github.com/racousin/data_science_practice_2025/pulls" target="_blank">
            View all pull requests <IconBrandGithub size={16} style={{verticalAlign: 'middle'}} />
          </Anchor>
        </Text>

      <Title order={2} mb="md" mt="xl">Step 8: Merge Your Pull Request</Title>

        <Text size="md" mb="md">
          Once your pull request has been approved by reviewers and all checks pass:
        </Text>

        <List spacing="md" size="md" mb="lg">
          <List.Item>The "Merge pull request" button will turn green</List.Item>
          <List.Item>Click "Merge pull request"</List.Item>
          <List.Item>Confirm the merge</List.Item>
          <List.Item>Your changes are now in the main branch!</List.Item>
        </List>

        <Image
          src="/assets/data-science-practice/module1/mergePR.png"
          alt="Merging a Pull Request"
          radius="md"
          mb="md"
        />

        <Alert icon={<IconRocket size="1rem" />} color="green" mt="md">
          Congratulations! Your code is now part of the main repository.
        </Alert>

      <Title order={2} mb="md" mt="xl">Step 9: Final Validation (Third Test)</Title>

        <Text size="md" mb="md">
          After merging, an automatic test validates your file content:
        </Text>

        <Card shadow="sm" padding="lg" radius="md" withBorder mb="md">
          <Title order={4} mb="sm">Content Validation Check</Title>
          <Text size="md" mb="sm">The system verifies:</Text>
          <List spacing="xs" size="sm">
            <List.Item>File exists at <code>your_username/module1/user</code></List.Item>
            <List.Item>Content follows the format: <code>username,firstname,surname</code></List.Item>
            <List.Item>No extra spaces or characters</List.Item>
            <List.Item>Exactly one line of text</List.Item>
          </List>
        </Card>

        <Alert icon={<IconAlertCircle size="1rem" />} color="orange" mb="md">
          <strong>If Validation Fails:</strong> Your work is already in main, but marked as incorrect.
          You'll need to create a new pull request to fix the issues.
        </Alert>

        <Title order={3} mt="lg" mb="sm">How to Fix Failed Validation:</Title>

        <CodeBlock
          code={`# Update your main branch
git checkout main
git pull origin main

# Create a fix branch
git checkout -b fix/your_username/module1

# Edit your file to correct the format
nano your_username/module1/user

# Commit and push the fix
git add your_username/module1/user
git commit -m "Fix user file format for module 1"
git push origin fix/your_username/module1`}
          language="bash"
        />

        <Text size="md" mt="md">
          Then create a new pull request with your fixes. The validation will run again after merge.
        </Text>

      <Title order={2} mb="md" mt="xl">Checking Your Results</Title>

        <Text size="md" mb="md">
          After your pull request is merged, you can check your exercise results:
        </Text>

        <List spacing="md" size="md">
          <List.Item>
            Visit the <Anchor href="/repositories" target="_blank">Repository Results Page</Anchor>
          </List.Item>
          <List.Item>Find your username in the list</List.Item>
          <List.Item>Check the status of Module 1 Exercise 1</List.Item>
          <List.Item>Green checkmark = Success! Red X = Needs fixing</List.Item>
        </List>

        <Space h="md" />

        <Divider my="lg" />

        <Title order={2} mb="md">Ready to Submit?</Title>

        <Text size="md" mb="md">
          Use the button below to access detailed submission instructions and commands:
        </Text>

        <EvaluationModal module={1} />

        <Alert icon={<IconChecks size="1rem" />} color="teal" mt="lg">
          <strong>Success Criteria:</strong> Your exercise is complete when all three validation tests pass
          and your file appears correctly in the main branch with the proper format.
        </Alert>

      <Title order={2} mb="md" mt="xl">Common Issues and Solutions</Title>

        <Grid>
          <Grid.Col span={{ md: 6 }}>
            <Card shadow="sm" padding="md" radius="md" withBorder>
              <Title order={4} mb="sm">Push Rejected</Title>
              <Text size="sm" color="dimmed" mb="xs">Error: File not in correct location</Text>
              <Text size="sm">
                Ensure your file path is exactly: <code>your_username/module1/user</code>
              </Text>
            </Card>
          </Grid.Col>

          <Grid.Col span={{ md: 6 }}>
            <Card shadow="sm" padding="md" radius="md" withBorder>
              <Title order={4} mb="sm">Validation Failed</Title>
              <Text size="sm" color="dimmed" mb="xs">Error: Invalid format</Text>
              <Text size="sm">
                Check for spaces between commas. Format must be: <code>username,firstname,surname</code>
              </Text>
            </Card>
          </Grid.Col>

          <Grid.Col span={{ md: 6 }} mt="md">
            <Card shadow="sm" padding="md" radius="md" withBorder>
              <Title order={4} mb="sm">Merge Conflicts</Title>
              <Text size="sm" color="dimmed" mb="xs">Error: Cannot merge</Text>
              <Text size="sm">
                Pull latest changes from main, resolve conflicts locally, then push again.
              </Text>
            </Card>
          </Grid.Col>

          <Grid.Col span={{ md: 6 }} mt="md">
            <Card shadow="sm" padding="md" radius="md" withBorder>
              <Title order={4} mb="sm">Review Requested Changes</Title>
              <Text size="sm" color="dimmed" mb="xs">Status: Changes requested</Text>
              <Text size="sm">
                Address reviewer feedback, push updates to the same branch, and re-request review.
              </Text>
            </Card>
          </Grid.Col>
        </Grid>
    </Container>
  );
};

export default Exercise1;