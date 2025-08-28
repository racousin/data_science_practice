import React from "react";
import { Container, Grid, Image } from '@mantine/core';
const Collaborating = () => {
  return (
    <Container fluid>
      <h2>Collaborating</h2>
      <p>
        Collaboration is essential in software development for scaling projects and improving code quality. 
        Git, along with hosting services like GitHub, provides powerful tools to enhance collaboration among developers. 
        This guide will walk you through the process of creating, reviewing, and merging pull requests on GitHub.
      </p>
      <Grid className="mt-4">
        <Grid.Col>
          <h3 id="creating-pull-request">Creating a Pull Request</h3>
          <p>
            After you've made changes in your feature branch and pushed it to GitHub, follow these steps to create a pull request:
          </p>
          <ol>
            <li>Go to your repository on GitHub.</li>
            <li>Click on the "Pull requests" tab.</li>
            <li>Click the "New pull request" button.</li>
            <li>Select your feature branch as the compare branch and the main branch as the base branch.</li>
            <li>Review your changes and click "Create pull request".</li>
            <li>Add a title and description for your pull request, explaining the changes you've made.</li>
            <li>Click "Create pull request" to submit it.</li>
          </ol>
          <Image src="/assets/data-science-practice/module1/createPR.png" alt="Creating a Pull Request on GitHub" fluid className="my-3" />
        </Grid.Col>
      </Grid>
      <Grid className="mt-4">
        <Grid.Col>
          <h3 id="review-process">The Review Process</h3>
          <p>
            Once you've created a pull request, it's time for others to review your code:
          </p>
          <ol>
            <li>
              <strong>Request reviewers:</strong> Click on the gear icon in the "Reviewers" section of your pull request and select team members to review your code.
              <Image src="/assets/data-science-practice/module1/askReview.png" alt="Reviewing a Pull Request on GitHub" fluid className="my-3" />
            </li>
            <li>
              <strong>Reviewers examine the code:</strong> They will look at your changes, leave comments, and suggest improvements.
              <Image src="/assets/data-science-practice/module1/review.png" alt="Reviewing a Pull Request on GitHub" fluid className="my-3" />
            </li>
            <li>
              <strong>Address feedback:</strong> Make any necessary changes based on the reviews and push new commits to your branch.
            </li>
            <li>
              <strong>Re-request review:</strong> After making changes, you can re-request a review.
            </li>
          </ol>
        </Grid.Col>
      </Grid>
      <Grid className="mt-4">
        <Grid.Col>
          <h3 id="merging-pull-request">Merging a Pull Request</h3>
          <p>
            Once your pull request has been approved, you can merge it into the main branch:
          </p>
          <ol>
            <li>Go to your pull request page on GitHub.</li>
            <li>If all checks have passed and you have the necessary approvals, the "Merge pull request" button will be green.</li>
            <li>Click "Merge pull request".</li>
            <li>Confirm the merge by clicking "Confirm merge".</li>
          </ol>
          <Image src="/assets/data-science-practice/module1/mergePR.png" alt="Merging a Pull Request on GitHub" fluid className="my-3" />
        </Grid.Col>
      </Grid>
      <Grid className="mt-4">
        <Grid.Col>
          <h3 id="best-practices">Best Practices</h3>
          <ul>
            <li>Keep your pull requests small and focused on a single feature or bug fix.</li>
            <li>Write clear and descriptive titles and descriptions for your pull requests.</li>
            <li>Respond promptly to review comments and be open to feedback.</li>
            <li>If you're a reviewer, be constructive and specific in your feedback.</li>
          </ul>
        </Grid.Col>
      </Grid>
    </Container>
  );
};
export default Collaborating;