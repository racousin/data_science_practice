import React from "react";
import { Container, Grid, Stack, Tabs, Image, Title, Text, List, Code, Flex } from '@mantine/core';
import CodeBlock from "components/CodeBlock";

const WorkingWithRemoteRepositories = () => {
  return (
    <Container fluid>
      <div data-slide>
        <Grid>
          <Grid.Col>
            <Title order={3} id="why-remote-repositories" mb="md">
              Why Remote Repositories are Useful
            </Title>
            <Text size="md" mb="md">
              Remote repositories on platforms like GitHub allow developers to
              store versions of their projects online, facilitating
              collaboration, backup, and public sharing. They enable teams to
              work together on projects from different locations, track
              changes, merge contributions, and maintain a history of all
              modifications.
            </Text>
          </Grid.Col>
        </Grid>
      </div>

      <div data-slide>
        <Grid className="justify-content-center">
          <Grid.Col span={{ xs: 12 }} md={10} lg={8}>
            <Flex direction="column" align="center">
              <Image
                src="/assets/data-science-practice/module1/Git_Remote_Workflow.png"
                alt="Git_Remote_Workflow"
                fluid
                style={{ maxWidth: 'min(800px, 90vw)', height: 'auto' }}
              />
              <Text>Git_Remote_Workflow</Text>
            </Flex>
          </Grid.Col>
        </Grid>
      </div>

      <div data-slide>
        <Grid>
          <Grid.Col>
            <Title order={3} id="creating-repository-github" mb="md">
              Creating a Repository on GitHub
            </Title>
            <Text size="md" mb="md">
              Creating a repository on GitHub is the first step toward remote
              project management. Here's how to set one up:
            </Text>
            <List type="ordered" spacing="sm">
              <List.Item>Log in to your GitHub account.</List.Item>
              <List.Item>Navigate to the Repositories tab and click 'New'.</List.Item>
              <List.Item>
                Enter a name for your repository and select the visibility
                (public or private).
              </List.Item>
              <List.Item>
                Optionally, initialize the repository with a README,
                .gitignore, or license.
              </List.Item>
              <List.Item>Click 'Create repository'.</List.Item>
            </List>
          </Grid.Col>
        </Grid>
      </div>

      <div data-slide>
        <Grid>
          <Grid.Col>
            <Title order={2} mb="md">Connecting to Remote Repositories</Title>
            <List type="ordered" spacing="sm">
              <List.Item>Navigate to the Repository of interest.</List.Item>
              <List.Item>Click &lt;&gt; Code ▼.</List.Item>
              <List.Item>Select SSH.</List.Item>
              <List.Item>Copy the &lt;remote_repository_url&gt;</List.Item>
            </List>
          </Grid.Col>
        </Grid>
      </div>

      <div data-slide>
        <div className="mytab">
          <Tabs defaultValue="clone">
            <Tabs.List>
              <Tabs.Tab value="clone" className="mx-1">
                Clone Remote Repository
              </Tabs.Tab>
              <Tabs.Tab value="add" className="mx-1">
                Add to Existing Local Repository
              </Tabs.Tab>
            </Tabs.List>

            <Tabs.Panel value="add">
              <Title order={4} mb="md">Add a Remote Repository</Title>
              <Text size="md" mb="md">
                Link your existing local repository to a remote
                repository on GitHub using the following command:
              </Text>
              <CodeBlock
                code={`git remote add origin <remote_repository_url>`}
              />
              <Text size="md" mb="md">
                This command sets the specified URL as 'origin', which
                is the conventional name used by Git to reference the
                primary remote.
              </Text>
            </Tabs.Panel>
            <Tabs.Panel value="clone">
              <Title order={4} mb="md">Clone a Remote Repository</Title>
              <Text size="md" mb="md">
                To work on an existing project, clone the remote
                repository with this command:
              </Text>
              <CodeBlock code={`git clone <remote_repository_url>`} />
              <Text size="md" mb="md">
                Cloning creates a local copy of the repository,
                including all historical commits and branches.
              </Text>
            </Tabs.Panel>
          </Tabs>
        </div>
      </div>

      <div data-slide>
        <Grid>
          <Grid.Col>
            <Title order={3} id="view-remote-repositories" mb="md">View Remote Repositories</Title>
            <Text size="md" mb="md">
              The <Code>git remote -v</Code> command is used to view all the
              remote repositories that your current repository knows about.
              Here's what the command does:
            </Text>
            <List spacing="sm">
              <List.Item>
                <strong>List Remotes:</strong> This command lists all remote
                handles (short names) for the remote URLs configured.
              </List.Item>
              <List.Item>
                <strong>Show URLs:</strong> Alongside each remote handle, the
                command also shows the associated URLs, which can be either
                fetch (data fetch from) or push (data send to) URLs.
              </List.Item>
            </List>
          </Grid.Col>
        </Grid>
      </div>

      <div data-slide>
        <CodeBlock code="git remote -v" />
        <CodeBlock
          code={`$ git remote -v
origin	git@github.com:racousin/data_science_practice_2025.git (fetch)
origin	git@github.com:racousin/data_science_practice_2025.git (push)
`}
          language=""
        />
        <Text size="md" mb="md">
          This output is particularly useful for verifying which remotes
          are set up for fetch and push operations, ensuring that you have
          the correct access paths for collaboration and deployment.
        </Text>
      </div>

      <div data-slide>
        <Grid>
          <Grid.Col>
            <Title order={3} id="fetch-changes" mb="md">Fetch Changes from a Remote Repository</Title>
            <Text size="md" mb="md">
              The <Code>git fetch origin</Code> command is used to fetch
              branches, tags, and other data from a remote repository
              identified by 'origin'. This command prepares your local
              repository for a merge by updating your remote tracking branches
              with the latest changes from the remote without altering your
              current working directory.
            </Text>
          </Grid.Col>
        </Grid>
      </div>

      <div data-slide>
        <List spacing="sm">
          <List.Item>
            <strong>Download Data:</strong> This command downloads
            commits, files, and refs from a remote repository into your
            local repo's working directory.
          </List.Item>
          <List.Item>
            <strong>Update Tracking Branches:</strong>{" "}
            <Code>git fetch</Code> updates your tracking branches under{" "}
            <Code>refs/remotes/origin/</Code>, which represent the state
            of your branches at the remote repository.
          </List.Item>
          <List.Item>
            <strong>No Workspace Changes:</strong> Fetching does not
            change your own local branches and does not modify your
            current workspace. It fetches the data but leaves your current
            branch unchanged, ensuring that your local development is not
            automatically disrupted by remote changes.
          </List.Item>
        </List>
        <CodeBlock code="git fetch origin" />
      </div>

      <div data-slide>
        <Text size="md" mb="md">
          After fetching, you may want to integrate these updates into
          your local branch, which involves an additional step:
        </Text>
        <Title order={3} id="merge-fetched-changes" mb="md">Merge Fetched Changes</Title>
        <Text size="md" mb="md">
          To merge the fetched changes into your current branch, you use
          the <Code>git merge</Code> command. This would typically involve
          merging the fetched branch (like <Code>origin/main</Code>) into
          your current branch:
        </Text>
        <CodeBlock code="git merge origin/main" />
      </div>

      <div data-slide>
        <Text size="md" mb="md">
          This command will merge the latest changes from the remote
          'main' branch into your current branch, allowing you to
          synchronize your local development with the latest updates from
          the remote repository.
        </Text>
        <Text size="md" mb="md">
          It's important to note that if there are any conflicts between
          the new changes and your local changes, you'll need to resolve
          them manually before completing the merge.
        </Text>
      </div>

      <div data-slide>
        <Grid>
          <Grid.Col>
            <Title order={3} id="pull-changes" mb="md">Pull Changes from a Remote Repository</Title>
            <Text size="md" mb="md">
              The <Code>git pull origin main</Code> command combines two
              distinct operations performed by Git:
            </Text>
            <List spacing="sm">
              <List.Item>
                <strong>Fetch:</strong> First, <Code>git pull</Code> executes{" "}
                <Code>git fetch</Code> which downloads content from the
                specified remote repository under the branch named 'main'.
                This step involves retrieving all the new commits, branches,
                and files from the remote repository.
              </List.Item>
              <List.Item>
                <strong>Merge:</strong> After fetching, Git automatically
                attempts to merge the new commits into your current local
                branch. If you are on the 'main' branch locally, it will merge
                the changes from 'origin/main' into your local 'main' branch.
              </List.Item>
            </List>
          </Grid.Col>
        </Grid>
      </div>

      <div data-slide>
        <CodeBlock code="git pull origin main" />
        <Text size="md" mb="md">
          This command is crucial for keeping your local development
          environment updated with changes made by other collaborators in
          the repository. It ensures that you are working on the latest
          version of the project, reducing conflicts and inconsistencies.
        </Text>
        <Grid className="justify-content-center">
          <Grid.Col span={{ xs: 12 }} md={10} lg={8}>
            <Flex direction="column" align="center">
              <Image
                src="/assets/data-science-practice/module1/view.png"
                alt="Git_Fetch_Merge_Pull"
                fluid
                style={{ maxWidth: 'min(700px, 70vw)', height: 'auto' }}
              />
              <Text>Git_Fetch_Merge_Pull</Text>
            </Flex>
          </Grid.Col>
        </Grid>
      </div>

      <div data-slide>
        <Grid>
          <Grid.Col>
            <Title order={3} id="push-changes" mb="md">Push Changes to a Remote Repository</Title>
            <Text size="md" mb="md">
              The <Code>git push origin main</Code> command sends the commits
              made on your local branch 'main' to the remote repository named
              'origin'. Here's how it works:
            </Text>
            <List spacing="sm">
              <List.Item>
                <strong>Upload Local Commits:</strong> This command pushes all
                the commits from your local 'main' branch to the remote 'main'
                branch managed by the 'origin' repository.
              </List.Item>
              <List.Item>
                <strong>Update Remote Branch:</strong> If successful, the
                remote 'main' branch will now reflect all the changes you've
                made locally. This is essential for sharing your work with
                other collaborators.
              </List.Item>
              <List.Item>
                <strong>Permissions and Conflicts:</strong> You must have
                write access to the remote repository, and your local branch
                should be up-to-date with the remote changes to avoid
                conflicts during the push.
              </List.Item>
            </List>
          </Grid.Col>
        </Grid>
      </div>

      <div data-slide>
        <CodeBlock code="git push origin main" />
        <Text size="md" mb="md">
          Using this command effectively allows teams to maintain a
          synchronized and collaborative development process, ensuring
          that all contributions are integrated and tracked in the remote
          repository.
        </Text>
      </div>

      <div data-slide>
        <Grid>
          <Grid.Col>
            <Title order={3} id="example-case" mb="md">Example case with Remote Repositories</Title>
            <Text size="md" mb="md">
              Here's a practical example of working with a remote repository:
            </Text>
            <List type="ordered" spacing="sm">
              <List.Item>
                <strong>Create and Clone Repository:</strong> Follow the steps
                above to create a repository on GitHub and clone it.
              </List.Item>
              <List.Item>
                <strong>Add a File:</strong> Create a new file 'example.txt',
                add some content to it, and save it in your project directory.
              </List.Item>
              <List.Item>
                <strong>Stage the File:</strong> Run{" "}
                <Code>git add example.txt</Code> to stage the file.
              </List.Item>
              <List.Item>
                <strong>Commit the Change:</strong> Commit the staged file
                with <Code>git commit -m "Add example.txt"</Code>.
              </List.Item>
            </List>
          </Grid.Col>
        </Grid>
      </div>

      <div data-slide>
        <List type="ordered" spacing="sm" start={5}>
          <List.Item>
            <strong>Push the Commit:</strong> Push your commit to GitHub
            with <Code>git push origin main</Code>.
          </List.Item>
          <List.Item>
            <strong>Verify on GitHub:</strong> Check your GitHub
            repository online to see the 'example.txt' file.
          </List.Item>
          <List.Item>
            <strong>Make Changes on GitHub:</strong> Edit 'example.txt' on
            GitHub and commit the changes online.
          </List.Item>
          <List.Item>
            <strong>Pull Changes:</strong> Pull the changes back to your
            local repository with <Code>git pull origin main</Code>.
          </List.Item>
        </List>
        <Text size="md" mb="md">
          This workflow covers creating, modifying, and syncing changes
          between local and remote repositories, demonstrating the
          collaborative possibilities of Git and GitHub.
        </Text>
      </div>
    </Container>
  );
};

export default WorkingWithRemoteRepositories;