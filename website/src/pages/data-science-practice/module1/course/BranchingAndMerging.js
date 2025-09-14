import React from "react";
import { Container, Grid, Image, Title, Text, List, Code, Table, Flex } from '@mantine/core';
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { a11yDark } from "react-syntax-highlighter/dist/esm/styles/prism";
import CodeBlock from "components/CodeBlock";

const BranchingAndMerging = () => {
  return (
    <Container fluid>
      <div data-slide>
        <Title order={2} mb="md">Branching and Merging</Title>
        <Text size="md" mb="md">
          Branching and merging are vital features of Git that facilitate
          simultaneous and non-linear development among teams. Branching allows
          multiple developers to work on different features at the same time
          without interfering with each other, while merging brings those changes
          together into a single branch.
        </Text>
      </div>

      <div data-slide>
        <Grid>
          <Grid.Col>
            <Title order={3} id="working-with-branches" mb="md">Working with Branches</Title>
            <List type="ordered" spacing="sm">
              <List.Item>
                <strong>List all local Branches:</strong>
                <CodeBlock code={`git branch`} />
                <CodeBlock
                  code={`$ git branch
  master
  my_branch1
* my_branch2
  my_branch3`}
                  language=""
                />
                The <Code>*</Code> indicates the current branch.
              </List.Item>
              <List.Item>
                <strong>Create a Branch:</strong> Use{" "}
                <CodeBlock code={`git checkout -b newbranch`} />
                <CodeBlock
                  code={`$ git checkout -b newbranch
Switched to a new branch 'newbranch'
`}
                  language=""
                />
              </List.Item>
            </List>
          </Grid.Col>
        </Grid>
      </div>

      <div data-slide>
        <Grid>
          <Grid.Col>
            <Title order={3} id="merging-branches" mb="md">Merging Branches</Title>
            <Text size="md" mb="md">
              Once development on a branch is complete, the changes can be merged
              back into the main branch (e.g. 'main'). Here are different merge
              types:
            </Text>
          </Grid.Col>
        </Grid>
      </div>

      <div data-slide>
        <List spacing="sm">
          <List.Item>
            <strong>Fast-forward Merge:</strong>
            <Text size="md" mb="md">
              Occurs when the target branch hasn't diverged from the source
              branch. Git simply moves the pointer forward.
            </Text>
            <CodeBlock
              code={`git checkout main
git merge feature-branch`}
            />
            <CodeBlock
              code={`$ git checkout main
Switched to branch 'main'
$ git merge feature-branch
Updating 22a36a3..3951f63
Fast-forward
 example.txt | 2 +-
 1 file changed, 1 insertion(+), 1 deletion(-)`}
              language=""
            />
          </List.Item>
        </List>
      </div>

      <div data-slide>
        <List spacing="sm">
          <List.Item>
            <strong>Three-way Merge:</strong>
            <Text size="md" mb="md">
              Occurs when the target branch has diverged from the source
              branch. Git creates a new commit to merge the histories.
            </Text>
            <CodeBlock
              code={`git checkout main
git merge feature-branch`}
            />
            <CodeBlock
              code={`$ git checkout main
Switched to branch 'main'
$ git merge feature-branch
Merge made by the 'recursive' strategy.
 example.txt | 2 +-
 1 file changed, 1 insertion(+), 1 deletion(-)`}
              language=""
            />
          </List.Item>
        </List>
      </div>

      <div data-slide>
        <List spacing="sm">
          <List.Item>
            <strong>Squash Merge:</strong>
            <Text size="md" mb="md">
              Combines all changes from the source branch into a single commit
              in the target branch.
            </Text>
            <CodeBlock
              code={`git checkout main
git merge --squash feature-branch
git commit -m "Squashed feature-branch changes"`}
            />
          </List.Item>
          <List.Item>
            <strong>Rebase:</strong>
            <Text size="md" mb="md">
              Moves the entire feature branch to begin on the tip of the main
              branch, effectively incorporating all new commits in main.
            </Text>
            <CodeBlock
              code={`git checkout feature-branch
git rebase main
git checkout main
git merge feature-branch`}
            />
          </List.Item>
        </List>
      </div>

      <div data-slide>
        <List spacing="sm">
          <List.Item>
            <strong>No-fast-forward Merge:</strong>
            <Text size="md" mb="md">
              Forces a new merge commit even when a fast-forward merge is
              possible. Useful for maintaining a record of merges.
            </Text>
            <CodeBlock
              code={`git checkout main
git merge --no-ff feature-branch`}
            />
          </List.Item>
        </List>
      </div>

      <div data-slide>
        <Grid className="justify-content-center">
          <Grid.Col span={{ xs: 12 }} md={10} lg={8}>
            <Flex direction="column" align="center">
              <Image
                src="/assets/data-science-practice/module1/Git_Fast-forward_Merge.png"
                alt="Git_Fast-forward_Merge"
                style={{ maxWidth: 'min(800px, 90vw)', height: 'auto' }}
                fluid
              />
              <Text>Git_Fast-forward_Merge</Text>
            </Flex>
          </Grid.Col>
        </Grid>
      </div>

      <div data-slide>
        <Grid className="justify-content-center">
          <Grid.Col span={{ xs: 12 }} md={10} lg={8}>
            <Flex direction="column" align="center">
              <Image
                src="/assets/data-science-practice/module1/Git_Rebase_Merge.png"
                alt="Git_Rebase_Merge"
                fluid
                style={{ maxWidth: 'min(800px, 90vw)', height: 'auto' }}
              />
              <Text>Git_Rebase_Merge</Text>
            </Flex>
          </Grid.Col>
        </Grid>
      </div>

      <div data-slide>
        <Grid className="justify-content-center">
          <Grid.Col span={{ xs: 12 }} md={10} lg={8}>
            <Flex direction="column" align="center">
              <Image
                src="/assets/data-science-practice/module1/Git_Squash_Merge.png"
                alt="Git_Squash_Merge"
                style={{ maxWidth: 'min(800px, 90vw)', height: 'auto' }}
                fluid
              />
              <Text>Git_Squash_Merge</Text>
            </Flex>
          </Grid.Col>
        </Grid>
      </div>

      <div data-slide>
        <Grid className="justify-content-center">
          <Grid.Col span={{ xs: 12 }} md={10} lg={8}>
            <Flex direction="column" align="center">
              <Image
                src="/assets/data-science-practice/module1/Git_Three-way_Merge.png"
                alt="Git_Three-way_Merge"
                style={{ maxWidth: 'min(800px, 90vw)', height: 'auto' }}
                fluid
              />
              <Text>Git_Three-way_Merge</Text>
            </Flex>
          </Grid.Col>
        </Grid>
      </div>

      <div data-slide>
        <Grid>
          <Grid.Col>
            <Title order={4} mb="md">Advantages of Different Merge Strategies</Title>
            <Table striped>
              <Table.Thead>
                <Table.Tr>
                  <Table.Th>Merge Type</Table.Th>
                  <Table.Th>Advantages</Table.Th>
                </Table.Tr>
              </Table.Thead>
              <Table.Tbody>
                <Table.Tr>
                  <Table.Td>Fast-forward</Table.Td>
                  <Table.Td>
                    - Simplest and cleanest history<br />
                    - No additional merge commits<br />
                    - Preserves linear history
                  </Table.Td>
                </Table.Tr>
                <Table.Tr>
                  <Table.Td>Three-way</Table.Td>
                  <Table.Td>
                    - Preserves complete history of both branches<br />
                    - Clearly shows where branches diverged and merged<br />
                    - Useful for complex feature integrations
                  </Table.Td>
                </Table.Tr>
                <Table.Tr>
                  <Table.Td>Squash</Table.Td>
                  <Table.Td>
                    - Simplifies feature history into a single commit<br />
                    - Keeps main branch history clean and concise<br />
                    - Easier to revert entire features if needed
                  </Table.Td>
                </Table.Tr>
                <Table.Tr>
                  <Table.Td>Rebase</Table.Td>
                  <Table.Td>
                    - Creates a linear, clean history<br />
                    - Avoids unnecessary merge commits
                  </Table.Td>
                </Table.Tr>
                <Table.Tr>
                  <Table.Td>No-fast-forward</Table.Td>
                  <Table.Td>
                    - Always creates a merge commit<br />
                    - Preserves branch structure and merge points<br />
                    - Useful for tracking when and where merges occurred
                  </Table.Td>
                </Table.Tr>
              </Table.Tbody>
            </Table>
          </Grid.Col>
        </Grid>
      </div>

      <div data-slide>
        <Grid>
          <Grid.Col>
            <Title order={3} id="resolving-merge-conflicts" mb="md">Resolving Merge Conflicts</Title>
            <Text size="md" mb="md">
              Conflicts occur when the same parts of the same file are changed in
              different branches:
            </Text>
          </Grid.Col>
        </Grid>
      </div>

      <div data-slide>
        <List type="ordered" spacing="sm">
          <List.Item>
            <strong>Identify Conflicts:</strong> During a merge, Git will tell
            you if there are conflicts that need manual resolution.
            <CodeBlock
              code={`$ git merge newbranch
Auto-merging example.txt
CONFLICT (content): Merge conflict in example.txt
Automatic merge failed; fix conflicts and then commit the result.
$ git status
On branch main
You have unmerged paths.
  (fix conflicts and run "git commit")
  (use "git merge --abort" to abort the merge)
Unmerged paths:
  (use "git add <file>..." to mark resolution)
	both modified:   example.txt
`}
              language=""
              showCopy={false}
            />
          </List.Item>
        </List>
      </div>

      <div data-slide>
        <List type="ordered" spacing="sm" start={2}>
          <List.Item>
            <strong>Edit Files:</strong> Open the conflicted files and make
            the necessary changes to resolve conflicts.
          </List.Item>
          <List.Item>
            <strong>Mark as Resolved:</strong> Use <Code>git add</Code> on the
            resolved files to mark them as resolved.
            <CodeBlock code={`git add example.txt`} />
          </List.Item>
          <List.Item>
            <strong>Complete the Merge:</strong> Use <Code>git commit</Code>{" "}
            to complete the merge.
            <CodeBlock
              code={`git commit -m "Resolved merge conflict by including both suggestions."`}
            />
          </List.Item>
        </List>
      </div>

      <div data-slide>
        <Grid>
          <Grid.Col>
            <Title order={3} id="example-case" mb="md">Example: Adding a Feature via Branch</Title>
            <Text size="md" mb="md">
              Imagine you are working on a project and need to add a new feature
              without disrupting the main development line. Here's how you can
              handle it with Git branching and merging:
            </Text>
          </Grid.Col>
        </Grid>
      </div>

      <div data-slide>
        <List type="ordered" spacing="sm">
          <List.Item>
            <strong>Create a Feature Branch:</strong> Suppose you want to add
            a new login feature. You would start by creating a new branch
            dedicated to this feature.
            <CodeBlock code={`git branch login-feature`} />
          </List.Item>
          <List.Item>
            <strong>Switch to the Feature Branch:</strong> Move to the
            'login-feature' branch to work on this feature.
            <CodeBlock code={`git checkout login-feature`} />
          </List.Item>
          <List.Item>
            <strong>Develop the Feature:</strong> Make all necessary changes
            for the new feature. For example, create new files or modify
            existing ones, test the feature, etc.
            <CodeBlock
              code={`git add .
git commit -m "Add login feature"`}
            />
          </List.Item>
        </List>
      </div>

      <div data-slide>
        <List type="ordered" spacing="sm" start={4}>
          <List.Item>
            <strong>Switch Back to Main Branch:</strong> Once the feature
            development is complete and tested, switch back to the main branch
            to prepare for merging.
            <CodeBlock code={`git checkout main`} />
          </List.Item>
          <List.Item>
            <strong>Merge the Feature Branch:</strong> Merge the changes from
            'login-feature' into 'main'. Assuming no conflicts, this merge
            will integrate the new feature into the main project.
            <CodeBlock code={`git merge login-feature`} />
          </List.Item>
          <List.Item>
            <strong>Delete the Feature Branch:</strong> After the feature has
            been successfully merged, you can delete the branch to keep the
            repository clean.
            <CodeBlock code={`git branch -d login-feature`} />
          </List.Item>
        </List>
      </div>

      <div data-slide>
        <Text size="md" mb="md">
          This workflow keeps the main line stable while allowing development
          of new features in parallel. It also ensures that any ongoing work
          is not affected by the new changes until they are fully ready to be
          integrated.
        </Text>
      </div>
    </Container>
  );
};

export default BranchingAndMerging;