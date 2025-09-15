import React from "react";
import { Container, Title, Text, Table } from '@mantine/core';
import CodeBlock from "components/CodeBlock";

const Cheatsheet = () => {
  return (
    <Container fluid>
      <div data-slide>
        <Title order={2} mb="md">Git Commands Cheatsheet</Title>
        <Text size="md" mb="md">
          A comprehensive reference of essential Git commands organized by category.
        </Text>
      </div>

      <div data-slide>
        <Title order={3} mb="md">Repository Setup</Title>
        <Table>
          <Table.Thead>
            <Table.Tr>
              <Table.Th>Command</Table.Th>
              <Table.Th>Description</Table.Th>
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            <Table.Tr>
              <Table.Td><code>git init</code></Table.Td>
              <Table.Td>Initialize a new Git repository</Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td><code>git clone &lt;url&gt;</code></Table.Td>
              <Table.Td>Clone a remote repository</Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td><code>git config --global user.name "Name"</code></Table.Td>
              <Table.Td>Set your username globally</Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td><code>git config --global user.email "email"</code></Table.Td>
              <Table.Td>Set your email globally</Table.Td>
            </Table.Tr>
          </Table.Tbody>
        </Table>
      </div>

      <div data-slide>
        <Title order={3} mb="md">Basic Commands</Title>
        <Table>
          <Table.Thead>
            <Table.Tr>
              <Table.Th>Command</Table.Th>
              <Table.Th>Description</Table.Th>
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            <Table.Tr>
              <Table.Td><code>git status</code></Table.Td>
              <Table.Td>Show working directory status</Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td><code>git add &lt;file&gt;</code></Table.Td>
              <Table.Td>Add file to staging area</Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td><code>git add .</code></Table.Td>
              <Table.Td>Add all changes to staging area</Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td><code>git commit -m "message"</code></Table.Td>
              <Table.Td>Commit staged changes</Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td><code>git commit -am "message"</code></Table.Td>
              <Table.Td>Add all changes and commit</Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td><code>git diff</code></Table.Td>
              <Table.Td>Show unstaged changes</Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td><code>git diff --staged</code></Table.Td>
              <Table.Td>Show staged changes</Table.Td>
            </Table.Tr>
          </Table.Tbody>
        </Table>
      </div>

      <div data-slide>
        <Title order={3} mb="md">Branching</Title>
        <Table>
          <Table.Thead>
            <Table.Tr>
              <Table.Th>Command</Table.Th>
              <Table.Th>Description</Table.Th>
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            <Table.Tr>
              <Table.Td><code>git branch</code></Table.Td>
              <Table.Td>List all branches</Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td><code>git branch &lt;name&gt;</code></Table.Td>
              <Table.Td>Create new branch</Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td><code>git checkout &lt;branch&gt;</code></Table.Td>
              <Table.Td>Switch to branch</Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td><code>git checkout -b &lt;name&gt;</code></Table.Td>
              <Table.Td>Create and switch to new branch</Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td><code>git switch &lt;branch&gt;</code></Table.Td>
              <Table.Td>Switch to branch (modern syntax)</Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td><code>git merge &lt;branch&gt;</code></Table.Td>
              <Table.Td>Merge branch into current branch</Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td><code>git branch -d &lt;branch&gt;</code></Table.Td>
              <Table.Td>Delete branch</Table.Td>
            </Table.Tr>
          </Table.Tbody>
        </Table>
      </div>

      <div data-slide>
        <Title order={3} mb="md">Remote Repositories</Title>
        <Table>
          <Table.Thead>
            <Table.Tr>
              <Table.Th>Command</Table.Th>
              <Table.Th>Description</Table.Th>
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            <Table.Tr>
              <Table.Td><code>git remote -v</code></Table.Td>
              <Table.Td>List remote repositories</Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td><code>git remote add origin &lt;url&gt;</code></Table.Td>
              <Table.Td>Add remote repository</Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td><code>git push origin &lt;branch&gt;</code></Table.Td>
              <Table.Td>Push branch to remote</Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td><code>git pull</code></Table.Td>
              <Table.Td>Fetch and merge from remote</Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td><code>git fetch</code></Table.Td>
              <Table.Td>Fetch changes from remote</Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td><code>git push -u origin &lt;branch&gt;</code></Table.Td>
              <Table.Td>Push and set upstream branch</Table.Td>
            </Table.Tr>
          </Table.Tbody>
        </Table>
      </div>

      <div data-slide>
        <Title order={3} mb="md">History & Logs</Title>
        <Table>
          <Table.Thead>
            <Table.Tr>
              <Table.Th>Command</Table.Th>
              <Table.Th>Description</Table.Th>
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            <Table.Tr>
              <Table.Td><code>git log</code></Table.Td>
              <Table.Td>Show commit history</Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td><code>git log --oneline</code></Table.Td>
              <Table.Td>Show condensed commit history</Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td><code>git log --graph</code></Table.Td>
              <Table.Td>Show commit history with graph</Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td><code>git show &lt;commit&gt;</code></Table.Td>
              <Table.Td>Show commit details</Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td><code>git blame &lt;file&gt;</code></Table.Td>
              <Table.Td>Show who changed each line</Table.Td>
            </Table.Tr>
          </Table.Tbody>
        </Table>
      </div>

      <div data-slide>
        <Title order={3} mb="md">Undoing Changes</Title>
        <Table>
          <Table.Thead>
            <Table.Tr>
              <Table.Th>Command</Table.Th>
              <Table.Th>Description</Table.Th>
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            <Table.Tr>
              <Table.Td><code>git reset HEAD &lt;file&gt;</code></Table.Td>
              <Table.Td>Unstage file</Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td><code>git checkout -- &lt;file&gt;</code></Table.Td>
              <Table.Td>Discard file changes</Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td><code>git reset --soft HEAD~1</code></Table.Td>
              <Table.Td>Undo last commit, keep changes</Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td><code>git reset --hard HEAD~1</code></Table.Td>
              <Table.Td>Undo last commit, discard changes</Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td><code>git revert &lt;commit&gt;</code></Table.Td>
              <Table.Td>Create new commit that undoes changes</Table.Td>
            </Table.Tr>
          </Table.Tbody>
        </Table>
      </div>

      <div data-slide>
        <Title order={3} mb="md">Stashing</Title>
        <Table>
          <Table.Thead>
            <Table.Tr>
              <Table.Th>Command</Table.Th>
              <Table.Th>Description</Table.Th>
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            <Table.Tr>
              <Table.Td><code>git stash</code></Table.Td>
              <Table.Td>Stash current changes</Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td><code>git stash pop</code></Table.Td>
              <Table.Td>Apply and remove latest stash</Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td><code>git stash list</code></Table.Td>
              <Table.Td>List all stashes</Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td><code>git stash apply</code></Table.Td>
              <Table.Td>Apply latest stash without removing</Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td><code>git stash drop</code></Table.Td>
              <Table.Td>Delete latest stash</Table.Td>
            </Table.Tr>
          </Table.Tbody>
        </Table>
      </div>

      <div data-slide>
        <Title order={3} mb="md">Tags</Title>
        <Table>
          <Table.Thead>
            <Table.Tr>
              <Table.Th>Command</Table.Th>
              <Table.Th>Description</Table.Th>
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            <Table.Tr>
              <Table.Td><code>git tag</code></Table.Td>
              <Table.Td>List all tags</Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td><code>git tag &lt;name&gt;</code></Table.Td>
              <Table.Td>Create lightweight tag</Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td><code>git tag -a &lt;name&gt; -m "message"</code></Table.Td>
              <Table.Td>Create annotated tag</Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td><code>git push origin &lt;tag&gt;</code></Table.Td>
              <Table.Td>Push specific tag</Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td><code>git push origin --tags</code></Table.Td>
              <Table.Td>Push all tags</Table.Td>
            </Table.Tr>
          </Table.Tbody>
        </Table>
      </div>
    </Container>
  );
};

export default Cheatsheet;