import React from "react";
import { Container, Grid, Image, Flex, Text, Title, Code } from '@mantine/core';
import CodeBlock from "components/CodeBlock";

const FirstStepsWithGit = () => {
  return (
    <Container fluid>
      <div data-slide>
        <Title order={2} mb="md">First Steps with Git</Title>
      </div>

      <div data-slide>
        <Grid>
          <Grid.Col>
            <Title order={3} id="creating-repository" mb="md">Creating a Git Repository</Title>
            <Text size="md" mb="md">
              Before you can start using Git to track changes, you need to create
              a new repository. This process begins by creating a new directory
              for your project, navigating into it, and then initializing it as a
              Git repository.
            </Text>
            <Text size="md" mb="md">
              <strong>Step 1: Create a new directory for your project:</strong>
            </Text>
            <CodeBlock code="mkdir my_project" />
            <Text size="md" mb="md">
              This command creates a new folder called 'my_project' where your
              project files will reside.
            </Text>
          </Grid.Col>
        </Grid>
      </div>

      <div data-slide>
        <Text size="md" mb="md">
          <strong>Step 2: Navigate into your project directory:</strong>
        </Text>
        <CodeBlock code="cd my_project" />
        <Text size="md" mb="md">
          This command moves the terminal's current working directory to the
          'my_project' folder.
        </Text>
        <Text size="md" mb="md">
          <strong>
            Step 3: Initialize the directory as a Git repository:
          </strong>
        </Text>
        <CodeBlock code="git init" />
        <Text size="md" mb="md">
          This command creates a new Git repository in the current directory.
          It sets up the necessary Git infrastructure within the '.git'
          directory. Here, Git will store all the metadata for the
          repository's change history.
        </Text>
      </div>

      <div data-slide>
        <Text size="md" mb="md">
          Once these steps are completed, you have successfully set up a new
          Git repository and can begin tracking changes to the files within
          this directory.
        </Text>
      </div>

      <div data-slide>
        <Grid>
          <Grid.Col>
            <Title order={3} id="staging-changes" mb="md">Staging Changes</Title>
            <Text size="md" mb="md">
              Staging changes in Git involves preparing changes made to files in
              your working directory to be included in the next commit. This
              process allows you to selectively add files to your next commit
              while leaving others unchanged.
            </Text>
          </Grid.Col>
        </Grid>
      </div>

      <div data-slide>
        <Grid>
          <Grid.Col>
            <Title order={4} mb="md">Adding a New File and Tracking Changes</Title>
            <Text size="md" mb="md">
              <strong>
                Step 1: Create a new file to add to your Git repository:
              </strong>
            </Text>
            <CodeBlock code={`echo 'Initial content' > example.txt`} />
            <Text size="md" mb="md">
              This command creates a new text file named 'example.txt' with some
              initial content.
            </Text>
          </Grid.Col>
        </Grid>
      </div>

      <div data-slide>
        <Title order={4} mb="md">Step 2: Check the status of your repository:</Title>
        <Text size="md" mb="md">
          Use the <Code>git status</Code> command to see which changes are
          pending to be added to your next commit:
        </Text>
        <CodeBlock code="git status" />
        <CodeBlock
          code={`$ git status
On branch master
No commits yet
Untracked files:
  (use "git add <file>..." to include in what will be committed)
	example.txt
nothing added to commit but untracked files present (use "git add" to track)
`}
          showCopy={false}
          language=""
        />
      </div>

      <div data-slide>
        <Text size="md" mb="md">
          You will see 'example.txt' listed as an untracked file because Git
          has noticed a new file in the directory but it hasn't been added to
          the staging area yet.
        </Text>
        <Title order={4} mb="md">Step 3: Stage the new file:</Title>
        <Text size="md" mb="md">
          To add this new file to the staging area, use the{" "}
          <Code>git add</Code> command:
        </Text>
        <CodeBlock code="git add example.txt" />
      </div>

      <div data-slide>
        <Text size="md" mb="md">
          This command moves 'example.txt' into the staging area, making it
          ready to be included in the next commit. You can verify this by
          running <Code>git status</Code> again, which will now show
          'example.txt' as staged.
        </Text>
        <CodeBlock
          code={`$ git status
On branch master
No commits yet
Changes to be committed:
  (use "git rm --cached <file>..." to unstage)
	new file:   example.txt
`}
          language=""
        />
      </div>

      <div data-slide>
        <Title order={4} mb="md">Step 4: Stage all changes in the directory:</Title>
        <Text size="md" mb="md">
          If you have multiple files to stage, you can add all modified files
          to the staging area using:
        </Text>
        <CodeBlock code="git add ." />
        <Text size="md" mb="md">
          This command adds all new and modified files to the staging area.
          It's useful when you have several files that need to be committed
          together.
        </Text>
      </div>

      <div data-slide>
        <Title order={4} mb="md">Step 5: Unstage a file:</Title>
        <Text size="md" mb="md">
          If you decide that a file should not be included in the next commit,
          you can unstage it using the <Code>git reset</Code> command:
        </Text>
        <CodeBlock code="git reset HEAD example.txt" />
        <Text size="md" mb="md">
          This command will remove 'example.txt' from the staging area but the
          file will remain in your working directory with any changes intact.
          Running <Code>git status</Code> will now show the file as not staged
          for commit. NB: If you don't have any commits yet, using{" "}
          <Code>git reset HEAD &quot;&lt;file&gt;&quot;</Code> will result in
          an error.
        </Text>
      </div>

      <div data-slide>
        <Grid>
          <Grid.Col>
            <Title order={3} id="committing-changes" mb="md">What Does Committing Mean?</Title>
            <Text size="md" mb="md">
              Committing in Git refers to the process of saving your staged
              changes to the local repository's history. A commit is essentially a
              snapshot of your repository at a specific point in time, allowing
              you to record the progress of your project in manageable increments.
            </Text>
          </Grid.Col>
        </Grid>
      </div>

      <div data-slide>
        <Grid>
          <Grid.Col>
            <Title order={4} mb="md">What Will Be Committed?</Title>
            <Text size="md" mb="md">
              Only changes that have been staged (using <Code>git add</Code>) will
              be included in a commit. Unstaged changes remain in your working
              directory and are not included in the commit.
            </Text>
          </Grid.Col>
        </Grid>
      </div>

      <div data-slide>
        <Grid>
          <Grid.Col>
            <Title order={4} mb="md">How to Create a Commit</Title>
            <Text size="md" mb="md">
              To create a commit, you use the <Code>git commit</Code> command
              along with a message describing the changes made. This message is
              important for maintaining a clear, accessible history of project
              changes and should be informative and concise.
            </Text>
            <CodeBlock code="git commit -m 'Your commit message'" />
            <CodeBlock
              code={`$ git commit -m 'Your commit message'
[master (root-commit) 1c17586] Your commit message
 1 file changed, 1 insertion(+)
 create mode 100644 example.txt
`}
              language=""
            />
          </Grid.Col>
        </Grid>
      </div>

      <div data-slide>
        <Text size="md" mb="md">
          This command captures a snapshot of the project's currently staged
          changes. The commit message should clearly describe what the changes
          do, making it easier for others (and your future self) to understand
          the purpose of the changes without needing to read the code.
        </Text>
        <Grid className="justify-content-center">
          <Grid.Col span={{ xs: 12 }} md={10} lg={8}>
            <Flex direction="column" align="center">
              <Image
                src="/assets/data-science-practice/module1/Git_Workflow.png"
                alt="Git Workflow Diagram"
                style={{ maxWidth: 'min(600px, 70vw)', height: 'auto' }}
                fluid
              />
              <Text>Git Workflow Diagram</Text>
            </Flex>
          </Grid.Col>
        </Grid>
      </div>

      <div data-slide>
        <Title order={4} mb="md">Why Committing is Important</Title>
        <Text size="md" mb="md">
          Commits serve as checkpoints where individual changes can be saved
          to the project history. This allows you to revert selected changes
          if needed or compare differences over time. They are crucial for
          collaborative projects, enabling multiple developers to work on
          different features simultaneously without conflict.
        </Text>
        <Text size="md" mb="md">
          Committing frequently ensures that your changes are securely
          recorded in your local repository, allowing for detailed tracking of
          your project's evolution. It also facilitates collaborative
          workflows such as branching and merging, and supports continuous
          integration practices.
        </Text>
      </div>

      <div data-slide>
        <Grid>
          <Grid.Col>
            <Title order={3} id="viewing-commit-history" mb="md">Understanding Commit History</Title>
            <Text size="md" mb="md">
              Commit history in Git is a record of all previous commits in the
              repository. It allows you to review changes, revert to previous
              states, and understand the chronological progression of your
              project.
            </Text>
          </Grid.Col>
        </Grid>
      </div>

      <div data-slide>
        <Grid>
          <Grid.Col>
            <Title order={4} mb="md">Basic Command to View Commit History</Title>
            <Text size="md" mb="md">
              To see a detailed list of the commit history, you can use the{" "}
              <Code>git log</Code> command:
            </Text>
            <CodeBlock code="git log" />
            <CodeBlock
              code={`commit 58ad9c9a1ca32944e9440c354631f5985a262d6e (HEAD -> master)
Author: username <username@mail.com>
Date:   Thu Jul 4 12:12:29 2024 +0200
    commit message 3
commit e380ec1ec5b257e96ee15e3648eeec351cf1009c
Author: username <username@mail.com>
Date:   Thu Jul 4 12:12:23 2024 +0200
    commit message 2
commit 6bd8e71d76237dfa4b72a8268c0f423e7bc91793
Author: username <username@mail.com>
Date:   Thu Jul 4 12:12:14 2024 +0200
    commit message 1
`}
              language=""
            />
          </Grid.Col>
        </Grid>
      </div>

      <div data-slide>
        <Text size="md" mb="md">
          This command displays the commit IDs, author information, dates, and
          commit messages in a detailed log format.
        </Text>
        <Title order={4} mb="md">Condensed Commit History</Title>
        <Text size="md" mb="md">
          For a more concise view of the commit history, use the{" "}
          <Code>--oneline</Code> option:
        </Text>
        <CodeBlock code="git log --oneline" />
        <Text size="md" mb="md">
          This option shows each commit as a single line, making it easier to
          browse through many commits quickly.
        </Text>
      </div>

      <div data-slide>
        <Title order={4} mb="md">Visualizing Commit History as a Graph</Title>
        <Text size="md" mb="md">
          To see the commit history represented as a graph, use the{" "}
          <Code>--graph</Code> option along with <Code>--oneline</Code> and{" "}
          <Code>--all</Code> to show all branches:
        </Text>
        <CodeBlock code="git log --graph --oneline --all" />
        <CodeBlock
          code={`| * cc56e19 fix/Use correctly backlog and po
|/
*   e035944 Merge pull request #7 from multi/ml
|\
| * 3e0abe6 update
| * 66598a8 comments
| *   fe0be5c Merge branch 'refacto/multi/ml/classif' of github.com:main/ml
ncept into refacto/multi_ml
| |\
| | * 6018bef Update simulation.py
| | * 64ba655 Update main.py
| * | 002d42e fix
| |/
| * 5bb7e7c update
|/
*   ef3061f Merge pull request #6 from multi/ml/auto_ml
|\
| * 5da7b55 - Fix issues with recent data.
| * c98d77a update ML with hotfix for dates
`}
          language=""
        />
      </div>

      <div data-slide>
        <Text size="md" mb="md">
          This graph view provides a visual representation of branches and
          merge points in your commit history.
        </Text>
        <Grid className="justify-content-center">
          <Grid.Col span={{ xs: 12 }} md={10} lg={8}>
            <Flex direction="column" align="center">
              <Image
                src="/assets/data-science-practice/module1/Git_Commit_History.png"
                alt="Git_Commit_History"
                style={{ maxWidth: 'min(800px, 90vw)', height: 'auto' }}
                fluid
              />
              <Text>Git_Commit_History</Text>
            </Flex>
          </Grid.Col>
        </Grid>
      </div>

      <div data-slide>
        <Grid>
          <Grid.Col>
            <Title order={3} id="undoing-changes" mb="md">Overview of Undoing Changes</Title>
            <Text size="md" mb="md">
              Git provides several tools to revert or undo changes after they have
              been committed. These tools are crucial for maintaining the
              integrity and accuracy of your project history.
            </Text>
          </Grid.Col>
        </Grid>
      </div>

      <div data-slide>
        <Grid>
          <Grid.Col>
            <Title order={4} mb="md">Committing and Realizing a Mistake</Title>
            <Text size="md" mb="md">Imagine you've modified a file and committed the changes:</Text>
            <CodeBlock code={`echo 'Modify content' > example.txt`} />
            <CodeBlock code="git add example.txt" />
            <CodeBlock code="git commit -m 'Add modify content'" />
            <Text size="md" mb="md">
              After reviewing, you realize there's a mistake that needs
              correction.
            </Text>
          </Grid.Col>
        </Grid>
      </div>

      <div data-slide>
        <Grid>
          <Grid.Col>
            <Title order={3} id="finding-commit-hash" mb="md">Finding a Commit Hash</Title>
            <Text size="md" mb="md">
              Before you can revert changes or checkout a previous version, you
              need to identify the commit hash. The commit hash is a unique
              identifier for each commit. You can find this by viewing the commit
              log:
            </Text>
            <CodeBlock code="git log" />
          </Grid.Col>
        </Grid>
      </div>

      <div data-slide>
        <CodeBlock
          code={`commit 58ad9c9a1ca32944e9440c354631f5985a262d6e (HEAD -> master)
Author: username <username@mail.com>
Date:   Thu Jul 4 12:12:29 2024 +0200
    Add modify content
commit e380ec1ec5b257e96ee15e3648eeec351cf1009c
Author: username <username@mail.com>
Date:   Thu Jul 4 12:12:23 2024 +0200
    Your commit message
commit 6bd8e71d76237dfa4b72a8268c0f423e7bc91793
Author: username <username@mail.com>
Date:   Thu Jul 4 12:12:14 2024 +0200
    commit message 1
`}
          language=""
        />
        <Text size="md" mb="md">
          This command displays a list of recent commits, each with a unique
          hash at the top, author information, date, and commit message. Look
          for the hash associated with the commit you are interested in
          reverting or checking out.
        </Text>
      </div>

      <div data-slide>
        <Title order={4} mb="md">Viewing the Difference</Title>
        <Text size="md" mb="md">To see what was changed with a previous commit, you can use:</Text>
        <CodeBlock code="git diff <commit_hash>" />
        <CodeBlock
          code={`$ git diff e380ec1ec5b257e96ee15e3648eeec351cf1009cq
diff --git a/example.txt b/example.txt
index 8430408..9201842 100644
--- a/example.txt
+++ b/example.txt
@@ -1 +1 @@
-Initial content
+Modify content
`}
          language=""
        />
        <Text size="md" mb="md">
          This command shows the differences between the current HEAD and the
          previous commit.
        </Text>
      </div>

      <div data-slide>
        <Title order={4} mb="md">Reverting a Commit</Title>
        <Text size="md" mb="md">
          To undo the changes made by a previous commit, use the{" "}
          <Code>git revert</Code> command. This will create a new commit that
          undoes all changes made in the previous commit:
        </Text>
        <CodeBlock code="git revert <commit_hash>" />
        <CodeBlock
          code={`$ git revert 58ad9c9a1ca32944e9440c354631f5985a262d6e
[master 22a36a3] Revert "Add modify content"
 1 file changed, 1 insertion(+), 1 deletion(-)
`}
          language=""
        />
        <Text size="md" mb="md">
          This is safe for shared branches as it does not alter commit
          history.
        </Text>
      </div>

      <div data-slide>
        <Grid className="justify-content-center">
          <Grid.Col span={{ xs: 12 }} md={10} lg={8}>
            <Flex direction="column" align="center">
              <Image
                src="/assets/data-science-practice/module1/Git_Commit_History_Revert.png"
                alt="Git_Commit_History_Revert"
                style={{ maxWidth: 'min(800px, 90vw)', height: 'auto' }}
                fluid
              />
              <Text>Git_Commit_Revert</Text>
            </Flex>
          </Grid.Col>
        </Grid>
      </div>

      <div data-slide>
        <Title order={4} mb="md">Checking Out to a Previous Version</Title>
        <Text size="md" mb="md">
          If reverting is not suitable, you can checkout a previous version of
          a file or project to see or restore it as it was:
        </Text>
        <CodeBlock code="git checkout <commit_hash>" />
        <Text size="md" mb="md">
          This command will go back in the past. You can then return to main{" "}
          <Code>HEAD</Code> commit using
        </Text>
        <CodeBlock code="git checkout main" />
      </div>
    </Container>
  );
};

export default FirstStepsWithGit;