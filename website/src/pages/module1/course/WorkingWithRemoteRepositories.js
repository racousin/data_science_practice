import React from "react";
import { Container, Row, Col, Nav, Tab } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";

const WorkingWithRemoteRepositories = () => {
  return (
    <Container fluid>
      <Row>
        <Col>
          {/* Introduction to Remote Repositories */}
          <Row>
            <Col>
              <h3 id="why-remote-repositories">
                Why Remote Repositories are Useful
              </h3>
              <p>
                Remote repositories on platforms like GitHub allow developers to
                store versions of their projects online, facilitating
                collaboration, backup, and public sharing. They enable teams to
                work together on projects from different locations, track
                changes, merge contributions, and maintain a history of all
                modifications.
              </p>
            </Col>
          </Row>

          {/* Creating a Repository on GitHub */}
          <Row>
            <Col>
              <h3 id="creating-repository-github">
                Creating a Repository on GitHub
              </h3>
              <p>
                Creating a repository on GitHub is the first step toward remote
                project management. Here's how to set one up:
              </p>
              <ol>
                <li>Log in to your GitHub account.</li>
                <li>Navigate to the Repositories tab and click 'New'.</li>
                <li>
                  Enter a name for your repository and select the visibility
                  (public or private).
                </li>
                <li>
                  Optionally, initialize the repository with a README,
                  .gitignore, or license.
                </li>
                <li>Click 'Create repository'.</li>
              </ol>
            </Col>
          </Row>

          {/* Adding and Cloning Remote Repositories */}
          <Row>
            <Col>
              <h3 id="connecting-remote-repository">
                Connecting to a Remote Repository
              </h3>
              <p>
                Once your remote repository is set up, you can connect it to
                your local repository to push changes. Here are the steps for
                both starting from an existing local repo and cloning a new one:
              </p>
              <Tab.Container defaultActiveKey="add">
                <Nav variant="pills" className="flex-row">
                  <Nav.Item>
                    <Nav.Link eventKey="add">
                      Add to Existing Local Repo
                    </Nav.Link>
                  </Nav.Item>
                  <Nav.Item>
                    <Nav.Link eventKey="clone">Clone Remote Repo</Nav.Link>
                  </Nav.Item>
                </Nav>
                <Tab.Content>
                  <Tab.Pane eventKey="add">
                    <h5>Add Remote Repository</h5>
                    <p>
                      To link your existing local repository to your GitHub
                      repository, use the following command:
                    </p>
                    <CodeBlock
                      code={`git remote add origin <remote_repository_url>`}
                    />
                    <p>
                      This command sets your remote repository as 'origin',
                      which is the default name used by Git for the primary
                      remote.
                    </p>
                  </Tab.Pane>
                  <Tab.Pane eventKey="clone">
                    <h5>Clone Remote Repository</h5>
                    <p>
                      If you want to start working on a project with an existing
                      remote repository, you can clone it:
                    </p>
                    <CodeBlock code={`git clone <remote_repository_url>`} />
                    <p>
                      This command creates a local copy of the remote
                      repository, including all branches and commits.
                    </p>
                  </Tab.Pane>
                </Tab.Content>
              </Tab.Container>
            </Col>
          </Row>
          <Row>
            <Col>
              <h3 id="view-remote-repositories">View Remote Repositories</h3>
              <p>
                The <code>git remote -v</code> command is used to view all the
                remote repositories that your current repository knows about.
                Here's what the command does:
              </p>
              <ul>
                <li>
                  <strong>List Remotes:</strong> This command lists all remote
                  handles (short names) for the remote URLs configured.
                </li>
                <li>
                  <strong>Show URLs:</strong> Alongside each remote handle, the
                  command also shows the associated URLs, which can be either
                  fetch (data fetch from) or push (data send to) URLs.
                </li>
              </ul>
              <CodeBlock code="git remote -v" />
              <p>
                This output is particularly useful for verifying which remotes
                are set up for fetch and push operations, ensuring that you have
                the correct access paths for collaboration and deployment.
              </p>
            </Col>
          </Row>
          <Row>
            <Col>
              <h3 id="fetch-changes">Fetch Changes from a Remote Repository</h3>
              <p>
                The <code>git fetch origin</code> command is used to fetch
                branches, tags, and other data from a remote repository
                identified by 'origin'. Here's what the command involves:
              </p>
              <ul>
                <li>
                  <strong>Download Data:</strong> This command downloads
                  commits, files, and refs from a remote repository into your
                  local repo's working directory.
                </li>
                <li>
                  <strong>Update Tracking Branches:</strong>{" "}
                  <code>git fetch</code> updates your tracking branches under{" "}
                  <code>refs/remotes/origin/</code>.
                </li>
                <li>
                  <strong>No Workspace Changes:</strong> Fetching does not
                  change your own local branches and does not modify your
                  current workspace. It fetches the data but leaves your current
                  branch unchanged.
                </li>
              </ul>
              <CodeBlock code="git fetch origin" />
              <p>
                This command is essential before starting new work on your local
                repository to ensure you are working with the latest data from
                the remote repository.
              </p>
            </Col>
          </Row>
          {/* <Row>
            <Col>
              <h3 id="pull-changes">Pull Changes from a Remote Repository</h3>
              <p>
                The <code>git pull origin main</code> command combines two
                distinct operations performed by Git:
              </p>
              <ul>
                <li>
                  <strong>Fetch:</strong> First, <code>git pull</code> executes{" "}
                  <code>git fetch</code> which downloads content from the
                  specified remote repository under the branch named 'main'.
                  This step involves retrieving all the new commits, branches,
                  and files from the remote repository.
                </li>
                <li>
                  <strong>Merge:</strong> After fetching, Git automatically
                  attempts to merge the new commits into your current local
                  branch. If you are on the 'main' branch locally, it will merge
                  the changes from 'origin/main' into your local 'main' branch.
                </li>
              </ul>
              <CodeBlock code="git pull origin main" />
              <p>
                This command is crucial for keeping your local development
                environment updated with changes made by other collaborators in
                the repository. It ensures that you are working on the latest
                version of the project, reducing conflicts and inconsistencies.
              </p>
            </Col>
          </Row> */}
          <Row>
            <Col>
              <h3 id="push-changes">Push Changes to a Remote Repository</h3>
              <p>
                The <code>git push origin main</code> command sends the commits
                made on your local branch 'main' to the remote repository named
                'origin'. Here's how it works:
              </p>
              <ul>
                <li>
                  <strong>Upload Local Commits:</strong> This command pushes all
                  the commits from your local 'main' branch to the remote 'main'
                  branch managed by the 'origin' repository.
                </li>
                <li>
                  <strong>Update Remote Branch:</strong> If successful, the
                  remote 'main' branch will now reflect all the changes you've
                  made locally. This is essential for sharing your work with
                  other collaborators.
                </li>
                <li>
                  <strong>Permissions and Conflicts:</strong> You must have
                  write access to the remote repository, and your local branch
                  should be up-to-date with the remote changes to avoid
                  conflicts during the push.
                </li>
              </ul>
              <CodeBlock code="git push origin main" />
              <p>
                Using this command effectively allows teams to maintain a
                synchronized and collaborative development process, ensuring
                that all contributions are integrated and tracked in the remote
                repository.
              </p>
            </Col>
          </Row>

          <Row>
            <Col>
              <h3 id="example-case">Example case with Remote Repositories</h3>
              <p>
                Here's a practical example of working with a remote repository:
              </p>
              <ol>
                <li>
                  <strong>Create and Clone Repository:</strong> Follow the steps
                  above to create a repository on GitHub and clone it.
                </li>
                <li>
                  <strong>Add a File:</strong> Create a new file 'example.txt',
                  add some content to it, and save it in your project directory.
                </li>
                <li>
                  <strong>Stage the File:</strong> Run{" "}
                  <code>git add example.txt</code> to stage the file.
                </li>
                <li>
                  <strong>Commit the Change:</strong> Commit the staged file
                  with <code>git commit -m "Add example.txt"</code>.
                </li>
                <li>
                  <strong>Push the Commit:</strong> Push your commit to GitHub
                  with <code>git push origin main</code>.
                </li>
                <li>
                  <strong>Verify on GitHub:</strong> Check your GitHub
                  repository online to see the 'example.txt' file.
                </li>
                <li>
                  <strong>Make Changes on GitHub:</strong> Edit 'example.txt' on
                  GitHub and commit the changes online.
                </li>
                <li>
                  <strong>Fetch Changes:</strong> Fetch the changes back to your
                  local repository with <code>git fetch origin</code>.
                </li>
              </ol>
              <p>
                This workflow covers creating, modifying, and syncing changes
                between local and remote repositories, demonstrating the
                collaborative possibilities of Git and GitHub.
              </p>
            </Col>
          </Row>
        </Col>
      </Row>
    </Container>
  );
};

export default WorkingWithRemoteRepositories;
