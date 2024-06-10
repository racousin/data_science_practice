import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";

const ConfigureAndAccessGithub = () => {
  const commands = {
    configUser: "git config --global user.name 'Your Username'",
    configEmail: "git config --global user.email 'your_email@example.com'",
    sshKeyGen: "ssh-keygen -t rsa -b 4096 -C 'your_email@example.com'",
    sshStart: "eval $(ssh-agent -s)",
    sshAdd: "ssh-add ~/.ssh/id_rsa",
    testConnection: "ssh -T git@github.com",
  };

  return (
    <Container fluid>
      <h2>Configure and Access GitHub</h2>
      <Row>
        <Col md={12}>
          <p>
            Now that you have a GitHub account and Git installed, it’s time to
            configure your Git setup to securely connect and interact with
            GitHub.
          </p>
          <h3>Configure Git with Your Information</h3>
          <p>
            Start by setting your GitHub username and email address in Git,
            which will be used to identify the commits you make:
          </p>
          <CodeBlock code={commands.configUser} language="bash" />
          <CodeBlock code={commands.configEmail} language="bash" />

          <h3>Generating a New SSH Key</h3>
          <p>
            Securely connect to GitHub with SSH by generating a new SSH key:
          </p>
          <CodeBlock code={commands.sshKeyGen} language="bash" />
          <p>
            After generating the key, start the SSH agent in the background and
            add your new SSH key to it:
          </p>
          <CodeBlock code={commands.sshStart} language="bash" />
          <CodeBlock code={commands.sshAdd} language="bash" />
          <p>
            For detailed steps tailored to different operating systems, visit
            the{" "}
            <a
              href="https://docs.github.com/en/authentication/connecting-to-github-with-ssh"
              target="_blank"
              rel="noopener noreferrer"
            >
              official GitHub SSH guide
            </a>
            .
          </p>

          <h3>Add Your SSH Public Key to GitHub</h3>
          <p>
            To enable secure SSH access to your GitHub account, you need to add
            your SSH public key to your GitHub settings:
          </p>
          <ol>
            <li>Go to your GitHub account settings.</li>
            <li>Navigate to "SSH and GPG keys" under "Access."</li>
            <li>Click on "New SSH key" to add a new key.</li>
            <li>Paste your public key into the field and save it.</li>
          </ol>
          <p>
            This step is essential for authenticating your future SSH sessions
            with GitHub without using a username and password.
          </p>

          <h3>Test Your SSH Connection</h3>
          <p>Verify that SSH is properly set up by connecting to GitHub:</p>
          <CodeBlock code={commands.testConnection} language="bash" />
          <p>
            If the connection is successful, you'll see a message confirming you
            are authenticated.
          </p>
        </Col>
      </Row>
    </Container>
  );
};

export default ConfigureAndAccessGithub;
