import React from "react";
import { Container, Grid, Title, Text, List, Anchor, Code } from '@mantine/core';
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
      <div data-slide>
        <Title order={3} id="create-github-account" mb="md">Create a GitHub Account</Title>
        <Text size="md" mb="md">
          GitHub is a platform for hosting and collaborating on software
          development projects using Git. Creating a GitHub account is the first
          step towards managing your projects online, contributing to other
          projects, and collaborating with other developers.
        </Text>
        <Grid>
          <Grid.Col span={{ md: 12 }}>
            <List type="ordered" spacing="sm">
              <List.Item>
                Visit the GitHub homepage:{" "}
                <Anchor
                  href="https://www.github.com"
                  target="_blank"
                >
                  www.github.com
                </Anchor>
                .
              </List.Item>
              <List.Item>
                Click on the "Sign up" button in the upper-right corner of the
                homepage.
              </List.Item>
              <List.Item>Follow the steps!</List.Item>
            </List>
          </Grid.Col>
        </Grid>
      </div>

      <div data-slide>
        <Grid>
          <Grid.Col span={{ md: 12 }}>
            <Title order={3} id="configure-git" mb="md">Configure Git with Your Information</Title>
            <Text size="md" mb="md">
              Now that you have a GitHub account and Git installed, it's time to
              configure your Git setup to securely connect and interact with
              GitHub.
            </Text>
            <Text size="md" mb="md">
              Start by setting your GitHub username and email address in Git,
              which will be used to identify the commits you make:
            </Text>
            <CodeBlock code={commands.configUser} language="bash" />
            <CodeBlock code={commands.configEmail} language="bash" />
          </Grid.Col>
        </Grid>
      </div>

      <div data-slide>
        <Title order={3} id="connect-to-github-with-ssh" mb="md">Connect to GitHub with SSH</Title>
        <Title order={4} mb="md">Generating a New SSH Key</Title>
        <Text size="md" mb="md">
          Securely connect to GitHub with SSH by generating a new SSH key:
        </Text>
        <CodeBlock code={commands.sshKeyGen} language="bash" />
        <CodeBlock
          showCopy={false}
          code={`$ ssh-keygen -t rsa -b 4096 -C 'username@email.com'
Generating public/private rsa key pair.
Enter file in which to save the key (/home/username/.ssh/id_rsa):
Enter passphrase (empty for no passphrase):
Enter same passphrase again:
Your identification has been saved in /home/username/.ssh/id_rsa
Your public key has been saved in /home/username/.ssh/id_rsa.pub
The key fingerprint is:
SHA256:U508DNoGVSuUKX7KM2Y6+U8tBQujvvqulLsd7/ohS5Q username@email.com
The key's randomart image is:
+---[RSA 4096]----+
|        ..++.    |
|        .=o= o   |
|       .+.* B    |
|       o.=.+ .   |
|      E.So. .    |
|     +  B. o     |
|    o ==.oo .    |
|   . ++* o .     |
|    =*B*=..      |
+----[SHA256]-----+
`}
          language=""
        />
      </div>

      <div data-slide>
        <Text size="md" mb="md">
          After generating the key, start the SSH agent in the background and
          add your new SSH key to it:
        </Text>
        <CodeBlock code={commands.sshStart} language="bash" />
        <CodeBlock code={commands.sshAdd} language="bash" />
        <Text size="md" mb="md">
          For detailed steps tailored to different operating systems, visit
          the{" "}
          <Anchor
            href="https://docs.github.com/en/authentication/connecting-to-github-with-ssh"
            target="_blank"
          >
            official GitHub SSH guide
          </Anchor>
          .
        </Text>
      </div>

      <div data-slide>
        <Title order={4} mb="md">Add Your SSH Public Key to GitHub</Title>
        <Text size="md" mb="md">
          To enable secure SSH access to your GitHub account, you need to add
          your SSH public key to your GitHub settings:
        </Text>
        <List type="ordered" spacing="sm">
          <List.Item>Go to your GitHub account settings.</List.Item>
          <List.Item>Navigate to "SSH and GPG keys" under "Access."</List.Item>
          <List.Item>Click on "New SSH key" to add a new key.</List.Item>
          <List.Item>
            Paste your public key (<Code>cat .ssh/id_rsa.pub</Code>) into the
            field and save it.
          </List.Item>
        </List>
        <Text size="md" mb="md">
          This step is essential for authenticating your future SSH sessions
          with GitHub.
        </Text>
      </div>

      <div data-slide>
        <Title order={4} mb="md">Test Your SSH Connection</Title>
        <Text size="md" mb="md">Verify that SSH is properly set up by connecting to GitHub:</Text>
        <CodeBlock code={commands.testConnection} language="bash" />
        <CodeBlock
          code={`$ ssh -T git@github.com
Hi username! You've successfully authenticated, but GitHub does not provide shell access.
`}
          language=""
          showCopy={false}
        />
        <Text size="md" mb="md">
          If the connection is successful, you'll see a message confirming you
          are authenticated.
        </Text>
      </div>
    </Container>
  );
};

export default ConfigureAndAccessGithub;