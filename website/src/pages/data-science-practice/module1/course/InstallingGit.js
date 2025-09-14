import React from "react";
import { Container, Grid, Button, Title, Text, Anchor } from '@mantine/core';
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { a11yDark } from "react-syntax-highlighter/dist/esm/styles/prism";
import CodeBlock from "components/CodeBlock";
const InstallingGit = () => {
  const commands = {
    mac: "brew install git",
    windows: "choco install git",
    linux: "sudo apt install git",
    version: "git --version",
  };
  return (
    <Container fluid>
      <div data-slide>
        <Title order={2} mb="md">Installing Git</Title>
        <Text size="md" mb="md">
          To install Git on your computer, you can use a package manager specific
          to your operating system, or download the installer directly from the{" "}
          <Anchor
            href="https://git-scm.com/downloads"
            target="_blank"
          >
            official Git website
          </Anchor>
          .
        </Text>
      </div>

      <div data-slide>
        <Grid>
          <Grid.Col>
            <Title order={3} id="mac" mb="md">Mac</Title>
            <Text size="md" mb="md">
              If you have{" "}
              <Anchor
                href="https://brew.sh/"
                target="_blank"
              >
                Homebrew
              </Anchor>{" "}
              installed on your Mac, you can install Git by running the following
              command in the Terminal:
            </Text>
            <CodeBlock code={commands.mac} />
            <Text size="md" mb="md">
              Don't have Homebrew?{" "}
              <Anchor
                href="https://brew.sh/"
                target="_blank"
              >
                Install Homebrew here
              </Anchor>
              .
            </Text>
          </Grid.Col>
        </Grid>
      </div>

      <div data-slide>
        <Grid>
          <Grid.Col>
            <Title order={3} id="windows" mb="md">Windows</Title>
            <Text size="md" mb="md">
              For Windows users,{" "}
              <Anchor
                href="https://chocolatey.org/"
                target="_blank"
              >
                Chocolatey
              </Anchor>{" "}
              can be used to install Git easily:
            </Text>
            <CodeBlock code={commands.windows} />
            <Text size="md" mb="md">
              Alternatively, download the Git installer directly from the{" "}
              <Anchor
                href="https://git-scm.com/download/win"
                target="_blank"
              >
                Git website
              </Anchor>{" "}
              and follow the installation instructions.
            </Text>
            <Text size="md" mb="md">
              Don't have Chocolatey?{" "}
              <Anchor
                href="https://chocolatey.org/install"
                target="_blank"
              >
                Install Chocolatey here
              </Anchor>
              .
            </Text>
          </Grid.Col>
        </Grid>
      </div>

      <div data-slide>
        <Grid>
          <Grid.Col>
            <Title order={3} id="linux" mb="md">Linux</Title>
            <Text size="md" mb="md">
              If you are using a Debian-based Linux distribution, such as Ubuntu,
              you can install Git using the following command in your terminal:
            </Text>
            <CodeBlock code={commands.linux} />
            <Text size="md" mb="md">
              For other Linux distributions, you can find specific installation
              instructions on the{" "}
              <Anchor
                href="https://git-scm.com/download/linux"
                target="_blank"
              >
                Git website
              </Anchor>
              .
            </Text>
          </Grid.Col>
        </Grid>
      </div>

      <div data-slide>
        <Grid>
          <Grid.Col>
            <Title order={3} mb="md">Validate Your Installation</Title>
            <Text size="md" mb="md">
              To verify that Git has been installed correctly, open a terminal or
              command prompt and type the following command:
            </Text>
            <CodeBlock code={commands.version} />
            <CodeBlock
              code={`$ git --version
git version 2.34.1`}
              showCopy={false}
              language=""
            />
            <Text size="md" mb="md">
              This command should display the installed version of Git, confirming
              that the software is ready for use.
            </Text>
          </Grid.Col>
        </Grid>
      </div>

      <div data-slide>
        <Grid>
          <Grid.Col>
            <Title order={3} mb="md">Troubleshooting</Title>
            <Text size="md" mb="md">
              If you encounter any issues during installation, consult the{" "}
              <Anchor
                href="https://git-scm.com/book/en/v2/Getting-Started-Installing-Git"
                target="_blank"
              >
                Git installation guide
              </Anchor>
            </Text>
          </Grid.Col>
        </Grid>
      </div>
    </Container>
  );
};
export default InstallingGit;
