import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";

const SettingUpPythonEnvironment = () => {
  return (
    <Container fluid>
      <h1 className="my-4">Setting Up a Python Environment</h1>
      <p>
        Setting up a dedicated Python environment for your projects can help
        manage dependencies and ensure that different projects can run
        independently on the same machine. This section will guide you through
        the setup of a Python virtual environment using <code>virtualenv</code>,
        a popular tool for creating isolated Python environments.
      </p>
      <Row>
        <Col md={12}>
          <h3 id="install-virtualenv">Install virtualenv</h3>
          <p>
            The first step in creating a virtual environment is installing the{" "}
            <code>virtualenv</code> package if it's not already installed.
          </p>
          <CodeBlock code={`pip install virtualenv`} language="bash" />
          <p>
            This command installs <code>virtualenv</code> globally, allowing you
            to create isolated environments anywhere on your system.
          </p>

          <h3 id="create-environment">Create a Virtual Environment</h3>
          <p>
            Once <code>virtualenv</code> is installed, you can create a new
            environment specifically for your project.
          </p>
          <CodeBlock code={`virtualenv myenv`} language="bash" />
          <p>
            This command creates a new directory called <code>myenv</code> that
            contains a fresh, isolated Python installation. You can replace
            "myenv" with a name of your choice for your environment.
          </p>

          <h3 id="activate-environment">Activate the Virtual Environment</h3>
          <p>
            Activating the environment will set up your shell to use the Python
            and pip executables from the virtual environment, making it the
            active Python instance.
          </p>
          <CodeBlock code={`source myenv/bin/activate`} language="bash" />
          <p>For Windows users, the activation command differs slightly:</p>
          <CodeBlock code={`myenv\\Scripts\\activate`} language="bash" />
          <p>
            This command changes the shell’s environment to use the Python and
            pip from the <code>myenv</code> directory.
          </p>

          <h3 id="deactivate-environment">
            Deactivate the Virtual Environment
          </h3>
          <p>
            When you're done working in the virtual environment and wish to
            return to the system's default Python settings, you can deactivate
            it.
          </p>
          <CodeBlock code={`deactivate`} language="bash" />
          <p>
            This command will revert your Python environment to the system’s
            default settings.
          </p>
        </Col>
      </Row>
    </Container>
  );
};

export default SettingUpPythonEnvironment;
