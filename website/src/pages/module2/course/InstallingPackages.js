import React from "react";
import { Row, Col } from 'react-bootstrap';
import { Container, Grid } from '@mantine/core';
import CodeBlock from "components/CodeBlock";
const InstallingPackages = () => {
  return (
    <Container fluid>
      <h1 className="my-4">Installing Packages</h1>
      <Row>
        <Col>
          <h3 id="install-package">Install a Package</h3>
          <p>To install a package using pip, use the following command:</p>
          <CodeBlock code={`pip install numpy`} language="bash"/>
          <p>
            This command installs the latest version of the specified package,
            in this case, numpy.
          </p>
          <h3 id="install-specific-version">
            Install a Specific Version of a Package
          </h3>
          <p>
            To install a specific version of a package, use the following
            command:
          </p>
          <CodeBlock code={`pip install numpy==1.19.5`} language="bash"/>
          <p>
            This command installs version 1.19.5 of the numpy package.
            Specifying the version is useful when you need a particular version
            that is compatible with your code.
          </p>
          <h3 id="install-from-requirements">
            Install Packages from a Requirements File
          </h3>
          <p>
            To install packages listed in a requirements file, use the following
            command:
          </p>
          <CodeBlock code={`pip install -r requirements.txt`} language="bash"/>
          <p>
            A requirements file lists all the packages your project depends on,
            along with their versions. This command reads the requirements file
            and installs all the listed packages.
          </p>
          <h4>Example of a requirements file:</h4>
          <CodeBlock code={`numpy==1.19.5\npandas==1.1.5\nscipy==1.5.4`} />
          <p>
            This example requirements file specifies exact versions for numpy,
            pandas, and scipy. Using a requirements file ensures consistency
            across different environments.
          </p>
          <h3 id="display-installed-packages">
            Display Installed Packages
          </h3>
          <p>To see a list of all the installed Python packages in your environment, use the following command:</p>
          <CodeBlock code={`pip freeze`} language="bash"/>
          <p>This will output a list of installed packages with their versions.</p>
          <h3 id="create-a-requirements-file">
            Create a Requirements File
          </h3>
          <p>To save the current environment’s dependencies to a <code>requirements.txt</code> file, run:</p>
          <CodeBlock code={`pip freeze > requirements.txt`} language="bash"/>
          <p>This command captures all installed packages and their versions in the <code>requirements.txt</code> file. This file can be used later to recreate the same environment.</p>
        </Col>
      </Row>
    </Container>
  );
};
export default InstallingPackages;
