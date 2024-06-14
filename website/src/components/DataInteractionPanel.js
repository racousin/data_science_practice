import React from "react";
import { Button, Container, Row, Col } from "react-bootstrap";
import { FaDownload, FaExternalLinkAlt } from "react-icons/fa";

const DataInteractionPanel = ({
  trainDataUrl,
  testDataUrl,
  notebookUrl,
  notebookHtmlUrl,
}) => {
  const openInColab = (url) => {
    const colabUrl = `https://colab.research.google.com/github/${url}`;
    window.open(colabUrl, "_blank");
  };

  const downloadFile = (url) => {
    window.open(url, "_self");
  };

  return (
    <Container className="my-4">
      <Row className="mb-2">
        <Col>
          <h3>Download Data</h3>
          <Button variant="primary" onClick={() => downloadFile(trainDataUrl)}>
            Train Data <FaDownload />
          </Button>{" "}
          <Button variant="primary" onClick={() => downloadFile(testDataUrl)}>
            Test Data <FaDownload />
          </Button>
        </Col>
      </Row>
      <Row className="mb-2">
        <Col>
          <h3>Work with Notebooks</h3>
          <Button variant="success" onClick={() => downloadFile(notebookUrl)}>
            Download Notebook <FaDownload />
          </Button>{" "}
          <Button variant="info" onClick={() => openInColab(notebookUrl)}>
            Open in Colab <FaExternalLinkAlt />
          </Button>
        </Col>
      </Row>
      <Row>
        <Col>
          <h3>View Notebook Output</h3>
          <iframe
            src={notebookHtmlUrl}
            style={{ width: "100%", height: "500px", border: "none" }}
          ></iframe>
        </Col>
      </Row>
    </Container>
  );
};

export default DataInteractionPanel;
