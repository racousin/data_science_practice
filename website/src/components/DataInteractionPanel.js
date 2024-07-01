import React from "react";
import { Button, Container, Row, Col, Table } from "react-bootstrap";
import { FaDownload, FaExternalLinkAlt } from "react-icons/fa";

const DataInteractionPanel = ({
  trainDataUrl,
  testDataUrl,
  notebookUrl,
  notebookHtmlUrl,
  notebookColabUrl,
  requirementsUrl,
  dataUrl,
  metadata,
}) => {
  const openInColab = (url) => {
    const colabUrl = `https://colab.research.google.com/github/racousin/data_science_practice/blob/master/${notebookColabUrl}`;
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
          {dataUrl && (
            <Button variant="primary" onClick={() => downloadFile(dataUrl)}>
              Data <FaDownload />
            </Button>
          )}
          {trainDataUrl && testDataUrl && (
            <>
              <Button
                variant="primary"
                onClick={() => downloadFile(trainDataUrl)}
              >
                Train Data <FaDownload />
              </Button>{" "}
              <Button
                variant="primary"
                onClick={() => downloadFile(testDataUrl)}
              >
                Test Data <FaDownload />
              </Button>
            </>
          )}
        </Col>
      </Row>
      {metadata && (
        <Row className="mt-3">
          <Col>
            <h3>Dataset Metadata</h3>
            <Table striped bordered hover size="sm">
              <thead>
                <tr>
                  <th>Variable Name</th>
                  <th>Description</th>
                </tr>
              </thead>
              <tbody>
                {metadata.map((item) => (
                  <tr key={item.name}>
                    <td>{item.name}</td>
                    <td>{item.description}</td>
                  </tr>
                ))}
              </tbody>
            </Table>
          </Col>
        </Row>
      )}
      <Row className="mb-2">
        <Col>
          <h3>Notebook</h3>
          <Button variant="success" onClick={() => downloadFile(notebookUrl)}>
            Download Notebook <FaDownload />
          </Button>{" "}
          <Button variant="info" onClick={() => openInColab(notebookUrl)}>
            Open in Colab <FaExternalLinkAlt />
          </Button>
          {requirementsUrl && (
            <Button
              variant="primary"
              onClick={() => downloadFile(requirementsUrl)}
            >
              Requirements.txt <FaDownload />
            </Button>
          )}
        </Col>
      </Row>
      <Row>
        <Col>
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