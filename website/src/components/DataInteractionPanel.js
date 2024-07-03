import React, { useState } from "react";
import { Button, Container, Row, Col, Table, Spinner } from "react-bootstrap";
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
  const [iframeLoading, setIframeLoading] = useState(true);
  const openInColab = (url) => {
    const colabUrl = `https://colab.research.google.com/github/racousin/data_science_practice/blob/master/${url}`;
    window.open(colabUrl, "_blank");
  };

  const downloadFile = (url) => {
    const link = document.createElement("a");
    link.href = url;
    link.download = url.split("/").pop(); // Assumes the URL ends with the filename
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const openInNewTab = (url) => {
    window.open(url, "_blank");
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
            <p>
              <strong>Description:</strong> {metadata.description}
            </p>
            <p>
              <strong>Source:</strong> {metadata.source}
            </p>
            <p>
              <strong>Target Variable:</strong> {metadata.target}
            </p>
            <Table striped bordered hover size="sm">
              <thead>
                <tr>
                  <th>Variable Name</th>
                  <th>Description</th>
                </tr>
              </thead>
              <tbody>
                {metadata.listData.map((item, index) => (
                  <tr key={index}>
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
          <Button variant="info" onClick={() => openInColab(notebookColabUrl)}>
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
          {iframeLoading && (
            <div
              style={{
                display: "flex",
                justifyContent: "center",
                alignItems: "center",
                height: "500px",
              }}
            >
              <Spinner animation="border" role="status">
                <span className="visually-hidden">Loading...</span>
              </Spinner>
            </div>
          )}
          <iframe
            src={notebookHtmlUrl}
            style={{
              width: "100%",
              height: "500px",
              border: "none",
              display: iframeLoading ? "none" : "block",
            }}
            onLoad={() => setIframeLoading(false)}
          ></iframe>
          <Button
            variant="secondary"
            style={{
              position: "absolute",
              right: "1rem",
              top: "1rem",
              zIndex: 1000,
            }}
            onClick={() => openInNewTab(notebookHtmlUrl)}
          >
            <FaExternalLinkAlt />
          </Button>
        </Col>
      </Row>
    </Container>
  );
};

export default DataInteractionPanel;
