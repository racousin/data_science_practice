import React, { useState } from 'react';
import { Container, Grid, Title, Button, Table, Text, Loader, ScrollArea, Box, Group } from '@mantine/core';
import { IconDownload, IconExternalLink, IconBrandCodesandbox, IconFileText } from '@tabler/icons-react';

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
    window.open(colabUrl, '_blank');
  };

  const downloadFile = (url) => {
    const link = document.createElement('a');
    link.href = url;
    link.download = url.split('/').pop();
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const openInNewTab = (url) => {
    window.open(url, '_blank');
  };

  const ButtonWithIcon = ({ icon: Icon, onClick, children }) => (
    <Button variant="light" color="blue" onClick={onClick}>
      <Icon size={20} style={{ marginRight: '0.5rem' }} />
      {children}
    </Button>
  );

  return (
    <Container fluid mt="xl">
      <Box w="100%" mb="xl">
        <Title order={2} size="h3" weight={600} color="#2F80ED" mb="md">Download Data</Title>
        <Group position="left" spacing="md">
          {trainDataUrl && (
            <ButtonWithIcon icon={IconDownload} onClick={() => downloadFile(trainDataUrl)}>
              Train Data
            </ButtonWithIcon>
          )}
          {testDataUrl && (
            <ButtonWithIcon icon={IconDownload} onClick={() => downloadFile(testDataUrl)}>
              Test Data
            </ButtonWithIcon>
          )}
          {dataUrl && (
            <ButtonWithIcon icon={IconDownload} onClick={() => downloadFile(dataUrl)}>
              Data
            </ButtonWithIcon>
          )}
        </Group>
      </Box>

      {metadata && (
        <Box mb="xl">
          <Title order={2} size="h3" weight={600} color="#2F80ED" mb="md">Dataset Metadata</Title>
          <Table fontSize="sm" mb="md">
            <tbody>
              <tr>
                <td><strong>Description:</strong></td>
                <td>{metadata.description}</td>
              </tr>
              <tr>
                <td><strong>Source:</strong></td>
                <td>{metadata.source}</td>
              </tr>
              <tr>
                <td><strong>Target Variable:</strong></td>
                <td>{metadata.target}</td>
              </tr>
            </tbody>
          </Table>
          <ScrollArea>
            <Table striped highlightOnHover fontSize="sm">
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
          </ScrollArea>
        </Box>
      )}

      <Box mb="xl">
        <Title order={2} size="h3" weight={600} color="#2F80ED" mb="md">Notebook</Title>
        <Group position="left" spacing="md">
          <ButtonWithIcon icon={IconDownload} onClick={() => downloadFile(notebookUrl)}>
            Download Notebook
          </ButtonWithIcon>
          <ButtonWithIcon icon={IconBrandCodesandbox} onClick={() => openInColab(notebookColabUrl)}>
            Open in Colab
          </ButtonWithIcon>
          {requirementsUrl && (
            <ButtonWithIcon icon={IconFileText} onClick={() => downloadFile(requirementsUrl)}>
              Requirements.txt
            </ButtonWithIcon>
          )}
          <ButtonWithIcon icon={IconExternalLink} onClick={() => openInNewTab(notebookHtmlUrl)}>
            Open HTML
          </ButtonWithIcon>
        </Group>
      </Box>

      <Box>
        {iframeLoading && <Loader size="xl" variant="dots" color="blue" mx="auto" my="xl" />}
        <iframe
          src={notebookHtmlUrl}
          width="100%"
          height="700px"
          onLoad={() => setIframeLoading(false)}
          style={{ border: 'none', display: iframeLoading ? 'none' : 'block' }}
        />
      </Box>
    </Container>
  );
};

export default DataInteractionPanel;