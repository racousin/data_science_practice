import React from 'react';
import { CopyButton, Tooltip, ActionIcon, Box } from '@mantine/core';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { a11yDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { IconCheck, IconCopy } from '@tabler/icons-react';

const CodeBlock = ({ code, language = "bash", showCopy = true }) => {
  return (
    <Box pos="relative" style={{ marginBottom: '16px' }}  className={!showCopy ? "terminal-result" : ""}>
      {showCopy && (
        <CopyButton value={code} timeout={2000}>
          {({ copied, copy }) => (
            <Tooltip label={copied ? 'Copied!' : 'Copy'} position="left" withArrow>
              <ActionIcon
                onClick={copy}
                variant="subtle"
                color={copied ? 'teal' : 'gray'}
                style={{
                  position: 'absolute',
                  top: '8px',
                  right: '8px',
                  zIndex: 1,
                }}
              >
                {copied ? <IconCheck size="1rem" /> : <IconCopy size="1rem" />}
              </ActionIcon>
            </Tooltip>
          )}
        </CopyButton>
      )}
      <SyntaxHighlighter 
        language={language} 
        style={a11yDark}
        customStyle={{
          margin: 0,
          borderRadius: '4px',
        }}
      >
        {code}
      </SyntaxHighlighter>
    </Box>
  );
};

export default CodeBlock;