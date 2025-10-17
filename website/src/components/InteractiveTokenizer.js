import React, { useState, useEffect } from "react";
import { Text, Textarea, Badge, Group, Box, Select, Anchor, Button, Alert, Divider } from '@mantine/core';
import { IconDownload, IconAlertCircle } from '@tabler/icons-react';
import { encode as encodeO200k, decode as decodeO200k, vocabularySize as vocabO200k } from "gpt-tokenizer";
import { encode as encodeCl100k, decode as decodeCl100k, vocabularySize as vocabCl100k } from "gpt-tokenizer/encoding/cl100k_base";
import { encode as encodeP50k, decode as decodeP50k, vocabularySize as vocabP50k } from "gpt-tokenizer/encoding/p50k_base";
import { encode as encodeR50k, decode as decodeR50k, vocabularySize as vocabR50k } from "gpt-tokenizer/encoding/r50k_base";

const InteractiveTokenizer = () => {
  const [inputText, setInputText] = useState("The tokenization of uncommon words like 'anticonstitutionnellement' and émojis 🤖 varies across tokenizers!");
  const [selectedModel, setSelectedModel] = useState("o200k");
  const [inputIds, setInputIds] = useState("976, 1238, 11222, 4165, 382, 14667,0");

  const [decodedText, setDecodedText] = useState("");
  const [decodeError, setDecodeError] = useState("");

  const models = [
    { value: "o200k", label: "GPT-4o / GPT-5 ", encode: encodeO200k, decode: decodeO200k, vocabSize: vocabO200k },
    { value: "cl100k", label: "GPT-4 / GPT-3.5", encode: encodeCl100k, decode: decodeCl100k, vocabSize: vocabCl100k },
    { value: "r50k", label: "GPT-2", encode: encodeR50k, decode: decodeR50k, vocabSize: vocabR50k }
  ];

  const currentModel = models.find(m => m.value === selectedModel);

  const getTokens = (text) => {
    if (!text || !currentModel) return [];
    const tokenIds = currentModel.encode(text);
    return tokenIds.map(tokenId => ({
      id: tokenId,
      text: currentModel.decode([tokenId])
    }));
  };

  const tokens = getTokens(inputText);

  // Generate colors for tokens based on their ID
  const getTokenColor = (index) => {
    const colors = ['blue', 'green', 'red', 'orange', 'grape', 'cyan', 'pink', 'teal'];
    return colors[index % colors.length];
  };

  // Handle reverse tokenization (IDs to text)
  const handleDecodeIds = (idsString) => {
    setInputIds(idsString);
    setDecodeError("");
    setDecodedText("");

    if (!idsString.trim()) {
      return;
    }

    try {
      // Parse the input - split by comma, space, or both
      const idStrings = idsString.split(/[,\s]+/).filter(s => s.trim() !== "");

      // Convert to numbers and validate
      const tokenIds = idStrings.map((str, index) => {
        const num = parseInt(str.trim(), 10);
        if (isNaN(num)) {
          throw new Error(`Invalid number at position ${index + 1}: "${str}"`);
        }
        if (num < 0) {
          throw new Error(`Negative token ID at position ${index + 1}: ${num}`);
        }
        if (num >= currentModel.vocabSize) {
          throw new Error(`Token ID ${num} exceeds vocabulary size (${currentModel.vocabSize})`);
        }
        return num;
      });

      // Decode the token IDs
      const decoded = currentModel.decode(tokenIds);
      setDecodedText(decoded);
    } catch (error) {
      setDecodeError(error.message);
    }
  };

  // Decode initial example on mount and when model changes
  useEffect(() => {
    if (inputIds && currentModel) {
      handleDecodeIds(inputIds);
    }
  }, [selectedModel, currentModel]);

  // Download vocabulary
  const downloadVocabulary = () => {
    if (!currentModel) return;

    const vocabulary = [];
    // Generate vocabulary by decoding each token ID
    for (let i = 0; i < currentModel.vocabSize; i++) {
      try {
        const token = currentModel.decode([i]);
        vocabulary.push({ id: i, token: token });
      } catch (e) {
        // Some IDs might not be valid
        vocabulary.push({ id: i, token: '[INVALID]' });
      }
    }

    // Convert to CSV format
    const csv = 'Token ID,Token\n' + vocabulary.map(v => `${v.id},"${v.token.replace(/"/g, '""')}"`).join('\n');

    // Create download
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `vocabulary_${selectedModel}.csv`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  return (
    <Box>
      <Select
        label="Select tokenizer"
        value={selectedModel}
        onChange={setSelectedModel}
        data={models}
        mb="xs"
      />

      <Group spacing="md" mb="md">
        <Text size="sm">
          Vocabulary size: <strong>{currentModel?.vocabSize?.toLocaleString()}</strong>
        </Text>
        <Button
          size="xs"
          variant="light"
          leftIcon={<IconDownload size={14} />}
          onClick={downloadVocabulary}
        >
          Download Vocabulary
        </Button>
      </Group>

      <Textarea
        label="Enter text to tokenize"
        placeholder="Type or paste text here..."
        value={inputText}
        onChange={(event) => setInputText(event.currentTarget.value)}
        minRows={3}
        mb="md"
      />

      <Text size="sm" mb="xs">
        Token count: <strong>{tokens.length}</strong>
      </Text>

      <Text size="sm" mb="xs" weight={500}>
        Tokens:
      </Text>
      <Group spacing="xs" mb="md">
        {tokens.map((token, index) => (
          <Badge
            key={index}
            color={getTokenColor(index)}
            variant="filled"
            size="lg"
            style={{ cursor: 'help' }}
            title={`Token ID: ${token.id}`}
          >
            {token.text}
          </Badge>
        ))}
      </Group>

      <Text size="sm" mb="xs" weight={500}>
        Token IDs:
      </Text>
      <Group spacing="xs" mb="md">
        {tokens.map((token, index) => (
          <Badge
            key={index}
            color="gray"
            variant="outline"
            size="lg"
            style={{ cursor: 'help' }}
            title={`Token: ${token.text}`}
          >
            {token.id}
          </Badge>
        ))}
      </Group>

      <Divider my="xl" />

      <Text size="md" mb="md" weight={500}>
        Reverse Tokenization (Token IDs → Text)
      </Text>

      <Textarea
        label="Enter token IDs"
        placeholder="Enter token IDs separated by commas or spaces (e.g., 123, 456, 789)"
        value={inputIds}
        onChange={(event) => handleDecodeIds(event.currentTarget.value)}
        minRows={2}
        mb="md"
      />

      {decodeError && (
        <Alert icon={<IconAlertCircle />} color="red" mb="md">
          {decodeError}
        </Alert>
      )}

      {decodedText && !decodeError && (
        <Box>
          <Text size="sm" mb="xs" weight={500}>
            Decoded text:
          </Text>
          <Text
            p="sm"
            style={{
              backgroundColor: '#f8f9fa',
              borderRadius: '4px',
              fontFamily: 'monospace'
            }}
          >
            {decodedText}
          </Text>
        </Box>
      )}

      <Text size="xs" mt="xl" c="dimmed">
        Powered by{' '}
        <Anchor
          href="https://github.com/niieani/gpt-tokenizer"
          target="_blank"
          rel="noopener noreferrer"
          size="xs"
        >
          gpt-tokenizer
        </Anchor>
      </Text>
    </Box>
  );
};

export default InteractiveTokenizer;
