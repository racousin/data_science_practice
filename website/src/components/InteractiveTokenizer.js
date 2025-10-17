import React, { useState } from "react";
import { Text, Textarea, Badge, Group, Box, Select, Anchor } from '@mantine/core';
import { encode as encodeO200k, decode as decodeO200k } from "gpt-tokenizer";
import { encode as encodeCl100k, decode as decodeCl100k } from "gpt-tokenizer/encoding/cl100k_base";
import { encode as encodeP50k, decode as decodeP50k } from "gpt-tokenizer/encoding/p50k_base";
import { encode as encodeR50k, decode as decodeR50k } from "gpt-tokenizer/encoding/r50k_base";

const InteractiveTokenizer = () => {
  const [inputText, setInputText] = useState("The tokenization of uncommon words like 'anticonstitutionnellement' and émojis 🤖 varies across tokenizers!");
  const [selectedModel, setSelectedModel] = useState("o200k");

  const models = [
    { value: "o200k", label: "GPT-4o / GPT-5 ", encode: encodeO200k, decode: decodeO200k },
    { value: "cl100k", label: "GPT-4 / GPT-3.5", encode: encodeCl100k, decode: decodeCl100k },
    { value: "r50k", label: "GPT-2", encode: encodeR50k, decode: decodeR50k }
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

  return (
    <Box>
      <Select
        label="Select tokenizer"
        value={selectedModel}
        onChange={setSelectedModel}
        data={models}
        mb="md"
      />

      <Textarea
        label="Enter text to tokenize"
        placeholder="Type or paste text here..."
        value={inputText}
        onChange={(event) => setInputText(event.currentTarget.value)}
        minRows={3}
        mb="md"
      />

      <Text size="sm" mb="xs">
        Token count: {tokens.length}
      </Text>

      <Group spacing="xs">
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

      <Text size="xs" mt="md" c="dimmed">
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
