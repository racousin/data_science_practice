import React from 'react';
import { Container, Title, Text, Code, Table, Stack, Paper, List } from '@mantine/core';
import { FileText, FileSpreadsheet, Database, FileJson, FileCode, FileImage } from 'lucide-react';
import CodeBlock from "components/CodeBlock";

const FileTypeIcon = ({ type }) => {
  const iconProps = { size: 24, strokeWidth: 1.5 };
  switch (type) {
    case 'csv': return <FileText {...iconProps} />;
    case 'xlsx': return <FileSpreadsheet {...iconProps} />;
    case 'parquet': return <Database {...iconProps} />;
    case 'json': return <FileJson {...iconProps} />;
    case 'xml': return <FileCode {...iconProps} />;
    case 'text': return <FileText {...iconProps} />;
    case 'image': return <FileImage {...iconProps} />;
    default: return null;
  }
};

const FileTypeItem = ({ type, name, description, metadata, sample, code }) => (
  <Stack spacing="md" mb="xl">
    <Title order={3}>{name}</Title>
    <Text>{description}</Text>
    <Title order={4}>Metadata</Title>
    <Table>
      <tbody>
        {Object.entries(metadata).map(([key, value]) => (
          <tr key={key}>
            <td><strong>{key}</strong></td>
            <td>{value}</td>
          </tr>
        ))}
      </tbody>
    </Table>
    {sample && (
      <>
        <Title order={4}>Sample Data</Title>
        <Paper p="xs" withBorder>
          <Code block>{sample}</Code>
        </Paper>
      </>
    )}
    <Title order={4}>Reading the File</Title>
    <CodeBlock language="python" code={code} />
  </Stack>
);

const Files = () => {
  const structuredFiles = [
    {
      type: 'csv',
      name: 'CSV (Comma-Separated Values)',
      description: 'Simple text format for tabular data, where each line represents a row and columns are separated by commas.',
      metadata: {
        'File Extension': '.csv',
        'Delimiter': 'Typically comma (,), but can be others like semicolon (;) or tab (\\t)',
        'Header': 'Optional, usually the first row',
        'Encoding': 'Usually UTF-8, but can vary',
      },
      sample: `id,name,age,city
1,John Doe,30,New York
2,Jane Smith,25,Los Angeles
3,Bob Johnson,35,Chicago`,
      code: `import pandas as pd

# Basic reading
df = pd.read_csv('data.csv')

# With specific options
df = pd.read_csv('data.csv', 
                 delimiter=';',  # for semicolon-separated files
                 header=None,    # if there's no header
                 encoding='latin1',  # for different encoding
                 names=['id', 'name', 'age', 'city'])  # custom column names

print(df.head())`
    },
    {
      type: 'xlsx',
      name: 'XLSX (Excel Spreadsheet)',
      description: 'Microsoft Excel file format for storing and organizing data in spreadsheets.',
      metadata: {
        'File Extension': '.xlsx',
        'Sheets': 'Can contain multiple sheets',
        'Header': 'Optional, usually the first row',
        'Cell Types': 'Can include text, numbers, dates, formulas',
      },
      sample: `(Excel file structure cannot be directly represented in text)`,
      code: `import pandas as pd

# Basic reading
df = pd.read_excel('data.xlsx')

# With specific options
df = pd.read_excel('data.xlsx', 
                   sheet_name='Sheet2',  # specify sheet name or index
                   header=None,          # if there's no header
                   names=['id', 'name', 'age', 'city'])  # custom column names

print(df.head())`
    },
    {
      type: 'parquet',
      name: 'Parquet',
      description: 'Columnar storage format designed for efficient data processing and compression.',
      metadata: {
        'File Extension': '.parquet',
        'Compression': 'Supports various algorithms (snappy, gzip, etc.)',
        'Encoding': 'Efficient encoding schemes for different data types',
        'Partitioning': 'Supports partitioning for distributed computing',
      },
      sample: `(Binary format, cannot be directly represented in text)`,
      code: `import pandas as pd

# Basic reading
df = pd.read_parquet('data.parquet')

# With specific options
df = pd.read_parquet('data.parquet', 
                     engine='pyarrow',  # or 'fastparquet'
                     columns=['name', 'age'])  # read only specific columns

print(df.head())`
    }
  ];

  const semiStructuredFiles = [
    {
      type: 'json',
      name: 'JSON (JavaScript Object Notation)',
      description: 'Lightweight, human-readable data format often used in web applications and APIs.',
      metadata: {
        'File Extension': '.json',
        'Structure': 'Key-value pairs, arrays',
        'Data Types': 'Strings, numbers, booleans, null, objects, arrays',
        'Encoding': 'Usually UTF-8',
      },
      sample: `{
  "employees": [
    {"name": "John Doe", "age": 30, "city": "New York"},
    {"name": "Jane Smith", "age": 25, "city": "Los Angeles"},
    {"name": "Bob Johnson", "age": 35, "city": "Chicago"}
  ]
}`,
      code: `import pandas as pd

# Basic reading
df = pd.read_json('data.json')

# With specific options
df = pd.read_json('data.json', 
                  orient='records',  # specify JSON structure
                  lines=True)        # for JSON Lines format

print(df.head())`
    },
    {
      type: 'xml',
      name: 'XML (eXtensible Markup Language)',
      description: 'Markup language that defines a set of rules for encoding documents in a format that is both human-readable and machine-readable.',
      metadata: {
        'File Extension': '.xml',
        'Structure': 'Tags, elements, attributes',
        'Encoding': 'Usually UTF-8, but can be specified in XML declaration',
        'Schema': 'Can be defined using DTD or XSD',
      },
      sample: `<?xml version="1.0" encoding="UTF-8"?>
<employees>
  <employee>
    <name>John Doe</name>
    <age>30</age>
    <city>New York</city>
  </employee>
  <employee>
    <name>Jane Smith</name>
    <age>25</age>
    <city>Los Angeles</city>
  </employee>
</employees>`,
      code: `import pandas as pd

# Basic reading
df = pd.read_xml('data.xml')

# With specific options
df = pd.read_xml('data.xml', 
                 xpath='//employee',  # specify XPath to parse
                 encoding='utf-8')    # specify encoding if needed

print(df.head())`
    }
  ];

  const unstructuredFiles = [
    {
      type: 'text',
      name: 'Text Files',
      description: 'Plain text files containing unstructured data like articles, documents, or logs.',
      metadata: {
        'File Extension': '.txt, .log, etc.',
        'Encoding': 'Various (UTF-8, ASCII, etc.)',
        'Structure': 'No fixed structure',
        'Content': 'Can be any text data',
      },
      sample: `This is a sample text file.
It can contain any kind of textual information.
There is no predefined structure to this data.`,
      code: `# Basic reading
with open('data.txt', 'r', encoding='utf-8') as file:
    text = file.read()

print(text[:100])  # Print first 100 characters

# Using pandas for large files
import pandas as pd

df = pd.read_csv('data.txt', sep='\\n', header=None, names=['text'])
print(df.head())`
    },
    {
      type: 'image',
      name: 'Image Files',
      description: 'Visual data stored in formats like JPEG, PNG, or TIFF.',
      metadata: {
        'File Extensions': '.jpg, .png, .tiff, etc.',
        'Color Modes': 'RGB, CMYK, Grayscale, etc.',
        'Compression': 'Lossy (JPEG) or Lossless (PNG)',
        'Metadata': 'Can include EXIF data (for photos)',
      },
      sample: `(Binary image data cannot be directly represented in text)`,
      code: `from PIL import Image
import numpy as np

# Open and display basic info
img = Image.open('image.jpg')
print(f"Format: {img.format}")
print(f"Size: {img.size}")
print(f"Mode: {img.mode}")

# Convert to numpy array for processing
img_array = np.array(img)
print(f"Shape: {img_array.shape}")
print(f"Data type: {img_array.dtype}")`
    }
  ];

  return (
    <Container fluid>
      <div data-slide>
        <Title order={1}>Files</Title>
        <Text size="lg" mt="md">
          Understanding different file formats and how to work with them is essential for data science.
          Files can be categorized based on their structure and data organization.
        </Text>
      </div>

      <div data-slide>
        <Title id="metadata" order={2}>File Metadata</Title>
        <Text>
          File metadata is crucial information about a file that describes its structure, content, and how to interpret it.
          Key metadata elements include:
        </Text>
        <List mt="md">
          <List.Item>File format and extension</List.Item>
          <List.Item>Encoding (e.g., UTF-8, ASCII)</List.Item>
          <List.Item>Structure (e.g., headers, delimiters for structured data)</List.Item>
          <List.Item>Size and creation/modification dates</List.Item>
          <List.Item>Author or owner information</List.Item>
        </List>
      </div>

      <div data-slide>
        <Title id="structured" order={2}>Structured Files</Title>
        <Text mb="md">
          Structured files have a well-defined format with organized rows and columns, making them easy to process and analyze.
        </Text>
        {structuredFiles.map(file => (
          <FileTypeItem key={file.type} {...file} />
        ))}
      </div>

      <div data-slide>
        <Title id="semi" order={2}>Semi-Structured Files</Title>
        <Text mb="md">
          Semi-structured files have some organization but are more flexible than structured formats, often using hierarchical or nested structures.
        </Text>
        {semiStructuredFiles.map(file => (
          <FileTypeItem key={file.type} {...file} />
        ))}
      </div>

      <div data-slide>
        <Title id="unstructured" order={2}>Unstructured Files</Title>
        <Text mb="md">
          Unstructured files contain data without a predefined format or organization, requiring special processing techniques to extract meaningful information.
        </Text>
        {unstructuredFiles.map(file => (
          <FileTypeItem key={file.type} {...file} />
        ))}
      </div>

    </Container>
  );
};

export default Files;