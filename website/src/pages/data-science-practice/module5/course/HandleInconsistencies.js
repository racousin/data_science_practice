import React from "react";
import { Container, Stack, Title, Text, List } from '@mantine/core';
import CodeBlock from "components/CodeBlock";
import DataInteractionPanel from "components/DataInteractionPanel";

const HandleInconsistencies = () => {
  const requirementsUrl =
    process.env.PUBLIC_URL + "/modules/module5/course/module5_requirements.txt";
  const dataUrl =
    process.env.PUBLIC_URL +
    "/modules/module5/course/module5_course_handle_inconsistencies.csv";
  const notebookUrl =
    process.env.PUBLIC_URL +
    "/modules/module5/course/handle_inconsistencies.ipynb";
  const notebookHtmlUrl =
    process.env.PUBLIC_URL +
    "/modules/module5/course/handle_inconsistencies.html";
  const notebookColabUrl =
    process.env.PUBLIC_URL +
    "website/public/modules/module5/course/handle_inconsistencies.ipynb";
  const metadata = {
    description:
      "This dataset contains demographic information including state, country, age, and date.",
    source: "Demographic Survey Records",
    target: null,
    listData: [
      {
        name: "State",
        description: "The state or province of residence",
      },
      {
        name: "Country",
        description: "The country of residence",
      },
      {
        name: "Age",
        description: "The age of the individual in years",
      },
      {
        name: "Date",
        description: "The date the information was recorded (YYYY-MM-DD)",
      },
    ],
  };

  return (
    <Container fluid>
      <Title order={1} my="md">Handle Inconsistencies</Title>

      <Stack>
        <Title order={2} id="types-of-inconsistencies">Types of Inconsistencies</Title>
        <List>
          <List.Item>
            <Text><span style={{ fontWeight: 700 }}>Formatting inconsistencies:</span> Variations in date
            formats, text capitalization, or use of special characters.</Text>
          </List.Item>
          <List.Item>
            <Text><span style={{ fontWeight: 700 }}>Type inconsistencies:</span> Mixed data types within a
            column (e.g., numbers and strings).</Text>
          </List.Item>
          <List.Item>
            <Text><span style={{ fontWeight: 700 }}>Contradictory data:</span> Records that conflict with
            each other, often due to data entry errors or merging issues.</Text>
          </List.Item>
        </List>

        <Title order={2} id="detecting-inconsistencies">Detecting Inconsistencies</Title>
        <Text>Effective detection is the first step towards cleaning:</Text>
        <CodeBlock
          language="python"
          code={`# Example dataset with type inconsistencies

df_type = pd.DataFrame({'mixed_column': [1, 2, 3, 'one', 'two', 'three']})

# Detect type inconsistencies
print(df_type['mixed_column'].apply(type).value_counts())

# Output:
# <class 'str'>     3
# <class 'int'>     3`}/>

        <CodeBlock
          language="python"
          code={`# Example dataset with date format inconsistencies

import re

df_date = pd.DataFrame({'date': ['2023-01-01', '01/02/2023', '2023-03-03', '04-05-2023']})

# Detect date format issues
print(df_date['date'].apply(lambda x: not re.match(r'\\d{4}-\\d{2}-\\d{2}', x)).sum())

# Output:
# 2
`}
        />

<CodeBlock
          language="python"
          code={`# Example dataset with contradictory data

df_contradictory = pd.DataFrame({
    'is_positive': [True, True, False, False],
    'value': [5, -2, 3, 3]
})

# Detect contradictory data
print(df_contradictory[(df_contradictory['is_positive'] & (df_contradictory['value'] <= 0)) |
                       (~df_contradictory['is_positive'] & (df_contradictory['value'] => 0))])

# Output:
#   is_positive  value
# 1         True    -2
# 3         True     3`}
        />

        <Title order={2} id="solutions-to-inconsistencies">Solutions to Inconsistencies</Title>
        <Text>
          Depending on the issue, you may choose to treat, remove, or modify
          inconsistent data:
        </Text>
        <List>
          <List.Item>
            <Text><span style={{ fontWeight: 700 }}>Casting types:</span> Ensure all data in a column is of
            the same type.</Text>
          </List.Item>
          <List.Item>
            <Text><span style={{ fontWeight: 700 }}>Standardizing text:</span> Convert all text data to a
            consistent format (e.g., all lower case).</Text>
          </List.Item>
        </List>

        <CodeBlock
          language="python"
          code={`# Casting types
# Define a mapping dictionary for text-to-number conversion
text_to_num = {'one': 1, 'two': 2, 'three': 3}

# Convert string values to their numeric equivalents
df_type['mixed_column'] = df_type['mixed_column'].map(lambda x: text_to_num.get(x, x))

# Convert the entire column to numeric type
df_type['mixed_column'] = pd.to_numeric(df_type['mixed_column'])
# Output:
# After casting types:
#    mixed_column
# 0           1
# 1           2
# 2           3
# 3           1
# 4           2
# 5           3`}/>

      
        <CodeBlock
          language="python"
          code={`# Standardizing text
df_text = pd.DataFrame({'text_column': ['     Hello WORLD ', ' data  Science ', '  PYTHoN']})
df_text['text_column'] = df_text['text_column'].str.lower().str.strip()
# After standardizing text:
print(df_text)

# Output:
#     text_column
# 0   hello world
# 1  data science
# 2        python`}
        />

        <Title order={2} id="advanced-text-manipulation">Advanced Text Manipulation</Title>
        <Text>
          Manage special characters and whitespace to improve text data
          quality:
        </Text>
        <CodeBlock
          language="python"
          code={`import re

# Dataset with a column containing formatted strings
df_formatted = pd.DataFrame({
    'formatted_column': [
        'ID: 123 | Name: John Doe | Age: 30',
        'ID: 456 | Name: Jane Smith | Age: 25',
        'ID: 789 | Name: Bob Johnson | Age: 40'
    ]
})

# Extract information using regex
df_formatted['id'] = df_formatted['formatted_column'].str.extract(r'ID: (\d+)')
df_formatted['name'] = df_formatted['formatted_column'].str.extract(r'Name: ([^|]+)')
df_formatted['age'] = df_formatted['formatted_column'].str.extract(r'Age: (\d+)')

print("After extracting information:")
print(df_formatted)

# Output:
# After extracting information:
#                                  formatted_column   id         name age
# 0  ID: 123 | Name: John Doe | Age: 30             123     John Doe   30
# 1  ID: 456 | Name: Jane Smith | Age: 25           456   Jane Smith   25
# 2  ID: 789 | Name: Bob Johnson | Age: 40          789  Bob Johnson   40`}
        />
      </Stack>

      <div id="notebook-example"></div>
      <DataInteractionPanel
        dataUrl={dataUrl}
        notebookUrl={notebookUrl}
        notebookHtmlUrl={notebookHtmlUrl}
        notebookColabUrl={notebookColabUrl}
        requirementsUrl={requirementsUrl}
        metadata={metadata}
      />
    </Container>
  );
};

export default HandleInconsistencies;