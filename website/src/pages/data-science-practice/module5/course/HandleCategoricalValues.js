import React from 'react';
import { Container, Title, Text, Stack, List, Group, Paper } from '@mantine/core';
import { IconChartBar, IconHash, IconTags, IconBinaryTree } from '@tabler/icons-react';
import CodeBlock from 'components/CodeBlock';
import DataInteractionPanel from 'components/DataInteractionPanel';

const HandleCategoricalValues = () => {
  const metadata = {
    description: "This dataset includes characteristics of different mushrooms, aiming to classify them as either poisonous or edible based on various physical attributes.",
    source: "Mycology Research Data",
    target: "class",
    listData: [
      {
        name: "class",
        description: "Indicates whether the mushroom is poisonous (p) or edible (e).",
        dataType: "Categorical",
        example: "p (poisonous), e (edible)",
      },
      {
        name: "cap-diameter",
        description: "Numerical measurement likely representing the diameter of the mushroom's cap.",
        dataType: "Continuous",
        example: "15 cm",
      },
      {
        name: "cap-shape",
        description: "Descriptive categories for the shape of the mushroom cap (e.g., x for convex, f for flat).",
        dataType: "Categorical",
        example: "x (convex)",
      },
      {
        name: "stem-width",
        description: "Numerical measurement likely representing the width of the mushroom's stem.",
        dataType: "Continuous",
        example: "2 cm",
      },
      {
        name: "has-ring",
        description: "Boolean indicating the presence of a ring (t for true, f for false).",
        dataType: "Boolean",
        example: "t (true)",
      },
    ],
  };

  return (
    <Container fluid>
      <Title order={1} id="handle-categorical-values" mt="xl" mb="md">Handling Categorical Values</Title>
      
      <Stack spacing="xl">
        <Title order={3} id="types-of-categorical-data">Types of Categorical Data</Title>
          <Text>
            Categorical data can be classified into two main types:
          </Text>
          <List>
            
            <List.Item><span style={{ fontWeight: 700 }}>Nominal :</span> Categories without inherent order (e.g., colors, types of cuisine)</List.Item>
            <List.Item><span style={{ fontWeight: 700 }}>Ordinal :</span> Categories with a logical order (e.g., rankings, education level)</List.Item>
          </List>
          <CodeBlock
            language="python"
            code={`
import pandas as pd

# Sample dataset
data = {
    'color': ['red', 'blue', 'green', 'red', 'blue'],
    'size': ['small', 'medium', 'large', 'medium', 'small'],
    'rating': ['low', 'medium', 'high', 'medium', 'low']
}
df = pd.DataFrame(data)

print("Original DataFrame:")
print(df)

# Identify data types
print("Data types:")
print(df.dtypes)

# Identify unique values
print("Unique values in each column:")
for column in df.columns:
    print(f"{column}: {df[column].nunique()} - {df[column].unique()}")

# Output:
# Original DataFrame:
#   color    size rating
# 0   red   small    low
# 1  blue  medium medium
# 2 green   large   high
# 3   red  medium medium
# 4  blue   small    low

# Data types:
# color     object
# size      object
# rating    object
# dtype: object

# Unique values in each column:
# color: 3 - ['red' 'blue' 'green']
# size: 3 - ['small' 'medium' 'large']
# rating: 3 - ['low' 'medium' 'high']
            `}
          />
        

        <Title order={3} id="identify-and-visualize-categorical-values">Identify and Visualize Categorical Values</Title>
          <Text>
            Identifying and visualizing categorical data is crucial for understanding distribution and influences within the dataset. Use techniques like count plots and bar plots to gain insights.
          </Text>
          <CodeBlock
            language="python"
            code={`
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Sample dataset
data = {
    'color': ['red', 'blue', 'green', 'red', 'blue', 'green', 'red', 'blue', 'green', 'red'],
    'size': ['small', 'medium', 'large', 'medium', 'small', 'large', 'small', 'medium', 'large', 'medium'],
    'rating': [4, 3, 5, 4, 2, 5, 3, 4, 5, 3]
}
df = pd.DataFrame(data)

# Count plot for 'color'
plt.figure(figsize=(10, 6))
sns.countplot(x='color', data=df, palette='viridis')
plt.title('Count Plot of Colors')
plt.xlabel('Color')
plt.ylabel('Count')
plt.show()

# Bar plot for average rating by color
plt.figure(figsize=(10, 6))
sns.barplot(x='color', y='rating', data=df, ci=None, palette='viridis')
plt.title('Average Rating by Color')
plt.xlabel('Color')
plt.ylabel('Average Rating')
plt.show()

# Stacked bar plot for size distribution by color
size_color = pd.crosstab(df['color'], df['size'])
size_color_pct = size_color.div(size_color.sum(1), axis=0)
size_color_pct.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Size Distribution by Color')
plt.xlabel('Color')
plt.ylabel('Percentage')
plt.legend(title='Size', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
            `}
          />
        

        <Group grow align="flex-start" spacing="xl">
          <Title order={3} id="one-hot-encoding">One-Hot Encoding</Title>
            <Text>
              One-hot encoding creates a new binary column for each category. It's useful for nominal data and models expecting numerical input.
            </Text>
            <CodeBlock
              language="python"
              code={`
import pandas as pd

# Sample dataset
data = {
    'color': ['red', 'blue', 'green', 'red', 'blue'],
    'size': ['small', 'medium', 'large', 'medium', 'small']
}
df = pd.DataFrame(data)

print("Original DataFrame:")
print(df)

# One-hot encoding
df_encoded = pd.get_dummies(df, columns=['color', 'size'])

print("One-hot encoded DataFrame:")
print(df_encoded)

# Output:
# Original DataFrame:
#   color    size
# 0   red   small
# 1  blue  medium
# 2 green   large
# 3   red  medium
# 4  blue   small

# One-hot encoded DataFrame:
#    color_blue  color_green  color_red  size_large  size_medium  size_small
# 0           0            0          1           0            0           1
# 1           1            0          0           0            1           0
# 2           0            1          0           1            0           0
# 3           0            0          1           0            1           0
# 4           1            0          0           0            0           1
              `}
            />
          

          <Title order={3} id="label-encoding">Label Encoding</Title>
            <Text>
              Label encoding converts each category to a number. It's particularly useful for ordinal data where the order matters.
            </Text>
            <CodeBlock
              language="python"
              code={`
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Sample dataset
data = {
    'color': ['red', 'blue', 'green', 'red', 'blue'],
    'size': ['small', 'medium', 'large', 'medium', 'small']
}
df = pd.DataFrame(data)

print("Original DataFrame:")
print(df)

# Label encoding
le = LabelEncoder()
df['color_encoded'] = le.fit_transform(df['color'])
df['size_encoded'] = le.fit_transform(df['size'])

print("Label encoded DataFrame:")
print(df)

print("Encoding mappings:")
print("Color:", dict(zip(le.classes_, le.transform(le.classes_))))
print("Size:", dict(zip(le.classes_, le.transform(le.classes_))))

# Output:
# Original DataFrame:
#   color    size
# 0   red   small
# 1  blue  medium
# 2 green   large
# 3   red  medium
# 4  blue   small

# Label encoded DataFrame:
#   color    size  color_encoded  size_encoded
# 0   red   small              2             2
# 1  blue  medium              0             1
# 2 green   large              1             0
# 3   red  medium              2             1
# 4  blue   small              0             2

# Encoding mappings:
# Color: {'blue': 0, 'green': 1, 'red': 2}
# Size: {'large': 0, 'medium': 1, 'small': 2}
              `}
            />
          
        </Group>

        <Title order={3} id="advanced-techniques">Advanced Techniques</Title>
          <Stack spacing="md">
              <Title order={4}>Handling Unseen Categories</Title>
              <Text>
                For categories in the test set not present during training, consider assigning an 'unknown' label or using the most frequent category.
              </Text>
              <CodeBlock
                language="python"
                code={`
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

# Training data
train_data = {'color': ['red', 'blue', 'green', 'red', 'blue']}
df_train = pd.DataFrame(train_data)

# Test data with unseen category
test_data = {'color': ['yellow', 'red', 'purple', 'blue', 'orange']}
df_test = pd.DataFrame(test_data)

# Instantiate OrdinalEncoder with unknown value handling
oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

# Fit the encoder on the training data
oe.fit(df_train[['color']])

# Transform the training data
train_encoded = oe.transform(df_train[['color']])

# Transform the test data, handling unknown categories
test_encoded = oe.transform(df_test[['color']])

# Flatten the results for easier reading
train_encoded = train_encoded.flatten()
test_encoded = test_encoded.flatten()

print("Training data encoded:")
print(train_encoded)
print("Test data encoded:")
print(test_encoded)
print("Encoding categories:", oe.categories_)

# Output:
# Training data encoded:
# [2 0 1 2 0]
# 
# Test data encoded:
# [-1 2 -1 0 -1]
# 
# Encoding categories: [['blue' 'green' 'red']]
                `}
              />
              <Title order={4}>Using Categorical Data in Models</Title>
              <Text>
                Some algorithms (e.g., CatBoost, decision trees) can handle categorical data directly without explicit encoding.
              </Text>
              <CodeBlock
                language="python"
                code={`
import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool

# Sample dataset
data = {
    'color': ['red', 'blue', 'green', 'red', 'blue', 'green', 'red', 'blue', 'green', 'red'],
    'size': ['small', 'medium', 'large', 'medium', 'small', 'large', 'small', 'medium', 'large', 'medium'],
    'price': [10, 15, 20, 12, 8, 22, 11, 16, 21, 13],
    'sold': [1, 0, 1, 1, 0, 1, 0, 1, 1, 0]
}
df = pd.DataFrame(data)

# Split the data
X = df.drop('sold', axis=1)
y = df['sold']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create CatBoost pools
train_pool = Pool(X_train, y_train, cat_features=['color', 'size'])
test_pool =Pool(X_test, cat_features=['color', 'size'])

# Initialize and train the model
model = CatBoostClassifier(iterations=100, depth=5, learning_rate=0.1, loss_function='Logloss', verbose=False)
model.fit(train_pool)
                `}
              />
          </Stack>
        

        <Title order={3} id="considerations-and-best-practices">Considerations and Best Practices</Title>
          <List>
            <List.Item>
              <Text><span style={{ fontWeight: 700 }}>Cardinality:</span> For high-cardinality categorical variables (many unique values), consider grouping less frequent categories or using embedding techniques.</Text>
            </List.Item>
            <List.Item>
              <Text><span style={{ fontWeight: 700 }}>Domain knowledge:</span> Incorporate domain expertise when encoding ordinal variables to ensure the correct order is maintained.</Text>
            </List.Item>
            <List.Item>
              <Text><span style={{ fontWeight: 700 }}>Dimensionality:</span> Be cautious with one-hot encoding for high-cardinality variables, as it can significantly increase the number of features.</Text>
            </List.Item>
            <List.Item>
              <Text><span style={{ fontWeight: 700 }}>Model selection:</span> Choose encoding methods that are compatible with your selected machine learning algorithm.</Text>
            </List.Item>
            <List.Item>
              <Text><span style={{ fontWeight: 700 }}>Consistency:</span> Apply the same encoding strategy to both training and test datasets to ensure consistency.</Text>
            </List.Item>
          </List>
        
      </Stack>
      <div id="notebook-example"></div>
      <DataInteractionPanel
        dataUrl="/modules/data-science-practice/module5/course/module5_course_handling_categorical.csv"
        notebookUrl="/modules/data-science-practice/module5/course/handling_categorical.ipynb"
        notebookHtmlUrl="/modules/data-science-practice/module5/course/handling_categorical.html"
        notebookColabUrl="/website/public/modules/data-science-practice/module5/course/handling_categorical.ipynb"
        requirementsUrl="/modules/data-science-practice/module5/course/module5_requirements.txt"
        metadata={metadata}
      />
    </Container>
  );
};



export default HandleCategoricalValues;