import React from 'react';
import { Container, Title, Text, Stack, List, Group } from '@mantine/core';
import { IconScale, IconMathFunction, IconChartBar, IconAdjustments } from '@tabler/icons-react';
import { InlineMath, BlockMath } from 'react-katex';
import CodeBlock from 'components/CodeBlock';
import DataInteractionPanel from 'components/DataInteractionPanel';

const ScalingAndNormalization = () => {
  const metadata = {
    description: "This dataset contains information about housing in California districts, derived from the 1990 census.",
    source: "California Housing Dataset from StatLib repository",
    target: "PRICE",
    listData: [
      { name: "MedInc", description: "Median income in block group (in tens of thousands of US Dollars)" },
      { name: "HouseAge", description: "Median house age in block group (in years)" },
      { name: "AveRooms", description: "Average number of rooms per household" },
      { name: "AveBedrms", description: "Average number of bedrooms per household" },
      { name: "Population", description: "Block group population" },
      { name: "AveOccup", description: "Average number of household members" },
      { name: "Latitude", description: "Block group latitude (in degrees)" },
      { name: "Longitude", description: "Block group longitude (in degrees)" },
      { name: "PRICE", description: "Median house value in block group (in hundreds of thousands of US Dollars)" },
    ],
  };

  return (
    <Container fluid>
      <Title order={1} id="scaling-and-normalization" mt="xl" mb="md">Scaling and Normalization</Title>
      
      <Text>
        Scaling and normalization are crucial preprocessing steps in many machine learning workflows. They help to standardize the range of features in a dataset, ensuring that all features contribute equally to the analysis and model performance.
      </Text>

      <Stack spacing="xl" mt="xl">
        <Section
          icon={<IconScale size={24} />}
          title="Why Scale and Normalize?"
          id="why-scale-normalize"
        >
          <Text>
            Different features in a dataset often have varying scales, units, and ranges. Scaling and normalization address several issues:
          </Text>
          
          <List>
            <List.Item>Prevent features with larger magnitudes from dominating the learning process</List.Item>
            <List.Item>Improve convergence speed for many optimization algorithms</List.Item>
            <List.Item>Enhance the stability and performance of many machine learning models</List.Item>
            <List.Item>Make features comparable and interpretable across different scales</List.Item>
          </List>

          <Text mt="md">
            Without proper scaling or normalization:
          </Text>

          <List>
            <List.Item>Distance-based algorithms may be disproportionately influenced by features with larger scales</List.Item>
            <List.Item>Gradient descent-based algorithms may converge slowly or struggle to find the optimal solution</List.Item>
            <List.Item>Some models (e.g., neural networks) may struggle to learn effectively from features with widely different scales</List.Item>
          </List>
        </Section>

        <Section
          icon={<IconMathFunction size={24} />}
          title="Scaling Methods"
          id="scaling-methods"
        >
          <Text>
            Scaling methods transform features to a common scale without distorting differences in the ranges of values.
          </Text>
          
          <Title order={4} mt="md">1. Min-Max Scaling</Title>
          <Text>
            Scales features to a fixed range, typically [0, 1].
          </Text>
          <BlockMath math="X_{scaled} = \frac{X - X_{min}}{X_{max} - X_{min}}" />
          <CodeBlock
            language="python"
            code={`
from sklearn.preprocessing import MinMaxScaler
import numpy as np

X = np.array([[1, 2], [3, 4], [5, 6]])
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
print(X_scaled)
            `}
          />

          <Title order={4} mt="md">2. Standard Scaling (Z-score Normalization)</Title>
          <Text>
            Transforms features to have zero mean and unit variance.
          </Text>
          <BlockMath math="X_{scaled} = \frac{X - \mu}{\sigma}" />
          <CodeBlock
            language="python"
            code={`
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(X_scaled)
            `}
          />

          <Title order={4} mt="md">3. Robust Scaling</Title>
          <Text>
            Uses statistics that are robust to outliers.
          </Text>
          <BlockMath math="X_{scaled} = \frac{X - \text{median}(X)}{\text{IQR}(X)}" />
          <CodeBlock
            language="python"
            code={`
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
print(X_scaled)
            `}
          />

          <Title order={4} mt="md">4. MaxAbs Scaling</Title>
          <Text>
            Scales each feature by its maximum absolute value.
          </Text>
          <BlockMath math="X_{scaled} = \frac{X}{\max(|X|)}" />
          <CodeBlock
            language="python"
            code={`
from sklearn.preprocessing import MaxAbsScaler

scaler = MaxAbsScaler()
X_scaled = scaler.fit_transform(X)
print(X_scaled)
            `}
          />
        </Section>

        <Section
          icon={<IconChartBar size={24} />}
          title="Normalization Methods"
          id="normalization-methods"
        >
          <Text>
            Normalization typically refers to adjusting values measured on different scales to a common scale, often scaling the length of vectors to unity.
          </Text>
          
          <Title order={4} mt="md">1. L1 Normalization (Least Absolute Deviations)</Title>
          <Text>
            Ensures the sum of absolute values is 1 for each sample.
          </Text>
          <BlockMath math="X_{normalized} = \frac{X}{||X||_1}" />
          <CodeBlock
            language="python"
            code={`
from sklearn.preprocessing import Normalizer

normalizer = Normalizer(norm='l1')
X_normalized = normalizer.fit_transform(X)
print(X_normalized)
            `}
          />

          <Title order={4} mt="md">2. L2 Normalization (Least Squares)</Title>
          <Text>
            Ensures the sum of squares is 1 for each sample.
          </Text>
          <BlockMath math="X_{normalized} = \frac{X}{||X||_2}" />
          <CodeBlock
            language="python"
            code={`
normalizer = Normalizer(norm='l2')
X_normalized = normalizer.fit_transform(X)
print(X_normalized)
            `}
          />

          <Title order={4} mt="md">3. Max Normalization</Title>
          <Text>
            Scales the features by dividing each value by the maximum value for that feature.
          </Text>
          <BlockMath math="X_{normalized} = \frac{X}{\max(X)}" />
          <CodeBlock
            language="python"
            code={`
def max_normalize(X):
    return X / np.max(X, axis=0)

X_normalized = max_normalize(X)
print(X_normalized)
            `}
          />
        </Section>

        <Section
          icon={<IconAdjustments size={24} />}
          title="Choosing the Right Method"
          id="choosing-method"
        >
          <Text>
            The choice between scaling and normalization methods depends on the specific requirements of your data and the algorithm you're using:
          </Text>
          
          <List>
            <List.Item><span style={{ fontWeight: 700 }}>Min-Max Scaling:</span> When you need bounded values and your data doesn't have significant outliers.</List.Item>
            <List.Item><span style={{ fontWeight: 700 }}>Standard Scaling:</span> When you assume your data follows a normal distribution and when dealing with algorithms sensitive to feature magnitude (e.g., neural networks, SVMs).</List.Item>
            <List.Item><span style={{ fontWeight: 700 }}>Robust Scaling:</span> When your data contains many outliers.</List.Item>
            <List.Item><span style={{ fontWeight: 700 }}>MaxAbs Scaling:</span> When dealing with sparse data and you want to preserve zero values.</List.Item>
            <List.Item><span style={{ fontWeight: 700 }}>L1 Normalization:</span> For sparse data or when you want many features to be zero.</List.Item>
            <List.Item><span style={{ fontWeight: 700 }}>L2 Normalization:</span> When the magnitude of vectors is important (e.g., in cosine similarity).</List.Item>
          </List>

          <Text mt="md">
            It's often beneficial to experiment with different scaling and normalization techniques to determine which works best for your specific dataset and machine learning task.
          </Text>
        </Section>

        <Section
  title="Practical Considerations"
  id="practical-considerations"
>
  <List>
    <List.Item>
      <Text><span style={{ fontWeight: 700 }}>Apply to Training and Test Data:</span> Always fit the scaler on the training data and apply the same transformation to both training and test data.</Text>
    </List.Item>
    <List.Item>
      <Text><span style={{ fontWeight: 700 }}>Handle Outliers:</span> Consider removing or capping outliers before scaling, especially for methods sensitive to extreme values.</Text>
    </List.Item>
    <List.Item>
      <Text><span style={{ fontWeight: 700 }}>Feature Types:</span> Be aware of the nature of your features. Avoid scaling one-hot encoded variables. These variables represent distinct categories, not numerical magnitudes, so scaling them would distort their meaning.</Text>
    </List.Item>
    <List.Item>
      <Text><span style={{ fontWeight: 700 }}>Model Requirements:</span> Some models (e.g., tree-based models) may not require feature scaling, while others (e.g., neural networks) often benefit greatly from it.</Text>
    </List.Item>
  </List>
</Section>

      </Stack>
      <div id="notebook-example"></div>
      <DataInteractionPanel
        dataUrl="/modules/module5/course/module5_course_scaling_and_normalization"
        notebookUrl="/modules/module5/course/scaling_and_normalization.ipynb"
        notebookHtmlUrl="/modules/module5/course/scaling_and_normalization.html"
        notebookColabUrl="/website/public/modules/module5/course/scaling_and_normalization.ipynb"
        requirementsUrl="/modules/module5/course/module5_requirements.txt"
        metadata={metadata}
      />
    </Container>
  );
};

const Section = ({ icon, title, id, children }) => (
  <Stack spacing="sm">
    <Group spacing="xs">
      {icon}
      <Title order={2} id={id}>{title}</Title>
    </Group>
    {children}
  </Stack>
);

export default ScalingAndNormalization;