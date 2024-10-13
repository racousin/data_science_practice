import React from 'react';
import { Container, Title, Text, Stack, List, Group } from '@mantine/core';
import { IconTools, IconMath, IconChartBar, IconClock } from '@tabler/icons-react';
import CodeBlock from 'components/CodeBlock';
import DataInteractionPanel from 'components/DataInteractionPanel';

const FeatureEngineering = () => {
  const metadata = {
    description: "This dataset contains hourly weather and holiday information along with corresponding bike rental counts, aimed at predicting bike rental demand based on various environmental and temporal factors.",
    source: "City Bike Share Program and Local Weather Station",
    target: "count",
    listData: [
      { name: "datetime", description: "Date and time of the observation, typically in hourly intervals.", dataType: "Datetime", example: "2022-01-01 13:00:00" },
      { name: "holiday", description: "Boolean indicating whether the day is a holiday (1) or not (0).", dataType: "Boolean", example: "1 (holiday), 0 (not holiday)" },
      { name: "temp", description: "Temperature in Celsius at the time of observation.", dataType: "Continuous", example: "25.5" },
      { name: "humidity", description: "Relative humidity as a percentage.", dataType: "Continuous", example: "65.0" },
      { name: "windspeed", description: "Wind speed in km/h at the time of observation.", dataType: "Continuous", example: "12.7" },
      { name: "pressure", description: "Atmospheric pressure in hectopascals (hPa).", dataType: "Continuous", example: "1015.2" },
      { name: "count", description: "Number of bikes rented in the given hour.", dataType: "Discrete", example: "145" },
    ],
  };

  return (
    <Container fluid>
      <Title order={1} id="feature-engineering" mt="xl" mb="md">Feature Engineering</Title>
      
      <Text>
        Feature engineering is the process of transforming raw data into formats that are better suited for machine learning models. It enhances model performance by incorporating domain-specific knowledge and data insights. This process is crucial for tailoring data to the specific needs of the business and predictive models.
      </Text>

      <Stack spacing="xl" mt="xl">
        <Section
          icon={<IconTools size={24} />}
          title="Decomposition and Feature Extraction"
          id="decomposition-extraction"
        >
          <Text>
            Decomposition involves breaking down complex features into simpler components to reveal additional insights. Feature extraction creates new features from existing ones to capture important information more effectively.
          </Text>
          
          <Title order={4} mt="md">1. DateTime Decomposition</Title>
          <CodeBlock
            language="python"
            code={`
import pandas as pd

df['datetime'] = pd.to_datetime(df['datetime'])
df['year'] = df['datetime'].dt.year
df['month'] = df['datetime'].dt.month
df['day'] = df['datetime'].dt.day
df['hour'] = df['datetime'].dt.hour
df['dayofweek'] = df['datetime'].dt.dayofweek
df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
            `}
          />

          <Title order={4} mt="md">2. Text Feature Extraction</Title>
          <CodeBlock
            language="python"
            code={`
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=1000)
tfidf_features = tfidf.fit_transform(df['text_column'])
tfidf_df = pd.DataFrame(tfidf_features.toarray(), columns=tfidf.get_feature_names_out())
            `}
          />

          <Title order={4} mt="md">3. Domain-Specific Feature Extraction</Title>
          <CodeBlock
            language="python"
            code={`
# Example: Creating a 'temperature_feel' feature
df['temperature_feel'] = df['temp'] - 0.55 * (1 - df['humidity'] / 100) * (df['temp'] - 14.5)
            `}
          />
        </Section>

        <Section
          icon={<IconMath size={24} />}
          title="Mathematical Transformations"
          id="mathematical-transformations"
        >
          <Text>
            Mathematical transformations can help normalize data distributions, handle skewness, or create more informative features.
          </Text>
          
          <Title order={4} mt="md">1. Log Transformation</Title>
          <CodeBlock
            language="python"
            code={`
import numpy as np

df['log_windspeed'] = np.log1p(df['windspeed'])
            `}
          />

          <Title order={4} mt="md">2. Polynomial Features</Title>
          <CodeBlock
            language="python"
            code={`
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(df[['temp', 'humidity']])
poly_df = pd.DataFrame(poly_features, columns=poly.get_feature_names(['temp', 'humidity']))
            `}
          />

          <Title order={4} mt="md">3. Box-Cox Transformation</Title>
          <CodeBlock
            language="python"
            code={`
from scipy import stats

df['box_cox_count'], _ = stats.boxcox(df['count'])
            `}
          />
        </Section>

        <Section
          icon={<IconChartBar size={24} />}
          title="Binning and Aggregation"
          id="binning-aggregation"
        >
          <Text>
            Binning transforms continuous data into categorical data, while aggregation provides summary statistics for groups.
          </Text>
          
          <Title order={4} mt="md">1. Equal-Width Binning</Title>
          <CodeBlock
            language="python"
            code={`
df['temp_bins'] = pd.cut(df['temp'], bins=4, labels=['Cold', 'Cool', 'Warm', 'Hot'])
            `}
          />

          <Title order={4} mt="md">2. Quantile Binning</Title>
          <CodeBlock
            language="python"
            code={`
df['humidity_bins'] = pd.qcut(df['humidity'], q=4, labels=['Low', 'Medium-Low', 'Medium-High', 'High'])
            `}
          />

          <Title order={4} mt="md">3. Time-based Aggregation</Title>
          <CodeBlock
            language="python"
            code={`
daily_agg = df.groupby(df['datetime'].dt.date).agg({
    'count': ['mean', 'max', 'min'],
    'temp': 'mean',
    'humidity': 'mean'
})
daily_agg.columns = ['_'.join(col).strip() for col in daily_agg.columns.values]
            `}
          />
        </Section>

        <Section
          icon={<IconClock size={24} />}
          title="Time Series Features"
          id="time-series-features"
        >
          <Text>
            Time series data often requires specific feature engineering techniques to capture temporal patterns and dependencies.
          </Text>
          
          <Title order={4} mt="md">1. Lag Features</Title>
          <CodeBlock
            language="python"
            code={`
for lag in [1, 3, 6, 24]:  # 1 hour, 3 hours, 6 hours, 1 day
    df[f'count_lag_{lag}'] = df['count'].shift(lag)
            `}
          />

          <Title order={4} mt="md">2. Rolling Window Statistics</Title>
          <CodeBlock
            language="python"
            code={`
for window in [6, 12, 24]:  # 6 hours, 12 hours, 1 day
    df[f'count_rolling_mean_{window}'] = df['count'].rolling(window=window).mean()
    df[f'count_rolling_std_{window}'] = df['count'].rolling(window=window).std()
            `}
          />

          <Title order={4} mt="md">3. Seasonal Features</Title>
          <CodeBlock
            language="python"
            code={`
df['day_of_year'] = df['datetime'].dt.dayofyear
df['week_of_year'] = df['datetime'].dt.isocalendar().week
df['month_progress'] = df['day'] / df['datetime'].dt.days_in_month
            `}
          />
        </Section>

        <Section
          title="Considerations and Best Practices"
          id="considerations-best-practices"
        >
          <List>
            <List.Item>
              <Text><span style={{ fontWeight: 700 }}>Domain Knowledge:</span> Leverage industry expertise to create meaningful features that capture important aspects of the data.</Text>
            </List.Item>
            <List.Item>
              <Text><span style={{ fontWeight: 700 }}>Feature Selection:</span> After creating new features, use feature selection techniques to identify the most relevant ones for your model.</Text>
            </List.Item>
            <List.Item>
              <Text><span style={{ fontWeight: 700 }}>Avoid Data Leakage:</span> Ensure that features are created using only information that would be available at the time of prediction in a real-world scenario.</Text>
            </List.Item>
            <List.Item>
              <Text><span style={{ fontWeight: 700 }}>Interpretability:</span> Consider the trade-off between complex feature engineering and model interpretability.</Text>
            </List.Item>
            <List.Item>
              <Text><span style={{ fontWeight: 700 }}>Scalability:</span> Design your feature engineering pipeline to handle large datasets efficiently.</Text>
            </List.Item>
          </List>
        </Section>
      </Stack>

      <DataInteractionPanel
        dataUrl="/modules/module5/course/module5_course_feature_engineering.csv"
        notebookUrl="/modules/module5/course/feature_engineering.ipynb"
        notebookHtmlUrl="/modules/module5/course/feature_engineering.html"
        notebookColabUrl="/website/public/modules/module5/course/feature_engineering.ipynb"
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

export default FeatureEngineering;