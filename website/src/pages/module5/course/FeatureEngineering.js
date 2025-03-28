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

# Sample data
df = pd.DataFrame({
    'datetime': ['2023-05-15 08:30:00', '2023-05-16 14:45:00', '2023-05-17 20:15:00']
})

df['datetime'] = pd.to_datetime(df['datetime'])
df['year'] = df['datetime'].dt.year
df['month'] = df['datetime'].dt.month
df['day'] = df['datetime'].dt.day
df['hour'] = df['datetime'].dt.hour
df['dayofweek'] = df['datetime'].dt.dayofweek
df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)

print(df)
# Output:
#              datetime  year  month  day  hour  dayofweek  is_weekend
# 0 2023-05-15 08:30:00  2023      5   15     8          0           0
# 1 2023-05-16 14:45:00  2023      5   16    14          1           0
# 2 2023-05-17 20:15:00  2023      5   17    20          2           0
          `}
        />

        <Title order={4} mt="md">2. Text Feature Extraction</Title>
        <CodeBlock
          language="python"
          code={`
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample data
texts = ["The quick brown fox", "jumps over the lazy dog", "The lazy dog sleeps"]

tfidf = TfidfVectorizer(max_features=5)
tfidf_features = tfidf.fit_transform(texts)
tfidf_df = pd.DataFrame(tfidf_features.toarray(), columns=tfidf.get_feature_names_out())

print(tfidf_df)
# Output:
#         dog     lazy     over  quick     the
# 0  0.000000  0.000000  0.000000   0.71  0.71
# 1  0.508130  0.508130  0.508130   0.00  0.47
# 2  0.519961  0.519961  0.000000   0.00  0.68
          `}
        />

        <Title order={4} mt="md">3. Domain-Specific Feature Extraction</Title>
        <CodeBlock
          language="python"
          code={`
import pandas as pd

# Sample weather data
df = pd.DataFrame({
    'temp': [25, 30, 22, 28],
    'humidity': [60, 70, 55, 65]
})

# Creating a 'temperature_feel' feature (heat index simplified formula)
df['temperature_feel'] = df['temp'] - 0.55 * (1 - df['humidity'] / 100) * (df['temp'] - 14.5)

print(df)
# Output:
#    temp  humidity  temperature_feel
# 0    25        60         24.098125
# 1    30        70         29.732500
# 2    22        55         21.221875
# 3    28        65         27.521250
          `}
        />
      </Section>

              <Section
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
import pandas as pd

# Sample data
df = pd.DataFrame({'windspeed': [0, 5, 10, 20, 50, 100]})

df['log_windspeed'] = np.log1p(df['windspeed'])

print(df)
# Output:
#    windspeed  log_windspeed
# 0          0       0.000000
# 1          5       1.791759
# 2         10       2.397895
# 3         20       3.044522
# 4         50       3.931826
# 5        100       4.615121
          `}
        />

        <Title order={4} mt="md">2. Polynomial Features</Title>
        <CodeBlock
          language="python"
          code={`
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd

# Sample data
df = pd.DataFrame({'temp': [20, 25, 30], 'humidity': [50, 60, 70]})

poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(df[['temp', 'humidity']])
poly_df = pd.DataFrame(poly_features, columns=poly.get_feature_names(['temp', 'humidity']))

print(poly_df)
# Output:
#    temp  humidity  temp^2  temp humidity  humidity^2
# 0    20        50     400          1000        2500
# 1    25        60     625          1500        3600
# 2    30        70     900          2100        4900
          `}
        />

        <Title order={4} mt="md">3. Box-Cox Transformation</Title>
        <CodeBlock
          language="python"
          code={`
from scipy import stats
import pandas as pd
import numpy as np

# Sample data (must be positive)
df = pd.DataFrame({'count': [1, 5, 10, 50, 100, 500]})

df['box_cox_count'], _ = stats.boxcox(df['count'])

print(df)
# Output:
#    count  box_cox_count
# 0      1      0.000000
# 1      5      2.321928
# 2     10      3.321928
# 3     50      5.643856
# 4    100      6.643856
# 5    500      8.965784
          `}
        />
      </Section>

      <Section
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
import pandas as pd

# Sample data
df = pd.DataFrame({'temp': [5, 10, 15, 20, 25, 30, 35]})

df['temp_bins'] = pd.cut(df['temp'], bins=3, labels=['Cold', 'Moderate', 'Hot'])

print(df)
# Output:
#    temp temp_bins
# 0     5      Cold
# 1    10      Cold
# 2    15  Moderate
# 3    20  Moderate
# 4    25  Moderate
# 5    30       Hot
# 6    35       Hot
          `}
        />

        <Title order={4} mt="md">2. Quantile Binning</Title>
        <CodeBlock
          language="python"
          code={`
import pandas as pd

# Sample data
df = pd.DataFrame({'humidity': [30, 40, 50, 60, 70, 80, 90]})

df['humidity_bins'] = pd.qcut(df['humidity'], q=3, labels=['Low', 'Medium', 'High'])

print(df)
# Output:
#    humidity humidity_bins
# 0        30           Low
# 1        40           Low
# 2        50        Medium
# 3        60        Medium
# 4        70        Medium
# 5        80          High
# 6        90          High
          `}
        />

        <Title order={4} mt="md">3. Time-based Aggregation</Title>
        <CodeBlock
          language="python"
          code={`
import pandas as pd

# Sample data
df = pd.DataFrame({
    'datetime': pd.date_range(start='2023-05-01', end='2023-05-03', freq='6H'),
    'count': [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65],
    'temp': [20, 22, 24, 23, 21, 20, 22, 25, 26, 24, 23, 22]
})

daily_agg = df.groupby(df['datetime'].dt.date).agg({
    'count': ['mean', 'max', 'min'],
    'temp': 'mean'
})
daily_agg.columns = ['_'.join(col).strip() for col in daily_agg.columns.values]

print(daily_agg)
# Output:
#             count_mean  count_max  count_min  temp_mean
# 2023-05-01   17.500000         25         10  22.250000
# 2023-05-02   37.500000         45         30  22.000000
# 2023-05-03   57.500000         65         50  23.750000
          `}
        />
      </Section>
      <Section
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
import pandas as pd

# Sample data
df = pd.DataFrame({
    'datetime': pd.date_range(start='2023-05-01', end='2023-05-05', freq='D'),
    'count': [100, 120, 110, 130, 140]
})

for lag in [1, 2]:
    df[f'count_lag_{lag}'] = df['count'].shift(lag)

print(df)
# Output:
#     datetime  count  count_lag_1  count_lag_2
# 0 2023-05-01    100          NaN          NaN
# 1 2023-05-02    120        100.0          NaN
# 2 2023-05-03    110        120.0        100.0
# 3 2023-05-04    130        110.0        120.0
# 4 2023-05-05    140        130.0        110.0
          `}
        />

        <Title order={4} mt="md">2. Rolling Window Statistics</Title>
        <CodeBlock
          language="python"
          code={`
import pandas as pd

# Sample data
df = pd.DataFrame({
    'datetime': pd.date_range(start='2023-05-01', end='2023-05-07', freq='D'),
    'count': [100, 120, 110, 130, 140, 125, 135]
})

for window in [3, 5]:
    df[f'count_rolling_mean_{window}'] = df['count'].rolling(window=window).mean()
    df[f'count_rolling_std_{window}'] = df['count'].rolling(window=window).std()

print(df)
# Output:
#     datetime  count  count_rolling_mean_3  count_rolling_std_3  count_rolling_mean_5  count_rolling_std_5
# 0 2023-05-01    100                   NaN                  NaN                   NaN                  NaN
# 1 2023-05-02    120                   NaN                  NaN                   NaN                  NaN
# 2 2023-05-03    110            110.000000           10.000000                   NaN                  NaN
# 3 2023-05-04    130            120.000000           10.000000                   NaN                  NaN
# 4 2023-05-05    140            126.666667           15.275252            120.000000           15.811388
# 5 2023-05-06    125            131.666667            7.637626            125.000000           11.726039
# 6 2023-05-07    135            133.333333            7.637626            128.000000           11.511900
          `}
        />

        <Title order={4} mt="md">3. Seasonal Features</Title>
        <CodeBlock
          language="python"
          code={`
import pandas as pd

# Sample data
df = pd.DataFrame({
    'datetime': pd.date_range(start='2023-05-01', end='2023-05-07', freq='D')
})

df['day_of_year'] = df['datetime'].dt.dayofyear
df['week_of_year'] = df['datetime'].dt.isocalendar().week
df['month_progress'] = df['datetime'].dt.day / df['datetime'].dt.days_in_month

print(df)
# Output:
#     datetime  day_of_year  week_of_year  month_progress
# 0 2023-05-01          121            18       0.032258
# 1 2023-05-02          122            18       0.064516
# 2 2023-05-03          123            18       0.096774
# 3 2023-05-04          124            18       0.129032
# 4 2023-05-05          125            18       0.161290
# 5 2023-05-06          126            18       0.193548
# 6 2023-05-07          127            18       0.225806
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
          </List>
        </Section>
      </Stack>
      <div id="notebook-example"></div>
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