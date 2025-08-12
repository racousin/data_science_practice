import React from 'react';
import { Container, Title, Text, Stack, Table, Group } from '@mantine/core';
import { IconCheck, IconX } from '@tabler/icons-react';
import CodeBlock from 'components/CodeBlock';
import { InlineMath, BlockMath } from 'react-katex';
import 'katex/dist/katex.min.css';
import { Link } from 'react-router-dom';

const TimeSeriesModels = () => {
  return (
    <Container fluid>
      <Title order={1} mt="xl" mb="md">Time Series Models</Title>

      <Text mb="xl">
        Time series analysis involves working with data points indexed in time order. It's crucial in many fields, including finance, economics, and weather forecasting.
      </Text>

      <Stack spacing="xl">
        <ModelSection
          title="ARIMA Models"
          id="arima"
          math={<BlockMath math="X_t = c + \phi_1 X_{t-1} + \cdots + \phi_p X_{t-p} + \theta_1 \varepsilon_{t-1} + \cdots + \theta_q \varepsilon_{t-q} + \varepsilon_t" />}
          description="ARIMA (AutoRegressive Integrated Moving Average) is a popular class of models for time series forecasting. It combines autoregression (AR), differencing (I), and moving average (MA) components."
          hyperparameters={[
            { name: 'p', description: 'Order of the AR term' },
            { name: 'd', description: 'Degree of differencing' },
            { name: 'q', description: 'Order of the MA term' },
          ]}
          checks={{
            unscaled: false,
            missing: false,
            categorical: false,
            regression: true,
            classification: false
          }}
          bayesianOptimization={`
from skopt import BayesSearchCV
from skopt.space import Integer
from statsmodels.tsa.arima.model import ARIMA

def arima_mse(order, X, y):
    try:
        model = ARIMA(y, order=order)
        results = model.fit()
        return -results.mse
    except:
        return 1000000  # Return a large error for failed fits

param_space = {
    'p': Integer(0, 5),
    'd': Integer(0, 2),
    'q': Integer(0, 5)
}

opt = BayesSearchCV(
    estimator=None,
    search_spaces=param_space,
    n_iter=50,
    scoring=arima_mse,
    cv=[(slice(None), slice(None))]  # single train/test split
)

opt.fit(X=None, y=time_series_data)
best_order = tuple(opt.best_params_.values())
          `}
        />

        <ModelSection
          title="Prophet"
          id="prophet"
          math={<BlockMath math="y(t) = g(t) + s(t) + h(t) + \epsilon_t" />}
          description="Prophet is a procedure for forecasting time series data based on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects."
          hyperparameters={[
            { name: 'changepoint_prior_scale', description: 'Flexibility of the trend' },
            { name: 'seasonality_prior_scale', description: 'Strength of the seasonality' },
            { name: 'holidays_prior_scale', description: 'Strength of the holiday effects' },
            { name: 'seasonality_mode', description: 'Additive or multiplicative seasonality' },
          ]}
          checks={{
            unscaled: true,
            missing: true,
            categorical: false,
            regression: true,
            classification: false
          }}
          bayesianOptimization={`
from skopt import BayesSearchCV
from skopt.space import Real, Categorical
from fbprophet import Prophet
import pandas as pd

def prophet_mape(params, X, y):
    model = Prophet(**params)
    model.fit(pd.DataFrame({'ds': X, 'y': y}))
    forecast = model.predict(pd.DataFrame({'ds': X}))
    return -np.mean(np.abs((y - forecast['yhat'].values) / y)) * 100

param_space = {
    'changepoint_prior_scale': Real(0.001, 0.5, prior='log-uniform'),
    'seasonality_prior_scale': Real(0.01, 10, prior='log-uniform'),
    'holidays_prior_scale': Real(0.01, 10, prior='log-uniform'),
    'seasonality_mode': Categorical(['additive', 'multiplicative'])
}

opt = BayesSearchCV(
    estimator=None,
    search_spaces=param_space,
    n_iter=50,
    scoring=prophet_mape,
    cv=[(slice(None), slice(None))]  # single train/test split
)

opt.fit(X=time_series_dates, y=time_series_values)
best_params = opt.best_params_
          `}
        />

        <Stack spacing="md">
          <Title order={2} id="lstm">Long Short-Term Memory (LSTM) Networks</Title>
          <Text>
            Long Short-Term Memory (LSTM) networks, a type of recurrent neural network, can be very effective for time series prediction. LSTMs are capable of learning long-term dependencies, making them particularly useful for complex time series data.
          </Text>
          <Text>
            For a detailed explanation of LSTM networks and their application in time series analysis, please refer to our dedicated <Link to="/module9/course">LSTM course module</Link>.
          </Text>
        </Stack>
      </Stack>
    </Container>
  );
};

const ModelSection = ({ title, id, math, description, hyperparameters, checks, bayesianOptimization }) => (
  <Stack spacing="md">
    <Title order={2} id={id}>{title}</Title>
    {math}
    <Text>{description}</Text>
    <HyperparameterTable hyperparameters={hyperparameters} />
    <ChecksTable checks={checks} />
    <Title order={3}>Bayesian Optimization Cheat Sheet</Title>
    <CodeBlock language="python" code={bayesianOptimization} />
  </Stack>
);

const HyperparameterTable = ({ hyperparameters }) => (
  <Table>
    <thead>
      <tr>
        <th>Parameter</th>
        <th>Description</th>
      </tr>
    </thead>
    <tbody>
      {hyperparameters.map((param, index) => (
        <tr key={index}>
          <td><code>{param.name}</code></td>
          <td>{param.description}</td>
        </tr>
      ))}
    </tbody>
  </Table>
);

const ChecksTable = ({ checks }) => (
  <Table>
    <thead>
      <tr>
        <th>Check</th>
        <th>Status</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Robust to unscaled data</td>
        <td>{checks.unscaled ? <IconCheck color="green" /> : <IconX color="red" />}</td>
      </tr>
      <tr>
        <td>Handles missing values</td>
        <td>{checks.missing ? <IconCheck color="green" /> : <IconX color="red" />}</td>
      </tr>
      {/* <tr>
        <td>Handles categorical data</td>
        <td>{checks.categorical ? <IconCheck color="green" /> : <IconX color="red" />}</td>
      </tr> */}
      <tr>
        <td>Supports regression</td>
        <td>{checks.regression ? <IconCheck color="green" /> : <IconX color="red" />}</td>
      </tr>
      <tr>
        <td>Supports classification</td>
        <td>{checks.classification ? <IconCheck color="green" /> : <IconX color="red" />}</td>
      </tr>
    </tbody>
  </Table>
);

export default TimeSeriesModels;