import React from 'react';
import { Container, Title, Text, Stack, List } from '@mantine/core';
import CodeBlock from 'components/CodeBlock';
import { InlineMath, BlockMath } from 'react-katex';
import 'katex/dist/katex.min.css';

const TimeSeriesModels = () => {
  return (
    <Container fluid>
      <Title order={1} mt="xl" mb="md">Time Series Models</Title>

      <Stack spacing="xl">
        <div data-slide>
          <Title order={2}>ARIMA Models</Title>
          <BlockMath math="y_t = c + \phi_1 y_{t-1} + \cdots + \phi_p y_{t-p} + \theta_1 \epsilon_{t-1} + \cdots + \theta_q \epsilon_{t-q} + \epsilon_t" />

          <Title order={3} mt="lg">Model Components</Title>
          <Text mb="md">
            ARIMA(p,d,q) combines three components: AutoRegressive (AR), Integrated (I), and Moving Average (MA). The model captures temporal dependencies through past values and past errors.
          </Text>

          <List spacing="sm">
            <List.Item>
              <strong>AR(p)</strong>: Uses p past observations. <InlineMath math="\phi_1 y_{t-1} + ... + \phi_p y_{t-p}" />
            </List.Item>
            <List.Item>
              <strong>I(d)</strong>: Differencing d times to achieve stationarity. <InlineMath math="\Delta^d y_t" />
            </List.Item>
            <List.Item>
              <strong>MA(q)</strong>: Uses q past forecast errors. <InlineMath math="\theta_1 \epsilon_{t-1} + ... + \theta_q \epsilon_{t-q}" />
            </List.Item>
          </List>

          <Title order={3} mt="lg">Key Hyperparameters</Title>
          <List spacing="sm">
            <List.Item>
              <strong>p (AR order)</strong>: Number of lag observations. Use PACF to identify. Typical: 0-5
            </List.Item>
            <List.Item>
              <strong>d (differencing)</strong>: Times to difference for stationarity. Usually 0-2. Check with ADF test
            </List.Item>
            <List.Item>
              <strong>q (MA order)</strong>: Size of moving average window. Use ACF to identify. Typical: 0-5
            </List.Item>
          </List>

          <CodeBlock language="python" code={`# ARIMA model fitting
from statsmodels.tsa.arima.model import ARIMA
model = ARIMA(y_train, order=(p=2, d=1, q=2))
fitted = model.fit()`} />
        </div>

        <div data-slide>
          <Title order={2}>ARIMA Training & Prediction</Title>

          <Title order={3} mt="lg">Training Process</Title>
          <Text mb="md">
            ARIMA uses Maximum Likelihood Estimation (MLE) to fit parameters. First, ensure stationarity through differencing, then optimize AR and MA coefficients.
          </Text>

          <CodeBlock language="python" code={`# Check stationarity and determine d
from statsmodels.tsa.stattools import adfuller
adf_test = adfuller(y_train)
d = 0 if adf_test[1] < 0.05 else 1  # Difference if non-stationary`} />

          <Title order={3} mt="lg">Model Selection</Title>
          <Text mb="md">
            Use information criteria (AIC/BIC) or grid search to find optimal (p,d,q). Auto-ARIMA automates this process.
          </Text>

          <CodeBlock language="python" code={`# Automatic parameter selection
from pmdarima import auto_arima
model = auto_arima(y_train, seasonal=False, stepwise=True,
                   suppress_warnings=True, max_p=5, max_q=5)`} />

          <Title order={3} mt="lg">Prediction Process</Title>
          <Text mb="md">
            ARIMA recursively predicts future values using fitted parameters. Confidence intervals widen for longer horizons due to error propagation.
          </Text>

          <CodeBlock language="python" code={`# Multi-step ahead forecasting
forecast = fitted.forecast(steps=30)  # 30 periods ahead
# Or with confidence intervals
forecast, stderr, conf_int = fitted.forecast(30, alpha=0.05)`} />

          <Title order={3} mt="lg">Bayesian Optimization</Title>
          <CodeBlock language="python" code={`from skopt import gp_minimize
def objective(params):
    p, d, q = params
    model = ARIMA(y_train, order=(p, d, q))
    return model.fit().aic

result = gp_minimize(objective, [(0,5), (0,2), (0,5)], n_calls=30)`} />
        </div>

        <div data-slide>
          <Title order={2}>SARIMA: Seasonal ARIMA</Title>
          <BlockMath math="ARIMA(p,d,q) \times (P,D,Q)_s" />

          <Title order={3} mt="lg">Seasonal Components</Title>
          <Text mb="md">
            SARIMA extends ARIMA with seasonal terms. Captures both non-seasonal and seasonal patterns with period s (e.g., s=12 for monthly data with yearly seasonality).
          </Text>

          <List spacing="sm">
            <List.Item>
              <strong>P</strong>: Seasonal AR order (typically 0-2)
            </List.Item>
            <List.Item>
              <strong>D</strong>: Seasonal differencing (usually 0-1)
            </List.Item>
            <List.Item>
              <strong>Q</strong>: Seasonal MA order (typically 0-2)
            </List.Item>
            <List.Item>
              <strong>s</strong>: Seasonal period (4=quarterly, 12=monthly, 52=weekly)
            </List.Item>
          </List>

          <CodeBlock language="python" code={`# SARIMA with monthly seasonality
from statsmodels.tsa.statespace.sarimax import SARIMAX
model = SARIMAX(y_train, order=(1,1,1), seasonal_order=(1,1,1,12))
fitted = model.fit(disp=False)`} />

          <Title order={3} mt="lg">Seasonal Decomposition</Title>
          <Text mb="md">
            Decompose series into trend, seasonal, and residual components to understand patterns before modeling.
          </Text>

          <CodeBlock language="python" code={`from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(y, model='additive', period=12)
# Access: decomposition.trend, .seasonal, .resid`} />
        </div>

        <div data-slide>
          <Title order={2}>Prophet</Title>
          <BlockMath math="y(t) = g(t) + s(t) + h(t) + \epsilon_t" />

          <Title order={3} mt="lg">Model Components</Title>
          <Text mb="md">
            Prophet decomposes time series into trend, seasonality, and holidays. Designed for business time series with strong seasonal patterns and missing data.
          </Text>

          <List spacing="sm">
            <List.Item>
              <strong>g(t)</strong>: Trend - piecewise linear or logistic growth with automatic changepoints
            </List.Item>
            <List.Item>
              <strong>s(t)</strong>: Seasonality - Fourier series for periodic patterns (yearly, weekly, daily)
            </List.Item>
            <List.Item>
              <strong>h(t)</strong>: Holidays - irregular events with customizable windows
            </List.Item>
          </List>

          <Title order={3} mt="lg">Key Hyperparameters</Title>
          <List spacing="sm">
            <List.Item>
              <strong>changepoint_prior_scale</strong>: Flexibility of trend (0.001-0.5). Higher = more flexible
            </List.Item>
            <List.Item>
              <strong>seasonality_prior_scale</strong>: Strength of seasonality (0.01-10). Higher = stronger seasonal patterns
            </List.Item>
            <List.Item>
              <strong>seasonality_mode</strong>: 'additive' (default) or 'multiplicative' for percentage-based seasonality
            </List.Item>
            <List.Item>
              <strong>growth</strong>: 'linear' (default) or 'logistic' for saturating forecasts
            </List.Item>
          </List>

          <CodeBlock language="python" code={`# Prophet model with custom settings
from prophet import Prophet
model = Prophet(changepoint_prior_scale=0.05,
                seasonality_mode='multiplicative')
model.fit(df[['ds', 'y']])  # Requires ds (date) and y columns`} />
        </div>

        <div data-slide>
          <Title order={2}>Prophet Training & Prediction</Title>

          <Title order={3} mt="lg">Training Process</Title>
          <Text mb="md">
            Prophet uses Stan for Bayesian inference via MAP estimation. Automatically detects changepoints and fits seasonal components using Fourier series.
          </Text>

          <CodeBlock language="python" code={`# Prepare data and fit
df = pd.DataFrame({'ds': dates, 'y': values})
model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
model.add_country_holidays(country_name='US')  # Add holidays
model.fit(df)`} />

          <Title order={3} mt="lg">Custom Seasonality</Title>
          <Text mb="md">
            Add custom seasonal patterns beyond default yearly/weekly/daily. Useful for business-specific cycles.
          </Text>

          <CodeBlock language="python" code={`# Add monthly and quarterly seasonality
model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
model.add_seasonality(name='quarterly', period=91.25, fourier_order=7)`} />

          <Title order={3} mt="lg">Prediction with Uncertainty</Title>
          <Text mb="md">
            Prophet provides uncertainty intervals through Bayesian framework. Intervals account for trend and seasonal uncertainty.
          </Text>

          <CodeBlock language="python" code={`# Generate future dates and forecast
future = model.make_future_dataframe(periods=365)
forecast = model.predict(future)
# Access: forecast[['yhat', 'yhat_lower', 'yhat_upper']]`} />

          <Title order={3} mt="lg">Hyperparameter Tuning</Title>
          <CodeBlock language="python" code={`from sklearn.model_selection import ParameterGrid
params = {'changepoint_prior_scale': [0.001, 0.01, 0.1],
          'seasonality_prior_scale': [0.01, 0.1, 1.0]}
# Grid search with cross-validation
for p in ParameterGrid(params):
    m = Prophet(**p).fit(train)
    cv_results = cross_validation(m, horizon='30 days')`} />
        </div>

        <div data-slide>
          <Title order={2}>Exponential Smoothing (ETS)</Title>
          <BlockMath math="\hat{y}_{t+h|t} = \ell_t + hb_t + s_{t+h-m(k+1)}" />

          <Title order={3} mt="lg">Model Components</Title>
          <Text mb="md">
            ETS models use exponentially weighted averages of past observations. Components can be Error/Trend/Seasonal with Additive or Multiplicative forms.
          </Text>

          <List spacing="sm">
            <List.Item>
              <strong>Simple (SES)</strong>: No trend or seasonality. <InlineMath math="\alpha" /> controls smoothing
            </List.Item>
            <List.Item>
              <strong>Holt's</strong>: Adds linear trend. <InlineMath math="\beta" /> controls trend smoothing
            </List.Item>
            <List.Item>
              <strong>Holt-Winters</strong>: Adds seasonality. <InlineMath math="\gamma" /> controls seasonal smoothing
            </List.Item>
          </List>

          <CodeBlock language="python" code={`# Holt-Winters with additive seasonality
from statsmodels.tsa.holtwinters import ExponentialSmoothing
model = ExponentialSmoothing(y_train, seasonal='add', seasonal_periods=12)
fitted = model.fit()`} />

          <Title order={3} mt="lg">Training Process</Title>
          <Text mb="md">
            Optimizes smoothing parameters (<InlineMath math="\alpha, \beta, \gamma" />) via MLE. State-space framework enables automatic selection.
          </Text>

          <CodeBlock language="python" code={`# Automatic ETS model selection
from statsmodels.tsa.statespace.exponential_smoothing import ExponentialSmoothing
model = ExponentialSmoothing(y_train, seasonal=12, initialization_method='estimated')
fitted = model.fit()  # Automatically selects best parameters`} />

          <Title order={3} mt="lg">Prediction</Title>
          <Text mb="md">
            Forecasts use final level, trend, and seasonal states. Simple models have narrower intervals than complex ones.
          </Text>

          <CodeBlock language="python" code={`# Forecast with confidence intervals
forecast = fitted.forecast(steps=24)
# Or get prediction intervals
forecast_df = fitted.get_forecast(steps=24).summary_frame()`} />
        </div>

        <div data-slide>
          <Title order={2}>Time Series Cross-Validation</Title>

          <Title order={3} mt="lg">Walk-Forward Validation</Title>
          <Text mb="md">
            Time series require special CV to respect temporal order. Train on past, test on future, expand training window.
          </Text>

          <CodeBlock language="python" code={`from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5, test_size=30)
for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X[train_idx], X[test_idx]`} />

          <Title order={3} mt="lg">Sliding vs Expanding Window</Title>
          <Text mb="md">
            Sliding: Fixed training size. Expanding: Growing training size. Sliding better for non-stationary data.
          </Text>

          <CodeBlock language="python" code={`# Sliding window (fixed size=100)
for i in range(100, len(data)-30):
    train = data[i-100:i]
    test = data[i:i+30]`} />

          <Title order={3} mt="lg">Prophet Cross-Validation</Title>
          <Text mb="md">
            Prophet includes built-in time series CV with configurable initial training period and horizon.
          </Text>

          <CodeBlock language="python" code={`from prophet.diagnostics import cross_validation
cv_results = cross_validation(model, initial='730 days',
                              period='180 days', horizon='365 days')`} />
        </div>

      </Stack>
    </Container>
  );
};

export default TimeSeriesModels;