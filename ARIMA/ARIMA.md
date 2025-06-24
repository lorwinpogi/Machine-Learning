# ARIMA Model Components

ARIMA modeling consists of three main components: **autoregressive (AR)**, **moving average (MA)**, and **integrated (I)**.

- The **integration** process, often achieved through *differencing*, ensures data **stationarity** by addressing trends and seasonal effects.
- Once the data is stationary:
  - **Autoregressive (AR)** models utilize previous values.
  - **Moving Average (MA)** models account for past errors.

Together, these components help create accurate **time series forecasts**.


## Practical Modeling Steps

Residual effects from previous sales predictions can impact current data, which is why it's important to **validate models against real outcomes**.

When analyzing sales data in the **mountain region**, both **high-end** and **low-end** products will be combined for total sales, with **2017 data set aside for validation**.

Utilizing the **Auto ARIMA** function helps determine the optimal ARIMA model by identifying necessary components, such as the **inclusion of previous values without the need for differencing**.

## ARIMA Model Components and Stationarity

ARIMA stands for **AutoRegressive (AR)**, **Integrated (I)**, and **Moving Average (MA)** components, each addressing different aspects of time series data dependency.

- **Integrated (I)** refers to **differencing the data** to achieve **stationarity**, meaning the effects of past data diminish over time, making the series stable and predictable around a constant mean.
- **Stationarity** is crucial because non-stationary data (e.g., trending or seasonal data) cannot be reliably predicted by simple historical averages.
- **Differencing** (subtracting consecutive observations) helps stabilize the mean by removing trends or seasonal patterns (e.g., monthly sales with yearly seasonality).



## Differencing and Stationarity

**Differencing** transforms the series by computing changes between time periods, effectively removing trends or seasonal effects to create stationarity.

- A **single time period difference** can remove a linear trend.
- **Seasonal differencing** (e.g., 12 months for annual seasonality) handles repeating seasonal patterns.



## Autoregressive (AR) Component

The **AR model** uses lags or previous values of the series to predict current values, assuming past observations have a residual influence on the present.

- These are called **long memory models** because the influence of past values diminishes slowly over time rather than disappearing immediately.


## Moving Average (MA) Component

The **MA model** captures the influence of previous **errors or shocks** (differences between observed and predicted values) on current observations.

- These are known as **short memory models** because the impact of past errors fades quickly and does not persist over many periods.



## Model Preparation and Validation

Before modeling, it is essential to **split data into training and validation sets** to evaluate model performance on unseen data and ensure reliable predictions.

- For example, **sales data for the mountain region** can be split, reserving the latest year (e.g., **2017**) for validation while training the model on earlier data.



## Auto ARIMA Function in R

R provides an **Auto ARIMA** function that automatically estimates the best ARIMA model by selecting appropriate orders for AR, differencing (I), and MA components based on the data.

- The function outputs the model specification, such as:
  - The number of **AR lags** to include
  - Whether **differencing** is needed
- Sometimes, **no differencing** is required if the data is already stationary.

## Autoregressive (AR) Component in ARIMA

The **Autoregressive (AR)** component of the ARIMA model refers to a model that uses the **dependent relationship between an observation and a number of lagged observations** (previous values).

### Key Concept

In an AR model, the current value of the time series is expressed as a **linear combination of its past values** and a random error term.

The general form of an AR model of order `p` (denoted as AR(p)) is:

Yₜ = c + φ₁Yₜ₋₁ + φ₂Yₜ₋₂ + ... + φₚYₜ₋ₚ + εₜ


Where:
- `Yₜ` is the value of the series at time `t`
- `c` is a constant
- `φ₁, φ₂, ..., φₚ` are the autoregressive coefficients
- `εₜ` is white noise (random error term with a mean of zero)

### How It Works

- The AR model assumes that **past values have a linear and diminishing influence** on current values.
- For example, in an AR(1) model:


Yₜ = c + φ₁Yₜ₋₁ + εₜ

The current value depends only on its immediately previous value.

### Characteristics

- AR models are best suited for **stationary time series**, where statistical properties like mean and variance do not change over time.
- The coefficients `φ` determine the **strength and direction** of influence from past values.
- If φ is positive, a high value in the past increases the likelihood of a high value in the present. If φ is negative, a high past value reduces the likelihood of a high present value.

### Use Cases

- AR models are widely used in **economics, finance, and forecasting**, such as:
- Stock price modeling
- Temperature forecasting
- Demand prediction

### Model Identification

- The **Partial Autocorrelation Function (PACF)** is typically used to determine the order `p` of an AR model. A sharp drop-off in PACF after lag `p` suggests an AR(p) model.


### Summary

The **Autoregressive component (AR)** captures the memory of the system by modeling how **past observations directly influence** the current value. It’s a key building block of the ARIMA model, making it effective for time series forecasting where past values are strong predictors of future behavior.


## Integrated (I) Component in ARIMA

The **Integrated (I)** component of the ARIMA model refers to the process of **differencing the time series data** to make it **stationary**—a key requirement for accurate time series forecasting.

### What Is Stationarity?

A **stationary time series** has a constant mean, constant variance, and autocovariance that does not depend on time. In contrast, non-stationary series may show trends, seasonality, or changing variance over time.

### Why Integration Is Needed

Most real-world time series data—such as sales, temperature, or stock prices—are **non-stationary**, making them unsuitable for direct modeling using AR or MA components. To overcome this, we use **differencing**, the technique at the core of the Integrated (I) part of ARIMA.



### Differencing Explained

**Differencing** transforms the data by subtracting the current observation from the previous one:

Y'ₜ = Yₜ - Yₜ₋₁


- The result, `Y'ₜ`, is the **first difference** of the series.
- If the first differencing doesn't achieve stationarity, second differencing may be used:

Y''ₜ = (Yₜ - Yₜ₋₁) - (Yₜ₋₁ - Yₜ₋₂)


- The number of differences required to make the series stationary is denoted as `d` in ARIMA(p, d, q).

### Example

Consider monthly sales data with a clear upward trend. Direct modeling would be ineffective because of the trend. Differencing once (`d = 1`) would remove the trend, stabilizing the mean and making the data suitable for AR and MA modeling.


### Seasonal Differencing

If the time series exhibits **seasonal patterns**, seasonal differencing may be applied:

Yₜ - Yₜ₋ₛ


Where `s` is the length of the season (e.g., 12 for monthly data with yearly seasonality).


### How to Identify the Need for Differencing

- **Visual inspection** of the time series plot
- **Augmented Dickey-Fuller (ADF) test** or **KPSS test** for stationarity
- **Autocorrelation Function (ACF)** plot showing slow decay indicates non-stationarity



### Summary

The **Integrated (I)** component of ARIMA ensures that non-stationary time series become **stationary** by applying differencing. This transformation allows the AR and MA components to effectively model the underlying patterns in the data.

In ARIMA(p, d, q):
- `d` represents the number of differences needed to make the series stationary.


## Moving Average (MA) Component in ARIMA

The **Moving Average (MA)** component of the ARIMA model captures the **relationship between an observation and a residual error** from a **lagged** time step. Unlike the moving average used for smoothing, the MA in ARIMA refers to modeling past **forecast errors** rather than data points.


### Key Concept

In an MA model, the current value of the time series is expressed as a **linear function of past forecast errors**.

The general form of an MA model of order `q` (denoted as MA(q)) is:



Yₜ = μ + εₜ + θ₁εₜ₋₁ + θ₂εₜ₋₂ + ... + θ_qεₜ₋_q


Where:
- `Yₜ` is the value of the time series at time `t`
- `μ` is the mean of the series
- `εₜ` is the white noise (random error) at time `t`
- `θ₁, θ₂, ..., θ_q` are the **moving average coefficients**
- `q` is the number of lagged forecast errors included

---

### How It Works

- Instead of using past **values** (as in AR), MA models use past **errors**—the differences between actual and predicted values—to improve the forecast.
- The idea is that errors from previous time steps might carry useful information about the structure or shocks in the system.



### Characteristics

- MA models are **short-memory models**: the impact of past errors dies out quickly over time.
- They are effective when the noise or shocks to the system have a temporary effect.
- The **Autocorrelation Function (ACF)** is used to identify the appropriate order `q`. A sharp cutoff in ACF after lag `q` suggests an MA(q) model.


### Example

If an unexpected dip in sales occurred due to a temporary event (like a supply chain delay), an MA model would use that residual error to correct the forecast for the next period, assuming that the dip was not part of a long-term trend.



### When to Use MA

- When residuals from an AR or differenced model show autocorrelation.
- When forecasting accuracy improves by incorporating information from past forecast **errors** rather than values.


### Summary

The **Moving Average (MA)** component in ARIMA models the **error structure** of the time series. It captures the lingering effects of previous forecasting errors, allowing for **short-term corrections** in predictions.

In ARIMA(p, d, q):
- `q` is the number of lagged error terms used to correct the forecast.


