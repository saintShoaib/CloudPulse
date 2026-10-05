# 🚦 CloudPulse

Real-Time Urban Intelligence System
Weather • Air Quality • Cloud Database • AI Forecasting • Pro Analytics

CloudPulse is a cloud-based environmental intelligence dashboard that combines
real-time weather and air-quality data with IBM Cloudant and machine learning
to provide current environmental conditions, AQI estimates, and an estimated
next-hour AQI.

---

## ✨ Features

### 🌦️ Real-Time Environmental Data
- Temperature
- Humidity
- Wind speed
- PM2.5
- PM10
- CloudPulse AQI estimate
- AQI health category

### 🔎 Location Search
- Search for cities and places
- Live autocomplete suggestions
- Select a location directly from the search results
- Automatically updates environmental data for the selected location

### 🤖 AI Forecasting
CloudPulse uses a Random Forest regression model to estimate AQI approximately
one hour into the future.

The model uses:
- Temperature
- Humidity
- Wind speed
- PM2.5
- PM10

Historical data is aligned chronologically and used to train and evaluate
the forecasting model.

### ☁️ IBM Cloudant
Environmental readings are automatically stored in IBM Cloudant.

CloudPulse also displays the current synchronization status so users can see
whether environmental data is being stored successfully.

### 📊 Pro Analysis

The advanced analysis section provides:

- 7-day AQI trend
- AQI distribution
- Hourly AQI patterns
- PM2.5 and PM10 analysis
- Temperature and humidity context
- Anomaly analysis
- Actual vs model comparison
- MAE
- RMSE
- R²
- Directional accuracy
- Historical error band
- Statistical summary
- Feature correlations
- Data-quality information
- Model diagnostics
- CSV / HTML / PDF reports

### 📱 Responsive Design

CloudPulse uses one responsive interface for desktop, tablet and mobile.

The dashboard automatically adapts:
- navigation
- search
- metric cards
- charts
- analysis panels
- tables
- spacing and typography

No separate mobile application is required.

---

## 🧠 How CloudPulse Works

```text
              ┌─────────────────┐
              │   User Search   │
              └────────┬────────┘
                       ↓
              ┌─────────────────┐
              │ Location / Geo  │
              └────────┬────────┘
                       ↓
              ┌─────────────────┐
              │   Open-Meteo    │
              │ Weather + AQ    │
              └────────┬────────┘
                       ↓
              ┌─────────────────┐
              │ Python Data     │
              │ Processing      │
              └──────┬─────┬────┘
                     │     │
                     ↓     ↓
              ┌─────────┐ ┌──────────────┐
              │Cloudant │ │ Random Forest│
              │ Storage │ │ Forecasting  │
              └────┬────┘ └──────┬───────┘
                   │              │
                   └──────┬───────┘
                          ↓
                 ┌──────────────────┐
                 │ Streamlit        │
                 │ CloudPulse UI    │
                 └──────────────────┘
