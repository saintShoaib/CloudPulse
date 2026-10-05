# CloudPulse — Pro Analysis Edition (ML Training Fixed)

CloudPulse is a real-time urban/environmental intelligence prototype with a search-first Streamlit interface, Open-Meteo environmental data, IBM Cloudant storage, and local machine-learning analysis.

## What was fixed in this build

The **Estimated AQI ~1 hour from now** model no longer silently fails when the weather and air-quality APIs return slightly different hourly timestamps.

### ML pipeline fixes
- Weather and air-quality histories are aligned with a nearest-hour merge (31-minute tolerance) instead of an exact timestamp merge.
- Historical values are converted to numeric values and cleaned before training.
- Local training now works with **24+ usable hourly observations** instead of unnecessarily requiring 36.
- The model uses a chronological 80/20 backtest and then refits on all usable history for the current estimate.
- RandomForest training is cached so the model is not retrained on every 60-second live refresh.
- Training failures are no longer hidden behind a generic “Unavailable” message.
- System Status now shows raw/usable history rows, train/test rows, model state, and the actual diagnostic reason when something fails.
- A packaged `cloudpulse_model.pkl` is used only as a fallback when a local location-specific model cannot be trained.

## Pro Analysis
- 7-day AQI trend
- AQI distribution
- typical hourly AQI pattern
- PM2.5 / PM10 trend
- temperature / humidity context
- statistical AQI anomalies
- model-vs-actual local backtest
- MAE, RMSE, R² and directional accuracy
- empirical 80% historical error band
- statistical profile and feature correlations
- CSV, HTML and executive PDF downloads

## Forecast wording
The UI says **Estimated AQI ~1 hour from now**. This means the ML model is estimating the AQI approximately one hour after the current observation. It is a model estimate, not a direct measurement or guaranteed future value.

## Data note
Open-Meteo weather and air-quality values are modelled environmental data. CloudPulse calculates an AQI estimate from PM2.5; it is **not an official CPCB station reading**.

## Secrets
Keep Cloudant credentials in Streamlit secrets or environment variables. Never upload `.streamlit/secrets.toml` or API keys to GitHub.

For local Streamlit use, create:

```toml
CLOUDANT_URL = "your-cloudant-url"
CLOUDANT_APIKEY = "your-new-api-key"
```

The project `.gitignore` excludes `.streamlit/secrets.toml`.


## V8 diagnostics fix
The Pro Analysis data-quality panel now reports the actual current prediction source (local Random Forest, packaged model, or forecast fallback) instead of checking only for the packaged model file.


## Sidebar update
- Removed the manual Save button.
- CloudPulse now automatically stores each newly fetched environmental reading in IBM Cloudant.
- Duplicate writes are prevented for the same location/timestamp during a Streamlit session.
- Sidebar shows Cloud Sync status and keeps Refresh now as the manual control.
## 📱 Mobile-ready layout

CloudPulse uses the same responsive interface on desktop and phones. On narrow screens:
- the sidebar follows Streamlit's automatic mobile behavior
- the search bar and place suggestions resize to the phone width
- metric cards use a compact two-column layout, then switch to one column on very narrow phones
- Pro Analysis columns stack/wrap instead of overflowing horizontally
- charts, tables, cards, and location details are constrained to the viewport

No separate mobile app is required.
