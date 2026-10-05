# 🚦 CloudPulse

## Real-Time Urban Intelligence System

CloudPulse is a cloud-based environmental intelligence platform that combines real-time weather and air-quality data, cloud storage, machine learning, and interactive analytics into a single responsive dashboard.

## ✨ What CloudPulse Does

CloudPulse allows users to search for a city or location and view its latest environmental conditions through an interactive dashboard.

The system provides:

• 🌦️ Real-time temperature, humidity, and wind speed  
• 🌫️ PM2.5 and PM10 air-quality data  
• 📊 CloudPulse AQI estimation  
• 🤖 Estimated AQI approximately 1 hour from now  
• 🔎 Live location search with autocomplete  
• ☁️ Automatic IBM Cloudant synchronization  
• 📈 7-day environmental trends  
• 🧠 Advanced Pro Analysis  
• 📱 Responsive desktop, tablet, and mobile interface  
• 📄 CSV, HTML, and PDF reports  
• 🔐 Secure cloud credential handling  

## 🧠 Artificial Intelligence

CloudPulse uses a Random Forest Regression model to estimate the AQI approximately one hour into the future.

The model uses the following environmental features:

• Temperature  
• Humidity  
• Wind speed  
• PM2.5  
• PM10  

Prediction flow:

Current Environmental Conditions
        ↓
Feature Processing
        ↓
Random Forest Regression
        ↓
Estimated AQI ~1 Hour From Now

The model is evaluated using:

• MAE — Mean Absolute Error  
• RMSE — Root Mean Squared Error  
• R² Score  
• Directional Accuracy  
• Historical Error Analysis  

The forecast is an estimate and should not be treated as a guaranteed prediction.

## ☁️ Cloud Architecture

CloudPulse follows a cloud-based data pipeline:

User
 ↓
CloudPulse Dashboard
 ↓
Location Search
 ↓
Open-Meteo APIs
 ↓
Python Data Processing
 ↓
 ┌─────────────────────┐
 ↓                     ↓
IBM Cloudant      Random Forest
 ↓                     ↓
 └──────────┬──────────┘
            ↓
    CloudPulse Dashboard

IBM Cloudant is used to store environmental readings and prediction-related data in the cloud.

Environmental readings are synchronized automatically rather than requiring the user to manually save them.

## 🛠️ Technology Stack

Python
• Core application and data processing

Streamlit
• Interactive web dashboard

Open-Meteo
• Weather and air-quality data

IBM Cloudant
• Cloud database and data storage

Scikit-learn
• Machine learning

Random Forest Regression
• Next-hour AQI estimation

Pandas
• Data processing and analysis

Joblib
• Machine-learning model serialization

HTML / CSS / JavaScript
• Responsive interface and location search

GitHub
• Source control and project collaboration

## 🌫️ AQI Estimation

CloudPulse calculates its displayed AQI estimate using PM2.5 concentration and the project's PM2.5 breakpoint scale.

AQI Categories:

0–50       → Good
51–100     → Satisfactory
101–200    → Moderate
201–300    → Poor
301–400    → Very Poor
401+       → Severe

IMPORTANT:
The AQI displayed by CloudPulse is a project-level estimate based on PM2.5. It should not be represented as an official CPCB monitoring-station reading.

## 📊 Pro Analysis

CloudPulse includes a dedicated Pro Analysis section for deeper environmental and machine-learning analysis.

It provides:

• 7-day AQI trends  
• AQI distribution  
• Hourly AQI patterns  
• PM2.5 vs PM10 analysis  
• Temperature and humidity analysis  
• Environmental anomaly analysis  
• Actual vs model comparison  
• Mean, median, minimum, maximum, and standard deviation  
• MAE, RMSE, and R²  
• Directional accuracy  
• Historical error analysis  
• Feature correlations  
• Missing-value and data-quality analysis  
• Model diagnostics  
• CSV reports  
• HTML reports  
• PDF reports  

## 🔎 Location Search

CloudPulse includes a custom interactive location-search system.

User types a location
        ↓
Live autocomplete suggestions
        ↓
User selects a place
        ↓
Latitude + Longitude are obtained
        ↓
CloudPulse loads environmental data
        ↓
Dashboard updates automatically

The search interface supports live suggestions, place hierarchy, keyboard navigation, scrolling, and responsive mobile interaction.

## 📱 Responsive Design

CloudPulse uses a single responsive web interface that works across:

Desktop
   ↓
Tablet
   ↓
Mobile

The interface automatically adapts the navigation, search bar, metric cards, charts, analysis panels, spacing, and typography to different screen sizes.

No separate mobile application is required.

## 🔐 Security

CloudPulse keeps sensitive credentials outside the source code.

Local Streamlit credentials are stored in:

.streamlit/secrets.toml

This file must NEVER be uploaded to GitHub.

The application uses:

CLOUDANT_URL
CLOUDANT_APIKEY

These values should be configured as secure deployment secrets or environment variables when the application is deployed.

## 🚀 Run CloudPulse Locally

Clone the repository:

git clone https://github.com/saintShoaib/CloudPulse.git

Move into the project:

cd CloudPulse

Create a virtual environment:

python -m venv .venv

Activate it on Windows:

.venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

Configure your Streamlit secrets:

.streamlit/secrets.toml

Add:

CLOUDANT_URL = "your-cloudant-url"
CLOUDANT_APIKEY = "your-cloudant-api-key"

Then start CloudPulse:

streamlit run app.py

Windows users can also use:

run_cloudpulse.bat

## 📁 Project Structure

CloudPulse/
│
├── app.py
├── requirements.txt
├── README.md
├── Dockerfile
├── .gitignore
├── run_cloudpulse
├── run_cloudpulse.bat
├── secrets.example
│
└── cloudpulse_search_component/
    └── index.html

Local-only files such as .venv, __pycache__, and .streamlit/secrets.toml should not be committed to GitHub.

## ⚠️ Data Disclaimer

CloudPulse uses modelled environmental data provided by Open-Meteo. It does not directly measure pollution using physical sensors.

The AQI shown by CloudPulse is calculated from PM2.5 using the project's AQI breakpoint logic and is not an official CPCB station reading.

The AI forecast is an estimate and is not guaranteed to represent future air quality.

CloudPulse is an educational and prototype system and should not replace official environmental monitoring or professional health guidance.

## 🎓 Project Objective

CloudPulse demonstrates how cloud computing, real-time APIs, data processing, machine learning, cloud databases, and interactive visualization can be combined to create a real-time urban environmental intelligence system.

The project focuses on keeping the architecture understandable while combining:

Cloud Computing
      +
Real-Time APIs
      +
Data Processing
      +
Machine Learning
      +
Cloud Database
      +
Interactive Visualization
      ↓
Real-Time Urban Intelligence System

## 🚦 CloudPulse

Real-Time Urban Intelligence System

Built with Python • Streamlit • IBM Cloudant • Open-Meteo • Scikit-learn
