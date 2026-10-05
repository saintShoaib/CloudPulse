import os
import time
import platform
import concurrent.futures
import html
import json
import urllib.parse
from datetime import datetime, timezone

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

import joblib
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit.components.v1 as components

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
except Exception:
    colors = A4 = getSampleStyleSheet = SimpleDocTemplate = Paragraph = Spacer = Table = TableStyle = None

try:
    import psutil
except Exception:
    psutil = None

from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibmcloudant.cloudant_v1 import CloudantV1

APP_TITLE = "CloudPulse"
DB_NAME = "cloudpulse_data"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
AIR_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
GEOCODER_URL = "https://nominatim.openstreetmap.org/search"
PHOTON_URL = "https://photon.komoot.io/api/"
WIKI_API = "https://commons.wikimedia.org/w/api.php"
MODEL_FILE = "cloudpulse_model.pkl"
START_TIME = time.time()

AQI_BREAKPOINTS = [(0, 30, 0, 50), (30, 60, 51, 100), (60, 90, 101, 200),
                   (90, 120, 201, 300), (120, 250, 301, 400), (250, 500, 401, 500)]

# Major world locations: visible at global zoom.
GLOBAL_POINTS = [
    ("London", 51.5074, -0.1278, "city"), ("New York", 40.7128, -74.0060, "city"),
    ("Los Angeles", 34.0522, -118.2437, "city"), ("Mexico City", 19.4326, -99.1332, "city"),
    ("São Paulo", -23.5505, -46.6333, "city"), ("Buenos Aires", -34.6037, -58.3816, "city"),
    ("Lagos", 6.5244, 3.3792, "city"), ("Cairo", 30.0444, 31.2357, "city"),
    ("Cape Town", -33.9249, 18.4241, "city"), ("Paris", 48.8566, 2.3522, "city"),
    ("Dubai", 25.2048, 55.2708, "city"), ("Mumbai", 19.0760, 72.8777, "city"),
    ("Delhi", 28.6139, 77.2090, "city"), ("Lucknow", 26.8467, 80.9462, "city"),
    ("Tokyo", 35.6762, 139.6503, "city"), ("Seoul", 37.5665, 126.9780, "city"),
    ("Singapore", 1.3521, 103.8198, "city"), ("Sydney", -33.8688, 151.2093, "city"),
    ("Jakarta", -6.2088, 106.8456, "city"), ("Beijing", 39.9042, 116.4074, "city"),
    ("Moscow", 55.7558, 37.6173, "city"), ("Istanbul", 41.0082, 28.9784, "city"),
]

# India state-capital tier. These appear as the camera moves into India.
INDIA_STATES = [
    ("Andhra Pradesh", "Amaravati", 16.5131, 80.5165), ("Arunachal Pradesh", "Itanagar", 27.0844, 93.6053),
    ("Assam", "Dispur", 26.1433, 91.7898), ("Bihar", "Patna", 25.5941, 85.1376),
    ("Chhattisgarh", "Raipur", 21.2514, 81.6296), ("Goa", "Panaji", 15.4909, 73.8278),
    ("Gujarat", "Gandhinagar", 23.2156, 72.6369), ("Haryana", "Chandigarh", 30.7333, 76.7794),
    ("Himachal Pradesh", "Shimla", 31.1048, 77.1734), ("Jharkhand", "Ranchi", 23.3441, 85.3096),
    ("Karnataka", "Bengaluru", 12.9716, 77.5946), ("Kerala", "Thiruvananthapuram", 8.5241, 76.9366),
    ("Madhya Pradesh", "Bhopal", 23.2599, 77.4126), ("Maharashtra", "Mumbai", 19.0760, 72.8777),
    ("Manipur", "Imphal", 24.8170, 93.9368), ("Meghalaya", "Shillong", 25.5788, 91.8933),
    ("Mizoram", "Aizawl", 23.7271, 92.7176), ("Nagaland", "Kohima", 25.6751, 94.1086),
    ("Odisha", "Bhubaneswar", 20.2961, 85.8245), ("Punjab", "Chandigarh", 30.7333, 76.7794),
    ("Rajasthan", "Jaipur", 26.9124, 75.7873), ("Sikkim", "Gangtok", 27.3389, 88.6065),
    ("Tamil Nadu", "Chennai", 13.0827, 80.2707), ("Telangana", "Hyderabad", 17.3850, 78.4867),
    ("Tripura", "Agartala", 23.8315, 91.2868), ("Uttar Pradesh", "Lucknow", 26.8467, 80.9462),
    ("Uttarakhand", "Dehradun", 30.3165, 78.0322), ("West Bengal", "Kolkata", 22.5726, 88.3639),
]

# Dense local tier for Uttar Pradesh. This is intentionally lightweight but gives the
# requested "more pins as I zoom" effect without calling an API for every point.
UP_DISTRICTS = [
    ("Agra", 27.1767, 78.0081), ("Aligarh", 27.8974, 78.0880), ("Ambedkar Nagar", 26.4055, 82.5402),
    ("Amethi", 26.1542, 81.8140), ("Amroha", 28.9031, 78.4698), ("Auraiya", 26.4620, 79.5096),
    ("Ayodhya", 26.7997, 82.2043), ("Azamgarh", 26.0736, 83.1859), ("Baghpat", 28.9447, 77.2184),
    ("Bahraich", 27.5743, 81.5941), ("Ballia", 25.7600, 84.1496), ("Balrampur", 27.4295, 82.1800),
    ("Banda", 25.4753, 80.3361), ("Barabanki", 26.9380, 81.1957), ("Bareilly", 28.3670, 79.4304),
    ("Basti", 26.7880, 82.7160), ("Bhadohi", 25.3950, 82.5700), ("Bijnor", 29.3724, 78.1358),
    ("Budaun", 28.0337, 79.1205), ("Bulandshahr", 28.4060, 77.8498), ("Chandauli", 25.2581, 83.2680),
    ("Chitrakoot", 25.1980, 80.9040), ("Deoria", 26.5024, 83.7791), ("Etah", 27.5588, 78.6626),
    ("Etawah", 26.7855, 79.0151), ("Farrukhabad", 27.3910, 79.5800), ("Fatehpur", 25.9300, 80.8100),
    ("Firozabad", 27.1591, 78.3957), ("Gautam Buddha Nagar", 28.4744, 77.5040), ("Ghaziabad", 28.6692, 77.4538),
    ("Ghazipur", 25.5800, 83.5800), ("Gonda", 27.1330, 81.9530), ("Gorakhpur", 26.7606, 83.3732),
    ("Hamirpur", 25.9550, 80.1480), ("Hapur", 28.7300, 77.7800), ("Hardoi", 27.3980, 80.1250),
    ("Hathras", 27.5950, 78.0500), ("Jalaun", 26.1450, 79.3300), ("Jaunpur", 25.7460, 82.6830),
    ("Jhansi", 25.4484, 78.5685), ("Kannauj", 27.0550, 79.9180), ("Kanpur Dehat", 26.4200, 79.8500),
    ("Kanpur Nagar", 26.4499, 80.3319), ("Kasganj", 27.8080, 78.6460), ("Kaushambi", 25.5300, 81.3800),
    ("Kheri", 27.9000, 80.7800), ("Kushinagar", 26.7400, 83.8900), ("Lalitpur", 24.6900, 78.4100),
    ("Lucknow", 26.8467, 80.9462), ("Maharajganj", 27.1300, 83.5600), ("Mahoba", 25.2900, 79.8700),
    ("Mainpuri", 27.2300, 79.0000), ("Mathura", 27.4924, 77.6737), ("Mau", 25.9500, 83.5600),
    ("Meerut", 28.9845, 77.7064), ("Mirzapur", 25.1460, 82.5700), ("Moradabad", 28.8386, 78.7733),
    ("Muzaffarnagar", 29.4727, 77.7085), ("Pilibhit", 28.6200, 79.8000), ("Pratapgarh", 25.9000, 81.9500),
    ("Prayagraj", 25.4358, 81.8463), ("Rae Bareli", 26.2345, 81.2409), ("Rampur", 28.8000, 79.0000),
    ("Saharanpur", 29.9680, 77.5552), ("Sambhal", 28.5900, 78.5700), ("Sant Kabir Nagar", 26.7700, 83.0300),
    ("Shahjahanpur", 27.8800, 79.9100), ("Shamli", 29.4500, 77.3100), ("Shravasti", 27.5200, 82.0500),
    ("Siddharthnagar", 27.2600, 83.0000), ("Sitapur", 27.5700, 80.6800), ("Sonbhadra", 24.6900, 83.0600),
    ("Sultanpur", 26.2700, 82.0700), ("Unnao", 26.5470, 80.4878), ("Varanasi", 25.3176, 82.9739),
]

st.set_page_config(page_title=APP_TITLE, page_icon="🌍", layout="wide", initial_sidebar_state="auto")

st.markdown("""
<style>
:root{--bg:#040914;--card:rgba(9,18,31,.78);--line:rgba(151,173,198,.13);--text:#edf5ff;--muted:#8195ab;--accent:#39e2d0}
.stApp{background:radial-gradient(900px 600px at 50% -20%,rgba(48,102,145,.14),transparent 60%),#040914;color:var(--text)}
.block-container{max-width:1500px;padding:1rem 2rem 3rem}
[data-testid="stHeader"]{background:transparent}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#07111e 0%,#050b14 55%,#03070d 100%);border-right:1px solid rgba(151,173,198,.10)}
[data-testid="stSidebar"] > div:first-child{padding:1.05rem .9rem 1.2rem}
[data-testid="stSidebar"] .stButton > button{border:1px solid rgba(151,173,198,.12);background:rgba(255,255,255,.035);color:#dce9f5;min-height:2.35rem;font-size:.78rem;font-weight:700;transition:all .18s ease}
[data-testid="stSidebar"] .stButton > button:hover{border-color:rgba(57,226,208,.35);background:rgba(57,226,208,.07);color:#fff;transform:translateY(-1px)}
[data-testid="stSidebar"] [data-testid="stExpander"]{border:1px solid rgba(151,173,198,.11);border-radius:15px;background:rgba(255,255,255,.018);overflow:hidden;margin-bottom:.55rem}
[data-testid="stSidebar"] [data-testid="stExpander"] summary{padding:.72rem .8rem}
.cp-side-brand{display:flex;align-items:center;justify-content:space-between;padding:4px 3px 10px}
.cp-side-logo{font-size:1.18rem;font-weight:900;letter-spacing:-.7px;background:linear-gradient(90deg,#fff,#75e7dc,#7cbaff);-webkit-background-clip:text;background-clip:text;color:transparent}
.cp-side-badge{font-size:.58rem;letter-spacing:1px;text-transform:uppercase;color:#7de6db;border:1px solid rgba(57,226,208,.18);background:rgba(57,226,208,.055);padding:4px 7px;border-radius:999px}
.cp-side-focus{padding:13px 13px 12px;border:1px solid rgba(151,173,198,.12);border-radius:16px;background:linear-gradient(135deg,rgba(16,31,50,.82),rgba(7,15,27,.72));box-shadow:0 12px 30px rgba(0,0,0,.14);margin-bottom:10px}
.cp-side-kicker{font-size:.58rem;text-transform:uppercase;letter-spacing:1.5px;color:#688098;margin-bottom:5px}
.cp-side-place{font-size:1rem;font-weight:850;color:#eef6ff;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.cp-side-address{font-size:.67rem;color:#7890a8;margin-top:4px;line-height:1.35}
.cp-side-live{display:flex;align-items:center;gap:7px;margin-top:10px;font-size:.66rem;color:#9bdcd4}
.cp-side-dot{width:7px;height:7px;border-radius:50%;background:#39e2d0;box-shadow:0 0 10px rgba(57,226,208,.8)}
.cp-side-section{font-size:.62rem;text-transform:uppercase;letter-spacing:1.4px;color:#61778e;margin:12px 2px 7px}
.cp-side-sync{display:flex;align-items:center;justify-content:space-between;gap:10px;padding:11px 12px;border:1px solid rgba(57,226,208,.11);border-radius:14px;background:linear-gradient(135deg,rgba(57,226,208,.055),rgba(255,255,255,.018));margin-top:8px}
.cp-side-sync>div{display:flex;align-items:center;gap:9px;min-width:0}
.cp-side-sync-icon{display:grid;place-items:center;width:28px;height:28px;border-radius:9px;background:rgba(57,226,208,.08);color:#7de6db;font-size:.9rem;flex:0 0 auto}
.cp-side-sync b{display:block;color:#dcebf7;font-size:.68rem;line-height:1.2}
.cp-side-sync span:not(.cp-side-sync-icon){display:block;color:#6f879d;font-size:.57rem;line-height:1.35;margin-top:2px}
.cp-side-sync strong{font-size:.57rem;color:#8fded5;white-space:nowrap}
.cp-side-info{font-size:.68rem;color:#7289a0;line-height:1.5;margin:3px 2px 8px}
.cp-side-mini-grid{display:grid;grid-template-columns:1fr 1fr;gap:7px;margin:8px 0}
.cp-side-mini{border:1px solid rgba(151,173,198,.09);border-radius:11px;background:rgba(255,255,255,.02);padding:8px 9px;font-size:.62rem;color:#738aa0}
.cp-side-mini b{display:block;color:#cfe0ef;font-size:.7rem;margin-top:2px}
.cp-side-footer{margin-top:14px;padding:10px 3px 0;border-top:1px solid rgba(151,173,198,.08);font-size:.59rem;color:#52687d;line-height:1.45}
.cp-brand{font:900 clamp(2.5rem,5vw,4.2rem)/.95 'Space Grotesk',sans-serif;letter-spacing:-3px;background:linear-gradient(90deg,#fff,#75e7dc,#7cbaff);-webkit-background-clip:text;background-clip:text;color:transparent}
.cp-tag{color:#8297ad;font-size:.88rem;margin-top:9px;max-width:820px}
.cp-pill{display:inline-block;padding:7px 11px;border-radius:999px;border:1px solid rgba(57,226,208,.17);background:rgba(57,226,208,.05);color:#8eece2;font-size:.72rem}
.cp-card{border:1px solid var(--line);border-radius:18px;background:var(--card);padding:15px 18px;backdrop-filter:blur(10px)}
.cp-kicker{font-size:.65rem;text-transform:uppercase;letter-spacing:1.5px;color:#6f8499}.cp-big{font-size:1.5rem;font-weight:850;margin-top:4px}.cp-muted{color:var(--muted);font-size:.74rem}
.cp-chip{display:inline-block;margin:10px 6px 0 0;padding:5px 8px;border:1px solid rgba(151,173,198,.12);border-radius:999px;color:#cddbea;font-size:.72rem;background:rgba(255,255,255,.025)}
.cp-detail{border:1px solid var(--line);border-radius:24px;background:rgba(8,17,30,.78);padding:22px;margin-top:18px}
.cp-location-card{border:1px solid rgba(151,173,198,.14);border-radius:20px;background:linear-gradient(135deg,rgba(11,23,39,.94),rgba(7,15,27,.86));padding:20px 22px;margin:2px 0 12px;box-shadow:0 12px 35px rgba(0,0,0,.14)}
.cp-location-name{font-size:1.55rem;font-weight:850;letter-spacing:-.5px;margin-top:3px}.cp-location-meta{margin-top:10px}.cp-updated{margin-top:10px;color:#637a91;font-size:.68rem}
.cp-insight{border-left:3px solid var(--accent);background:rgba(57,226,208,.045);padding:12px 14px;border-radius:0 12px 12px 0;color:#c4d3e2}
.cp-status-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px}.cp-status{padding:9px 10px;border:1px solid rgba(151,173,198,.1);border-radius:10px;background:rgba(255,255,255,.02);font-size:.73rem}
.cp-status b{float:right;color:#cfe3f7}.cp-note{font-size:.68rem;color:#61778e}
button{border-radius:12px!important}

/* ---------- Mobile / phone layout ---------- */
@media (max-width: 768px){
  /* Keep Streamlit's mobile toolbar out of the custom CloudPulse header. */
  [data-testid="stDeployButton"]{display:none!important}
  [data-testid="stToolbar"]{right:.45rem!important;top:.25rem!important}
  .block-container{max-width:100%;padding:4.15rem .72rem 2.2rem!important}
  [data-testid="stSidebar"] > div:first-child{padding:.8rem .72rem 1rem!important}
  .cp-brand{font-size:clamp(2.15rem,12vw,3.25rem);letter-spacing:-2px}
  .cp-tag{font-size:.78rem;line-height:1.45}
  .cp-pill{font-size:.64rem;padding:6px 9px}
  .cp-location-card{padding:15px 14px;border-radius:17px}
  .cp-location-name{font-size:1.28rem;line-height:1.15}
  .cp-location-meta{margin-top:6px}
  .cp-chip{font-size:.67rem;padding:5px 7px;margin:7px 4px 0 0}
  .cp-updated{font-size:.61rem;line-height:1.4}
  .cp-detail{padding:14px 11px;border-radius:18px;margin-top:12px}
  .cp-card{padding:12px 13px;border-radius:15px}
  .cp-insight{padding:10px 11px;font-size:.78rem;line-height:1.45}
  .cp-status-grid{grid-template-columns:1fr;gap:6px}
  .cp-status{font-size:.68rem;padding:8px 9px}
  .cp-side-place{font-size:.92rem}
  .cp-side-address{font-size:.63rem}
  .cp-side-sync{gap:7px;padding:9px 10px}
  .cp-side-sync b{font-size:.64rem}
  .cp-side-sync span:not(.cp-side-sync-icon){font-size:.54rem}
  .cp-side-sync strong{font-size:.52rem}
  [data-testid="stHorizontalBlock"]{gap:.55rem!important;flex-wrap:wrap!important}
  [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]{min-width:calc(50% - .3rem)!important;width:calc(50% - .3rem)!important;flex:1 1 calc(50% - .3rem)!important}
  [data-testid="stMetric"]{padding:.2rem 0!important}
  [data-testid="stMetricLabel"]{font-size:.67rem!important}
  [data-testid="stMetricValue"]{font-size:1.05rem!important}
  [data-testid="stPlotlyChart"]{margin-left:-2px;margin-right:-2px}
  [data-testid="stDataFrame"]{max-width:100%;overflow-x:auto}
}

@media (max-width: 390px){
  .block-container{padding-top:3.95rem!important;padding-left:.55rem!important;padding-right:.55rem!important}
  [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]{min-width:100%!important;width:100%!important;flex:1 1 100%!important}
  .cp-location-card{padding:13px 11px}
  .cp-location-name{font-size:1.16rem}
  .cp-chip{font-size:.63rem}
}
</style>
""", unsafe_allow_html=True)


def secret(name):
    value = os.getenv(name)
    if value:
        return value
    try:
        return st.secrets[name]
    except Exception:
        return None


CLOUDANT_URL = secret("CLOUDANT_URL")
CLOUDANT_APIKEY = secret("CLOUDANT_APIKEY")


def aqi_value(pm25):
    if pm25 is None or pd.isna(pm25):
        return 0
    c = max(0.0, float(pm25))
    if c > 500:
        return 500
    for lo, hi, ilo, ihi in AQI_BREAKPOINTS:
        if c <= hi:
            return int(round(ilo + (ihi - ilo) * (c - lo) / (hi - lo)))
    return 500


def aqi_status(aqi):
    aqi = float(aqi)
    if aqi <= 50: return "Good"
    if aqi <= 100: return "Satisfactory"
    if aqi <= 200: return "Moderate"
    if aqi <= 300: return "Poor"
    if aqi <= 400: return "Very Poor"
    return "Severe"


def status_icon(status):
    return {"Good":"🟢","Satisfactory":"🟡","Moderate":"🟠","Poor":"🔴","Very Poor":"🟣","Severe":"⚫"}.get(status,"⚪")


def format_result(r):
    a = r.get("address", {}) or {}
    names = r.get("namedetails", {}) or {}
    name = names.get("name:en") or names.get("name") or r.get("display_name", "").split(",")[0]
    locality = a.get("neighbourhood") or a.get("suburb") or a.get("quarter") or a.get("village") or a.get("town")
    city = a.get("city") or a.get("municipality") or a.get("county") or a.get("town")
    state = a.get("state")
    country = a.get("country")
    pieces = [p for p in [locality, city, state, country] if p and p != name]
    address = " · ".join(dict.fromkeys(pieces))
    kind = r.get("type", "")
    category = r.get("class", "")
    type_map = {
        "city":"City", "town":"Town", "village":"Village", "suburb":"Locality",
        "neighbourhood":"Neighbourhood", "university":"University", "college":"College",
        "hospital":"Hospital", "airport":"Airport", "road":"Road", "building":"Building",
        "attraction":"Landmark", "museum":"Landmark", "park":"Park", "administrative":"Administrative area"
    }
    label = type_map.get(kind, "Place")
    if category == "boundary" and kind == "administrative":
        label = "Administrative area"
    return {"name": name, "address": address, "display": r.get("display_name", name),
            "lat": float(r["lat"]), "lon": float(r["lon"]), "type": kind, "category": label}


@st.cache_data(ttl=900, show_spinner=False)
def search_places(query):
    query = query.strip()
    if len(query) < 2:
        return []
    r = requests.get(
        GEOCODER_URL,
        params={
            "q": query, "format": "jsonv2", "addressdetails": 1, "namedetails": 1,
            "limit": 7, "dedupe": 1, "accept-language": "en"
        },
        headers={"User-Agent": "CloudPulse-BTech-Project/3.0 (location search)"},
        timeout=8,
    )
    r.raise_for_status()
    results = [format_result(x) for x in r.json()]
    # Remove near-duplicate names/coordinates while preserving Nominatim ranking.
    seen = set(); clean = []
    for item in results:
        key = (item["name"].lower(), round(item["lat"], 4), round(item["lon"], 4))
        if key not in seen:
            seen.add(key); clean.append(item)
    return clean


def search_suggestions(query):
    try:
        return [(f"{p['name']}  —  {p['address']}  [{p['category']}]", p) for p in suggest_places(query)]
    except requests.RequestException:
        return []


@st.cache_data(ttl=55, show_spinner=False)
def live_data(lat, lon):
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        weather_future = pool.submit(requests.get, WEATHER_URL, params={"latitude":lat,"longitude":lon,"current":"temperature_2m,relative_humidity_2m,wind_speed_10m","timezone":"auto"}, timeout=7)
        air_future = pool.submit(requests.get, AIR_URL, params={"latitude":lat,"longitude":lon,"current":"pm2_5,pm10","timezone":"auto"}, timeout=7)
        w, a = weather_future.result(), air_future.result()
    w.raise_for_status(); a.raise_for_status()
    wc, ac = w.json()["current"], a.json()["current"]
    pm25, pm10 = float(ac.get("pm2_5",0)), float(ac.get("pm10",0))
    aq = aqi_value(pm25)
    return {
        "weather":{"temperature_c":round(float(wc["temperature_2m"]),1),"humidity_percent":round(float(wc["relative_humidity_2m"]),1),"wind_speed_kmh":round(float(wc["wind_speed_10m"]),1)},
        "air":{"pm2_5":round(pm25,1),"pm10":round(pm10,1),"aqi":aq,"status":aqi_status(aq)},
        "updated":datetime.now(timezone.utc).isoformat(),"api_latency_ms":round((time.perf_counter()-started)*1000,1)
    }


@st.cache_data(ttl=1800, show_spinner=False)
def history_data(lat, lon):
    """Fetch and robustly align the hourly weather and air-quality histories.

    Open-Meteo endpoints can occasionally return slightly different hourly
    timestamps. An exact dataframe merge can therefore throw away most of
    the usable history and make the local ML model appear unavailable. We
    align the two hourly series by nearest timestamp within 31 minutes.
    """
    w = requests.get(
        WEATHER_URL,
        params={"latitude":lat,"longitude":lon,"hourly":"temperature_2m,relative_humidity_2m,wind_speed_10m","past_hours":192,"forecast_hours":1,"timezone":"UTC"},
        timeout=18,
    )
    a = requests.get(
        AIR_URL,
        params={"latitude":lat,"longitude":lon,"hourly":"pm2_5,pm10","past_hours":192,"forecast_hours":1,"timezone":"UTC"},
        timeout=18,
    )
    w.raise_for_status(); a.raise_for_status()
    wd, ad = w.json()["hourly"], a.json()["hourly"]

    wdf = pd.DataFrame({
        "time": pd.to_datetime(wd["time"], errors="coerce"),
        "temperature": pd.to_numeric(wd["temperature_2m"], errors="coerce"),
        "humidity": pd.to_numeric(wd["relative_humidity_2m"], errors="coerce"),
        "wind": pd.to_numeric(wd["wind_speed_10m"], errors="coerce"),
    }).dropna(subset=["time"]).sort_values("time")
    adf = pd.DataFrame({
        "time": pd.to_datetime(ad["time"], errors="coerce"),
        "pm25": pd.to_numeric(ad["pm2_5"], errors="coerce"),
        "pm10": pd.to_numeric(ad["pm10"], errors="coerce"),
    }).dropna(subset=["time"]).sort_values("time")

    # Nearest-hour alignment is much safer than an exact merge for two
    # independent API responses, while the 31-minute tolerance prevents
    # unrelated observations from being paired.
    df = pd.merge_asof(
        wdf, adf, on="time", direction="nearest", tolerance=pd.Timedelta("5min")
    )
    df = df.dropna(subset=["temperature","humidity","wind","pm25","pm10"]).sort_values("time")
    df["aqi"] = df["pm25"].apply(aqi_value).astype(float)
    return df.reset_index(drop=True)


@st.cache_data(ttl=3600, show_spinner=False)
def place_images(name, locality=""):
    for q in [name, f"{name} {locality}" if locality else "", locality]:
        if not q: continue
        try:
            r = requests.get(WIKI_API, params={"action":"query","generator":"search","gsrsearch":q,"gsrnamespace":6,"gsrlimit":6,"prop":"imageinfo","iiprop":"url","iiurlwidth":1200,"format":"json"}, timeout=10)
            r.raise_for_status()
            imgs=[]
            for p in r.json().get("query",{}).get("pages",{}).values():
                info=(p.get("imageinfo") or [{}])[0]; url=info.get("thumburl") or info.get("url")
                if url and url.lower().split("?")[0].endswith((".jpg",".jpeg",".png",".webp")):
                    imgs.append({"url":url,"title":p.get("title","").replace("File:","")})
            if imgs: return imgs[:4]
        except Exception:
            pass
    return []


@st.cache_resource
def cloudant_client():
    if not CLOUDANT_URL or not CLOUDANT_APIKEY:
        return None
    try:
        auth = IAMAuthenticator(CLOUDANT_APIKEY)
        client = CloudantV1(authenticator=auth)
        client.set_service_url(CLOUDANT_URL)
        return client
    except Exception:
        return None


def cloudant_save(record):
    client = cloudant_client()
    if not client:
        return False, "not configured"
    try:
        client.post_document(db=DB_NAME, document=record).get_result()
        return True, "connected"
    except Exception as exc:
        return False, str(exc)[:90]


def load_model_package():
    """Load a user-supplied trained model if one exists."""
    try:
        if not os.path.exists(MODEL_FILE) or os.path.getsize(MODEL_FILE) < 100:
            return None
        package = joblib.load(MODEL_FILE)
        if isinstance(package, dict) and "model" in package:
            return package
        return {"model": package, "features": ["temperature_c", "humidity_percent", "wind_speed_kmh", "pm2_5", "pm10"]}
    except Exception:
        return None

MODEL_FEATURES = ["temperature_c", "humidity_percent", "wind_speed_kmh", "pm2_5", "pm10"]
FEATURE_MAP = {"temperature_c":"temperature", "humidity_percent":"humidity", "wind_speed_kmh":"wind", "pm2_5":"pm25", "pm10":"pm10"}

def train_local_model(feature_frame):
    """Train a location-specific one-hour AQI model and return diagnostics."""
    frame = feature_frame.copy().sort_values("time").reset_index(drop=True)
    frame["time"] = pd.to_datetime(frame["time"], errors="coerce")
    frame["target_next_aqi"] = pd.to_numeric(frame["aqi"], errors="coerce").shift(-1)
    cols = [FEATURE_MAP[f] for f in MODEL_FEATURES]
    for c in cols + ["aqi", "target_next_aqi"]:
        frame[c] = pd.to_numeric(frame[c], errors="coerce")
    raw_rows = len(frame)
    frame = frame.dropna(subset=cols + ["target_next_aqi"]).reset_index(drop=True)
    usable = len(frame)
    diagnostic = {"status":"Training", "raw_rows":raw_rows, "usable_rows":usable, "train_rows":0, "test_rows":0, "reason":""}
    if usable < 24:
        diagnostic.update(status="Insufficient history", reason=f"Only {usable} usable hourly observations are available; at least 24 are required.")
        return None, diagnostic

    split = int(usable * 0.80)
    split = min(max(split, 18), usable - 4)
    train = frame.iloc[:split].copy()
    test = frame.iloc[split:].copy()
    diagnostic.update(train_rows=len(train), test_rows=len(test))
    try:
        model = RandomForestRegressor(n_estimators=180, max_depth=10, min_samples_leaf=2, random_state=42, n_jobs=-1)
        X_train, y_train = train[cols], train["target_next_aqi"].astype(float)
        X_test, y_test = test[cols], test["target_next_aqi"].astype(float)
        model.fit(X_train, y_train)
        pred_test = np.clip(model.predict(X_test).astype(float), 0, 500)
        actual = y_test.to_numpy()
        current = test["aqi"].to_numpy(dtype=float)
        abs_err = np.abs(actual - pred_test)
        mae = float(mean_absolute_error(actual, pred_test))
        rmse = float(np.sqrt(mean_squared_error(actual, pred_test)))
        r2 = float(r2_score(actual, pred_test)) if len(actual) >= 2 else float("nan")
        directional = float((np.sign(actual-current) == np.sign(pred_test-current)).mean()*100)
        q80 = float(np.quantile(abs_err, 0.80)) if len(abs_err) else float("nan")
        if len(actual) >= 40 and mae <= 25 and directional >= 65 and (np.isnan(r2) or r2 >= 0.60):
            reliability = "High"
        elif len(actual) >= 24 and mae <= 50 and (directional >= 55 or (not np.isnan(r2) and r2 >= 0.25)):
            reliability = "Moderate"
        else:
            reliability = "Low"
        evaluation = test[["time", "aqi"]].copy()
        evaluation["actual_next_aqi"] = actual
        evaluation["predicted_next_aqi"] = pred_test
        evaluation["absolute_error"] = abs_err
        final_model = RandomForestRegressor(n_estimators=180, max_depth=10, min_samples_leaf=2, random_state=42, n_jobs=-1)
        final_model.fit(frame[cols], frame["target_next_aqi"].astype(float))
        diagnostic.update(status="Trained", reason="Local historical model trained successfully.")
        return {
            "model": final_model, "features": MODEL_FEATURES, "source": "Local historical training",
            "samples": usable, "train_samples": len(train), "test_samples": len(test), "diagnostic": diagnostic,
            "backtest": {"n":len(test),"mae":mae,"rmse":rmse,"r2":r2,"directional_accuracy":directional,"q80_error":q80,"reliability":reliability,"predictions":evaluation},
        }, diagnostic
    except Exception as exc:
        diagnostic.update(status="Training failed", reason=f"{type(exc).__name__}: {str(exc)[:180]}")
        return None, diagnostic

@st.cache_resource(show_spinner=False)
def _history_model(hist):
    if hist is None or len(hist) < 24:
        return None, {"status":"Insufficient history", "raw_rows":0 if hist is None else len(hist), "usable_rows":0 if hist is None else len(hist), "train_rows":0, "test_rows":0, "reason":"At least 24 hourly observations are required."}
    return train_local_model(hist[["time","temperature","humidity","wind","pm25","pm10","aqi"]])


@st.cache_resource(show_spinner=False)
def train_location_model(lat, lon):
    try:
        hist = history_data(lat, lon).tail(336).copy()
        return _history_model(hist)
    except Exception as exc:
        return None, {"status":"History fetch failed", "raw_rows":0, "usable_rows":0, "train_rows":0, "test_rows":0, "reason":f"{type(exc).__name__}: {str(exc)[:180]}"}

@st.cache_data(ttl=300, show_spinner=False)
def forecast_fallback_aqi(lat, lon):
    """Use the upstream hourly air-quality forecast only when local ML is unavailable."""
    r = requests.get(
        AIR_URL,
        params={
            "latitude": lat, "longitude": lon,
            "hourly": "pm2_5,pm10",
            "forecast_hours": 4, "past_hours": 0,
            "timezone": "UTC"
        },
        timeout=12,
    )
    r.raise_for_status()
    hourly = r.json().get("hourly", {})
    times = pd.to_datetime(hourly.get("time", []), utc=True, errors="coerce")
    pm25 = pd.to_numeric(hourly.get("pm2_5", []), errors="coerce")
    pm10 = pd.to_numeric(hourly.get("pm10", []), errors="coerce")
    frame = pd.DataFrame({"time": times, "pm25": pm25, "pm10": pm10}).dropna().sort_values("time")
    if frame.empty:
        raise ValueError("The air-quality forecast returned no usable hourly observations.")
    now = pd.Timestamp.now(tz="UTC")
    future = frame[frame["time"] > now + pd.Timedelta(minutes=20)]
    row = future.iloc[0] if not future.empty else frame.iloc[min(1, len(frame)-1)]
    return int(max(0, min(500, round(aqi_value(float(row["pm25"]))))))

def predict_next(live, hist=None, lat=None, lon=None):
    """Estimate AQI about one hour ahead. Prefer local ML; use a clearly labelled API fallback."""
    diagnostic = {"status":"Training", "raw_rows":0, "usable_rows":0, "train_rows":0, "test_rows":0, "reason":""}
    package = None
    if hist is not None:
        local, diagnostic = _history_model(hist)
        if local and local.get("model") is not None:
            package = local
        else:
            package = load_model_package()
            if package:
                diagnostic.update(status="External model loaded", reason="Local model unavailable; packaged model used.")
    else:
        package = load_model_package()
        diagnostic = {"status":"External model loaded" if package else "No local model", "raw_rows":0, "usable_rows":0, "train_rows":0, "test_rows":0, "reason":"Packaged model used." if package else "No historical model was supplied."}

    if package:
        model = package["model"]
        features = package.get("features", MODEL_FEATURES)
        values = {"temperature_c":live["weather"]["temperature_c"],"humidity_percent":live["weather"]["humidity_percent"],"wind_speed_kmh":live["weather"]["wind_speed_kmh"],"pm2_5":live["air"]["pm2_5"],"pm10":live["air"]["pm10"]}
        try:
            value = int(round(float(model.predict(pd.DataFrame([[values[f] for f in features]], columns=features))[0])))
            diagnostic.update(status="ML prediction ready", reason=diagnostic.get("reason") or "Location-specific historical model used.")
            return max(0, min(500, value)), package.get("source", "Local ML model"), diagnostic
        except Exception as exc:
            diagnostic.update(status="ML prediction failed", reason=f"{type(exc).__name__}: {str(exc)[:180]}")

    if lat is not None and lon is not None:
        try:
            fallback = forecast_fallback_aqi(lat, lon)
            diagnostic.update(status="Forecast fallback", reason=(diagnostic.get("reason") or "Local ML model unavailable.") + " Using the upstream hourly air-quality forecast instead.")
            return fallback, "Air-quality forecast fallback", diagnostic
        except Exception as exc:
            diagnostic.update(status="Forecast unavailable", reason=f"{diagnostic.get('reason') or 'Local ML model unavailable.'} Fallback error: {type(exc).__name__}: {str(exc)[:140]}")
    return None, diagnostic.get("status", "Unavailable"), diagnostic

def model_backtest(hist):
    """Return a real chronological local backtest, training on the first 80% and testing on the final 20%."""
    local, _diagnostic = _history_model(hist)
    if local and local.get("backtest"):
        return local["backtest"]
    package = load_model_package()
    if not package:
        return None
    model = package["model"]
    features = package.get("features", MODEL_FEATURES)
    frame = hist.copy().sort_values("time").reset_index(drop=True)
    frame["target_next_aqi"] = frame["aqi"].shift(-1)
    cols = [FEATURE_MAP.get(f, f) for f in features]
    frame = frame.dropna(subset=cols + ["target_next_aqi"])
    if len(frame) < 12:
        return None
    X = pd.DataFrame({f: frame[FEATURE_MAP.get(f, f)] for f in features})
    y = frame["target_next_aqi"].astype(float)
    try: pred = np.clip(model.predict(X).astype(float), 0, 500)
    except Exception: return None
    actual=y.to_numpy(); current=frame["aqi"].to_numpy(dtype=float); abs_err=np.abs(actual-pred)
    mae=float(mean_absolute_error(actual,pred)); rmse=float(np.sqrt(mean_squared_error(actual,pred))); r2=float(r2_score(actual,pred)) if len(actual)>=2 else float("nan"); directional=float((np.sign(actual-current)==np.sign(pred-current)).mean()*100); q80=float(np.quantile(abs_err,.80))
    reliability="High" if len(actual)>=48 and r2>=.60 and directional>=65 and mae<=25 else "Moderate" if len(actual)>=24 and (r2>=.25 or directional>=55) and mae<=50 else "Low"
    out=frame[["time","aqi"]].copy(); out["actual_next_aqi"]=actual; out["predicted_next_aqi"]=pred; out["absolute_error"]=abs_err
    return {"n":len(frame),"mae":mae,"rmse":rmse,"r2":r2,"directional_accuracy":directional,"q80_error":q80,"reliability":reliability,"predictions":out}


def detect_anomalies(hist):
    """Flag unusual AQI observations using a robust rolling baseline and IQR."""
    df=hist.copy().sort_values("time").reset_index(drop=True)
    if len(df)<12:
        df["anomaly"]=False; df["anomaly_reason"]=""; return df
    aqi=df["aqi"].astype(float); rolling_med=aqi.rolling(24,min_periods=12).median(); abs_dev=(aqi-rolling_med).abs(); mad=abs_dev.rolling(24,min_periods=12).median()
    robust_z=abs_dev/(1.4826*mad.replace(0,np.nan)); q1,q3=aqi.quantile([0.25,0.75]); iqr=max(float(q3-q1),1.0)
    iqr_flag=(aqi<q1-1.5*iqr)|(aqi>q3+1.5*iqr); rolling_flag=(robust_z>3.5)&(abs_dev>=15)
    df["anomaly"]=(iqr_flag|rolling_flag).fillna(False); df["anomaly_reason"]=np.where(rolling_flag,"Abrupt deviation from rolling baseline",np.where(iqr_flag,"Distribution outlier","")); df["rolling_baseline"]=rolling_med; df["robust_z"]=robust_z
    return df


def hourly_pattern(hist):
    h=hist.copy(); h["hour"]=pd.to_datetime(h["time"]).dt.hour
    return h.groupby("hour",as_index=False).agg(mean_aqi=("aqi","mean"),mean_pm25=("pm25","mean"),mean_temperature=("temperature","mean"),mean_humidity=("humidity","mean"),samples=("aqi","size"))


def aqi_distribution(hist):
    bins=[-1,50,100,200,300,400,500]; labels=["Good (0–50)","Satisfactory (51–100)","Moderate (101–200)","Poor (201–300)","Very Poor (301–400)","Severe (401–500)"]
    cats=pd.cut(hist["aqi"].clip(0,500),bins=bins,labels=labels); return cats.value_counts().reindex(labels,fill_value=0).reset_index(name="hours").rename(columns={"index":"category"})


def pro_metrics(hist, live, pred, model_state):
    h=hist.tail(168).copy(); aqi=h["aqi"]; pm25=h["pm25"]; pm10=h["pm10"]; slope=float((aqi.iloc[-1]-aqi.iloc[0])/max(1,len(aqi)-1)) if len(aqi)>=2 else 0.0; bt=model_backtest(h); anomalies=detect_anomalies(h)
    return {"hours":len(h),"aqi_mean":float(aqi.mean()),"aqi_min":float(aqi.min()),"aqi_max":float(aqi.max()),"aqi_median":float(aqi.median()),"aqi_std":float(aqi.std()),"pm25_mean":float(pm25.mean()),"pm25_max":float(pm25.max()),"pm10_mean":float(pm10.mean()),"pm10_max":float(pm10.max()),"temp_mean":float(h["temperature"].mean()),"temp_min":float(h["temperature"].min()),"temp_max":float(h["temperature"].max()),"humidity_mean":float(h["humidity"].mean()),"wind_mean":float(h["wind"].mean()),"poor_hours":int((aqi>200).sum()),"very_poor_hours":int((aqi>300).sum()),"severe_hours":int((aqi>400).sum()),"aqi_slope_hour":slope,"model":model_state,"next_aqi":pred,"current_status":live["air"]["status"],"backtest":bt,"anomalies":anomalies}

def build_html_report(place, hist, live, pred, model_state):
    m=pro_metrics(hist,live,pred,model_state); bt=m["backtest"]; an=m["anomalies"]
    anomaly_rows=an[an["anomaly"]].tail(30).to_html(index=False,border=0,float_format=lambda x:f"{x:.2f}" if isinstance(x,float) else str(x))
    hourly=hourly_pattern(hist).to_html(index=False,border=0,float_format=lambda x:f"{x:.2f}" if isinstance(x,float) else str(x))
    dist=aqi_distribution(hist).to_html(index=False,border=0)
    pred_text="Unavailable — no usable trained model is available." if pred is None else f"{pred} ({aqi_status(pred)})"
    reliability=(f'{bt["reliability"]} · MAE {bt["mae"]:.1f} AQI · directional accuracy {bt["directional_accuracy"]:.0f}%' if bt else "Unavailable")
    return f'''<!doctype html><html><head><meta charset="utf-8"><title>CloudPulse Pro Analysis</title>
<style>body{{font-family:Arial,sans-serif;background:#08111e;color:#eaf2fb;padding:32px;line-height:1.45}}h1{{color:#78e7dc}}h2{{color:#bfe9e4;margin-top:28px}}table{{border-collapse:collapse;width:100%;margin:12px 0}}th,td{{padding:7px;border:1px solid #24364a;font-size:12px}}th{{background:#13243a}}.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}}.card{{background:#0d1a2b;padding:14px;border-radius:10px}}.note{{color:#a9bdcf;font-size:13px}}</style></head><body>
<h1>CloudPulse Pro Analysis</h1><p><b>{html.escape(place["name"])}</b><br>{html.escape(place.get("address", ""))}<br>{place["lat"]:.5f}, {place["lon"]:.5f}</p>
<div class="grid"><div class="card"><b>Current AQI</b><br>{live["air"]["aqi"]} ({html.escape(live["air"]["status"])})</div><div class="card"><b>Estimated AQI ~1 hour from now</b><br>{pred_text}</div><div class="card"><b>Forecast reliability</b><br>{html.escape(reliability)}</div><div class="card"><b>7-day mean AQI</b><br>{m["aqi_mean"]:.1f}</div><div class="card"><b>7-day AQI range</b><br>{m["aqi_min"]:.0f} – {m["aqi_max"]:.0f}</div><div class="card"><b>Anomalies detected</b><br>{int(an["anomaly"].sum())}</div></div>
<h2>Executive interpretation</h2><p>Average AQI was {m["aqi_mean"]:.1f}, with a range of {m["aqi_min"]:.0f}–{m["aqi_max"]:.0f}. There were {m["poor_hours"]} hours above AQI 200, {m["very_poor_hours"]} above 300 and {m["severe_hours"]} above 400. The overall hourly slope was {m["aqi_slope_hour"]:+.3f} AQI points/hour.</p>
<h2>Short-term forecast</h2><p class="note">“Estimated AQI ~1 hour from now” is a model estimate, not a direct measurement. Reliability is based on a backtest against the selected location's recent hourly history when a trained model is available.</p><p><b>Forecast:</b> {pred_text}<br><b>Reliability:</b> {html.escape(reliability)}</p>
<h2>AQI distribution</h2>{dist}<h2>Hourly patterns</h2>{hourly}<h2>Anomaly detection</h2>{anomaly_rows if int(an["anomaly"].sum()) else "<p>No unusual AQI observations were flagged in the 7-day window.</p>"}
<h2>Recent dataset</h2>{hist.tail(168).to_html(index=False,border=0,float_format=lambda x:f"{x:.2f}" if isinstance(x,float) else str(x))}
</body></html>'''


def build_pdf_report(path, place, hist, live, pred, model_state):
    if SimpleDocTemplate is None: return False
    m=pro_metrics(hist,live,pred,model_state); bt=m["backtest"]; an=m["anomalies"]
    doc=SimpleDocTemplate(path,pagesize=A4,rightMargin=32,leftMargin=32,topMargin=32,bottomMargin=32); styles=getSampleStyleSheet()
    story=[Paragraph("CloudPulse Executive Environmental Report",styles["Title"]),Paragraph(html.escape(place["name"]),styles["Heading2"]),Paragraph(html.escape(place.get("address","")),styles["BodyText"]),Paragraph(f'Coordinates: {place["lat"]:.5f}, {place["lon"]:.5f}',styles["BodyText"]),Spacer(1,12)]
    pred_text="Unavailable — no usable trained model is available." if pred is None else f'{pred} ({aqi_status(pred)})'
    reliability="Unavailable" if not bt else f'{bt["reliability"]} · MAE {bt["mae"]:.1f} AQI · 80% of backtest errors were within ±{bt["q80_error"]:.1f} AQI'
    data=[["Metric","Result"],["Current AQI",f'{live["air"]["aqi"]} ({live["air"]["status"]})'],["Estimated AQI ~1 hour from now",pred_text],["Forecast reliability",reliability],["7-day mean / median",f'{m["aqi_mean"]:.1f} / {m["aqi_median"]:.1f}'],["AQI minimum / maximum",f'{m["aqi_min"]:.0f} / {m["aqi_max"]:.0f}'],["PM2.5 mean / max",f'{m["pm25_mean"]:.1f} / {m["pm25_max"]:.1f} µg/m³'],["PM10 mean / max",f'{m["pm10_mean"]:.1f} / {m["pm10_max"]:.1f} µg/m³'],["Hours AQI > 200 / 300 / 400",f'{m["poor_hours"]} / {m["very_poor_hours"]} / {m["severe_hours"]}'],["Anomalies flagged",str(int(an["anomaly"].sum()))],["Model",model_state]]
    t=Table(data,colWidths=[210,270]); t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#d9f5f1")),("GRID",(0,0),(-1,-1),0.4,colors.grey),("VALIGN",(0,0),(-1,-1),"TOP"),("FONTNAME",(0,0),(-1,0),"Helvetica-Bold")]))
    story += [t,Spacer(1,16),Paragraph("Executive summary",styles["Heading2"])]
    trend="rising" if m["aqi_slope_hour"]>0.03 else "falling" if m["aqi_slope_hour"]<-0.03 else "stable"
    story += [Paragraph(f'Across the 7-day hourly window, AQI averaged {m["aqi_mean"]:.1f} and was {trend}. {m["poor_hours"]} hours exceeded AQI 200, {m["very_poor_hours"]} exceeded 300, and {m["severe_hours"]} exceeded 400.',styles["BodyText"]),Spacer(1,8),Paragraph("Forecast interpretation",styles["Heading2"]),Paragraph("The ~1 hour value is a model estimate of the AQI expected about one hour after the current observation. It is not a direct measurement. Reliability is based on local historical backtesting when enough observations and a trained model are available.",styles["BodyText"]),Spacer(1,8),Paragraph("AQI distribution",styles["Heading2"])]
    dist=aqi_distribution(hist); story.append(Table([["AQI band","Hours"]]+dist.values.tolist(),colWidths=[300,180],style=TableStyle([("GRID",(0,0),(-1,-1),0.35,colors.grey),("BACKGROUND",(0,0),(-1,0),colors.HexColor("#eef7f6"))])))
    story += [Spacer(1,10),Paragraph("Hourly pattern",styles["Heading2"])]
    hp=hourly_pattern(hist); hp_rows=[["Hour","Mean AQI","Mean PM2.5","Mean Temp","Mean Humidity"]]+[[int(r.hour),f'{r.mean_aqi:.1f}',f'{r.mean_pm25:.1f}',f'{r.mean_temperature:.1f}',f'{r.mean_humidity:.1f}'] for _,r in hp.iterrows()]
    story.append(Table(hp_rows,colWidths=[55,80,90,85,90],style=TableStyle([("GRID",(0,0),(-1,-1),0.25,colors.grey),("BACKGROUND",(0,0),(-1,0),colors.HexColor("#eef7f6")),("FONTSIZE",(0,0),(-1,-1),7)])))
    if bt: story += [Spacer(1,10),Paragraph("Model vs actual evaluation",styles["Heading2"]),Paragraph(f'Backtest samples: {bt["n"]}. MAE: {bt["mae"]:.1f} AQI. RMSE: {bt["rmse"]:.1f} AQI. R²: {bt["r2"]:.3f}. Directional accuracy: {bt["directional_accuracy"]:.1f}%. 80% of absolute errors were within ±{bt["q80_error"]:.1f} AQI.',styles["BodyText"])]
    doc.build(story); return True

SEARCH_COMPONENT = os.path.join(os.path.dirname(__file__), "cloudpulse_search_component")
cloudpulse_search = components.declare_component("cloudpulse_search", path=SEARCH_COMPONENT)

def commit_location(place):
    """Commit a search result as the single source of truth for CloudPulse."""
    clean = {
        "name": str(place.get("name") or "Selected location"),
        "address": str(place.get("address") or ""),
        "lat": float(place["lat"]),
        "lon": float(place["lon"]),
        "type": str(place.get("type") or "place"),
        "category": str(place.get("category") or "Place"),
    }
    st.session_state.selected = clean
    st.session_state.details = False
    st.session_state.search_text = clean["name"]
    st.session_state.search_error = ""
    # Environmental values belong to the selected coordinates. Never reuse the
    # previous location's cached response after a place selection.
    try:
        live_data.clear()
    except Exception:
        pass
    st.session_state.last_live = {}
    # Keep a compact recent-search history for the component.
    recent = st.session_state.get("recent_searches", [])
    signature = (clean["name"], round(clean["lat"], 5), round(clean["lon"], 5))
    recent = [r for r in recent if (r.get("name"), round(float(r.get("lat", 0)), 5), round(float(r.get("lon", 0)), 5)) != signature]
    recent.insert(0, clean)
    st.session_state.recent_searches = recent[:6]

def render_search_component():
    result = cloudpulse_search(
        initial_query=st.session_state.get("search_text", ""),
        recent=st.session_state.get("recent_searches", [])[:5],
        bias_lat=st.session_state.get("selected", {}).get("lat", 26.8467),
        bias_lon=st.session_state.get("selected", {}).get("lon", 80.9462),
        key="cloudpulse_location_search",
    )
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except Exception:
            result = None
    if isinstance(result, dict) and result.get("action") == "select":
        place = result.get("place") or {}
        try:
            place["lat"], place["lon"] = float(place["lat"]), float(place["lon"])
        except (KeyError, TypeError, ValueError):
            return
        commit_location(place)
        st.session_state.search_text = place.get("name") or result.get("query", "")
        # The component's setComponentValue event already causes Streamlit to
        # rerun. Do NOT call st.rerun() here; doing so can interrupt the value
        # handshake and leave the old live panel rendered.
    elif isinstance(result, dict) and result.get("action") == "clear":
        st.session_state.search_text = ""

# ---------------- State ----------------
if "selected" not in st.session_state:
    st.session_state.selected={"name":"Lucknow","address":"Lucknow · Uttar Pradesh · India","lat":26.8467,"lon":80.9462}
if "details" not in st.session_state: st.session_state.details=False
if "search_results" not in st.session_state: st.session_state.search_results=[]
if "search_text" not in st.session_state: st.session_state.search_text=""
if "recent_searches" not in st.session_state: st.session_state.recent_searches=[]
if "search_error" not in st.session_state: st.session_state.search_error=""
if "last_search_at" not in st.session_state: st.session_state.last_search_at=0.0
if "pro_analysis" not in st.session_state: st.session_state.pro_analysis=False
if "cloudant_state" not in st.session_state: st.session_state.cloudant_state="Automatic"
if "last_cloudant_signature" not in st.session_state: st.session_state.last_cloudant_signature=None
if "model_diagnostic" not in st.session_state:
    st.session_state.model_diagnostic={"status":"Not checked","raw_rows":0,"usable_rows":0,"train_rows":0,"test_rows":0,"reason":"The local model will be checked when environmental history is loaded."}

selected=st.session_state.selected

# ---------------- Sidebar ----------------
with st.sidebar:
    st.markdown("<div class='cp-side-brand'><div class='cp-side-logo'>CloudPulse ◉</div><div class='cp-side-badge'>LIVE</div></div>", unsafe_allow_html=True)
    st.markdown("<div class='cp-side-info'>Real-time environmental intelligence for any selected place.</div>", unsafe_allow_html=True)

    st.markdown("<div class='cp-side-focus'><div class='cp-side-kicker'>Current focus</div><div class='cp-side-place'>" + html.escape(selected["name"]) + "</div><div class='cp-side-address'>" + html.escape(selected.get("address", "")) + "</div><div class='cp-side-live'><span class='cp-side-dot'></span>Live updates · every 60 seconds</div></div>", unsafe_allow_html=True)

    st.markdown("<div class='cp-side-section'>Controls</div>", unsafe_allow_html=True)
    if st.button("↻ Refresh now", width="stretch", key="side_refresh"):
        live_data.clear()
        st.rerun()

    sync_state = st.session_state.get("cloudant_state", "Automatic")
    sync_class = "🟢" if sync_state in ("Automatic", "connected") else "🟡" if "failed" not in sync_state.lower() else "🔴"
    st.markdown(
        f'<div class="cp-side-sync"><div><span class="cp-side-sync-icon">☁</span><div><b>Cloud sync</b><span>Environmental readings are stored automatically</span></div></div><strong>{sync_class} {html.escape(sync_state.title())}</strong></div>',
        unsafe_allow_html=True,
    )

    st.markdown("<div class='cp-side-section'>Explore</div>", unsafe_allow_html=True)
    with st.expander("📈  Pro Analysis", expanded=False):
        st.markdown("<div class='cp-side-info'>Deep environmental analytics, trends, correlations, model evaluation and reports.</div>", unsafe_allow_html=True)
        if st.button("Open Pro Analysis", width="stretch", key="side_open_pro"):
            st.session_state.pro_analysis = True
            st.rerun()
        st.markdown("<div class='cp-side-mini-grid'><div class='cp-side-mini'>History<b>7 days</b></div><div class='cp-side-mini'>Refresh<b>60 sec</b></div></div>", unsafe_allow_html=True)

    with st.expander("◉  System Status", expanded=False):
        uptime=int(time.time()-START_TIME)
        cpu=psutil.cpu_percent(interval=0.05) if psutil else None
        mem=psutil.virtual_memory().percent if psutil else None
        cloudant_ok=cloudant_client() is not None
        model_diag=st.session_state.get("model_diagnostic", {})
        model_status=model_diag.get("status", "Not checked")
        model_icon="🟢" if model_status == "Trained" else "🟡" if model_status in ("Training", "Not checked", "External model loaded") else "🔴"
        st.markdown("### Runtime")
        st.markdown(f'<div class="cp-status-grid"><div class="cp-status">Application <b>🟢 Online</b></div><div class="cp-status">Uptime <b>{uptime//60}m {uptime%60}s</b></div><div class="cp-status">CPU load <b>{f"{cpu:.0f}%" if cpu is not None else "N/A"}</b></div><div class="cp-status">Memory <b>{f"{mem:.0f}%" if mem is not None else "N/A"}</b></div></div>',unsafe_allow_html=True)
        st.markdown("### Services")
        st.markdown(f'<div class="cp-status-grid"><div class="cp-status">Weather <b>🟢 Open-Meteo</b></div><div class="cp-status">Air quality <b>🟢 Open-Meteo</b></div><div class="cp-status">Search <b>Photon + Nominatim</b></div><div class="cp-status">Images <b>🟢 Wikimedia</b></div><div class="cp-status">Cloudant <b>{"🟢 Connected" if cloudant_ok else "🟡 Optional"}</b></div><div class="cp-status">ML <b>{model_icon} {html.escape(model_status)}</b></div></div>',unsafe_allow_html=True)
        last=st.session_state.get("last_live", {})
        latency = f"{last.get('api_latency_ms')} ms" if last.get('api_latency_ms') else "On demand"
        st.markdown("### Pipeline")
        st.markdown(f'<div class="cp-status-grid"><div class="cp-status">Updates <b>60 sec</b></div><div class="cp-status">Search cache <b>15 min</b></div><div class="cp-status">Data cache <b>55 sec</b></div><div class="cp-status">API latency <b>{latency}</b></div><div class="cp-status">Cloudant write <b>{last.get("cloudant","Not checked")}</b></div><div class="cp-status">Prediction <b>{last.get("model","Not checked")}</b></div></div>',unsafe_allow_html=True)
        st.markdown("### ML diagnostics")
        st.caption(f'History {model_diag.get("raw_rows",0)} · usable {model_diag.get("usable_rows",0)} · train {model_diag.get("train_rows",0)} · test {model_diag.get("test_rows",0)}')
        if model_diag.get("reason"):
            st.caption(model_diag["reason"])
        st.markdown("### Technical")
        st.caption("Python " + platform.python_version() + " · Streamlit runtime")

    st.markdown("<div class='cp-side-footer'>CloudPulse uses modelled weather and air-quality data. AQI is a PM2.5-based CloudPulse estimate, not an official CPCB station reading.</div>", unsafe_allow_html=True)

# ---------------- Header + search ----------------
st.markdown('<div class="cp-brand">CloudPulse ◉</div><div class="cp-tag">Find any place first. CloudPulse then loads its live environmental intelligence.</div><div style="margin-top:12px"><span class="cp-pill">● LOCATION EXPLORER</span></div>', unsafe_allow_html=True)
st.markdown("### 🔎 Find a location")
st.markdown('<div class="cp-note" style="margin:-7px 0 10px">Search like Google Maps — start typing and choose the exact place from live suggestions.</div>', unsafe_allow_html=True)
render_search_component()

# ---------------- Live data: unified location surface ----------------
def render_live_panel():
    # Always read the latest selection inside the live renderer. This is important
    # when the search component triggers a rerun/fragment refresh.
    current = dict(st.session_state.get("selected") or selected)
    try:
        live=live_data(current["lat"],current["lon"])
    except Exception as exc:
        st.error(f"Environmental data unavailable: {exc}")
        return
    try:
        hist_for_pred = history_data(current["lat"], current["lon"]).tail(336).copy()
        history_error = None
    except Exception as exc:
        hist_for_pred = None
        history_error = f"{type(exc).__name__}: {str(exc)[:180]}"
    try:
        pred,model_state,model_diag=predict_next(live,hist_for_pred,current["lat"],current["lon"])
        if history_error and model_diag.get("reason"):
            model_diag["reason"] = f"History fetch issue: {history_error}. {model_diag['reason']}"
        elif history_error:
            model_diag["reason"] = f"History fetch issue: {history_error}."
    except Exception as exc:
        pred,model_state=None,"Prediction error",{"status":"Prediction pipeline failed","raw_rows":0,"usable_rows":0,"train_rows":0,"test_rows":0,"reason":f"{type(exc).__name__}: {str(exc)[:180]}"}
    st.session_state.model_diagnostic = model_diag
    delta=(pred-live["air"]["aqi"]) if pred is not None else 0
    st.markdown(
        f"""<div class="cp-location-card">
        <div class="cp-location-main">
          <div><div class="cp-kicker">CURRENT LOCATION</div><div class="cp-location-name">{html.escape(current["name"])}</div><div class="cp-muted">{html.escape(current.get("address", ""))}</div></div>
        </div>
        <div class="cp-location-meta">
          <span class="cp-chip">{status_icon(live["air"]["status"])} AQI <b>{live["air"]["aqi"]}</b></span>
          <span class="cp-chip">🌡 {live["weather"]["temperature_c"]}°C</span>
          <span class="cp-chip">💧 {live["weather"]["humidity_percent"]}%</span>
          <span class="cp-chip">💨 {live["weather"]["wind_speed_kmh"]} km/h</span>
          <span class="cp-chip">📈 Estimated AQI ~1 hour from now <b>{pred if pred is not None else "—"}</b></span>
        </div>
        <div class="cp-updated">Live reading · {live["updated"][11:19]} UTC · {live["api_latency_ms"]} ms</div>
        </div>""", unsafe_allow_html=True)
    # Use a native Streamlit expander for the detailed view. This is deliberately
    # independent of custom HTML/component events, so it remains clickable even
    # when the search component is mounted above it.
    with st.expander("📊 View full environmental analysis", expanded=False):
        st.markdown("### Environmental analysis")
        st.caption(f"{current['name']} · {current['lat']:.5f}, {current['lon']:.5f}")
        c1,c2,c3,c4,c5=st.columns(5)
        c1.metric("🌫 AQI",live["air"]["aqi"],live["air"]["status"])
        c2.metric("🌡 Temperature",f"{live['weather']['temperature_c']} °C")
        c3.metric("💧 Humidity",f"{live['weather']['humidity_percent']} %")
        c4.metric("💨 Wind",f"{live['weather']['wind_speed_kmh']} km/h")
        c5.metric("📈 Estimated AQI ~1 hour from now", pred if pred is not None else "Unavailable", model_state if pred is not None else "Unavailable")
        if pred is None:
            insight = "A short-term AQI estimate is temporarily unavailable. Open System Status → ML diagnostics to see the exact training issue."
        else:
            insight=(f"The model estimates AQI may rise by about {delta} points in ~1 hour." if delta>10 else f"The model estimates AQI may fall by about {abs(delta)} points in ~1 hour." if delta<-10 else "The model estimates AQI will remain relatively stable over the next hour.")
        st.markdown(f'<div class="cp-insight">🧠 <b>CloudPulse insight</b><br>{insight}</div>',unsafe_allow_html=True)
        try:
            imgs=place_images(current["name"],current.get("address","").split(" · ")[0] if current.get("address") else "")
            if imgs:
                st.image(imgs[0]["url"],width="stretch")
        except Exception:
            pass
        try:
            hist=history_data(current["lat"],current["lon"])
            chart=go.Figure(); chart.add_trace(go.Scatter(x=hist.tail(48)["time"],y=hist.tail(48)["aqi"],mode="lines",line=dict(width=2))); chart.update_layout(height=250,margin=dict(l=0,r=0,t=10,b=0),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(color="#8fa4ba"),xaxis=dict(showgrid=False),yaxis=dict(showgrid=True,title="AQI")); st.plotly_chart(chart,width="stretch",config={"scrollZoom":False,"displayModeBar":False})
        except Exception:
            st.info("Recent trend is temporarily unavailable.")
        a,b,c=st.columns(3); a.metric("PM2.5",f"{live['air']['pm2_5']} µg/m³"); b.metric("PM10",f"{live['air']['pm10']} µg/m³"); c.metric("Status",f"{status_icon(live['air']['status'])} {live['air']['status']}")
    # Persist each newly fetched live reading automatically. A session-level
    # signature prevents duplicate Cloudant documents when Streamlit reruns
    # without a new API reading.
    save_signature = (current["name"], round(float(current["lat"]), 5), round(float(current["lon"]), 5), str(live.get("updated", "")))
    if st.session_state.get("last_cloudant_signature") != save_signature:
        try:
            ok, msg = cloudant_save({
                "type": "city_environment",
                "location": {
                    "name": current["name"],
                    "latitude": current["lat"],
                    "longitude": current["lon"],
                },
                "timestamp": live["updated"],
                "weather": live["weather"],
                "air_quality": live["air"],
            })
            st.session_state.cloudant_state = "connected" if ok else "Sync failed: " + msg
            if ok:
                st.session_state.last_cloudant_signature = save_signature
        except Exception as exc:
            st.session_state.cloudant_state = "Sync failed: " + str(exc)[:90]
    elif "cloudant_state" not in st.session_state:
        st.session_state.cloudant_state = "Automatic"

    st.session_state.last_live={"api_latency_ms":live["api_latency_ms"],"cloudant":st.session_state.get("cloudant_state", "Automatic"),"model":model_state,"updated":live["updated"]}

if hasattr(st,"fragment"):
    @st.fragment(run_every="60s")
    def live_fragment():
        render_live_panel()
    live_fragment()
else:
    render_live_panel()

# ---------------- Pro Analysis ----------------
if st.session_state.get("pro_analysis"):
    current=dict(st.session_state.get("selected") or selected)
    st.markdown("## 📈 Pro Analysis")
    st.caption("Analyst workspace: trends, distributions, anomalies, hourly patterns, model evaluation and downloadable reports.")
    if st.button("Close Pro Analysis", key="close_pro"):
        st.session_state.pro_analysis=False; st.rerun()
    try:
        live=live_data(current["lat"],current["lon"]); hist=history_data(current["lat"],current["lon"]).tail(168).copy(); pred,model_state,model_diag=predict_next(live,hist,current["lat"],current["lon"]); st.session_state.model_diagnostic=model_diag; m=pro_metrics(hist,live,pred,model_state); bt=m["backtest"]; anomalies=m["anomalies"]
        st.markdown("### Executive snapshot")
        a,b,c,d,e=st.columns(5); a.metric("Current AQI",live["air"]["aqi"],live["air"]["status"]); b.metric("7-day mean",f'{m["aqi_mean"]:.1f}'); c.metric("AQI range",f'{m["aqi_min"]:.0f}–{m["aqi_max"]:.0f}'); d.metric("Anomalies",int(anomalies["anomaly"].sum())); e.metric("AQI > 200",f'{m["poor_hours"]} h')
        st.markdown("### 📈 Estimated AQI ~1 hour from now")
        if pred is None:
            st.warning("A location-specific model is not available, so CloudPulse is not presenting a fake forecast.")
            if model_diag.get("reason"):
                st.caption(f"Training diagnostic: {model_diag['reason']}")
        else:
            f1,f2,f3,f4=st.columns(4); f1.metric("Estimated AQI ~1 hour from now",pred,aqi_status(pred))
            if bt:
                f2.metric("Reliability",bt["reliability"]); f3.metric("Typical error (MAE)",f'{bt["mae"]:.1f} AQI'); f4.metric("80% error band",f'±{bt["q80_error"]:.0f} AQI')
                st.info(f'**What this means:** {pred} is the model\'s estimate of the AQI approximately one hour after the current reading. In the local backtest, the model had an average absolute error of {bt["mae"]:.1f} AQI; 80% of its historical errors were within ±{bt["q80_error"]:.0f} AQI. This is a model estimate, not a direct measurement or a guaranteed future value.')
            else:
                st.info(f'**What this means:** {pred} is the model\'s estimated AQI approximately one hour from the current reading. Reliability cannot be assessed locally because there are not enough usable historical observations.')
        st.markdown("### AQI trend · 7 days")
        fig=go.Figure(); fig.add_trace(go.Scatter(x=hist["time"],y=hist["aqi"],mode="lines",name="AQI",line=dict(width=2))); fig.add_hline(y=200,line_dash="dot",annotation_text="Poor threshold"); fig.add_hline(y=300,line_dash="dot",annotation_text="Very poor threshold"); fig.update_layout(height=330,margin=dict(l=0,r=0,t=15,b=0),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(color="#9eb2c7"),xaxis=dict(showgrid=False),yaxis=dict(title="AQI",showgrid=True)); st.plotly_chart(fig,width="stretch",config={"scrollZoom":False,"displayModeBar":False,"displaylogo":False})
        left,right=st.columns(2)
        with left:
            st.markdown("### AQI distribution"); figd=go.Figure(); figd.add_trace(go.Histogram(x=hist["aqi"],nbinsx=20,name="AQI observations")); figd.update_layout(height=300,margin=dict(l=0,r=0,t=15,b=0),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(color="#9eb2c7"),xaxis=dict(title="AQI",showgrid=False),yaxis=dict(title="Hours",showgrid=True)); st.plotly_chart(figd,width="stretch",config={"scrollZoom":False,"displayModeBar":False})
        with right:
            st.markdown("### Typical hourly pattern"); hp=hourly_pattern(hist); figh=go.Figure(); figh.add_trace(go.Scatter(x=hp["hour"],y=hp["mean_aqi"],mode="lines+markers",name="Mean AQI")); figh.update_layout(height=300,margin=dict(l=0,r=0,t=15,b=0),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(color="#9eb2c7"),xaxis=dict(title="Hour of day",dtick=2,showgrid=False),yaxis=dict(title="Mean AQI",showgrid=True)); st.plotly_chart(figh,width="stretch",config={"scrollZoom":False,"displayModeBar":False})
        left,right=st.columns(2)
        with left:
            st.markdown("### Particulate matter"); fig2=go.Figure(); fig2.add_trace(go.Scatter(x=hist["time"],y=hist["pm25"],mode="lines",name="PM2.5")); fig2.add_trace(go.Scatter(x=hist["time"],y=hist["pm10"],mode="lines",name="PM10")); fig2.update_layout(height=290,margin=dict(l=0,r=0,t=10,b=0),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(color="#9eb2c7"),xaxis=dict(showgrid=False),yaxis=dict(title="µg/m³",showgrid=True)); st.plotly_chart(fig2,width="stretch",config={"scrollZoom":False,"displayModeBar":False})
        with right:
            st.markdown("### Weather context"); fig3=go.Figure(); fig3.add_trace(go.Scatter(x=hist["time"],y=hist["temperature"],mode="lines",name="Temperature °C")); fig3.add_trace(go.Scatter(x=hist["time"],y=hist["humidity"],mode="lines",name="Humidity %",yaxis="y2")); fig3.update_layout(height=290,margin=dict(l=0,r=0,t=10,b=0),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(color="#9eb2c7"),xaxis=dict(showgrid=False),yaxis=dict(title="Temperature °C",showgrid=True),yaxis2=dict(title="Humidity %",overlaying="y",side="right")); st.plotly_chart(fig3,width="stretch",config={"scrollZoom":False,"displayModeBar":False})
        st.markdown("### 🚨 Anomaly detection")
        anomaly_count=int(anomalies["anomaly"].sum())
        if anomaly_count:
            st.warning(f"CloudPulse flagged {anomaly_count} unusual AQI observations in the 7-day window. These are statistical anomalies, not proof of an environmental incident.")
            cols=[c for c in ["time","aqi","rolling_baseline","robust_z","anomaly_reason"] if c in anomalies.columns]; st.dataframe(anomalies.loc[anomalies["anomaly"],cols].tail(30).sort_values("time",ascending=False),width="stretch",hide_index=True)
        else: st.success("No strong statistical AQI anomalies were detected in the 7-day window.")
        st.markdown("### 🤖 Model vs actual · local backtest")
        if bt:
            q1,q2,q3,q4,q5=st.columns(5); q1.metric("Samples",bt["n"]); q2.metric("MAE",f'{bt["mae"]:.1f}'); q3.metric("RMSE",f'{bt["rmse"]:.1f}'); q4.metric("R²",f'{bt["r2"]:.3f}'); q5.metric("Direction",f'{bt["directional_accuracy"]:.0f}%')
            pv=bt["predictions"].tail(72); figm=go.Figure(); figm.add_trace(go.Scatter(x=pv["time"],y=pv["actual_next_aqi"],mode="lines",name="Actual next-hour AQI")); figm.add_trace(go.Scatter(x=pv["time"],y=pv["predicted_next_aqi"],mode="lines",name="Model prediction")); figm.update_layout(height=320,margin=dict(l=0,r=0,t=10,b=0),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(color="#9eb2c7"),xaxis=dict(showgrid=False),yaxis=dict(title="AQI",showgrid=True)); st.plotly_chart(figm,width="stretch",config={"scrollZoom":False,"displayModeBar":False})
            st.caption(f'Reliability classification: {bt["reliability"]}. This label is a rule-based summary of the local backtest metrics; the raw MAE/RMSE/R²/directional accuracy above are the primary diagnostics.')
        else: st.info("Model-vs-actual evaluation is unavailable until a trained CloudPulse model and enough historical observations are present.")
        st.markdown("### 📊 Statistical profile")
        profile=pd.DataFrame({"Metric":["AQI","PM2.5","PM10","Temperature","Humidity","Wind"],"Mean":[hist.aqi.mean(),hist.pm25.mean(),hist.pm10.mean(),hist.temperature.mean(),hist.humidity.mean(),hist.wind.mean()],"Median":[hist.aqi.median(),hist.pm25.median(),hist.pm10.median(),hist.temperature.median(),hist.humidity.median(),hist.wind.median()],"Min":[hist.aqi.min(),hist.pm25.min(),hist.pm10.min(),hist.temperature.min(),hist.humidity.min(),hist.wind.min()],"Max":[hist.aqi.max(),hist.pm25.max(),hist.pm10.max(),hist.temperature.max(),hist.humidity.max(),hist.wind.max()]})
        st.dataframe(profile.style.format({"Mean":"{:.2f}","Median":"{:.2f}","Min":"{:.2f}","Max":"{:.2f}"}),width="stretch",hide_index=True)
        st.markdown("### Feature relationships"); st.dataframe(hist[["aqi","pm25","pm10","temperature","humidity","wind"]].corr().round(2),width="stretch")
        st.markdown("### Data quality & model diagnostics")
        missing=int(hist.isna().sum().sum())
        diag=st.session_state.get("model_diagnostic", {}) or {}
        diag_status=str(diag.get("status", "Not checked"))
        if diag_status in ("Trained", "ML prediction ready"):
            model_type="Random Forest Regressor"
            feature_list=diag.get("features") or MODEL_FEATURES
            source_label="Local historical ML model"
        elif diag_status == "External model loaded":
            package=load_model_package()
            model_type=type(package["model"]).__name__ if package else "External model"
            feature_list=package.get("features", MODEL_FEATURES) if package else MODEL_FEATURES
            source_label="Packaged model"
        elif diag_status == "Forecast fallback":
            model_type="Forecast fallback"
            feature_list=[]
            source_label="Open-Meteo air-quality forecast"
        elif diag_status in ("Training failed", "History fetch failed", "Insufficient history", "Forecast unavailable", "Prediction pipeline failed", "ML prediction failed"):
            model_type="Unavailable"
            feature_list=[]
            source_label="No ML model available"
        else:
            model_type="Not checked"
            feature_list=[]
            source_label="Not checked"
        feature_names=", ".join(feature_list) if feature_list else "—"
        q1,q2,q3,q4=st.columns(4)
        q1.metric("Hourly rows",len(hist))
        q2.metric("Missing values",missing)
        q3.metric("Model",model_type)
        q4.metric("Feature count",len(feature_list))
        st.caption(f"Prediction source: {source_label} · Status: {diag_status} · Features: {feature_names}")
        if diag.get("reason") and diag_status not in ("Trained", "ML prediction ready"):
            st.caption(f"Diagnostic: {diag['reason']}")
        st.caption("AQI is a CloudPulse PM2.5-based estimate, not an official CPCB station reading.")
        st.markdown("### 📥 Data & reports"); st.caption("Reports are generated from the selected location's 7-day hourly dataset and clearly label model estimates versus environmental inputs.")
        csv_bytes=hist.to_csv(index=False).encode("utf-8"); html_report=build_html_report(current,hist,live,pred,model_state).encode("utf-8"); dl1,dl2,dl3=st.columns(3); safe_name="".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in current["name"])[:50]
        dl1.download_button("⬇ Download CSV dataset",data=csv_bytes,file_name=f"cloudpulse_{safe_name}_7day.csv",mime="text/csv",width="stretch"); dl2.download_button("⬇ Download HTML report",data=html_report,file_name=f"cloudpulse_{safe_name}_pro_report.html",mime="text/html",width="stretch")
        if SimpleDocTemplate is not None:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".pdf",delete=False) as tf: pdf_path=tf.name
            try:
                build_pdf_report(pdf_path,current,hist,live,pred,model_state); pdf_data=open(pdf_path,"rb").read(); dl3.download_button("⬇ Download executive PDF",data=pdf_data,file_name=f"cloudpulse_{safe_name}_executive_report.pdf",mime="application/pdf",width="stretch")
            finally:
                try: os.remove(pdf_path)
                except Exception: pass
        else: dl3.caption("Install reportlab for PDF export.")
    except Exception as exc: st.error(f"Pro analysis unavailable: {exc}")

# Footer
st.markdown('<div class="cp-note" style="margin-top:18px">CloudPulse uses Open-Meteo modelled weather/air-quality data. AQI is a CloudPulse PM2.5-based estimate, not an official CPCB station reading. Place suggestions use Photon with local-result bias and client-side caching. Technical diagnostics stay inside System Status.</div>',unsafe_allow_html=True)
