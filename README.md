# Digital Twin Wildfire Monitoring - Ucayali, Peru

Interactive Streamlit dashboard for wildfire monitoring in Ucayali, Peru.

## Features

- **Fire Map**: Interactive map with NASA FIRMS hotspots (2020-present), filterable by year layer
- **Temporal Analysis**: Monthly trends, yearly comparison, seasonal patterns, month x year heatmap
- **Weather & Risk**: 7-day weather forecast (Open-Meteo), fire risk index, multi-station comparison (Pucallpa, Aguaytia, Atalaya, Calleria)
- **Vegetation & Terrain**: Vegetation dryness index, hotspot zones, terrain risk zones
- **Executive Summary**: Key statistics, risk assessment, high-risk coordinates

## Run locally

```bash
pip install -r requirements.txt
streamlit run app_fire_dt_ucayali.py
```

## Data sources

- **NASA FIRMS** (Fire Information for Resource Management System): thermal anomaly hotspots from VIIRS S-NPP, NOAA-20, NOAA-21 and MODIS satellites
- **Open-Meteo**: weather forecast API (free, no key required)

## Update data

Click the "Update FIRMS Data" button to fetch the latest hotspots.

In cloud deployment, set the FIRMS API key in Streamlit Secrets as `FIRMS_API_KEY` (free key from https://firms.modaps.eosdis.nasa.gov/api/area/).

## Deploy

1. Push this repo to GitHub
2. Go to https://share.streamlit.io
3. Connect your GitHub account and select the repo
4. Add the `FIRMS_API_KEY` secret
5. Deploy
