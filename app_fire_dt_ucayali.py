# -*- coding: utf-8 -*-
"""
DIGITAL TWIN WILDFIRE MONITORING - STREAMLIT APP
=================================================
Interactive dashboard for wildfire risk monitoring in Ucayali, Peru.

Features:
  - FIRMS hotspot data (2020-present) with incremental daily updates
  - Multi-station weather forecast (Pucallpa, Aguaytia, Atalaya, Calleria)
  - Vegetation dryness index (from FIRMS brightness temperatures)
  - Terrain risk zones (based on Ucayali topography)
  - Interactive filters: date range, year, zone, sensor, confidence
  - Real-time fire risk model

Run:  streamlit run app_fire_dt_ucayali.py
"""

import os
import sys
import json
import warnings
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from folium.plugins import HeatMapWithTime
from streamlit_folium import st_folium
import requests

warnings.filterwarnings('ignore')

# Fix: folium's HeatMapWithTime computes map bounds incorrectly for time-series
# data (each time step holds many points), which breaks streamlit_folium's
# internal get_bounds() call. Patch the method once at startup.
from folium.plugins import heat_map_withtime

def _heatmap_self_bounds(self):
    bounds = [[None, None], [None, None]]
    for step in self.data:
        for pt in step:
            if len(pt) < 2:
                continue
            lat, lon = pt[0], pt[1]
            bounds[0][0] = lat if bounds[0][0] is None else min(bounds[0][0], lat)
            bounds[0][1] = lon if bounds[0][1] is None else min(bounds[0][1], lon)
            bounds[1][0] = lat if bounds[1][0] is None else max(bounds[1][0], lat)
            bounds[1][1] = lon if bounds[1][1] is None else max(bounds[1][1], lon)
    return bounds

heat_map_withtime.HeatMapWithTime._get_self_bounds = _heatmap_self_bounds

# =============================================================================
# CONFIGURATION
# =============================================================================

LAT_MIN, LAT_MAX = -11.5, -7.0
LON_MIN, LON_MAX = -76.0, -72.5
DATE_START = '2020-01-01'

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
OUTPUTS_DIR = BASE_DIR / 'outputs'
DATA_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

CSV_PATH = DATA_DIR / 'firms_ucayali_2020_hoy.csv.gz'
UPDATE_LOG = DATA_DIR / 'last_update.json'

def get_firms_api_key():
    """Get FIRMS API key from Streamlit secrets, env var, or local file."""
    try:
        if st.secrets is not None and 'FIRMS_API_KEY' in st.secrets:
            return st.secrets['FIRMS_API_KEY']
    except Exception:
        pass
    key = os.environ.get('FIRMS_MAP_KEY')
    if key:
        return key
    key_file = DATA_DIR / 'firms_api_key.txt'
    if key_file.exists():
        return key_file.read_text().strip()
    return None

MONTH_NAMES = {
    1: 'January', 2: 'February', 3: 'March', 4: 'April',
    5: 'May', 6: 'June', 7: 'July', 8: 'August',
    9: 'September', 10: 'October', 11: 'November', 12: 'December'
}
MONTH_SHORT = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dic']

# Weather stations in Ucayali
WEATHER_STATIONS = {
    'Pucallpa': {'lat': -8.38, 'lon': -74.57, 'elevation': 154},
    'Aguaytia': {'lat': -8.55, 'lon': -74.35, 'elevation': 200},
    'Atalaya': {'lat': -10.67, 'lon': -73.63, 'elevation': 250},
    'Calleria': {'lat': -8.42, 'lon': -74.53, 'elevation': 160},
}

# Zone definitions for Ucayali
ZONES = {
    'Coronel Portillo - Pucallpa': {'lat_range': (-9.5, -8.0), 'lon_range': (-76.0, -74.5)},
    'Coronel Portillo - East': {'lat_range': (-9.5, -8.0), 'lon_range': (-74.5, -72.5)},
    'Padre Abad - Aguaytia': {'lat_range': (-7.0, -8.5), 'lon_range': (-76.0, -74.0)},
    'Atalaya - Ucayali': {'lat_range': (-11.5, -9.5), 'lon_range': (-76.0, -74.0)},
    'Atalaya - Purus': {'lat_range': (-11.5, -9.5), 'lon_range': (-74.0, -72.5)},
    'Southern Border': {'lat_range': (-11.5, -10.5), 'lon_range': (-76.0, -72.5)},
}

PALETTE_YEAR = {
    2020: '#3498db', 2021: '#e74c3c', 2022: '#f39c12',
    2023: '#27ae60', 2024: '#9b59b6', 2025: '#e67e22', 2026: '#1abc9c'
}

# =============================================================================
# DATA LOADING & UPDATING
# =============================================================================

def assign_zone(lat, lon):
    if pd.isna(lat) or pd.isna(lon):
        return 'No data'
    for zone_name, bounds in ZONES.items():
        if bounds['lat_range'][0] <= lat <= bounds['lat_range'][1]:
            if bounds['lon_range'][0] <= lon <= bounds['lon_range'][1]:
                return zone_name
    if lat > -8.0 and lon >= -74.0:
        return 'Padre Abad - Aguaytia'
    if lat < -10.5:
        return 'Southern Border'
    return 'Other Ucayali'


def load_firms_data():
    """Load FIRMS data from CSV or download fresh."""
    if CSV_PATH.exists():
        for attempt in range(3):
            try:
                df = pd.read_csv(CSV_PATH, compression='gzip', low_memory=False)
                break
            except (PermissionError, OSError) as e:
                if attempt == 2:
                    st.error(f"Cannot read data file. It may be open in another program. "
                             f"Close it and try again.\n\nError: {e}")
                    st.stop()
                time.sleep(1)
        df['acq_date'] = pd.to_datetime(df['acq_date'])
        if 'zone' not in df.columns:
            df['zone'] = df.apply(lambda r: assign_zone(r['latitude'], r['longitude']), axis=1)
        if 'veg_dryness' not in df.columns:
            if 'bright_t31' in df.columns and 'brightness' in df.columns:
                df['veg_dryness'] = np.where(
                    df['bright_t31'] > 0,
                    df['brightness'] / df['bright_t31'],
                    1.0
                ).clip(0.8, 2.0)
        return df
    return None


def fetch_firms_api(days_back=10):
    """Download recent FIRMS NRT data via the FIRMS API (works on cloud)."""
    api_key = get_firms_api_key()
    if not api_key:
        return None

    sources = ['VIIRS_SNPP_NRT', 'VIIRS_NOAA20_NRT', 'VIIRS_NOAA21_NRT', 'MODIS_NRT']
    all_dfs = []
    bbox = f"{LON_MIN},{LAT_MIN},{LON_MAX},{LAT_MAX}"

    for src in sources:
        url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{api_key}/{src}/{bbox}/{days_back}"
        try:
            r = requests.get(url, timeout=30)
            if r.status_code != 200:
                continue
            df = pd.read_csv(pd.io.common.StringIO(r.text), low_memory=False)
            df['_source'] = src.replace('_NRT', '')
            all_dfs.append(df)
        except Exception as e:
            print(f"FIRMS API error for {src}: {e}")

    if not all_dfs:
        return None

    combined = pd.concat(all_dfs, ignore_index=True)
    combined.columns = [c.lower() for c in combined.columns]
    for col in ['latitude', 'longitude']:
        combined[col] = pd.to_numeric(combined[col], errors='coerce')
    combined['acq_date'] = pd.to_datetime(combined['acq_date'], errors='coerce')
    combined = combined.dropna(subset=['acq_date'])
    mask = (
        (combined['latitude'] >= LAT_MIN) & (combined['latitude'] <= LAT_MAX) &
        (combined['longitude'] >= LON_MIN) & (combined['longitude'] <= LON_MAX)
    )
    combined = combined[mask].copy()

    # Normalize column names to match archive data schema
    col_map = {'bright_ti4': 'brightness', 'bright_ti5': 'bright_t31'}
    combined = combined.rename(columns=col_map)
    combined = combined.loc[:, ~combined.columns.duplicated()]
    return combined


def update_firms_data():
    """Update FIRMS data: from local DBF files (dev) or FIRMS API (cloud)."""
    # Check last update
    last_date = DATE_START
    if UPDATE_LOG.exists():
        with open(UPDATE_LOG) as f:
            info = json.load(f)
            last_date = info.get('last_date', DATE_START)

    # Load existing CSV
    if CSV_PATH.exists():
        existing = pd.read_csv(CSV_PATH, compression='gzip', low_memory=False)
        existing['acq_date'] = pd.to_datetime(existing['acq_date'])
        last_csv_date = existing['acq_date'].max().strftime('%Y-%m-%d')
        if last_csv_date > last_date:
            last_date = last_csv_date

    # FIRMS files (archive + NRT) - check if local DBF files exist
    firns_files = {
        'VIIRS_SNPP': [
            BASE_DIR / 'DL_FIRE_SV-C2_769837' / 'fire_archive_SV-C2_769837.dbf',
            BASE_DIR / 'DL_FIRE_SV-C2_769837' / 'fire_nrt_SV-C2_769837.dbf',
        ],
        'VIIRS_NOAA20': [
            BASE_DIR / 'DL_FIRE_J1V-C2_769835' / 'fire_archive_J1V-C2_769835.dbf',
            BASE_DIR / 'DL_FIRE_J1V-C2_769835' / 'fire_nrt_J1V-C2_769835.dbf',
        ],
        'VIIRS_NOAA21': [
            BASE_DIR / 'DL_FIRE_J2V-C2_769836' / 'fire_nrt_J2V-C2_769836.dbf',
        ],
        'MODIS': [
            BASE_DIR / 'DL_FIRE_M-C61_769833' / 'fire_archive_M-C61_769833.dbf',
            BASE_DIR / 'DL_FIRE_M-C61_769833' / 'fire_nrt_M-C61_769833.dbf',
        ],
    }
    any_dbf = any(fp.exists() for paths in firns_files.values() for fp in paths)

    if any_dbf:
        # Local mode: rebuild full dataset from DBF files
        try:
            from dbfread import DBF
        except ImportError:
            os.system(f'"{sys.executable}" -m pip install dbfread -q')
            from dbfread import DBF
        all_dfs = []
        for name, fpaths in firns_files.items():
            for fpath in fpaths:
                if fpath.exists():
                    df = DBF(str(fpath), encoding='latin-1', raw=False)
                    df = pd.DataFrame(iter(df))
                    df['_source'] = name
                    df.columns = [c.lower() for c in df.columns]
                    all_dfs.append(df)

        if all_dfs:
            combined = pd.concat(all_dfs, ignore_index=True)

            # Filter Ucayali
            for col in ['latitude', 'longitude']:
                combined[col] = pd.to_numeric(combined[col], errors='coerce')
            mask = (
                (combined['latitude'] >= LAT_MIN) & (combined['latitude'] <= LAT_MAX) &
                (combined['longitude'] >= LON_MIN) & (combined['longitude'] <= LON_MAX)
            )
            combined = combined[mask].copy()
            combined['acq_date'] = pd.to_datetime(combined['acq_date'], errors='coerce')
            combined = combined.dropna(subset=['acq_date'])
            combined = combined[combined['acq_date'] >= DATE_START].copy()

            # Deduplicate by (lat, lon, acq_date, _source)
            before = len(combined)
            combined = combined.drop_duplicates(subset=['latitude', 'longitude', 'acq_date', '_source'])
            dupes = before - len(combined)
            if dupes:
                print(f"Removed {dupes} duplicate records (archive <-> NRT overlap)")
    else:
        # Cloud mode: use FIRMS API for recent data, merge with existing CSV
        new_data = fetch_firms_api(days_back=10)
        if new_data is None:
            st.warning("No local data files found and FIRMS API unavailable. "
                       "Set the FIRMS API key in Streamlit secrets (FIRMS_API_KEY).")
            return None
        new_data = new_data[new_data['acq_date'] >= DATE_START].copy()

        if CSV_PATH.exists():
            try:
                existing = pd.read_csv(CSV_PATH, compression='gzip', low_memory=False)
                existing['acq_date'] = pd.to_datetime(existing['acq_date'])
                combined = pd.concat([existing, new_data], ignore_index=True)
                combined = combined.drop_duplicates(
                    subset=['latitude', 'longitude', 'acq_date', '_source'])
            except Exception:
                combined = new_data
        else:
            combined = new_data

        # Add derived columns for API data
        for col in ['year', 'month', 'year_month_str']:
            if col not in combined.columns:
                if col == 'year':
                    combined['year'] = combined['acq_date'].dt.year
                elif col == 'month':
                    combined['month'] = combined['acq_date'].dt.month
                else:
                    combined['year_month_str'] = combined['acq_date'].dt.strftime('%Y-%m')
        if 'zone' not in combined.columns:
            combined['zone'] = combined.apply(lambda r: assign_zone(r['latitude'], r['longitude']), axis=1)

    # Add derived columns
    combined['year'] = combined['acq_date'].dt.year
    combined['month'] = combined['acq_date'].dt.month
    combined['year_month_str'] = combined['acq_date'].dt.strftime('%Y-%m')
    combined['zone'] = combined.apply(lambda r: assign_zone(r['latitude'], r['longitude']), axis=1)

    # Vegetation dryness index (from brightness temps)
    if 'bright_t31' in combined.columns and 'brightness' in combined.columns:
        combined['veg_dryness'] = np.where(
            combined['bright_t31'] > 0,
            combined['brightness'] / combined['bright_t31'],
            1.0
        )
        combined['veg_dryness'] = combined['veg_dryness'].clip(0.8, 2.0)

    combined.to_csv(CSV_PATH, index=False, compression='gzip')

    # Update log
    with open(UPDATE_LOG, 'w') as f:
        json.dump({
            'last_date': combined['acq_date'].max().strftime('%Y-%m-%d'),
            'updated_at': datetime.now().isoformat(),
            'total_records': len(combined)
        }, f)

    return combined


def get_weather_forecast(station_name, forecast_days=7):
    """Get weather forecast for a station."""
    station = WEATHER_STATIONS.get(station_name)
    if station is None:
        return None

    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            'latitude': station['lat'],
            'longitude': station['lon'],
            'daily': 'temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max,relative_humidity_2m_min',
            'timezone': 'America/Lima',
            'forecast_days': forecast_days
        }
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        daily = data.get('daily', {})

        df = pd.DataFrame({
            'date': pd.to_datetime(daily.get('time', [])),
            'temp_max': daily.get('temperature_2m_max', []),
            'temp_min': daily.get('temperature_2m_min', []),
            'precip_mm': daily.get('precipitation_sum', []),
            'wind_max_kmh': daily.get('wind_speed_10m_max', []),
            'rh_min_pct': daily.get('relative_humidity_2m_min', []),
        })

        # Compute fire risk
        df['fire_risk'] = 0.0
        for i, row in df.iterrows():
            score = 0.0
            if row['temp_max'] > 30: score += 0.3
            if row['temp_max'] > 33: score += 0.2
            if row['rh_min_pct'] < 50: score += 0.2
            if row['rh_min_pct'] < 30: score += 0.1
            if row['wind_max_kmh'] > 20: score += 0.1
            if row['wind_max_kmh'] > 30: score += 0.1
            if row['precip_mm'] > 0: score -= 0.3
            df.at[i, 'fire_risk'] = max(0.0, min(1.0, score))

        df['station'] = station_name
        return df

    except Exception:
        return None


def compute_spatial_risk(df, resolution=0.1):
    """Compute spatial risk grid from recent hotspots."""
    now = pd.Timestamp.now()
    last_30d = now - timedelta(days=30)
    recent = df[(df['acq_date'] >= last_30d) & (df['acq_date'] <= now)].copy()

    if len(recent) == 0:
        return pd.DataFrame()

    recent['lat_bin'] = (recent['latitude'] / resolution).round() * resolution
    recent['lon_bin'] = (recent['longitude'] / resolution).round() * resolution

    density = recent.groupby(['lat_bin', 'lon_bin']).agg(
        hotspot_count=('latitude', 'count'),
        avg_frp=('frp', 'mean') if 'frp' in recent.columns else ('latitude', 'count'),
        avg_veg_dry=('veg_dryness', 'mean') if 'veg_dryness' in recent.columns else ('latitude', 'count'),
    ).reset_index()

    max_count = density['hotspot_count'].max()
    density['risk_score'] = density['hotspot_count'] / max_count if max_count > 0 else 0

    return density


def build_heatmap_time_data(df, granularity='W', max_points=8000):
    """Build (data, index) for HeatMapWithTime: a list of time steps, each with
    [lat, lon, intensity] points where intensity = fire radiative power (FRP).
    Sampling is balanced across time buckets so every step is represented.
    """
    if df.empty:
        return [], []

    df_sorted = df.sort_values('acq_date').copy()
    df_sorted['_bucket'] = df_sorted['acq_date'].dt.to_period(granularity)

    n_buckets = df_sorted['_bucket'].nunique()
    per_bucket = max(1, int(max_points / n_buckets))

    if 'frp' in df_sorted.columns:
        df_sorted['_intensity'] = df_sorted['frp'].fillna(0).clip(lower=0.5)
    else:
        df_sorted['_intensity'] = 1.0

    data, index = [], []
    label_fmt = '%Y-%m' if granularity == 'M' else '%Y-%m-%d'
    for b, g in df_sorted.groupby('_bucket'):
        sample = g.sample(min(per_bucket, len(g)), random_state=42)
        pts = [[lat, lon, intensity] for lat, lon, intensity in
               zip(sample['latitude'], sample['longitude'], sample['_intensity'])]
        data.append(pts)
        index.append(b.start_time.strftime(label_fmt))

    return data, index


# =============================================================================
# STREAMLIT APP
# =============================================================================

def main():
    st.set_page_config(
        page_title="Fire DT - Ucayali",
        page_icon="🔥",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS
    st.markdown("""
    <style>
    .main-header { font-size: 2.5rem; font-weight: bold; color: #e74c3c; }
    .metric-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                   padding: 20px; border-radius: 10px; color: white; text-align: center; }
    .risk-low { color: #27ae60; font-weight: bold; }
    .risk-moderate { color: #f39c12; font-weight: bold; }
    .risk-high { color: #e74c3c; font-weight: bold; }
    .risk-very-high { color: #8b0000; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<p class="main-header">Digital Twin Wildfire Monitoring</p>', unsafe_allow_html=True)
    st.markdown("**Ucayali Region, Peru** | FIRMS Satellite Data + Weather Forecast + Risk Model")
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.header("Filters")

        # Update button
        if st.button("Update FIRMS Data", type="primary"):
            with st.spinner("Updating FIRMS data..."):
                update_firms_data()
            st.success("Data updated!")
            st.rerun()

        if UPDATE_LOG.exists():
            with open(UPDATE_LOG) as f:
                info = json.load(f)
            st.caption(f"Last update: {info.get('updated_at', 'N/A')[:16]}")
            st.caption(f"Total records: {info.get('total_records', 0):,}")

        st.caption("**Data age**: The FIRMS archive lags ~3 months behind real-time. "
                   "NRT (near real-time) data covers recent weeks. "
                   "'Last 30 Days' is relative to the most recent data record, not today's date.")

        st.markdown("---")

        # Load data
        df = load_firms_data()
        if df is None:
            st.error("No data found. Click 'Update FIRMS Data' above.")
            st.stop()

        # Filter: Date range (sole temporal filter)
        min_d = df['acq_date'].min().date()
        max_d = df['acq_date'].max().date()
        date_range = st.date_input(
            "Date Range",
            value=(max_d - timedelta(days=90), max_d),
            min_value=min_d,
            max_value=max_d,
            help="Select the start and end dates. Years are automatically derived from this range."
        )

        # Filter: Zone
        zones = sorted(df['zone'].unique())
        zone_opts = ["All Zones"] + zones
        selected_zone = st.selectbox("Zone", zone_opts,
                                     help="Choose a specific zone or 'All Zones' to include everything.")

        # Filter: Sensor
        sensors = sorted(df['_source'].unique())
        sensor_help_text = (
            "**VIIRS_SNPP**: Suomi-NPP (2012-present) - 375m, day/night\n"
            "**VIIRS_NOAA20**: NOAA-20 (2018-present) - 375m, successor to SNPP\n"
            "**VIIRS_NOAA21**: NOAA-21 (2024-present) - 375m, latest VIIRS sensor\n"
            "**MODIS**: Terra/Aqua (2000-present) - 1km, long-term record"
        )
        sensor_opts = ["All Sensors"] + sensors
        selected_sensor = st.selectbox("Satellite Sensor", sensor_opts,
                                       help=sensor_help_text)

        st.markdown("---")

    # Apply filters
    mask = pd.Series([True] * len(df))
    if date_range:
        if isinstance(date_range, (tuple, list)) and len(date_range) == 2 and all(date_range):
            mask &= (df['acq_date'].dt.date >= date_range[0]) & (df['acq_date'].dt.date <= date_range[1])
        elif not isinstance(date_range, (tuple, list)) and date_range:
            mask &= (df['acq_date'].dt.date == date_range)
    if selected_zone != "All Zones":
        mask &= (df['zone'] == selected_zone)
    if selected_sensor != "All Sensors":
        mask &= (df['_source'] == selected_sensor)

    filtered = df[mask].copy()

    # ---- TOP METRICS ----
    st.markdown("## Key Metrics")
    col1, col2, col3, col4, col5 = st.columns(5)

    total_hotspots = len(filtered)
    max_data_date = filtered['acq_date'].max()
    last_30d_count = len(filtered[filtered['acq_date'] >= max_data_date - timedelta(days=30)])
    active_zones = filtered['zone'].nunique()
    peak_month = filtered.groupby('month').size().idxmax() if len(filtered) > 0 else 0
    avg_veg = filtered['veg_dryness'].mean() if 'veg_dryness' in filtered.columns else 0

    col1.metric("Total Hotspots", f"{total_hotspots:,}")
    col2.metric("Last 30 Days", f"{last_30d_count:,}",
                help=f"Hotspots detected in the 30 days before the last recorded date ({max_data_date.date()}). "
                      f"The count of recent fires, relative to the most recent satellite observation.")
    col3.metric("Active Zones", active_zones)
    col4.metric("Peak Month", MONTH_SHORT[peak_month - 1] if peak_month > 0 else "N/A")
    col5.metric("Veg. Dryness", f"{avg_veg:.2f}")

    st.caption(f"Last data record: {max_data_date.date()}. "
               f"'Last 30 Days' counts hotspots in the 30 days before that date.")

    st.markdown("---")

    # ---- TABS ----
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Fire Map", "Temporal Analysis", "Weather & Risk",
        "Vegetation & Terrain", "Executive Summary"
    ])

    # ---- TAB 1: FIRE MAP ----
    with tab1:
        st.subheader("Fire Hotspot Map")
        st.caption("Interactive map showing all detected fire hotspots. Click markers for details.")

        m = folium.Map(location=[-8.5, -74.5], zoom_start=7, tiles='CartoDB positron')

        # Add markers
        for year in sorted(filtered['year'].unique()):
            year_df = filtered[filtered['year'] == year]
            fg = folium.FeatureGroup(name=f'{year} ({len(year_df):,})')
            sample = year_df.sample(min(500, len(year_df)), random_state=42)

            for _, row in sample.iterrows():
                conf = str(row.get('confidence', '')).lower()
                color = 'red' if conf in ['high', 'nominal', 'h', 'n'] else 'orange'
                popup = f"""
                <b>{row['acq_date'].strftime('%d/%m/%Y')}</b><br>
                <b>FRP:</b> {row.get('frp', 'N/A')} MW<br>
                <b>Confidence:</b> {row.get('confidence', 'N/A')}<br>
                <b>Sensor:</b> {row.get('_source', 'N/A')}<br>
                <b>Zone:</b> {row.get('zone', 'N/A')}<br>
                <b>Coords:</b> {row['latitude']:.4f}, {row['longitude']:.4f}
                """
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=2, color=color, fill=True, fill_opacity=0.5,
                    popup=folium.Popup(popup, max_width=250)
                ).add_to(fg)
            fg.add_to(m)

        folium.LayerControl().add_to(m)

        # Legend
        legend_html = """
        <div style="position: fixed; bottom: 30px; left: 30px; z-index: 1000;
             background-color: white; padding: 12px; border-radius: 8px;
             border: 2px solid #333; font-family: Arial; font-size: 12px;
             box-shadow: 2px 2px 6px rgba(0,0,0,0.3);">
            <b>LEGEND</b><br>
            <span style="color: red;">●</span> High/Nominal confidence<br>
            <span style="color: orange;">●</span> Low confidence<br>
            <span style="font-size:10px; color:#666;">FRP = Fire Radiative Power (MW)<br>
            Toggle years in layer control</span>
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))

        st_folium(m, width=1200, height=600)

        st.markdown("---")
        st.subheader("Animated Fire Heat Map")
        st.caption("Watch how fire intensity (heat) concentrates and spreads across Ucayali over time. "
                   "Colors go from blue (low intensity) to red (very intense fires), so you can see "
                   "where and when fire fronts concentrate.")

        span_days = (filtered['acq_date'].max() - filtered['acq_date'].min()).days
        if span_days <= 120:
            default_anim = "Day"
        elif span_days <= 730:
            default_anim = "Week"
        else:
            default_anim = "Month"

        granularity_opts = ["Day", "Week", "Month"]
        anim_granularity = st.selectbox(
            "Animation granularity",
            granularity_opts,
            index=granularity_opts.index(default_anim),
            help="'Day' shows fine detail for short periods, 'Week' balances detail and speed, "
                 "'Month' is best for multi-year overviews."
        )

        bucket_map = {"Day": 'D', "Week": 'W', "Month": 'M'}
        heat_data, heat_index = build_heatmap_time_data(filtered, granularity=bucket_map[anim_granularity])

        if heat_data:
            m_anim = folium.Map(location=[-8.5, -74.5], zoom_start=7, tiles='CartoDB positron')
            HeatMapWithTime(
                heat_data,
                index=heat_index,
                radius=25,
                blur=0.8,
                min_opacity=0.1,
                max_opacity=0.75,
                scale_radius=True,
                auto_play=False,
                display_index=True,
                position='bottomright',
                min_speed=0.1,
                max_speed=10,
            ).add_to(m_anim)
            st_folium(m_anim, width=1200, height=600)
            st.caption("Each frame = fire radiative power (FRP, MW) detected in that period. "
                       "Intensity is shown as a heat gradient: blue = low, green/yellow = moderate, "
                       "red = very intense. Use the time slider or play button to watch the evolution.")
        else:
            st.info("No data in the selected range to animate.")

    # ---- TAB 2: TEMPORAL ANALYSIS ----
    with tab2:
        st.subheader("Temporal Analysis")

        col_a, col_b = st.columns(2)

        with col_a:
            # Monthly series
            serie = filtered.groupby(filtered['acq_date'].dt.to_period('M')).size().reset_index(name='hotspots')
            serie['date'] = serie['acq_date'].dt.to_timestamp()
            fig = px.bar(serie, x='date', y='hotspots',
                         title='Monthly Hotspot Count',
                         color_discrete_sequence=['#e74c3c'])
            fig.update_layout(xaxis_title='Date', yaxis_title='Hotspots')
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Each bar represents the number of fire hotspots detected by NASA satellites in one month. "
                        "Higher bars = more fire activity.")

        with col_b:
            # By year
            by_year = filtered.groupby('year').size().reset_index(name='hotspots')
            fig = px.bar(by_year, x='year', y='hotspots',
                         title='Hotspots by Year',
                         color='year', color_discrete_map=PALETTE_YEAR)
            fig.update_layout(xaxis_title='Year', yaxis_title='Hotspots')
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Total fire hotspots detected per calendar year. "
                        "This helps identify trends across years.")

        col_c, col_d = st.columns(2)

        with col_c:
            # Seasonal pattern
            seasonal = filtered.groupby('month').size() / max(filtered['year'].nunique(), 1)
            seasonal_df = pd.DataFrame({
                'month': range(1, 13),
                'name': MONTH_SHORT,
                'avg_hotspots': [seasonal.get(m, 0) for m in range(1, 13)]
            })
            colors = ['#e74c3c' if m in [6,7,8,9,10] else '#3498db' for m in range(1,13)]
            fig = px.bar(seasonal_df, x='name', y='avg_hotspots',
                         title='Seasonal Pattern (Average by Month)',
                         color_discrete_sequence=colors)
            fig.update_layout(xaxis_title='Month', yaxis_title='Avg Hotspots')
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Red bars = dry season (Jun-Oct), Blue bars = rainy season (Nov-May). "
                        "Shows which months typically have the most fires.")

        with col_d:
            # Heatmap
            pivot = filtered.pivot_table(index='month', columns='year', values='latitude', aggfunc='count')
            pivot.index = [MONTH_SHORT[m-1] for m in pivot.index]
            fig = px.imshow(pivot, text_auto='.0f', color_continuous_scale='YlOrRd',
                           title='Heatmap: Month x Year')
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Darker red = more hotspots. Read across a row to see how the same month changed "
                        "across years. Read down a column to see fire seasonality within a year.")

        # Interannual comparison
        st.subheader("Interannual Comparison")
        inter = filtered.pivot_table(index='month', columns='year', values='latitude', aggfunc='count').fillna(0)
        fig = go.Figure()
        for year in inter.columns:
            fig.add_trace(go.Scatter(
                x=list(range(1,13)), y=inter[year].values,
                name=str(year), mode='lines+markers',
                line=dict(color=PALETTE_YEAR.get(year, '#888'), width=2)
            ))
        fig.update_layout(
            title='Same Month, Different Years',
            xaxis_title='Month', yaxis_title='Hotspots',
            xaxis=dict(tickvals=list(range(1,13)), ticktext=MONTH_SHORT)
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Each line is one year. Compare the same month across years to see if fire patterns "
                    "are shifting earlier, later, or getting more intense over time.")

    # ---- TAB 3: WEATHER & RISK ----
    with tab3:
        st.subheader("Weather Forecast & Fire Risk")

        # Station selector
        station = st.selectbox("Select Weather Station", list(WEATHER_STATIONS.keys()), index=0)

        weather_df = get_weather_forecast(station)

        if weather_df is not None:
            col_e, col_f = st.columns(2)

            with col_e:
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(
                    go.Bar(x=weather_df['date'], y=weather_df['precip_mm'],
                           name='Precipitation (mm)', marker_color='#3498db', opacity=0.6),
                    secondary_y=False
                )
                fig.add_trace(
                    go.Scatter(x=weather_df['date'], y=weather_df['temp_max'],
                              name='Temp Max', mode='lines+markers',
                              line=dict(color='#e74c3c', width=2)),
                    secondary_y=True
                )
                fig.add_trace(
                    go.Scatter(x=weather_df['date'], y=weather_df['temp_min'],
                              name='Temp Min', mode='lines+markers',
                              line=dict(color='#3498db', width=2, dash='dash')),
                    secondary_y=True
                )
                fig.update_layout(title=f'Weather Forecast - {station}')
                fig.update_yaxes(title_text="Precipitation (mm)", secondary_y=False)
                fig.update_yaxes(title_text="Temperature (C)", secondary_y=True)
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Bars = rain, Lines = temperature. "
                           "More rain lowers fire risk. Higher temperatures (especially above 30C) increase it.")

            with col_f:
                risk_colors = []
                risk_labels = []
                for r in weather_df['fire_risk']:
                    if r >= 0.75: risk_colors.append('#8b0000'); risk_labels.append('VERY HIGH')
                    elif r >= 0.5: risk_colors.append('#e74c3c'); risk_labels.append('HIGH')
                    elif r >= 0.25: risk_colors.append('#f39c12'); risk_labels.append('MODERATE')
                    else: risk_colors.append('#27ae60'); risk_labels.append('LOW')

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=weather_df['date'], y=weather_df['fire_risk'],
                    marker_color=risk_colors,
                    text=risk_labels, textposition='outside'
                ))
                fig.add_hrect(y0=0, y1=0.25, fillcolor='green', opacity=0.1, line_width=0)
                fig.add_hrect(y0=0.25, y1=0.5, fillcolor='yellow', opacity=0.1, line_width=0)
                fig.add_hrect(y0=0.5, y1=0.75, fillcolor='orange', opacity=0.1, line_width=0)
                fig.add_hrect(y0=0.75, y1=1.0, fillcolor='red', opacity=0.1, line_width=0)
                fig.add_hline(y=0.5, line_dash='dash', line_color='red', opacity=0.7)
                fig.update_layout(
                    title='Fire Risk Index - 7 Day Forecast',
                    yaxis_title='Risk Index (0-1)', yaxis_range=[0, 1.1]
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption("How the risk score works: "
                           "**Temp>30C** (+0.3), **Temp>33C** (+0.2), "
                           "**Humidity<50%** (+0.2), **Wind>20km/h** (+0.1), "
                           "**Rain>0mm** (-0.3). "
                           "Score range: 0 (safe) to 1 (extreme).")

            # Multi-station comparison
            st.subheader("Multi-Station Comparison")
            all_weather = []
            for s in WEATHER_STATIONS:
                wdf = get_weather_forecast(s)
                if wdf is not None:
                    all_weather.append(wdf)

            if all_weather:
                combined_weather = pd.concat(all_weather, ignore_index=True)
                fig = px.line(combined_weather, x='date', y='fire_risk', color='station',
                             title='Fire Risk Comparison Across Stations',
                             markers=True)
                fig.add_hline(y=0.5, line_dash='dash', line_color='red', opacity=0.5)
                fig.update_layout(yaxis_title='Fire Risk', yaxis_range=[0, 1.1])
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Compare fire risk across 4 Ucayali stations. "
                           "Pucallpa and Aguaytia are lower elevation; Atalaya is further south. "
                           "Differences help identify where conditions are most dangerous.")

        else:
            st.warning("Could not fetch weather data. Check internet connection.")

    # ---- TAB 4: VEGETATION & TERRAIN ----
    with tab4:
        st.subheader("Vegetation & Terrain Analysis")

        col_g, col_h = st.columns(2)

        with col_g:
            # Vegetation dryness distribution
            if 'veg_dryness' in filtered.columns:
                fig = px.histogram(filtered, x='veg_dryness', nbins=50,
                                  title='Vegetation Dryness Index Distribution',
                                  color_discrete_sequence=['#27ae60'])
                fig.add_vline(x=1.2, line_dash='dash', line_color='red',
                             annotation_text='Dry threshold')
                fig.update_layout(xaxis_title='Dryness Ratio (brightness/T31)',
                                 yaxis_title='Count')
                st.plotly_chart(fig, use_container_width=True)

                st.info("""
                **Vegetation Dryness Index** = Brightness Temperature / Band T31

                - **< 1.1**: Healthy vegetation (high moisture content)
                - **1.1 - 1.3**: Moderate dryness (some fire risk)
                - **> 1.3**: Very dry vegetation (high fire risk)

                Higher values = drier vegetation = more fuel for fires
                """)
            else:
                st.warning("Vegetation dryness data not available in current dataset.")

        with col_h:
            # Risk by zone
            zone_risk = filtered.groupby('zone').agg(
                hotspots=('latitude', 'count'),
                avg_frp=('frp', 'mean') if 'frp' in filtered.columns else ('latitude', 'count'),
                avg_dryness=('veg_dryness', 'mean') if 'veg_dryness' in filtered.columns else ('latitude', 'count'),
            ).reset_index().sort_values('hotspots', ascending=True)

            fig = px.bar(zone_risk, x='hotspots', y='zone', orientation='h',
                          title='Hotspots by Zone',
                          color='hotspots', color_continuous_scale='YlOrRd')
            fig.update_layout(xaxis_title='Total Hotspots', yaxis_title='Zone')
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Which zones have the most fire activity. "
                       "Pucallpa has the highest due to agricultural burning and proximity to roads.")

        # Terrain risk zones (based on topography)
        st.subheader("Terrain Risk Zones")
        st.caption("Based on Ucayali topography: lowlands have more agriculture/pasture fires, highlands have forest fires")

        terrain_data = pd.DataFrame({
            'Zone': ['Lowland Pucallpa\n(Agriculture)', 'Eastern Lowlands\n(Forest edge)',
                     'Western Highlands\n(Padre Abad)', 'Atalaya Basin\n(Purus river)',
                     'Southern Border\n(Deep forest)'],
            'Elevation (m)': [154, 200, 800, 250, 500],
            'Fire Type': ['Agricultural\nburning', 'Deforestation\nfires',
                         'Forest fires\n(slope-driven)', 'Riparian\nfires', 'Wildland\nfires'],
            'Risk Factor': [0.7, 0.8, 0.5, 0.6, 0.4]
        })

        fig = px.scatter(terrain_data, x='Elevation (m)', y='Risk Factor',
                         size='Elevation (m)', color='Fire Type',
                         title='Fire Risk by Terrain Type',
                         hover_data=['Zone'])
        fig.update_layout(yaxis_title='Risk Factor', yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Each point is a terrain type. Lower elevation areas (Pucallpa, Eastern lowlands) "
                   "have higher risk due to agriculture, roads, and human activity. "
                   "Higher elevation forest areas have lower risk but more intense fires when they occur.")

        # DEM note
        st.info("""
        **For full DEM/slope analysis**, download SRTM 30m data from:
        - [NASA Earthdata](https://earthdata.nasa.gov/)
        - [USGS EarthExplorer](https://earthexplorer.usgs.gov/)

        Coverage: -12 to -7 lat, -77 to -72 lon (full Ucayali region)
        """)

    # ---- TAB 5: EXECUTIVE SUMMARY ----
    with tab5:
        st.subheader("Executive Summary")

        total = len(filtered)
        years = sorted(filtered['year'].unique())

        if total == 0:
            st.warning("No data matches the selected filters. Try adjusting your selection.")
            st.stop()

        period_str = f"{years[0]}" if len(years) == 1 else f"{years[0]} - {years[-1]}"

        col_i, col_j = st.columns(2)

        with col_i:
            st.markdown(f"""
            ### Data Coverage
            - **Period**: {period_str}
            - **Total hotspots**: {total:,}
            - **Sensors**: VIIRS S-NPP, NOAA-20, MODIS C6.1
            - **Area**: Ucayali Region, Peru
            """)

            st.markdown("### Statistics by Year")
            for y in years:
                count = len(filtered[filtered['year'] == y])
                pct = count / total * 100 if total > 0 else 0
                st.progress(pct / 100, text=f"{y}: {count:,} hotspots ({pct:.1f}%)")

        with col_j:
            st.markdown("### Temporal Pattern")
            if len(filtered) > 0:
                peak = filtered.groupby('month').size().idxmax()
                st.markdown(f"- **Peak fire month**: {MONTH_NAMES.get(peak, peak)}")
                st.markdown("- **High season**: June - October (dry)")
                st.markdown("- **Low season**: December - March (rainy)")

            st.markdown("### Top Zones")
            top_zones = filtered['zone'].value_counts().head(5)
            for zone, count in top_zones.items():
                st.markdown(f"- **{zone}**: {count:,} hotspots")

        # Risk summary
        st.markdown("---")
        st.markdown("### Current Risk Assessment")

        if weather_df is not None:
            avg_risk = weather_df['fire_risk'].mean()
            max_risk = weather_df['fire_risk'].max()

            col_k, col_l, col_m = st.columns(3)

            with col_k:
                if avg_risk >= 0.5: st.error(f"ALERT: Avg Risk = {avg_risk:.2f}")
                elif avg_risk >= 0.3: st.warning(f"WATCH: Avg Risk = {avg_risk:.2f}")
                else: st.success(f"NORMAL: Avg Risk = {avg_risk:.2f}")

            with col_l:
                st.metric("Max Risk (7d)", f"{max_risk:.2f}")

            with col_m:
                risk_cells = compute_spatial_risk(filtered)
                high_risk_cells = len(risk_cells[risk_cells['risk_score'] > 0.5]) if len(risk_cells) > 0 else 0
                st.metric("High Risk Cells (30d)", high_risk_cells)

        # High risk coordinates
        if len(risk_cells) > 0:
            high_risk = risk_cells[risk_cells['risk_score'] > 0.5]
            if len(high_risk) > 0:
                st.markdown("### High Risk Coordinates (Last 30 Days)")
                for _, r in high_risk.nlargest(10, 'risk_score').iterrows():
                    st.markdown(
                        f"- **({r['lat_bin']:.1f}, {r['lon_bin']:.1f})** "
                        f"- Risk: {r['risk_score']:.2f} "
                        f"- Hotspots: {r['hotspot_count']}"
                    )

    # Footer
    st.markdown("---")
    st.caption("Data: NASA FIRMS (thermal anomalies) + Open-Meteo (weather forecast)")


if __name__ == '__main__':
    main()
