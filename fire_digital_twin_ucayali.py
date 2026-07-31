# -*- coding: utf-8 -*-
"""
DIGITAL TWIN WILDFIRE MONITORING - UCAYALI, PERU
==================================================
Real-time monitoring, historical analysis, and fire risk prediction
using NASA FIRMS satellite data and Open-Meteo weather forecasts.

Data sources:
  - FIRMS archive shapefiles (VIIRS S-NPP, NOAA-20, MODIS C6.1)
  - FIRMS NRT (Near Real-Time) shapefiles
  - Open-Meteo API (weather forecast)

Period: 2020-01-01 to present
Region: Ucayali, Peru (-7.0 to -11.5 lat, -72.5 to -76.0 lon)

Author: Johana Castillo (CGIAR/IFPRI)
Date: July 2026
"""

import os
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

warnings.filterwarnings('ignore')

MONTH_NAMES = {
    1: 'January', 2: 'February', 3: 'March', 4: 'April',
    5: 'May', 6: 'June', 7: 'July', 8: 'August',
    9: 'September', 10: 'October', 11: 'November', 12: 'December'
}
MONTH_SHORT = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

# =============================================================================
# CONFIGURATION
# =============================================================================

LAT_MIN, LAT_MAX = -11.5, -7.0
LON_MIN, LON_MAX = -76.0, -72.5
DATE_START = '2020-01-01'

BASE_DIR = Path(r'D:\OneDrive - CGIAR\Johana Castillo\2026\MFL\1_Report_Ucayali')
DATA_DIR = BASE_DIR / 'data'
OUTPUTS_DIR = BASE_DIR / 'outputs'
DATA_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

FIRMS_FILES = {
    'VIIRS_SNPP': BASE_DIR / 'DL_FIRE_SV-C2_769837' / 'fire_archive_SV-C2_769837.dbf',
    'VIIRS_NOAA20': BASE_DIR / 'DL_FIRE_J1V-C2_769835' / 'fire_archive_J1V-C2_769835.dbf',
    'MODIS': BASE_DIR / 'DL_FIRE_M-C61_769833' / 'fire_archive_M-C61_769833.dbf',
    'VIIRS_SNPP_NRT': BASE_DIR / 'DL_FIRE_SV-C2_769837' / 'fire_nrt_SV-C2_769837.dbf',
    'VIIRS_NOAA20_NRT': BASE_DIR / 'DL_FIRE_J1V-C2_769835' / 'fire_nrt_J1V-C2_769835.dbf',
    'MODIS_NRT': BASE_DIR / 'DL_FIRE_M-C61_769833' / 'fire_nrt_M-C61_769833.dbf',
}

PALETTE_YEAR = {
    2020: '#3498db', 2021: '#e74c3c', 2022: '#f39c12',
    2023: '#27ae60', 2024: '#9b59b6', 2025: '#e67e22', 2026: '#1abc9c'
}
COLORS = ['#e74c3c', '#f39c12', '#27ae60', '#3498db', '#9b59b6', '#1abc9c']

# =============================================================================
# 1. LOAD FIRMS DATA
# =============================================================================

def ensure_dbfread():
    try:
        from dbfread import DBF
        return True
    except ImportError:
        print("  Installing dbfread...")
        os.system(f'"{sys.executable}" -m pip install dbfread -q')
        return True


def load_dbf(filepath):
    from dbfread import DBF
    table = DBF(str(filepath), encoding='latin-1', raw=False)
    df = pd.DataFrame(iter(table))
    return df


def load_all_firms():
    print("=" * 60)
    print("STEP 1: LOADING FIRMS DATA")
    print("=" * 60)

    ensure_dbfread()
    all_dfs = []

    for name, filepath in FIRMS_FILES.items():
        if filepath.exists():
            print(f"  Loading {name}...")
            df = load_dbf(filepath)
            df['_source'] = name
            print(f"    -> {len(df):,} records")
            all_dfs.append(df)
        else:
            print(f"  [WARN] Not found: {filepath.name}")

    if not all_dfs:
        print("  [ERROR] No FIRMS files found")
        sys.exit(1)

    print("\n  Concatenating datasets...")
    combined = pd.concat(all_dfs, ignore_index=True)
    combined.columns = [c.lower() for c in combined.columns]
    print(f"  Total raw: {len(combined):,} records")
    return combined


def filter_ucayali(df):
    print("\n  Filtering by Ucayali bounding box...")
    for col in ['latitude', 'longitude']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    mask = (
        (df['latitude'] >= LAT_MIN) & (df['latitude'] <= LAT_MAX) &
        (df['longitude'] >= LON_MIN) & (df['longitude'] <= LON_MAX)
    )
    df = df[mask].copy()
    print(f"    -> {len(df):,} records in Ucayali")

    if 'acq_date' in df.columns:
        df['acq_date'] = pd.to_datetime(df['acq_date'], errors='coerce')
        df = df.dropna(subset=['acq_date'])
        df = df[df['acq_date'] >= DATE_START].copy()
        print(f"    -> {len(df):,} records since {DATE_START}")

    df['year'] = df['acq_date'].dt.year
    df['month'] = df['acq_date'].dt.month
    df['month_name'] = df['month'].map(MONTH_NAMES)
    df['year_month'] = df['acq_date'].dt.to_period('M')
    df['year_month_str'] = df['acq_date'].dt.strftime('%Y-%m')

    return df


def save_consolidated(df):
    outpath = DATA_DIR / 'firms_ucayali_2020_hoy.csv'
    df.to_csv(outpath, index=False)
    print(f"\n  Saved: {outpath}")
    print(f"  Total records: {len(df):,}")
    return outpath


# =============================================================================
# ZONE ASSIGNMENT BY COORDINATES
# =============================================================================

def assign_zones(df):
    def _zone(row):
        lat, lon = row.get('latitude', 0), row.get('longitude', 0)
        if pd.isna(lat) or pd.isna(lon):
            return 'No data'
        if lat > -8.0 and lon < -74.0:
            return 'Padre Abad - Aguaytia'
        elif lat > -8.5 and lon >= -74.0:
            return 'Padre Abad - Aguaytia'
        elif -9.5 <= lat <= -8.0 and lon < -74.5:
            return 'Coronel Portillo - Pucallpa'
        elif -9.5 <= lat <= -8.0 and lon >= -74.5:
            return 'Coronel Portillo - East'
        elif lat < -9.5 and lon < -74.0:
            return 'Atalaya - Ucayali'
        elif lat < -9.5 and lon >= -74.0:
            return 'Atalaya - Purus'
        elif lat < -10.5:
            return 'Southern Border'
        else:
            return 'Ucayali - Central'
    return df.apply(_zone, axis=1)


# =============================================================================
# 2. HISTORICAL ANALYSIS
# =============================================================================

def historical_analysis(df):
    print("\n" + "=" * 60)
    print("STEP 2: HISTORICAL ANALYSIS (2020-2026)")
    print("=" * 60)

    stats = {}

    stats['by_year'] = df.groupby('year').size()
    print("\n  Hotspots by year:")
    for y, c in stats['by_year'].items():
        print(f"    {y}: {c:,}")

    stats['monthly_series'] = df.groupby('year_month_str').size().reset_index(name='hotspots')
    stats['monthly_series']['date'] = pd.to_datetime(stats['monthly_series']['year_month_str'] + '-01')
    stats['monthly_series'] = stats['monthly_series'].sort_values('date')

    n_years = max(df['year'].nunique(), 1)
    stats['seasonal'] = df.groupby('month').size() / n_years

    df['zone'] = assign_zones(df)
    stats['admin_col'] = 'zone'
    stats['top_zones'] = df['zone'].value_counts().head(15)

    stats['interannual'] = df.pivot_table(
        index='month', columns='year', values='latitude', aggfunc='count'
    ).fillna(0)

    return stats


# =============================================================================
# 3. CURRENT CONDITIONS (LIVE API)
# =============================================================================

def get_current_weather():
    print("\n" + "=" * 60)
    print("STEP 3: CURRENT CONDITIONS (Open-Meteo)")
    print("=" * 60)

    try:
        import requests

        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            'latitude': -8.38,
            'longitude': -74.57,
            'hourly': 'temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation,wind_gusts_10m',
            'daily': 'temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max,relative_humidity_2m_min',
            'timezone': 'America/Lima',
            'forecast_days': 7
        }

        resp = requests.get(url, params=params, timeout=15)
        data = resp.json()

        daily = data.get('daily', {})
        weather_df = pd.DataFrame({
            'date': pd.to_datetime(daily.get('time', [])),
            'temp_max': daily.get('temperature_2m_max', []),
            'temp_min': daily.get('temperature_2m_min', []),
            'precip_mm': daily.get('precipitation_sum', []),
            'wind_max_kmh': daily.get('wind_speed_10m_max', []),
            'rh_min_pct': daily.get('relative_humidity_2m_min', []),
        })

        print("  7-day forecast for Pucallpa:")
        for _, r in weather_df.iterrows():
            date_str = r['date'].strftime('%d/%m')
            print(f"    {date_str}: {r['temp_max']:.0f}-{r['temp_min']:.0f}C  "
                  f"wind:{r['wind_max_kmh']:.0f}km/h  "
                  f"rain:{r['precip_mm']:.1f}mm  "
                  f"RH:{r['rh_min_pct']:.0f}%")

        weather_df['fire_risk'] = 0.0
        for i, row in weather_df.iterrows():
            score = 0.0
            if row['temp_max'] > 30: score += 0.3
            if row['temp_max'] > 33: score += 0.2
            if row['rh_min_pct'] < 50: score += 0.2
            if row['rh_min_pct'] < 30: score += 0.1
            if row['wind_max_kmh'] > 20: score += 0.1
            if row['wind_max_kmh'] > 30: score += 0.1
            if row['precip_mm'] > 0: score -= 0.3
            weather_df.at[i, 'fire_risk'] = max(0.0, min(1.0, score))

        return weather_df

    except Exception as e:
        print(f"  [WARN] Open-Meteo error: {e}")
        return None


# =============================================================================
# 4. RISK MODEL
# =============================================================================

def compute_risk_model(df, weather_df):
    print("\n" + "=" * 60)
    print("STEP 4: COMPOSITE RISK MODEL")
    print("=" * 60)

    now = pd.Timestamp.now()
    last_30d = now - timedelta(days=30)
    recent = df[(df['acq_date'] >= last_30d) & (df['acq_date'] <= now)].copy()

    print(f"  Hotspots last 30 days: {len(recent):,}")

    if len(recent) > 0:
        recent['lat_bin'] = (recent['latitude'] / 0.1).round() * 0.1
        recent['lon_bin'] = (recent['longitude'] / 0.1).round() * 0.1
        density = recent.groupby(['lat_bin', 'lon_bin']).size().reset_index(name='count')
        max_count = density['count'].max()
        density['risk_history'] = density['count'] / max_count if max_count > 0 else 0
        print(f"  Risk cells: {len(density)}")
    else:
        density = pd.DataFrame(columns=['lat_bin', 'lon_bin', 'count', 'risk_history'])

    if weather_df is not None and len(weather_df) > 0:
        avg_risk = weather_df['fire_risk'].mean()
        print(f"  Average weather risk: {avg_risk:.2f}")

    return density


# =============================================================================
# 5. VISUALIZATIONS
# =============================================================================

def plot_dashboard(df, stats, weather_df, risk_grid):
    print("\n" + "=" * 60)
    print("STEP 5: GENERATING VISUALIZATIONS")
    print("=" * 60)

    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    fig.suptitle(
        'DIGITAL TWIN WILDFIRE MONITORING - UCAYALI, PERU\n'
        'Monitoring, Historical Analysis and Forecast',
        fontsize=16, fontweight='bold', y=0.98
    )

    # 1. Monthly time series
    ax = axes[0, 0]
    serie = stats['monthly_series']
    bar_colors = [PALETTE_YEAR.get(d.year, '#888') for d in serie['date']]
    ax.bar(range(len(serie)), serie['hotspots'], color=bar_colors, width=0.8)
    ax.set_title('Hotspots by Month (2020-2026)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Number of hotspots')
    step = max(1, len(serie) // 12)
    tick_pos = list(range(0, len(serie), step))
    tick_lab = [serie['year_month_str'].iloc[i] for i in tick_pos]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lab, rotation=45, fontsize=7)

    # 2. Hotspots by year
    ax = axes[0, 1]
    by_year = stats['by_year']
    bars = ax.bar(
        by_year.index.astype(str), by_year.values,
        color=[PALETTE_YEAR.get(y, '#888') for y in by_year.index]
    )
    ax.set_title('Hotspots by Year', fontsize=11, fontweight='bold')
    ax.set_ylabel('Total hotspots')
    for bar, val in zip(bars, by_year.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + max(by_year) * 0.01,
            f'{val:,}', ha='center', fontsize=8
        )

    # 3. Seasonal pattern
    ax = axes[0, 2]
    seasonal_vals = [stats['seasonal'].get(m, 0) for m in range(1, 13)]
    colors_seasonal = ['#3498db' if m not in [6,7,8,9,10] else '#e74c3c' for m in range(1,13)]
    ax.bar(MONTH_SHORT, seasonal_vals, color=colors_seasonal)
    ax.set_title('Seasonal Pattern (Average)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Average hotspots')
    ax.tick_params(axis='x', rotation=45)

    # 4. Spatial map
    ax = axes[1, 0]
    n_sample = min(5000, len(df))
    if n_sample > 0:
        sample = df.sample(n_sample, random_state=42)
        ax.scatter(
            sample['longitude'], sample['latitude'],
            c=[PALETTE_YEAR.get(y, '#888') for y in sample['year']],
            alpha=0.3, s=1
        )
    ax.set_title(f'Spatial Distribution ({len(df):,} records)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_xlim(LON_MIN, LON_MAX)
    ax.set_ylim(LAT_MIN, LAT_MAX)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=PALETTE_YEAR[y],
               markersize=8, label=str(y))
        for y in sorted(df['year'].unique()) if y in PALETTE_YEAR
    ]
    if legend_elements:
        ax.legend(handles=legend_elements, fontsize=7, loc='lower left')

    # 5. Interannual comparison
    ax = axes[1, 1]
    inter = stats['interannual']
    for col in inter.columns:
        if col in PALETTE_YEAR:
            ax.plot(inter.index, inter[col], marker='o', color=PALETTE_YEAR[col],
                    label=str(col), linewidth=2, markersize=4)
    ax.set_title('Interannual Comparison', fontsize=11, fontweight='bold')
    ax.set_xlabel('Month')
    ax.set_ylabel('Hotspots')
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(MONTH_SHORT, fontsize=8)
    ax.legend(fontsize=8)

    # 6. Heatmap month x year
    ax = axes[1, 2]
    pivot = df.pivot_table(index='month', columns='year', values='latitude', aggfunc='count')
    pivot.index = [MONTH_SHORT[m - 1] for m in pivot.index]
    sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax,
                cbar_kws={'label': 'Hotspots'})
    ax.set_title('Heatmap Month x Year', fontsize=11, fontweight='bold')
    ax.tick_params(axis='x', rotation=0)

    # 7. Confidence distribution
    ax = axes[2, 0]
    if 'confidence' in df.columns:
        def _normalize_conf(c):
            c = str(c).strip().lower()
            if c in ['h', 'high']:
                return 'High'
            elif c in ['n', 'nominal']:
                return 'Nominal'
            elif c in ['l', 'low']:
                return 'Low'
            else:
                try:
                    v = int(c)
                    if v >= 80: return 'High'
                    elif v >= 30: return 'Nominal'
                    else: return 'Low'
                except:
                    return 'Other'
        conf_norm = df['confidence'].apply(_normalize_conf).value_counts()
        conf_colors = {'High': '#27ae60', 'Nominal': '#f39c12', 'Low': '#e74c3c', 'Other': '#888'}
        ax.pie(
            conf_norm.values,
            labels=conf_norm.index,
            autopct='%1.1f%%',
            colors=[conf_colors.get(c, '#888') for c in conf_norm.index]
        )
        ax.set_title('FIRMS Confidence Distribution', fontsize=11, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'No confidence data', ha='center', va='center', fontsize=12)
        ax.set_title('Confidence', fontsize=11, fontweight='bold')

    # 8. Sensor breakdown
    ax = axes[2, 1]
    src_counts = df['_source'].value_counts()
    ax.barh(src_counts.index, src_counts.values, color=COLORS[:len(src_counts)])
    ax.set_title('Hotspots by Sensor', fontsize=11, fontweight='bold')
    ax.set_xlabel('Number of hotspots')

    # 9. Weather fire risk forecast with clear legend
    ax = axes[2, 2]
    if weather_df is not None and len(weather_df) > 0:
        risk_colors = [plt.cm.RdYlGn_r(r) for r in weather_df['fire_risk']]
        ax.bar(range(len(weather_df)), weather_df['fire_risk'], color=risk_colors, edgecolor='white', linewidth=0.5)
        ax.set_title('Fire Risk Forecast - 7 Days', fontsize=11, fontweight='bold')
        ax.set_ylabel('Risk Index (0-1)')
        ax.set_xticks(range(len(weather_df)))
        ax.set_xticklabels(
            [d.strftime('%d/%m\n%a') for d in weather_df['date']], rotation=0, fontsize=8
        )
        ax.set_ylim(0, 1.0)
        # Risk level zones with colors
        ax.axhspan(0, 0.25, alpha=0.1, color='green', label='LOW (0-0.25)')
        ax.axhspan(0.25, 0.5, alpha=0.1, color='yellow', label='MODERATE (0.25-0.5)')
        ax.axhspan(0.5, 0.75, alpha=0.1, color='orange', label='HIGH (0.5-0.75)')
        ax.axhspan(0.75, 1.0, alpha=0.1, color='red', label='VERY HIGH (>0.75)')
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, linewidth=1)
        # Add risk level text for each day
        for i, row in enumerate(weather_df.itertuples()):
            risk = row.fire_risk
            if risk >= 0.75: level, lc = 'VERY HIGH', 'darkred'
            elif risk >= 0.5: level, lc = 'HIGH', 'red'
            elif risk >= 0.25: level, lc = 'MODERATE', 'darkorange'
            else: level, lc = 'LOW', 'green'
            ax.text(i, risk + 0.03, level, ha='center', va='bottom', fontsize=7, fontweight='bold', color=lc)
        ax.legend(fontsize=7, loc='upper right', ncol=2)
    else:
        ax.text(0.5, 0.5, 'No forecast available', ha='center', va='center', fontsize=12)
        ax.set_title('Fire Risk Forecast', fontsize=11, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    outpath = OUTPUTS_DIR / 'dashboard_principal.png'
    fig.savefig(outpath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Dashboard saved: {outpath}")
    return outpath


def plot_serie_temporal_extendida(df):
    fig, ax = plt.subplots(figsize=(16, 6))

    serie = df.groupby('year_month_str').size().reset_index(name='hotspots')
    serie['date'] = pd.to_datetime(serie['year_month_str'] + '-01')
    serie = serie.sort_values('date')

    ax.fill_between(serie['date'], serie['hotspots'], alpha=0.3, color='#e74c3c')
    ax.plot(serie['date'], serie['hotspots'], color='#e74c3c', linewidth=1.5)
    serie['moving_avg'] = serie['hotspots'].rolling(3, center=True).mean()
    ax.plot(serie['date'], serie['moving_avg'], color='#2c3e50', linewidth=2,
            linestyle='--', label='3-month moving average')

    ax.set_title('FIRMS Hotspot Time Series - Ucayali (2020-2026)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Number of hotspots')
    ax.legend()
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.xticks(rotation=45)

    plt.tight_layout()
    outpath = OUTPUTS_DIR / 'serie_temporal_2020_2026.png'
    fig.savefig(outpath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Time series saved: {outpath}")
    return outpath


def plot_mapa_interactivo(df):
    try:
        import folium
        from folium.plugins import MarkerCluster
        from folium import Element
    except ImportError:
        print("  Installing folium...")
        os.system(f'"{sys.executable}" -m pip install folium -q')
        import folium
        from folium.plugins import MarkerCluster
        from folium import Element

    m = folium.Map(location=[-8.5, -74.5], zoom_start=7, tiles='CartoDB positron')

    for year in sorted(df['year'].unique()):
        fg = folium.FeatureGroup(name=f'Hotspots {year}')
        year_df = df[df['year'] == year]
        cluster = MarkerCluster().add_to(fg)
        sample = year_df.sample(min(2000, len(year_df)), random_state=42)

        for _, row in sample.iterrows():
            conf = str(row.get('confidence', '')).lower()
            color = 'red' if conf in ['high', 'nominal', 'h', 'n'] else 'orange'
            popup_html = (
                f"<b>{row['acq_date'].strftime('%d/%m/%Y')}</b><br>"
                f"FRP: {row.get('frp', 'N/A')}<br>"
                f"Confidence: {row.get('confidence', 'N/A')}<br>"
                f"Sensor: {row.get('_source', 'N/A')}<br>"
                f"Lat: {row['latitude']:.4f}, Lon: {row['longitude']:.4f}"
            )
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=2, color=color, fill=True, fill_opacity=0.5,
                popup=folium.Popup(popup_html, max_width=200)
            ).add_to(cluster)

        fg.add_to(m)

    folium.LayerControl().add_to(m)

    # Add legend
    legend_html = """
    <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000;
         background-color: white; padding: 15px; border-radius: 8px;
         border: 2px solid #333; font-family: Arial; font-size: 13px;
         box-shadow: 2px 2px 6px rgba(0,0,0,0.3);">
        <b style="font-size: 14px;">FIRMS HOTSPOT MAP</b><br>
        <span style="font-size: 11px; color: #666;">
        Fire detections from NASA satellites<br>
        Click layers to toggle by year
        </span><br><br>
        <span style="color: red;">&#9679;</span> <b>RED</b> = High/Nominal confidence<br>
        <span style="color: orange;">&#9679;</span> <b>ORANGE</b> = Low confidence<br><br>
        <span style="font-size: 10px; color: #888;">
        FRP = Fire Radiative Power (MW)<br>
        Higher FRP = More intense fire
        </span>
    </div>
    """
    m.get_root().html.add_child(Element(legend_html))

    outpath = OUTPUTS_DIR / 'mapa_focos_ucayali.html'
    m.save(str(outpath))
    print(f"  Interactive map saved: {outpath}")
    return outpath


def plot_mapa_riesgo(risk_grid):
    if len(risk_grid) == 0:
        print("  [WARN] No risk data for map")
        return None

    try:
        import folium
        from folium.plugins import HeatMap
        from folium import Element
    except ImportError:
        os.system(f'"{sys.executable}" -m pip install folium -q')
        import folium
        from folium.plugins import HeatMap
        from folium import Element

    m = folium.Map(location=[-8.5, -74.5], zoom_start=7, tiles='CartoDB dark_matter')
    heat_data = [
        [row['lat_bin'], row['lon_bin'], row['risk_history']]
        for _, row in risk_grid.iterrows()
    ]
    HeatMap(heat_data, radius=15, max_zoom=10, max_val=1.0).add_to(m)

    # Add HTML legend explaining the colors
    legend_html = """
    <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000;
         background-color: white; padding: 15px; border-radius: 8px;
         border: 2px solid #333; font-family: Arial; font-size: 13px;
         box-shadow: 2px 2px 6px rgba(0,0,0,0.3);">
        <b style="font-size: 14px;">FIRE RISK MAP</b><br>
        <span style="font-size: 11px; color: #666;">
        Based on hotspot density<br>
        in the last 30 days
        </span><br><br>
        <span style="color: #ff0000;">&#9632;</span> <b>RED / HOT</b> = Many fires detected = HIGH RISK<br>
        <span style="color: #ff8800;">&#9632;</span> <b>ORANGE</b> = Moderate fires = MEDIUM RISK<br>
        <span style="color: #ffff00;">&#9632;</span> <b>YELLOW</b> = Few fires = LOW RISK<br>
        <span style="color: #000033;">&#9632;</span> <b>DARK</b> = No fires detected<br><br>
        <span style="font-size: 10px; color: #888;">
        Risk = (# hotspots in cell / max # in any cell)<br>
        Higher concentration = Higher risk
        </span>
    </div>
    """
    m.get_root().html.add_child(Element(legend_html))

    outpath = OUTPUTS_DIR / 'mapa_riesgo.html'
    m.save(str(outpath))
    print(f"  Risk heatmap saved: {outpath}")
    return outpath


def plot_zone_summary(df, zone_col):
    if zone_col is None:
        print("  [WARN] No zone column")
        return None

    tabla = df.groupby(zone_col).agg(
        total_hotspots=('latitude', 'count'),
        active_years=('year', 'nunique'),
        avg_per_year=('latitude', lambda x: len(x) / max(df['year'].nunique(), 1)),
    ).sort_values('total_hotspots', ascending=False).head(15).round(1)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('off')
    table = ax.table(
        cellText=tabla.values,
        colLabels=['Total Hotspots', 'Active Years', 'Avg/Year'],
        rowLabels=tabla.index,
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    ax.set_title(
        'Top 15 Zones by FIRMS Hotspots (2020-2026)',
        fontsize=13, fontweight='bold', pad=20
    )

    plt.tight_layout()
    outpath = OUTPUTS_DIR / 'tabla_distritos.png'
    fig.savefig(outpath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Zone summary saved: {outpath}")
    return outpath


def plot_risk_explanation():
    """Standalone figure explaining the risk model to non-technical users."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle('HOW TO READ THE FIRE RISK MAPS\nGuide for Decision Makers',
                 fontsize=16, fontweight='bold', y=1.02)

    # Left panel: Risk level explanation
    ax = axes[0]
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    risk_levels = [
        (9.0, 'LOW RISK', '#27ae60', '0.0 - 0.25',
         'Few or no fires detected.\nNormal conditions.\nNo special action needed.'),
        (7.0, 'MODERATE RISK', '#f39c12', '0.25 - 0.50',
         'Some fire activity detected.\nConditions becoming dry.\nIncrease monitoring.'),
        (5.0, 'HIGH RISK', '#e74c3c', '0.50 - 0.75',
         'Significant fire activity.\nDry and windy conditions.\nPrepare response teams.'),
        (3.0, 'VERY HIGH RISK', '#8b0000', '0.75 - 1.00',
         'Dense fire activity detected.\nExtreme conditions likely.\nActivate emergency protocols.'),
    ]

    for y, label, color, score_range, description in risk_levels:
        # Color box
        rect = plt.Rectangle((0.5, y - 0.7), 2.0, 1.4, facecolor=color, alpha=0.7,
                              edgecolor='black', linewidth=2, transform=ax.transData)
        ax.add_patch(rect)
        ax.text(1.5, y, f'{label}\n{score_range}', ha='center', va='center',
                fontsize=11, fontweight='bold', color='white', transform=ax.transData)
        # Description
        ax.text(3.2, y, description, ha='left', va='center', fontsize=10,
                transform=ax.transData, color='#333')

    ax.text(5.0, 9.8, 'Risk Score', ha='center', fontsize=14, fontweight='bold', transform=ax.transData)
    ax.text(5.0, 9.3, 'How to interpret', ha='center', fontsize=11, color='#666', transform=ax.transData)

    # Right panel: How the model works
    ax = axes[1]
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    model_text = [
        (9.5, 'FIRE RISK MODEL', 14, 'bold', '#333'),
        (8.8, 'The risk score combines TWO sources:', 11, 'normal', '#333'),
        (7.8, '1. SATELLITE HISTORY (50%)', 12, 'bold', '#e74c3c'),
        (7.3, '   How many fires were detected in the area', 10, 'normal', '#555'),
        (6.9, '   in the last 30 days.', 10, 'normal', '#555'),
        (6.3, '   More fires = Higher risk', 10, 'bold', '#e74c3c'),
        (5.5, '2. WEATHER FORECAST (50%)', 12, 'bold', '#3498db'),
        (5.0, '   Temperature, humidity, wind, and rain', 10, 'normal', '#555'),
        (4.6, '   from Open-Meteo 7-day forecast.', 10, 'normal', '#555'),
        (4.0, '   Hot + Dry + Windy = Higher risk', 10, 'bold', '#3498db'),
        (3.2, 'WHAT EACH MAP SHOWS:', 12, 'bold', '#333'),
        (2.7, 'Dashboard (panel 9):', 10, 'bold', '#555'),
        (2.3, '   Daily weather risk for Pucallpa', 10, 'normal', '#777'),
        (1.7, 'Heatmap (HTML):', 10, 'bold', '#555'),
        (1.3, '   Spatial density of fires, last 30 days', 10, 'normal', '#777'),
        (0.7, 'Interactive map (HTML):', 10, 'bold', '#555'),
        (0.3, '   Individual fire detections, filterable by year', 10, 'normal', '#777'),
    ]

    for y, text, size, weight, color in model_text:
        ax.text(0.5, y, text, ha='left', va='center', fontsize=size,
                fontweight=weight, color=color, transform=ax.transData)

    plt.tight_layout()
    outpath = OUTPUTS_DIR / 'risk_explanation_guide.png'
    fig.savefig(outpath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Risk guide saved: {outpath}")
    return outpath


def plot_prediccion(weather_df):
    if weather_df is None or len(weather_df) == 0:
        print("  [WARN] No forecast data")
        return None

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(
        'Weather Forecast - Pucallpa, Ucayali\nFire Risk Index',
        fontsize=13, fontweight='bold'
    )

    x = range(len(weather_df))
    labels = [d.strftime('%d/%m\n%a') for d in weather_df['date']]

    ax1.bar(x, weather_df['precip_mm'], color='#3498db', alpha=0.6, label='Precipitation (mm)')
    ax1_t = ax1.twinx()
    ax1_t.plot(x, weather_df['temp_max'], 'o-', color='#e74c3c', label='Temp Max', linewidth=2)
    ax1_t.plot(x, weather_df['temp_min'], 's--', color='#3498db', label='Temp Min', linewidth=2)
    ax1.set_ylabel('Precipitation (mm)')
    ax1_t.set_ylabel('Temperature (C)')
    ax1.legend(loc='upper left', fontsize=9)
    ax1_t.legend(loc='upper right', fontsize=9)
    ax1.set_title('Meteorological Variables', fontsize=11)

    ax2.bar(x, weather_df['wind_max_kmh'], color='#f39c12', alpha=0.6, label='Max Wind (km/h)')
    ax2_t = ax2.twinx()
    ax2_t.plot(x, weather_df['fire_risk'], 'D-', color='#e74c3c',
               label='Fire Risk', linewidth=2, markersize=8)
    ax2_t.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5)
    ax2.set_ylabel('Max Wind (km/h)')
    ax2_t.set_ylabel('Risk Index (0-1)')
    ax2.legend(loc='upper left', fontsize=9)
    ax2_t.legend(loc='upper right', fontsize=9)
    ax2.set_title('Fire Risk Condition', fontsize=11)
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(labels)

    plt.tight_layout()
    outpath = OUTPUTS_DIR / 'prediccion_7dias.png'
    fig.savefig(outpath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Forecast saved: {outpath}")
    return outpath


# =============================================================================
# 6. EXECUTIVE SUMMARY
# =============================================================================

def generate_summary(df, stats, weather_df, risk_grid, zone_col):
    print("\n" + "=" * 60)
    print("STEP 6: EXECUTIVE SUMMARY")
    print("=" * 60)

    total = len(df)
    years = sorted(df['year'].unique())
    peak_month = stats['seasonal'].idxmax()

    lines = [
        '=' * 70,
        'DIGITAL TWIN WILDFIRE MONITORING - UCAYALI, PERU',
        'EXECUTIVE SUMMARY',
        f'Generated: {datetime.now().strftime("%d/%m/%Y %H:%M")}',
        '=' * 70,
        '',
        '1. DATA COVERAGE',
        f'   Period: {years[0]} - {years[-1]}',
        f'   Total hotspots detected: {total:,}',
        '   Sensors: VIIRS S-NPP, VIIRS NOAA-20, MODIS C6.1',
        '   Area: Ucayali Region, Peru',
        '',
        '2. STATISTICS BY YEAR',
    ]
    for y in years:
        count = stats['by_year'].get(y, 0)
        pct = count / total * 100 if total > 0 else 0
        lines.append(f'   {y}: {count:>8,} hotspots ({pct:5.1f}%)')

    lines += [
        '',
        '3. TEMPORAL PATTERN',
        f'   Peak fire month: {MONTH_NAMES.get(peak_month, peak_month)}',
        '   High season: June - October (dry)',
        '   Low season: December - March (rainy)',
        '',
        '4. CURRENT RISK (7-day forecast)',
    ]

    if weather_df is not None and len(weather_df) > 0:
        avg_risk = weather_df['fire_risk'].mean()
        max_risk = weather_df['fire_risk'].max()
        lines.append(f'   Average risk: {avg_risk:.2f}/1.00')
        lines.append(f'   Maximum risk: {max_risk:.2f}/1.00')
        if avg_risk >= 0.5:
            lines.append('   ALERT: Conditions favorable for wildfires')
        elif avg_risk >= 0.3:
            lines.append('   WATCH: Moderate fire risk conditions')
        else:
            lines.append('   NORMAL: Low fire risk conditions')
    else:
        lines.append('   No weather data available')

    if len(risk_grid) > 0:
        hotspots = risk_grid[risk_grid['risk_history'] > 0.5]
        lines += ['', '5. HIGH RISK ZONES (last 30 days)']
        lines.append(f'   High risk cells: {len(hotspots)}')
        if len(hotspots) > 0:
            lines.append('   Key coordinates:')
            for _, r in hotspots.nlargest(5, 'risk_history').iterrows():
                lines.append(f'     ({r["lat_bin"]:.1f}, {r["lon_bin"]:.1f}) - risk: {r["risk_history"]:.2f}')

    if zone_col and stats['top_zones'] is not None and len(stats['top_zones']) > 0:
        lines += ['', '6. TOP 10 ZONES']
        for zone, cnt in stats['top_zones'].head(10).items():
            lines.append(f'   {zone}: {cnt:,} hotspots')

    lines += [
        '',
        '=' * 70,
        'GENERATED FILES',
        '=' * 70,
        '   data/firms_ucayali_2020_hoy.csv',
        '   outputs/dashboard_principal.png',
        '   outputs/serie_temporal_2020_2026.png',
        '   outputs/mapa_focos_ucayali.html',
        '   outputs/mapa_riesgo.html',
        '   outputs/tabla_distritos.png',
        '   outputs/prediccion_7dias.png',
        '   outputs/risk_explanation_guide.png',
        '   outputs/resumen_ejecutivo.txt',
        '=' * 70,
    ]

    summary = '\n'.join(lines)

    outpath = OUTPUTS_DIR / 'resumen_ejecutivo.txt'
    with open(outpath, 'w', encoding='utf-8') as f:
        f.write(summary)

    print(summary)
    print(f"\n  Summary saved: {outpath}")
    return outpath


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("DIGITAL TWIN WILDFIRE MONITORING - UCAYALI, PERU")
    print(f"Date: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    print("=" * 60)

    raw_df = load_all_firms()
    df = filter_ucayali(raw_df)
    save_consolidated(df)

    stats = historical_analysis(df)
    weather_df = get_current_weather()
    risk_grid = compute_risk_model(df, weather_df)

    plot_dashboard(df, stats, weather_df, risk_grid)
    plot_serie_temporal_extendida(df)
    plot_mapa_interactivo(df)
    plot_mapa_riesgo(risk_grid)
    plot_zone_summary(df, 'zone')
    plot_prediccion(weather_df)
    plot_risk_explanation()

    generate_summary(df, stats, weather_df, risk_grid, 'zone')

    print("\n" + "=" * 60)
    print("COMPLETE - Files in outputs/")
    print("=" * 60)


if __name__ == '__main__':
    main()
