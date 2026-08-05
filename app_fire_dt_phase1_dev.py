# -*- coding: utf-8 -*-
"""
DIGITAL TWIN WILDFIRE MONITORING - PHASE 1 (DEV, NATIONAL PERU)
===============================================================
Local development version (DO NOT deploy yet). Phase 1 scope:
  - All Peru FIRMS hotspots (2020-present)
  - Context layers: native communities (MINAM GeoServidor) + protected areas
    (SERNANP: ANP nacional, ZR, ACR, ACP) + departments (GeoBoundaries ADM1)
  - Per-hotspot proximity to communities / protected areas
  - Priority table ("closest to communities/ANPs first")
  - False-positive filter (type + confidence/FRP + volcano exclusion)

Run:  streamlit run app_fire_dt_phase1_dev.py
"""

import json
import os
import sys
import warnings
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import folium
from folium.plugins import HeatMapWithTime
from streamlit_folium import st_folium

warnings.filterwarnings('ignore')

# Fix folium HeatMapWithTime bounds (same patch as deployed app).
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

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
CONTEXT_DIR = DATA_DIR / 'context'

FIRMS_PATH = DATA_DIR / 'firms_peru_2020_hoy.parquet'
FIRMS_CSV = DATA_DIR / 'firms_peru_2020_hoy_compact.csv.gz'
UPDATE_LOG = DATA_DIR / 'last_update_firms_peru.json'

def _first_existing(*names):
    for n in names:
        p = CONTEXT_DIR / n
        if p.exists():
            return p
    return CONTEXT_DIR / names[0]


COMMUNITIES_PATH = _first_existing('comunidades_minam_light.geojson', 'comunidades_minam.geojson')
ANP_FILES = {
    'ANP Nacional': _first_existing('anp_nacional_light.geojson', 'anp_nacional.geojson'),
    'Zona Reservada': _first_existing('anp_zr_light.geojson', 'anp_zr.geojson'),
    'Area de Conservacion Regional': _first_existing('anp_acr_light.geojson', 'anp_acr.geojson'),
    'Area de Conservacion Privada': _first_existing('anp_acp_light.geojson', 'anp_acp.geojson'),
}
DEPARTMENTS_PATH = _first_existing('peru_departamentos_light.geojson', 'peru_departamentos.geojson')

DATE_START = '2020-01-01'
MAP_CENTER = [-9.2, -74.5]
MAP_ZOOM = 5

# False-positive thresholds (shown to user for transparency)
MODIS_CONF_MIN = 30
VIIRS_LOW = {'l', 'low'}
FRP_MIN = 0.0
VOLCANO_RADIUS_KM = 25.0

# Priority tiers (km from nearest community or protected area)
TIER_CRIT = 'CRITICO (dentro)'
TIER_ALTO = 'ALTO (<2 km)'
TIER_MOD = 'MODERADO (<5 km)'
TIER_BAS = 'MONITOREO (>=5 km)'
PRIORITY_MAX_POINTS = 30000

# =============================================================================
# DATA LOADING (cached in memory)
# =============================================================================


@st.cache_resource(show_spinner='Cargando FIRMS nacional...')
def load_firms():
    df = None
    if FIRMS_PATH.exists():
        try:
            df = pd.read_parquet(FIRMS_PATH)
        except Exception:
            df = None
    if df is None and FIRMS_CSV.exists():
        df = pd.read_csv(FIRMS_CSV, compression='gzip', low_memory=False,
                         parse_dates=['acq_date'])
        for c in ('confidence', '_source', 'fp_reason', 'zone'):
            df[c] = df[c].astype('category')
        df['is_valid'] = df['is_valid'].astype(bool)
    if df is None:
        return None
    df['acq_date'] = pd.to_datetime(df['acq_date'])
    return df


def _feature_ring(feat):
    g = feat.get('geometry')
    if not g or not g.get('coordinates'):
        return None
    coords = g['coordinates']
    if g['type'] == 'Polygon':
        return coords[0]
    rings = []
    for poly in coords:
        rings.append(poly[0])
    return rings


@st.cache_resource(show_spinner='Cargando comunidades (MINAM)...')
def load_communities():
    with open(COMMUNITIES_PATH, encoding='utf-8') as f:
        gj = json.load(f)
    rows = []
    polys = []
    for feat in gj['features']:
        p = feat['properties']
        rings = _feature_ring(feat)
        if rings is None:
            continue
        if rings and isinstance(rings[0], list) and isinstance(rings[0][0], list):
            ring = rings[0]
        else:
            ring = rings
        ring = np.array(ring)
        if ring.ndim != 2 or len(ring) < 3:
            continue
        rows.append({
            'nombre': p.get('nombre') or 'Sin nombre',
            'etnia': p.get('etnia') or '',
            'federacion': p.get('federacion') or '',
            'distrito': p.get('distrito') or '',
            'provincia': p.get('provincia') or '',
            'poblacion': p.get('poblacion') or '',
            'dpto': p.get('nomdpto') or '',
            'centroid_lat': ring[:, 1].mean(),
            'centroid_lon': ring[:, 0].mean(),
        })
        polys.append({
            'name': p.get('nombre') or 'Sin nombre',
            'etnia': p.get('etnia') or '',
            'distrito': p.get('distrito') or '',
            'bbox': [ring[:, 0].min(), ring[:, 1].min(),
                     ring[:, 0].max(), ring[:, 1].max()],
            'path': ring,
        })
    centroids = pd.DataFrame(rows)
    return centroids, polys


@st.cache_resource(show_spinner='Cargando areas protegidas (SERNANP)...')
def load_anps():
    all_polys = []
    for cat, path in ANP_FILES.items():
        if not path.exists():
            continue
        with open(path, encoding='utf-8') as f:
            gj = json.load(f)
        for feat in gj['features']:
            p = feat['properties']
            rings = _feature_ring(feat)
            if rings is None:
                continue
            if rings and isinstance(rings[0], list) and isinstance(rings[0][0], list):
                ring = rings[0]
            else:
                ring = rings
            ring = np.array(ring)
            if ring.ndim != 2 or len(ring) < 3:
                continue
            name = (p.get('anp_nomb') or p.get('zr_nomb') or p.get('acr_nomb')
                    or p.get('acp_nomb') or 'Sin nombre')
            all_polys.append({
                'name': str(name),
                'cat': cat,
                'bbox': [ring[:, 0].min(), ring[:, 1].min(),
                         ring[:, 0].max(), ring[:, 1].max()],
                'centroid_lat': ring[:, 1].mean(),
                'centroid_lon': ring[:, 0].mean(),
                'path': ring,
            })
    return all_polys


@st.cache_resource(show_spinner='Cargando departamentos...')
def load_departments():
    with open(DEPARTMENTS_PATH, encoding='utf-8') as f:
        gj = json.load(f)
    polys = []
    names = []
    for feat in gj['features']:
        rings = _feature_ring(feat)
        if rings is None:
            continue
        if rings and isinstance(rings[0], list) and isinstance(rings[0][0], list):
            ring = rings[0]
        else:
            ring = rings
        ring = np.array(ring)
        if ring.ndim != 2 or len(ring) < 3:
            continue
        polys.append(ring)
        nm = feat['properties'].get('shapeName', '')
        names.append('Lima' if nm == 'Municipalidad Metropolitana de Lima' else nm)
    return polys, names


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371.0088 * 2 * np.arcsin(np.sqrt(a))


def _cell_index(value, cell):
    return int(np.floor(value / cell))


def build_centroid_buckets(lats, lons, cell=0.25):
    buckets = {}
    for i, (la, lo) in enumerate(zip(lats, lons)):
        key = (_cell_index(la, cell), _cell_index(lo, cell))
        buckets.setdefault(key, []).append(i)
    return buckets


def nearest_centroid_bucketed(p_lat, p_lon, c_lat, c_lon, buckets, cell=0.25):
    """Nearest centroid per point via coarse grid; brute-force fallback for far points."""
    n = len(p_lat)
    out_d = np.full(n, np.inf)
    out_i = np.full(n, -1, dtype=int)
    pending = []
    for k in range(n):
        la, lo = p_lat[k], p_lon[k]
        ci, cj = _cell_index(la, cell), _cell_index(lo, cell)
        best_d, best_i = np.inf, -1
        for dci in (-1, 0, 1):
            for dcj in (-1, 0, 1):
                cands = buckets.get((ci + dci, cj + dcj))
                if not cands:
                    continue
                d = haversine(la, lo, c_lat[cands], c_lon[cands])
                m = np.argmin(d)
                if d[m] < best_d:
                    best_d, best_i = d[m], cands[m]
        out_d[k], out_i[k] = best_d, best_i
        if best_i < 0:
            pending.append(k)
    if pending:
        for k in pending:
            d = haversine(p_lat[k], p_lon[k], c_lat, c_lon)
            m = np.argmin(d)
            out_d[k], out_i[k] = d[m], m
    return out_d, out_i


def build_polygon_buckets(polys, cell=0.25):
    """Map grid cell -> list of polygon indices whose bbox touches the cell."""
    buckets = {}
    for i, poly in enumerate(polys):
        xmin, ymin, xmax, ymax = poly['bbox']
        i0, j0 = _cell_index(xmin, cell), _cell_index(ymin, cell)
        i1, j1 = _cell_index(xmax, cell), _cell_index(ymax, cell)
        for ci in range(i0, i1 + 1):
            for cj in range(j0, j1 + 1):
                buckets.setdefault((ci, cj), []).append(i)
    return buckets


def _points_by_cell(p_lat, p_lon, cell):
    cells = {}
    for k in range(len(p_lat)):
        ci, cj = _cell_index(p_lon[k], cell), _cell_index(p_lat[k], cell)
        cells.setdefault((ci, cj), []).append(k)
    return cells


def _neighbor_candidates(cell, buckets, ring=1):
    cands = set()
    ci, cj = cell
    for dci in range(-ring, ring + 1):
        for dcj in range(-ring, ring + 1):
            cands.update(buckets.get((ci + dci, cj + dcj), []))
    return cands


def _bbox_mask(pts, bbox):
    xmin, ymin, xmax, ymax = bbox
    return ((pts[:, 0] >= xmin) & (pts[:, 0] <= xmax) &
            (pts[:, 1] >= ymin) & (pts[:, 1] <= ymax))


def inside_polygons_batched(p_lat, p_lon, polys, buckets, cell=0.25, ring=1):
    """Per-point list of polygon names it falls inside (deduped, batched by cell)."""
    from matplotlib.path import Path
    n = len(p_lat)
    names_by_point = [[] for _ in range(n)]
    by_cell = _points_by_cell(p_lat, p_lon, cell)
    for (ci, cj), pidx in by_cell.items():
        pts = np.column_stack([p_lon[pidx], p_lat[pidx]])
        for pi in _neighbor_candidates((ci, cj), buckets, ring):
            poly = polys[pi]
            sub = _bbox_mask(pts, poly['bbox'])
            if not sub.any():
                continue
            sub_idx = np.flatnonzero(sub)
            path = Path(poly['path'])
            mask = path.contains_points(pts[sub])
            for si, isin in zip(sub_idx, mask):
                if isin:
                    names_by_point[pidx[si]].append(poly['name'])
    return [sorted(set(names_by_point[k])) for k in range(n)]


def _downsample(verts, max_v=150):
    if len(verts) <= max_v:
        return verts
    idx = np.linspace(0, len(verts) - 1, max_v).astype(int)
    return verts[idx]


def min_dist_to_polys_batched(p_lat, p_lon, polys, buckets, cell=0.25, ring=3):
    """Per-point min distance (km) to polygon boundary; 0 if inside. Blank if >~200 km."""
    from matplotlib.path import Path
    n = len(p_lat)
    out_d = np.full(n, np.inf)
    out_i = np.full(n, -1, dtype=int)
    light = [(poly, _downsample(poly['path']), Path(poly['path'])) for poly in polys]
    by_cell = _points_by_cell(p_lat, p_lon, cell)
    for (ci, cj), pidx in by_cell.items():
        pts = np.column_stack([p_lon[pidx], p_lat[pidx]])
        for pi in _neighbor_candidates((ci, cj), buckets, ring):
            poly, verts, path = light[pi]
            sub = _bbox_mask(pts, poly['bbox'])
            if not sub.any():
                continue
            sub_idx = np.flatnonzero(sub)
            inside = path.contains_points(pts[sub])
            for si, isin in zip(sub_idx, inside):
                if isin:
                    out_d[pidx[si]] = 0.0
                    out_i[pidx[si]] = pi
            outside = sub_idx[~inside]
            if len(outside) == 0:
                continue
            o_pts = pts[sub][~inside]
            for si, op in zip(outside, o_pts):
                d = haversine(op[1], op[0], verts[:, 1], verts[:, 0]).min()
                if d < out_d[pidx[si]]:
                    out_d[pidx[si]] = d
                    out_i[pidx[si]] = pi
    out_d[out_i < 0] = np.inf
    return out_d, out_i


def compute_proximity(df, com_cent, com_polys, anp_polys,
                      com_buckets, anp_poly_buckets, poly_buckets_com, poly_buckets_anp):
    """Add proximity columns + priority tier to a hotspot dataframe (bounded size)."""
    if df.empty:
        return df.assign(nearest_com='', dist_com_km='', nearest_anp='', dist_anp_km='',
                         inside_com='', inside_anp='', tier='', min_dist_km=float('nan'))
    lat = df['latitude'].values
    lon = df['longitude'].values
    com_names = com_cent['nombre'].values
    anp_names = np.array([p['name'] for p in anp_polys])

    dc, ic = nearest_centroid_bucketed(lat, lon, com_cent['centroid_lat'].values,
                                       com_cent['centroid_lon'].values, com_buckets)
    da, ia = min_dist_to_polys_batched(lat, lon, anp_polys, anp_poly_buckets)

    inc = inside_polygons_batched(lat, lon, com_polys, poly_buckets_com)
    ina = inside_polygons_batched(lat, lon, anp_polys, poly_buckets_anp, ring=1)

    df = df.copy()
    df['dist_com_km'] = np.where(ic >= 0, np.round(dc, 1).astype(str), '')
    df['nearest_com'] = np.where(ic >= 0, com_names[ic], '')
    df['dist_anp_km'] = np.where(ia >= 0, np.round(da, 1).astype(str), '')
    df['nearest_anp'] = np.where(ia >= 0, anp_names[ia], '')
    df['inside_com'] = ['; '.join(h) for h in inc]
    df['inside_anp'] = ['; '.join(h) for h in ina]

    d_com = pd.to_numeric(df['dist_com_km'].replace('', np.nan))
    d_anp = pd.to_numeric(df['dist_anp_km'].replace('', np.nan))
    df['min_dist_km'] = d_com.combine(d_anp, lambda a, b: min(a, b) if pd.notna(a) and pd.notna(b) else (a if pd.notna(a) else b))

    crit = (df['inside_com'] != '') | (df['inside_anp'] != '')
    alto = df['min_dist_km'] < 2.0
    mod = df['min_dist_km'] < 5.0
    df['tier'] = TIER_BAS
    df.loc[mod, 'tier'] = TIER_MOD
    df.loc[alto, 'tier'] = TIER_ALTO
    df.loc[crit, 'tier'] = TIER_CRIT
    return df


def priority_rank(df):
    """Sort by priority: CRITICO -> ALTO -> MODERADO -> MONITOREO, then distance."""
    order = {TIER_CRIT: 0, TIER_ALTO: 1, TIER_MOD: 2, TIER_BAS: 3}
    if df.empty:
        return df
    df = df.copy()
    df['_porder'] = df['tier'].map(order).fillna(9)
    df = df.sort_values(['_porder', 'min_dist_km', 'frp'],
                        ascending=[True, True, False], na_position='last')
    return df.drop(columns='_porder')


def build_heatmap_time_data(df, granularity='W', max_points=8000):
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
        data.append([[la, lo, it] for la, lo, it in
                     zip(sample['latitude'], sample['longitude'], sample['_intensity'])])
        index.append(b.start_time.strftime(label_fmt))
    return data, index


def add_context_layers(m, com_cent, com_polys, anp_polys, dpt_polys, dpt_names,
                       show_com=True, show_anp=True, show_dpt=True):
    if show_dpt and dpt_polys:
        fg = folium.FeatureGroup(name='Departamentos', show=False)
        for ring, name in zip(dpt_polys, dpt_names):
            folium.Polygon(
                locations=[[la, lo] for lo, la in ring],
                color='#999', weight=1, fill=False,
                popup=name
            ).add_to(fg)
        fg.add_to(m)

    if show_anp and anp_polys:
        colors = {
            'ANP Nacional': '#1e8449',
            'Zona Reservada': '#145a32',
            'Area de Conservacion Regional': '#2e86c1',
            'Area de Conservacion Privada': '#8e44ad',
        }
        fg = folium.FeatureGroup(name='Areas protegidas (SERNANP)', show=True)
        for poly in anp_polys:
            folium.Polygon(
                locations=[[la, lo] for lo, la in poly['path']],
                color=colors.get(poly['cat'], '#333'),
                weight=1.5, fill=True, fill_opacity=0.12,
                popup=f"<b>{poly['name']}</b><br>{poly['cat']}"
            ).add_to(fg)
        fg.add_to(m)

    if show_com and len(com_cent) > 0:
        fg = folium.FeatureGroup(name='Comunidades nativas (MINAM)', show=False)
        for _, row in com_cent.iterrows():
            folium.CircleMarker(
                location=[row['centroid_lat'], row['centroid_lon']],
                radius=2, color='#e67e22', fill=True, fill_opacity=0.6,
                popup=folium.Popup(
                    f"<b>{row['nombre']}</b><br>Etnia: {row['etnia']}<br>"
                    f"Distrito: {row['distrito']}<br>Departamento: {row['dpto']}",
                    max_width=250)
            ).add_to(fg)
        fg.add_to(m)
    return m


def hotspot_color(conf):
    conf = str(conf).lower()
    if conf in ['h', 'high', 'n', 'nominal']:
        return 'red'
    if conf.replace('.', '', 1).isdigit() and float(conf) >= 80:
        return 'red'
    return 'orange'


def build_hotspot_map(df, com_cent, com_polys, anp_polys, dpt_polys, dpt_names,
                      show_com=True, show_anp=True, show_dpt=True, title_suffix='',
                      color_by='conf'):
    m = folium.Map(location=MAP_CENTER, zoom_start=MAP_ZOOM, tiles='CartoDB positron')
    m = add_context_layers(m, com_cent, com_polys, anp_polys, dpt_polys, dpt_names,
                           show_com, show_anp, show_dpt)

    fg = folium.FeatureGroup(name=f'Focos ({len(df):,})')
    sample = df.sample(min(2000, len(df)), random_state=42) if len(df) > 2000 else df
    for _, row in sample.iterrows():
        if 'is_valid' in row and not row['is_valid']:
            color, radius, fill = '#95a5a6', 3, 0.4
        elif color_by == 'tier' and row.get('tier'):
            tier = str(row['tier'])
            if 'CRITICO' in tier:
                color, radius, fill = '#8b0000', 6, 0.85
            elif 'ALTO' in tier:
                color, radius, fill = '#e74c3c', 5, 0.7
            elif 'MODERADO' in tier:
                color, radius, fill = '#f39c12', 4, 0.6
            else:
                color, radius, fill = '#27ae60', 3, 0.5
        else:
            color = hotspot_color(row.get('confidence', ''))
            radius, fill = 3, 0.55
        popup = (f"<b>{row['acq_date'].strftime('%d/%m/%Y')}</b><br>"
                 f"<b>FRP:</b> {row.get('frp', 'N/A')} MW<br>"
                 f"<b>Confianza:</b> {row.get('confidence', 'N/A')}<br>"
                 f"<b>Sensor:</b> {row.get('_source', 'N/A')}<br>"
                 f"<b>Depto:</b> {row.get('zone', 'N/A')}<br>"
                 f"<b>Prioridad:</b> {row.get('tier', '')}<br>"
                 f"<b>Comunidad mas cercana:</b> {row.get('nearest_com', '')} "
                 f"({row.get('dist_com_km', '')} km)<br>"
                 f"<b>Dentro de comunidad:</b> {row.get('inside_com', '')}<br>"
                 f"<b>ANP mas cercana:</b> {row.get('nearest_anp', '')} "
                 f"({row.get('dist_anp_km', '')} km)<br>"
                 f"<b>Dentro de ANP:</b> {row.get('inside_anp', '')}<br>"
                 f"<b>Coords:</b> {row['latitude']:.4f}, {row['longitude']:.4f}")
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=radius, color=color, fill=True, fill_opacity=fill,
            popup=folium.Popup(popup, max_width=280)
        ).add_to(fg)
    fg.add_to(m)

    folium.LayerControl().add_to(m)

    if color_by == 'tier':
        marker_legend = ('<span style="color: #8b0000;">●</span> Critico (dentro de comunidad/ANP)<br>'
                         '<span style="color: #e74c3c;">●</span> Alto (<2 km)<br>'
                         '<span style="color: #f39c12;">●</span> Moderado (<5 km)<br>'
                         '<span style="color: #27ae60;">●</span> Monitoreo (>=5 km)<br>')
    else:
        marker_legend = ('<span style="color: red;">●</span> Alta/Nominal confianza<br>'
                         '<span style="color: orange;">●</span> Baja confianza<br>')
    legend_html = f"""
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 1000;
         background-color: white; padding: 10px 12px; border-radius: 8px;
         border: 2px solid #333; font-family: Arial; font-size: 12px;
         box-shadow: 2px 2px 6px rgba(0,0,0,0.3);">
        <b>FOCOS DE CALOR</b><br>
        {marker_legend}
        <span style="color: #95a5a6;">●</span> Excluido (falso positivo)<br>
        <span style="color: #1e8449;">■</span> ANP Nacional / ZR<br>
        <span style="color: #2e86c1;">■</span> ACR<br>
        <span style="color: #8e44ad;">■</span> ACP<br>
        <span style="color: #e67e22;">●</span> Comunidad nativa (MINAM)
        {('<br><span style="color:#555; font-size:10px;">' + title_suffix + '</span>') if title_suffix else ''}
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    return m


# =============================================================================
# STREAMLIT APP
# =============================================================================

def main():
    st.set_page_config(page_title="Fire DT - Phase 1 (Peru)", page_icon="🔥",
                       layout="wide", initial_sidebar_state="expanded")

    st.markdown("""
    <style>
    .main-header { font-size: 2.2rem; font-weight: bold; color: #e74c3c; }
    .tier-crit { color: #8b0000; font-weight: bold; }
    .tier-alto { color: #e74c3c; font-weight: bold; }
    .tier-mod  { color: #f39c12; font-weight: bold; }
    .tier-bas  { color: #27ae60; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<p class="main-header">Digital Twin Wildfire - Phase 1 (DEV)</p>',
                unsafe_allow_html=True)
    st.markdown("**Todo el Peru** | FIRMS (NASA) + Comunidades nativas (MINAM) "
                "+ Areas protegidas (SERNANP)")
    st.caption("**DEV**: no desplegada aun. Valida en local antes de publicar.")
    st.markdown("---")

    # Load heavy data once
    with st.spinner('Cargando datos...'):
        df = load_firms()
        com_cent, com_polys = load_communities()
        anp_polys = load_anps()
        dpt_polys, dpt_names = load_departments()
    if df is None:
        st.error('No se encontraron datos FIRMS nacionales. Ejecuta '
                 'scripts/build_firms_peru.py primero.')
        st.stop()

    # Precompute grids once
    com_buckets = build_centroid_buckets(com_cent['centroid_lat'].values,
                                         com_cent['centroid_lon'].values)
    poly_buckets_com = build_polygon_buckets(com_polys)
    poly_buckets_anp = build_polygon_buckets(anp_polys)

    # Sidebar filters
    with st.sidebar:
        st.header('Filtros')
        min_d = df['acq_date'].min().date()
        max_d = df['acq_date'].max().date()
        date_range = st.date_input(
            'Rango de fechas',
            value=(max_d - timedelta(days=90), max_d),
            min_value=min_d, max_value=max_d)
        dpts = sorted(df['zone'].dropna().unique())
        sel_dpt = st.selectbox('Departamento', ['Todos'] + [d for d in dpts if d != 'No data'])
        sensors = sorted(df['_source'].unique())
        sel_sensor = st.selectbox('Sensor', ['Todos'] + sensors)
        show_valid = st.checkbox('Solo focos validos (filtro falsos positivos)', value=True)
        st.caption('Si se desactiva, los focos excluidos se muestran en gris en el mapa.')

        show_com = st.checkbox('Mostrar comunidades nativas (MINAM)', value=False)
        show_anp = st.checkbox('Mostrar areas protegidas (SERNANP)', value=True)
        show_dpt = st.checkbox('Mostrar departamentos', value=False)

        st.markdown('---')
        st.caption('**Fuentes de datos**')
        st.caption('Focos: FIRMS/NASA (VIIRS SNPP/NOAA-20/21 + MODIS), hasta '
                   f'{max_d:%d/%m/%Y}')
        st.caption('Comunidades: MINAM GeoServidor (ServicioTematico/30)')
        st.caption('ANP: SERNANP geoservicios (ANP/ZR/ACR/ACP)')
        st.caption('Departamentos: GeoBoundaries ADM1')
        st.caption('Filtro falsos positivos: tipo no-vegetacion + confianza '
                   f'(MODIS<{MODIS_CONF_MIN}, VIIRS low) + FRP<={FRP_MIN} + '
                   f'radio de volcan {VOLCANO_RADIUS_KM:.0f} km')

    # Apply temporal + department + sensor filters
    mask = pd.Series([True] * len(df))
    if date_range and isinstance(date_range, (tuple, list)) and len(date_range) == 2 and all(date_range):
        mask &= (df['acq_date'].dt.date >= date_range[0]) & (df['acq_date'].dt.date <= date_range[1])
    if sel_dpt != 'Todos':
        mask &= (df['zone'] == sel_dpt)
    if sel_sensor != 'Todos':
        mask &= (df['_source'] == sel_sensor)
    filtered_all = df[mask].copy()
    if show_valid:
        filtered = filtered_all[filtered_all['is_valid']].copy()
    else:
        filtered = filtered_all.copy()

    if filtered.empty:
        st.warning('Sin focos validos en el rango seleccionado. '
                   'Desmarca "Solo focos validos" para ver los excluidos.')
        st.stop()

    # Priority analysis: full proximity only for the most recent N valid fires
    valid_fires = filtered_all[filtered_all['is_valid']].sort_values(
        'acq_date', ascending=False)
    prio_set = valid_fires.head(PRIORITY_MAX_POINTS)
    with st.spinner('Calculando proximidad a comunidades y areas protegidas...'):
        ranked = priority_rank(compute_proximity(
            prio_set, com_cent, com_polys, anp_polys,
            com_buckets, poly_buckets_anp, poly_buckets_com, poly_buckets_anp))

    tab1, tab2, tab3, tab4 = st.tabs(
        ['Mapa de focos', 'Priorizacion', 'Falsos positivos', 'Resumen'])

    # ---------------- TAB 1: MAP ----------------
    with tab1:
        st.subheader('Mapa de focos de calor - Peru')
        sample_map = filtered.sample(min(2000, len(filtered)), random_state=42).copy()
        sample_map = compute_proximity(sample_map, com_cent, com_polys, anp_polys,
                                       com_buckets, poly_buckets_anp,
                                       poly_buckets_com, poly_buckets_anp)
        m = build_hotspot_map(sample_map, com_cent, com_polys, anp_polys,
                              dpt_polys, dpt_names,
                              show_com=show_com, show_anp=show_anp, show_dpt=show_dpt)
        st_folium(m, width='100%', height=620)
        st.caption(f'{len(filtered):,} focos en el periodo. Muestra aleatoria hasta 2000 '
                   'puntos en el mapa para mantener la fluidez. Ver capa '
                   '"Focos (n)" y detalles al hacer clic.')

        st.markdown('---')
        st.subheader('Animacion temporal de intensidad (FRP)')
        if len(filtered[filtered['is_valid']]) > 0:
            data, index = build_heatmap_time_data(filtered[filtered['is_valid']])
            if data:
                m_anim = folium.Map(location=MAP_CENTER, zoom_start=MAP_ZOOM,
                                    tiles='CartoDB positron')
                HeatMapWithTime(
                    data, index=index, radius=10, blur=15,
                    display_index=True, position='topright',
                    min_speed=0.1, max_speed=10,
                    gradient={0.0: 'blue', 0.25: 'cyan', 0.5: 'lime',
                              0.75: 'yellow', 1.0: 'red'},
                ).add_to(m_anim)
                legend_html = """
                <div style="position: fixed; bottom: 40px; left: 30px; z-index: 1000;
                     background-color: white; padding: 10px 12px; border-radius: 8px;
                     border: 2px solid #333; font-family: Arial; font-size: 12px;
                     box-shadow: 2px 2px 6px rgba(0,0,0,0.3);">
                    <b>INTENSIDAD (FRP)</b><br>
                    <div style="width: 150px; height: 12px; border-radius: 3px; margin: 6px 0;
                         background: linear-gradient(90deg, blue, cyan, lime, yellow, red);"></div>
                    <span>Baja</span><span style="margin-left: 90px;">Alta</span>
                </div>
                """
                m_anim.get_root().html.add_child(folium.Element(legend_html))
                st.iframe(m_anim.get_root().render(), width=1200, height=600)
                st.caption('Potencia radiativa del fuego (FRP, MW) por periodo. '
                           'Colores normalizados al foco mas intenso de cada paso temporal.')
        else:
            st.info('Sin focos validos en el periodo.')

    # ---------------- TAB 2: PRIORIZATION ----------------
    with tab2:
        st.subheader('Priorizacion de focos por proximidad a comunidades y areas protegidas')
        st.markdown(
            'Ordenado de mayor a menor prioridad. Prioridad = mas cercano a '
            'comunidades nativas y/o areas protegidas primero.')
        st.markdown(
            f'- **{TIER_CRIT}**: dentro de una comunidad o area protegida<br>'
            f'- **{TIER_ALTO}**: a menos de 2 km<br>'
            f'- **{TIER_MOD}**: entre 2 y 5 km<br>'
            f'- **{TIER_BAS}**: a 5 km o mas', unsafe_allow_html=True)

        tier_counts = ranked['tier'].value_counts().reindex(
            [TIER_CRIT, TIER_ALTO, TIER_MOD, TIER_BAS]).fillna(0).astype(int)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(TIER_CRIT, f'{tier_counts.get(TIER_CRIT, 0):,}')
        c2.metric(TIER_ALTO, f'{tier_counts.get(TIER_ALTO, 0):,}')
        c3.metric(TIER_MOD, f'{tier_counts.get(TIER_MOD, 0):,}')
        c4.metric(TIER_BAS, f'{tier_counts.get(TIER_BAS, 0):,}')

        show_top = st.selectbox('Cuandos mostrar', [100, 250, 500, 1000], index=2)
        top = ranked.head(show_top).copy()

        disp = top[[
            'acq_date', 'zone', 'latitude', 'longitude', 'confidence', 'frp',
            '_source', 'nearest_com', 'dist_com_km', 'nearest_anp', 'dist_anp_km',
            'inside_com', 'inside_anp', 'tier'
        ]].rename(columns={
            'acq_date': 'Fecha', 'zone': 'Departamento', 'latitude': 'Lat',
            'longitude': 'Lon', 'confidence': 'Confianza', 'frp': 'FRP (MW)',
            '_source': 'Sensor', 'nearest_com': 'Comunidad cercana',
            'dist_com_km': 'Dist. comunidad (km)', 'nearest_anp': 'ANP cercana',
            'dist_anp_km': 'Dist. ANP (km)', 'inside_com': 'Dentro de comunidad',
            'inside_anp': 'Dentro de ANP', 'tier': 'Prioridad'
        })
        disp['Fecha'] = disp['Fecha'].dt.strftime('%d/%m/%Y')
        disp = disp.fillna('')
        st.dataframe(disp, height=520, width='stretch')

        csv = disp.to_csv(index=False).encode('utf-8-sig')
        st.download_button('Descargar priorizacion (CSV)', csv,
                           'priorizacion_focos.csv', 'text/csv')

        st.markdown('---')
        st.subheader('Mapa de priorizacion (top 100)')
        m2 = build_hotspot_map(ranked.head(100), com_cent, com_polys, anp_polys,
                               dpt_polys, dpt_names,
                               show_com=True, show_anp=True, show_dpt=False,
                               title_suffix='Top 100 por prioridad', color_by='tier')
        st_folium(m2, width='100%', height=600)
        st.caption('La priorizacion se calcula sobre los '
                   f'{min(len(valid_fires), PRIORITY_MAX_POINTS):,} focos validos '
                   'mas recientes del periodo seleccionado.')

    # ---------------- TAB 3: FALSE POSITIVES ----------------
    with tab3:
        st.subheader('Filtro de falsos positivos')
        total = len(filtered_all)
        n_valid = int(filtered_all['is_valid'].sum())
        n_excl = total - n_valid
        c1, c2, c3 = st.columns(3)
        c1.metric('Total focos', f'{total:,}')
        c2.metric('Validos (vegetacion)', f'{n_valid:,}')
        c3.metric('Excluidos', f'{n_excl:,} ({100 * n_excl / total:.1f}%)')

        st.markdown('Criterios de exclusion aplicados (transparencia):')
        st.markdown(
            f'1. **Tipo no-vegetacion**: registros marcados por NASA como volcan, '
            f'plataforma marina/gas o fuente estatica (industria).<br>'
            f'2. **Baja confianza**: VIIRS = low; MODIS < {MODIS_CONF_MIN}%.<br>'
            f'3. **Sin FRP**: senal debil, FRP <= {FRP_MIN}.<br>'
            f'4. **Radio de volcan**: a menos de {VOLCANO_RADIUS_KM:.0f} km de un '
            f'volcan activo del cinturon sur (doble seguro).', unsafe_allow_html=True)

        if n_excl > 0:
            excl = filtered_all[~filtered_all['is_valid']]
            counts = excl['fp_reason'].value_counts()
            fig = px.bar(counts, orientation='h',
                         title='Focos excluidos por motivo',
                         color_discrete_sequence=['#7f8c8d'])
            fig.update_layout(xaxis_title='Focos', yaxis_title='', height=260)
            st.plotly_chart(fig, width='stretch')

            st.markdown('Muestra de focos excluidos:')
            ex_disp = excl[[
                'acq_date', 'zone', 'latitude', 'longitude', 'confidence', 'frp',
                'type', 'fp_reason', '_source'
            ]].sort_values('acq_date', ascending=False).head(200).rename(columns={
                'acq_date': 'Fecha', 'zone': 'Departamento', 'latitude': 'Lat',
                'longitude': 'Lon', 'confidence': 'Confianza', 'frp': 'FRP',
                'type': 'Tipo NASA', 'fp_reason': 'Motivo', '_source': 'Sensor'
            })
            ex_disp['Fecha'] = ex_disp['Fecha'].dt.strftime('%d/%m/%Y')
            st.dataframe(ex_disp, height=420, width='stretch')

    # ---------------- TAB 4: SUMMARY ----------------
    with tab4:
        st.subheader('Resumen')
        valid = filtered_all[filtered_all['is_valid']]
        c1, c2, c3 = st.columns(3)
        c1.metric('Focos validos', f'{len(valid):,}')
        c2.metric('Departamentos con fuego', f'{valid["zone"].nunique()}')
        c3.metric('FRP medio (MW)', f'{valid["frp"].mean():.1f}')

        if len(valid) > 0:
            by_dpt = valid.groupby('zone').size().reset_index(name='focos').sort_values(
                'focos', ascending=False)
            fig = px.bar(by_dpt.head(15), x='focos', y='zone', orientation='h',
                         title='Focos validos por departamento',
                         color_discrete_sequence=['#e74c3c'])
            fig.update_layout(xaxis_title='Focos', yaxis_title='', height=500)
            st.plotly_chart(fig, width='stretch')

        if UPDATE_LOG.exists():
            with open(UPDATE_LOG) as f:
                info = json.load(f)
            st.caption(f"Ultima actualizacion dataset: {info.get('updated_at', 'N/A')} | "
                       f"Datos hasta: {info.get('last_date', 'N/A')} | "
                       f"Registros: {info.get('total_records', 0):,}")


if __name__ == '__main__':
    main()
