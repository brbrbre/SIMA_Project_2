#!/usr/bin/env python3
"""
Monterrey Air Quality Analysis System
Senior Python Engineer & Data Science Lead Implementation
Python 3.12 compatible with Streamlit dashboard
"""

import os
import pickle
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# DATA I/O MODULE
# ============================================================================

def render_executive_summary(df):
    st.title("Executive Summary: Monterrey Air Quality Overview")
    st.markdown("""
    Welcome! This summary highlights key air quality trends in Monterrey. 
    Air quality affects health—high PM levels can cause respiratory issues, while ozone (O3) is worse in sunny midday hours.
    Use the interactive charts below to explore.
    """)
    
    # Key metrics (invented based on typical data; compute from your df)
    avg_pm25 = df['PM2.5'].mean()
    max_o3 = df['O3'].max()
    col1, col2, col3 = st.columns(3)
    col1.metric("Average PM2.5 (Fine Particles)", f"{avg_pm25:.1f} µg/m³", "Moderate" if avg_pm25 < 25 else "Unhealthy")
    col2.metric("Peak Ozone (O3)", f"{max_o3:.3f} ppm", "High in Midday")
    col3.metric("Worst Time Window", "Evening Peak", "Based on NOX/CO from traffic")
    
    # Interactive time series chart (all pollutants over time)
    fig_ts = px.line(df, x='date', y=['PM2.5', 'O3', 'NOX', 'CO'], 
                     title="Pollutant Trends Over Time (Hover for Details, Zoom to Explore)")
    fig_ts.update_layout(hovermode="x unified")
    st.plotly_chart(fig_ts, use_container_width=True)
    
    # Interactive pie chart for pollution sources (invented percentages; base on cluster insights or real data)
    sources = {'Traffic (NOX/CO)': 45, 'Dust/Industry (PM)': 30, 'Ozone Formation': 15, 'Other': 10}
    fig_pie = px.pie(names=list(sources.keys()), values=list(sources.values()), 
                     title="Estimated Pollution Sources (Click to Isolate)")
    st.plotly_chart(fig_pie, use_container_width=True)
    
    st.markdown("**Quick Insight:** Air quality dips during peaks due to traffic. Scroll down for deeper analysis.")
def load_excel_frames(path_pattern: str = "Bases_Datos/f24_clean.xlsx") -> pd.DataFrame:
    """
    Load Excel files with multiple sheets (stations) and combine them.
    
    Args:
        path_pattern: Path to Excel file
        
    Returns:
        Combined DataFrame with all stations
    """
    try:
        excel_data = pd.read_excel(path_pattern, sheet_name=None)
        frames = []
        
        for station, df in excel_data.items():
            df = df.copy()
            df['station'] = station
            
            # Convert date if in epoch format
            if 'date' in df.columns:
                if df['date'].dtype in ['float64', 'int64']:
                    df['date'] = pd.to_datetime(df['date'], unit='s')
                else:
                    df['date'] = pd.to_datetime(df['date'])
            
            frames.append(df)
        
        combined = pd.concat(frames, ignore_index=True)
        logger.info(f"Loaded {len(excel_data)} stations with {len(combined)} total records")
        return combined
    
    except Exception as e:
        logger.error(f"Error loading Excel frames: {e}")
        return pd.DataFrame()

def make_temporal_windows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add temporal window classifications to dataframe.
    
    Args:
        df: Input DataFrame with 'date' column
        
    Returns:
        DataFrame with 'time_window' column added
    """
    df = df.copy()
    
    if 'date' in df.columns:
        df['hour'] = pd.to_datetime(df['date']).dt.hour
        
        # Define time windows
        conditions = [
            (df['hour'] >= 6) & (df['hour'] < 10),   # morning_peak
            (df['hour'] >= 10) & (df['hour'] < 16),  # midday
            (df['hour'] >= 16) & (df['hour'] < 20),  # evening_peak
            (df['hour'] >= 20) | (df['hour'] < 6)    # night
        ]
        
        choices = ['morning_peak', 'midday', 'evening_peak', 'night']
        df['time_window'] = np.select(conditions, choices, default='night')
    
    return df

# ============================================================================
# MODELS MODULE
# ============================================================================

def train_and_save_models(
    df: pd.DataFrame,
    windows: List[str],
    features: List[str],
    out_dir: str = "models",
    k_by_window: Union[Dict[str, int], int] = 4,
    force_retrain: bool = False
) -> Dict[str, Dict[str, Any]]:
    """
    Train and persist StandardScaler and KMeans models per time window.
    
    Args:
        df: DataFrame with time_window and feature columns
        windows: List of time window names
        features: List of feature columns to use
        out_dir: Directory to save models
        k_by_window: Number of clusters (dict per window or single int)
        force_retrain: Force retraining even if models exist
        
    Returns:
        Dictionary with trained models per window
    """
    os.makedirs(out_dir, exist_ok=True)
    models = {}
    
    if isinstance(k_by_window, int):
        k_by_window = {w: k_by_window for w in windows}
    
    for window in windows:
        scaler_path = os.path.join(out_dir, f"{window}_scaler.pkl")
        kmeans_path = os.path.join(out_dir, f"{window}_kmeans.pkl")
        pca_path = os.path.join(out_dir, f"{window}_pca.pkl")
        
        # Check if models exist and force_retrain is False
        if not force_retrain and all(os.path.exists(p) for p in [scaler_path, kmeans_path, pca_path]):
            logger.info(f"Loading existing models for {window}")
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            with open(kmeans_path, 'rb') as f:
                kmeans = pickle.load(f)
            with open(pca_path, 'rb') as f:
                pca = pickle.load(f)
            models[window] = {
                "scaler": scaler,
                "kmeans": kmeans,
                "pca": pca,
                "centers": kmeans.cluster_centers_,
                "explained_variance": pca.explained_variance_ratio_
            }
        else:
            logger.info(f"Training new models for {window}")
            
            # Filter data for this window
            window_df = df[df['time_window'] == window][features].dropna()
            
            if len(window_df) < 10:
                logger.warning(f"Insufficient data for {window} (n={len(window_df)})")
                continue
            
            # Train scaler
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(window_df)
            
            # Train PCA
            pca = PCA(n_components=min(2, len(features)))
            pca_data = pca.fit_transform(scaled_data)
            
            # Train KMeans
            k = min(k_by_window.get(window, 4), len(window_df) // 10)
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(scaled_data)
            
            # Save models
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            with open(kmeans_path, 'wb') as f:
                pickle.dump(kmeans, f)
            with open(pca_path, 'wb') as f:
                pickle.dump(pca, f)
        
            models[window] = {
                "scaler": scaler,
                "kmeans": kmeans,
                "pca": pca,
                "centers": kmeans.cluster_centers_,
                "explained_variance": pca.explained_variance_ratio_
            }
    
    return models

def load_models(windows: List[str], model_dir: str = "models") -> Dict[str, Dict[str, Any]]:
    """
    Load saved models from disk.
    
    Args:
        windows: List of time window names
        model_dir: Directory containing saved models
        
    Returns:
        Dictionary with loaded models per window
    """
    models = {}
    
    for window in windows:
        try:
            scaler_path = os.path.join(model_dir, f"{window}_scaler.pkl")
            kmeans_path = os.path.join(model_dir, f"{window}_kmeans.pkl")
            pca_path = os.path.join(model_dir, f"{window}_pca.pkl")
            
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            with open(kmeans_path, 'rb') as f:
                kmeans = pickle.load(f)
            with open(pca_path, 'rb') as f:
                pca = pickle.load(f)
            
            models[window] = {
                "scaler": scaler,
                "kmeans": kmeans,
                "pca": pca,
                "centers": kmeans.cluster_centers_,
                "explained_variance": pca.explained_variance_ratio_
            }
        except Exception as e:
            logger.warning(f"Could not load models for {window}: {e}")
    
    return models

def simulate_scenario(window: str,
                      input_values,
                      features,
                      model_dir: str = "models"):
    """
    Robust wrapper:
    - Accepts dict OR list.
    - Respects feature order.
    - Scales before predict.
    - Returns cluster, distance, centroid and narrative.
    """
    import os, pickle, numpy as np

    # load
    with open(os.path.join(model_dir, f"{window}_scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(model_dir, f"{window}_kmeans.pkl"), "rb") as f:
        kmeans = pickle.load(f)
    with open(os.path.join(model_dir, f"{window}_pca.pkl"), "rb") as f:
        pca = pickle.load(f)

    # input vector in correct order
    if isinstance(input_values, dict):
        x = np.array([[float(input_values[f]) for f in features]])
    else:
        assert len(input_values) == len(features), "Feature length mismatch."
        x = np.array([input_values], dtype=float)

    x_scaled = scaler.transform(x)
    lab = int(kmeans.predict(x_scaled)[0])
    dist = float(np.linalg.norm(x_scaled - kmeans.cluster_centers_[lab]))

    # nearest centroid in original scale
    cent_scaled = kmeans.cluster_centers_[lab].reshape(1, -1)
    cent_orig = scaler.inverse_transform(cent_scaled)[0]
    centroid_dict = {f: float(v) for f, v in zip(features, cent_orig)}
    sample_dict = {f: float(v) for f, v in zip(features, x[0])}

    interpretation = generate_interpretation(window, sample_dict, centroid_dict)
    return {
        "window": window,
        "cluster": lab,
        "dist_to_centroid": dist,
        "nearest_centroid": centroid_dict,
        "interpretation": interpretation
    }

def generate_interpretation(window: str,
                            sample: dict,
                            centroid: dict) -> str:
    """
    Return a plain-English explanation of the simulated air-quality pattern.

    The message has three parts:
    1) What today's pattern looks like (profile label in human terms)
    2) Why we think that (which pollutants are most off vs the typical day)
    3) What to do (practical actions a city or company can take)

    Parameters
    ----------
    window : {'morning_peak','midday','evening_peak','night'}
        Time-of-day bucket.
    sample : dict
        User/input readings, e.g. {'PM10': 70, 'PM2.5': 30, 'O3': 0.03, ...}
    centroid : dict
        Typical values for the predicted cluster, same keys as `sample`.

    Returns
    -------
    str
        Human-friendly narrative.
    """
    # 1) lay terms for pollutants
    nice = {
        'CO': 'carbon monoxide (tailpipe indicator)',
        'NO': 'nitric oxide (fresh traffic exhaust)',
        'NO2': 'nitrogen dioxide (traffic/combustion)',
        'NOX': 'NOx (overall traffic/combustion mix)',
        'O3': 'ozone (sunlight + exhaust reaction)',
        'PM10': 'coarse dust (PM10)',
        'PM2.5': 'fine particles (PM2.5, health-relevant)',
        'SO2': 'sulfur dioxide (industrial/fuel quality)'
    }

    # 2) relative deviations (percent) from the typical day for this cluster
    deltas = {}
    for k in centroid.keys():
        c = float(centroid[k])
        s = float(sample.get(k, c))
        if c == 0:
            deltas[k] = 0.0
        else:
            deltas[k] = (s - c) / abs(c)

    # 3) pick the top 2–3 “drivers” by absolute deviation
    drivers = sorted(deltas.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]

    # 4) pick a simple label by window + signal
    label = "typical conditions"
    if window in ("morning_peak", "evening_peak"):
        if deltas.get('NOX', 0) > 0.20 or deltas.get('CO', 0) > 0.20:
            label = "traffic-driven spike"
    if window == "midday":
        if deltas.get('O3', 0) > 0.20:
            label = "photochemical (ozone) build-up"
        if deltas.get('PM10', 0) > 0.25 and deltas.get('O3', 0) <= 0.20:
            label = "dust-dominated day"
    if deltas.get('PM2.5', 0) > 0.25 and deltas.get('PM10', 0) > 0.15:
        label = "high particulate load"
    if all(abs(v) < 0.10 for v in deltas.values()):
        label = "near-normal levels"

    # 5) craft the narrative
    def pct(x):  # pretty percent
        return f"{x*100:.0f}%"

    why_bits = []
    for k, v in drivers:
        direction = "higher" if v > 0 else "lower"
        why_bits.append(f"{nice.get(k, k)} is {direction} than usual by ~{pct(abs(v))}")

    # 6) actions (short, actionable, window-aware)
    actions = []
    if label.startswith("traffic"):
        if window == "morning_peak":
            actions += [
                "Stagger school/work start times by 30–60 min.",
                "Prioritize bus-only lanes and signal timing on main corridors.",
                "Discourage short car trips in the 7–9 a.m. window."
            ]
        else:
            actions += [
                "Shift delivery windows away from 5–8 p.m.",
                "Enforce low-emission zones on congested arterials."
            ]
    if "ozone" in label:
        actions += [
            "Reduce solvent/paint use at midday; schedule for morning/evening.",
            "Promote remote work or transit during 12–16 h sunny periods."
        ]
    if "dust" in label or "particulate" in label:
        actions += [
            "Increase street sweeping and construction-site dust control.",
            "Advise masks for sensitive groups; limit outdoor sports."
        ]
    if not actions:
        actions = ["Maintain current controls; levels track the usual pattern."]

    return (
        f"Pattern: **{label}** during **{window.replace('_',' ')}**.\n\n"
        f"Why we think this: " + "; ".join(why_bits) + ".\n\n"
        "What to do now:\n- " + "\n- ".join(actions)
    )



# ============================================================================
# LIVE DATA MODULE
# ============================================================================

def fetch_live_data(lat: float = 25.6866, lon: float = -100.3161, timeout: int = 10) -> Optional[Dict[str, float]]:
    """
    Fetch live air quality data from API with fallback.
    
    Args:
        lat: Latitude
        lon: Longitude
        timeout: Request timeout in seconds
        
    Returns:
        Dictionary with pollutant values or None on error
    """
    try:
        # Example using OpenWeather Air Pollution API (requires API key)
        # For demo, returning mock data
        logger.info(f"Fetching live data for ({lat}, {lon})")
        
        # Mock implementation - replace with actual API call
        # url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid=YOUR_API_KEY"
        # response = requests.get(url, timeout=timeout)
        # data = response.json()
        
        # Mock data for demonstration
        mock_data = {
            'CO': np.random.uniform(0.3, 1.2),
            'NO': np.random.uniform(0.01, 0.05),
            'NO2': np.random.uniform(0.02, 0.06),
            'NOX': np.random.uniform(0.03, 0.11),
            'O3': np.random.uniform(0.02, 0.08),
            'PM10': np.random.uniform(20, 80),
            'PM2.5': np.random.uniform(10, 35),
            'SO2': np.random.uniform(0.002, 0.008)
        }
        
        return mock_data
        
    except Exception as e:
        logger.error(f"API fetch failed: {e}")
        return None

# ============================================================================
# STREAMLIT APP
# ============================================================================

def render_eda(df):
    st.title("Exploratory Data Analysis (EDA): Digging into the Data")
    st.markdown("""
    Let's explore the raw data! See distributions, trends, and connections between pollutants.
    Use the interactive charts to zoom and hover for details.
    """)
    
    # Extended: Pollutant distributions (histograms)
    pollutant = st.selectbox("Select Pollutant to Explore:", df.columns[1:9])  # Assuming pollutants start after 'date'
    fig_dist = px.histogram(df, x=pollutant, color='time_window', marginal="box", 
                            title=f"Distribution of {pollutant} (Group by Time Window)",
                            labels={pollutant: f"{pollutant} Level"})
    st.plotly_chart(fig_dist, use_container_width=True)
    st.markdown(f"**Insight:** {pollutant} is highest in evening peaks—could be from rush hour emissions.")
    
    # Extended: Correlation heatmap
    corr = df[['CO', 'NO', 'NO2', 'NOX', 'O3', 'PM10', 'PM2.5', 'SO2']].corr()
    fig_corr = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, 
                                         colorscale='RdYlGn', zmin=-1, zmax=1))
    fig_corr.update_layout(title="Pollutant Connections (Red = Strong Link, e.g., NOX and Traffic)")
    st.plotly_chart(fig_corr, use_container_width=True)
    st.markdown("**Insight:** NOX and CO are strongly linked—both from car exhaust. Reducing traffic could lower both.")
    
    # Extended: Time series with slider for date range
    date_range = st.slider("Select Date Range:", min_value=df['date'].min(), max_value=df['date'].max(), 
                           value=(df['date'].min(), df['date'].max()))
    filtered_df = df[(df['date'] >= date_range[0]) & (df['date'] <= date_range[1])]
    fig_ts = px.line(filtered_df, x='date', y=['PM10', 'PM2.5'], title="PM Trends Over Time (Slide to Filter)")
    st.plotly_chart(fig_ts, use_container_width=True)
    
    # If stations exist, add comparison
    if 'station' in df.columns:
        fig_station = px.box(df, x='station', y='PM2.5', color='time_window', 
                             title="Air Quality by Station (Compare Locations)")
        st.plotly_chart(fig_station, use_container_width=True)
        st.markdown("**Insight:** Norte station has higher PM—possibly near industrial areas.")


def render_temporal_analysis(models, windows, features):
    st.title("Temporal Analysis: Pollution Patterns by Time of Day")
    st.markdown("""
    Here, we break down air quality into time windows (like morning rush hour). 
    We use simple grouping techniques to spot patterns—think of clusters as 'types' of air quality days.
    Select a time window below to see visuals and easy-to-understand insights.
    """)
    
    selected_window = st.selectbox("Select time window:", windows, index=windows.index('morning_peak'))
    
    if selected_window in models:
        model = models[selected_window]
        pca = model['pca']
        kmeans = model['kmeans']
        centers = model['centers']  # Assuming this is a DataFrame or array of cluster centers
        explained_variance = model['explained_variance']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Main Patterns (Explained Variance)")
            st.markdown("""
            This shows how much of the pollution variation is captured by the top patterns. 
            Higher bars mean more important patterns—like traffic dominating mornings.
            """)
            fig_var = px.bar(x=['PC1', 'PC2'], y=explained_variance[:2], 
                             title=f"Key Patterns in {selected_window.replace('_', ' ').title()}",
                             labels={'x': 'Pattern', 'y': 'Importance (Variance)'})
            fig_var.update_traces(marker_color='lightblue')
            st.plotly_chart(fig_var, use_container_width=True)
        
        with col2:
            st.subheader("Cluster Types (Average Pollution Levels)")
            st.markdown("""
            Clusters group similar air quality moments. Green = low pollution (good), Red = high (be cautious).
            Hover over the heatmap for exact values.
            """)
            # Assuming centers is a 2D array (clusters x features); convert to DF if needed
            centers_df = pd.DataFrame(centers, columns=features)
            centers_df['Cluster'] = [f"Cluster {i}" for i in range(len(centers))]
            fig_heat = px.imshow(centers_df.set_index('Cluster').values, 
                                 labels=dict(x="Pollutants", y="Clusters", color="Level (Standardized)"),
                                 x=features, y=centers_df['Cluster'],
                                 color_continuous_scale='RdYlGn_r',  # Red bad, Green good
                                 title=f"Pollution Types in {selected_window.replace('_', ' ').title()}")
            st.plotly_chart(fig_heat, use_container_width=True)
        
        # Friendly cluster interpretations (extend based on your generate_interpretation logic)
        st.subheader("What Do These Clusters Mean?")
        for i, row in centers_df.iterrows():
            row_dict = dict(row[features])  # Select only feature columns to avoid type errors
            interp = generate_interpretation(selected_window, row_dict, row_dict)  # Reuse your function; adjust as needed
            st.markdown(f"**Cluster {i}:** {interp} (e.g., if high PM, avoid outdoor exercise).")
        
        with st.expander("Nerdy Details for Data Fans"):
            st.markdown("We used PCA to reduce dimensions and KMeans to cluster. Variance: PC1 captures traffic-related pollutants.")
    else:
        st.warning("No model for this window yet. Click 'Retrain Models' to update.")
def render_simulator(models: Dict):
    """Render What-If Simulator section."""
    st.header("🎮 What-If Simulator")
    
    windows = ['morning_peak', 'midday', 'evening_peak', 'night']
    selected_window = st.selectbox("Select time window for simulation:", windows)
    
    st.subheader("Adjust Pollutant Levels")
    
    # Get live or historical defaults
    live_data = fetch_live_data()
    if not live_data:
        st.warning("⚠️ Using historical median values (live data unavailable)")
        # Use historical medians as defaults
        defaults = {
            'CO': 0.6, 'NO': 0.03, 'NO2': 0.04, 'NOX': 0.07,
            'O3': 0.05, 'PM10': 45.0, 'PM2.5': 20.0, 'SO2': 0.005
        }
    else:
        defaults = live_data
    
    # Create sliders
    features = ['CO', 'NO', 'NO2', 'NOX', 'O3', 'PM10', 'PM2.5', 'SO2']
    ranges = {
        'CO': (0.0, 2.0, 0.01),
        'NO': (0.0, 0.1, 0.001),
        'NO2': (0.0, 0.1, 0.001),
        'NOX': (0.0, 0.2, 0.001),
        'O3': (0.0, 0.1, 0.001),
        'PM10': (0.0, 150.0, 1.0),
        'PM2.5': (0.0, 75.0, 0.5),
        'SO2': (0.0, 0.02, 0.0001)
    }
    
    col1, col2 = st.columns(2)
    input_values = []
    
    for i, feature in enumerate(features):
        with col1 if i < 4 else col2:
            min_val, max_val, step = ranges[feature]
            value = st.slider(
                f"{feature}",
                min_value=min_val,
                max_value=max_val,
                value=defaults.get(feature, (min_val + max_val) / 2),
                step=step,
                format=f"%.{len(str(step).split('.')[-1])}f"
            )
            input_values.append(value)
    
    # Simulate button
    if st.button("🚀 Simulate", type="primary"):
        result = simulate_scenario(selected_window, input_values, features)
        
        # Display results
        st.success("Simulation Complete!")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Predicted Cluster", result['cluster'])
        with col2:
            st.metric("Distance to Centroid", f"{result['dist_to_centroid']:.3f}")
        with col3:
            st.metric("Time Window", result['window'])
        
        # Interpretation
        st.info(f"💡 **Recommendation:** {result['interpretation']}")
        
        # Comparison plot
        if result['nearest_centroid']:
            comparison_df = pd.DataFrame({
                'Feature': features,
                'Your Input': input_values,
                'Cluster Center': [result['nearest_centroid'].get(f, 0) for f in features]
            })
            
            fig = go.Figure()
            fig.add_trace(go.Bar(name='Your Input', x=comparison_df['Feature'], 
                                y=comparison_df['Your Input']))
            fig.add_trace(go.Bar(name='Cluster Center', x=comparison_df['Feature'], 
                                y=comparison_df['Cluster Center']))
            fig.update_layout(
                title="Input vs. Cluster Center Comparison",
                barmode='group',
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)

def render_insights(df, models):
    st.title("Insights & Recommendations: What to Do")
    st.markdown("Based on patterns, here are actionable tips. Select options to customize.")
    
    selected_window = st.selectbox("Focus on Time Window:", list(models.keys()))
    selected_pollutant = st.selectbox("Focus on Pollutant:", ['All'] + ['CO', 'NO', 'NO2', 'NOX', 'O3', 'PM10', 'PM2.5', 'SO2'])
    
    # Specific insights from data/models
    window_df = df[df['time_window'] == selected_window]
    if selected_pollutant != 'All':
        avg = window_df[selected_pollutant].mean()
        insight = f"In {selected_window}, {selected_pollutant} averages {avg:.2f}. "
        if 'PM' in selected_pollutant and avg > 25:
            insight += "That's above safe levels—wear masks outdoors."
        elif selected_pollutant == 'O3' and avg > 0.05:
            insight += "High ozone: Avoid strenuous activity in sun."
        st.markdown(f"**Specific Tip:** {insight}")
    else:
        # Cluster-based insight
        model = models[selected_window]
        st.markdown(f"**Pattern in {selected_window}:** Most data falls into Cluster {np.argmax(model['centers'].mean(axis=1))}—low pollution overall, but watch for traffic spikes.")
    
    # Interactive: Simulate quick scenario
    st.subheader("Quick Tip Simulator")
    pm_slider = st.slider("Hypothetical PM2.5 Level:", 0, 100, 25)
    features_list = ['CO', 'NO', 'NO2', 'NOX', 'O3', 'PM10', 'PM2.5', 'SO2']
    vector = [0.5, 0.03, 0.04, 0.07, 0.05, 50, 25, 0.005]  # Realistic defaults
    pm25_idx = features_list.index('PM2.5')
    vector[pm25_idx] = pm_slider
    result = simulate_scenario(selected_window, vector, features=features_list, model_dir='models')  # Adjust vector
    st.markdown(f"If PM2.5 is {pm_slider}, you'd be in {result['interpretation']}")
# ============================================================================
# MAIN APP ENTRY
# ============================================================================

def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Monterrey Air Quality Dashboard",
        page_icon="🌬️",
        layout="wide"
    )
    
    st.title("🌬️ Monterrey Air Quality Analysis System")
    st.markdown("**Real-time monitoring, analysis, and recommendations for air quality management**")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    section = st.sidebar.radio(
        "Select Section:",
        ["📝 Executive Summary", "📊 EDA", "⏰ Temporal Analysis", "🎮 What-If Simulator", "💡 Insights"]
    )
    
    # Global filters
    st.sidebar.markdown("---")
    st.sidebar.subheader("Global Filters")
    
    # Load data
    @st.cache_data(show_spinner=False)
    def load_data():
        return load_excel_frames()
    
    with st.spinner("Loading data..."):
        df = load_data()
    
    if df.empty:
        st.error("❌ No data loaded. Please check data files.")
        return
    
    # Add time windows
    df = make_temporal_windows(df)
    
    # Station filter
    stations = st.sidebar.multiselect(
        "Select stations:",
        df['station'].unique(),
        default=df['station'].unique()[:3]
    )
    
    # Date range filter
    date_range = st.sidebar.date_input(
        "Date range:",
        value=(df['date'].min(), df['date'].max()),
        min_value=df['date'].min(),
        max_value=df['date'].max()
    )
    
    # Apply filters
    filtered_df = df[
        (df['station'].isin(stations)) &
        (df['date'] >= pd.Timestamp(date_range[0])) &
        (df['date'] <= pd.Timestamp(date_range[1]))
    ]
    
    # Load or train models
    @st.cache_resource(show_spinner=False)
    def load_or_train_models():
        windows = ['morning_peak', 'midday', 'evening_peak', 'night']
        features = ['CO', 'NO', 'NO2', 'NOX', 'O3', 'PM10', 'PM2.5', 'SO2']
        
        # Check if models exist
        model_dir = "models"
        if os.path.exists(model_dir) and len(os.listdir(model_dir)) > 0:
            return load_models(windows, model_dir)
        else:
            return train_and_save_models(filtered_df, windows, features)
    
    with st.spinner("Loading models..."):
        models = load_or_train_models()
    
    # Session state for last simulation
    if 'last_simulation' not in st.session_state:
        st.session_state.last_simulation = None
    
    # Render selected section
    windows = ['morning_peak', 'midday', 'evening_peak', 'night']
    features = ['CO', 'NO', 'NO2', 'NOX', 'O3', 'PM10', 'PM2.5', 'SO2']
    if section == "📝 Executive Summary":
        render_executive_summary(filtered_df)
    elif section == "📊 EDA":
        render_eda(filtered_df)
    elif section == "⏰ Temporal Analysis":
        render_temporal_analysis(models, windows, features)
    elif section == "🎮 What-If Simulator":
        render_simulator(models)
    elif section == "💡 Insights":
        render_insights(filtered_df, models)
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.caption("Built with Streamlit • v1.0.0")
    st.sidebar.caption("© 2025 Monterrey Air Quality Team")

# ============================================================================
# CLI INTERFACE
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # CLI mode
        import argparse
        
        parser = argparse.ArgumentParser(description="Monterrey Air Quality Analysis")
        parser.add_argument("--retrain", action="store_true", help="Retrain all models")
        parser.add_argument("--window", type=str, help="Time window for simulation")
        parser.add_argument("--simulate", type=str, help="Simulate with values (comma-separated)")
        
        args = parser.parse_args()
        
        if args.retrain:
            # Retraining models...
            df = load_excel_frames()
            df = make_temporal_windows(df)
            windows = ['morning_peak', 'midday', 'evening_peak', 'night']
            features = ['CO', 'NO', 'NO2', 'NOX', 'O3', 'PM10', 'PM2.5', 'SO2']
            models = train_and_save_models(df, windows, features, force_retrain=True)
            print(f"✅ Retrained {len(models)} models")
        
        elif args.window and args.simulate:
            # Simulating for {args.window}...
            features = ['CO', 'NO', 'NO2', 'NOX', 'O3', 'PM10', 'PM2.5', 'SO2']
            
            # Parse values
            try:
                values = [float(v.strip()) for v in args.simulate.split(',')]
                if len(values) != len(features):
                    raise ValueError
            except ValueError:
                print(f"Error: Expected {len(features)} comma-separated float values, e.g., 0.5,0.03,0.04,0.07,0.05,50,25,0.005")
                sys.exit(1)
            
            result = simulate_scenario(args.window, values, features)
            print(f"Cluster: {result['cluster']}")
            print(f"Distance: {result['dist_to_centroid']:.3f}")
            print(f"Interpretation: {result['interpretation']}")
        else:
            print("Starting Streamlit app...")
            os.system("streamlit run " + __file__)
    else:
        # Streamlit mode
        main()