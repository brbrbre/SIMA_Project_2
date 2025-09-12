import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, adfuller
from statsmodels.multivariate.manova import MANOVA
import statsmodels.api as sm
import os

# Configuración de página (SOLO UNA VEZ)
st.set_page_config(
    page_title="Análisis NOX Nuevo León", 
    page_icon="🌍", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .success-card {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #28a745;
    }
    .warning-card {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #ffc107;
    }
    .danger-card {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

# Función para cargar datos reales
@st.cache_data
def load_real_data():
    """Carga datos reales desde Excel o session_state."""
    
    # Opción 1: Desde session_state (si ya están cargados)
    if "f24_clean" in st.session_state:
        return st.session_state["f24_clean"]
    
    # Opción 2: Cargar desde Excel
    excel_paths = [
        "f24_clean.xlsx",           # En el mismo directorio
        "../f24_clean.xlsx",        # Un nivel arriba
        "data/f24_clean.xlsx",      # En carpeta data
        "../data/f24_clean.xlsx"    # data un nivel arriba
    ]
    
    for path in excel_paths:
        if os.path.exists(path):
            try:
                st.info(f"📁 Cargando datos desde: {path}")
                dfs = pd.read_excel(path, sheet_name=None)
                
                # Convertir columnas de fecha
                for name, dfi in dfs.items():
                    if "date" in dfi.columns:
                        dfi["date"] = pd.to_datetime(dfi["date"], errors="coerce")
                
                # Guardar en session_state para futuro uso
                st.session_state["f24_clean"] = dfs
                st.success(f"✅ Datos cargados exitosamente: {len(dfs)} estaciones")
                return dfs
                
            except Exception as e:
                st.warning(f"⚠️ Error cargando {path}: {str(e)}")
                continue
    
    # Opción 3: Datos sintéticos si no se encuentra el Excel
    st.warning("⚠️ No se encontró f24_clean.xlsx. Generando datos sintéticos...")
    return generate_sample_data()

# Función para generar datos de ejemplo
@st.cache_data
def generate_sample_data():
    """Genera datos sintéticos para prueba."""
    np.random.seed(42)
    
    stations = ["NOROESTE2", "CENTRO", "SURESTE", "NORESTE"]
    data_dict = {}
    
    intervention_date = pd.Timestamp('2023-04-05')
    
    for station in stations:
        dates = pd.date_range('2022-01-01', '2024-12-31', freq='H')
        dates = dates[dates.dayofweek < 5]  # Solo días hábiles
        
        n_points = len(dates)
        hour = dates.hour
        
        # Patrón diurno
        day_pattern = 50 + 30 * np.sin(2 * np.pi * hour / 24 - np.pi/4) + 20 * np.sin(4 * np.pi * hour / 24)
        
        # Efectos adicionales
        seasonal = 10 * np.sin(2 * np.pi * dates.dayofyear / 365)
        time_trend = np.arange(n_points) * 0.01
        
        # Efecto de intervención
        after_intervention = (dates >= intervention_date).astype(int)
        intervention_effect = -15 * after_intervention * (1 - np.exp(-0.1 * np.arange(n_points)))
        
        # NOX final
        noise = np.random.normal(0, 8, n_points)
        nox = day_pattern + seasonal + time_trend + intervention_effect + noise
        nox = np.maximum(nox, 5)
        
        # Variables meteorológicas
        prs = 1013 + np.random.normal(0, 10, n_points)
        rainf = np.maximum(0, np.random.exponential(0.5, n_points))
        rh = np.clip(50 + 20 * np.sin(2 * np.pi * dates.dayofyear / 365) + np.random.normal(0, 10, n_points), 0, 100)
        sr = np.maximum(0, 800 * np.sin(np.pi * hour / 12) + np.random.normal(0, 100, n_points))
        tout = 20 + 10 * np.sin(2 * np.pi * dates.dayofyear / 365 - np.pi/6) + np.random.normal(0, 3, n_points)
        wsr = np.maximum(0, np.random.exponential(2, n_points))
        wdr = np.random.uniform(0, 360, n_points)
        
        df = pd.DataFrame({
            'date': dates, 
            'CO': nox * 0.3 + np.random.normal(0, 2, n_points),  # CO correlacionado con NOX
            'NO': nox * 0.4 + np.random.normal(0, 3, n_points),   # NO correlacionado con NOX
            'NO2': nox * 0.6 + np.random.normal(0, 2, n_points),  # NO2 correlacionado con NOX
            'NOX': nox, 
            'O3': np.maximum(0, 80 - nox * 0.3 + np.random.normal(0, 10, n_points)),  # O3 anti-correlacionado
            'PM10': nox * 0.8 + np.random.normal(0, 5, n_points),
            'PRS': prs, 
            'RAINF': rainf, 
            'RH': rh, 
            'SO2': np.maximum(0, np.random.exponential(2, n_points)),
            'SR': sr, 
            'TOUT': tout, 
            'WSR': wsr, 
            'WDR': wdr
        })
        
        data_dict[station] = df
    
    return data_dict

# Función de preparación de datos
def prepare_data(data_dict, station):
    """Prepara datos para análisis MANOVA e ITS."""
    if station not in data_dict:
        raise ValueError(f"Estación '{station}' no encontrada en los datos")
    
    df = data_dict[station].copy()
    
    # Verificar columnas requeridas
    required_cols = ['date', 'NOX', 'PRS', 'RH', 'TOUT', 'WSR']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Columnas faltantes: {missing_cols}")
    
    # Preparar variables temporales
    df["hour"] = df["date"].dt.hour
    df["date_only"] = df["date"].dt.date
    df["after"] = (df["date"] >= "2023-04-05").astype(int)
    df["time_numeric"] = (df["date"] - df["date"].min()).dt.days

    # Pivot para MANOVA
    nox_wide = df.pivot_table(values="NOX", index="date_only", columns="hour", aggfunc="mean")
    nox_wide = nox_wide.dropna()  # Eliminar días incompletos
    nox_wide.columns = [f"h_{int(c)}" for c in nox_wide.columns]

    # Meteorología diaria
    meteo_daily = df.groupby("date_only").agg({
        'PRS': 'mean', 'RAINF': 'sum', 'RH': 'mean',
        'SR': 'mean', 'TOUT': 'mean', 'WSR': 'mean',
        'after': 'first', 'time_numeric': 'first'
    }).reset_index()

    # Combinar datos
    analysis_data = nox_wide.reset_index().merge(meteo_daily, on="date_only", how="inner")
    
    return df, analysis_data

# Función para diagnósticos
def run_diagnostics(df):
    """Ejecuta diagnósticos de autocorrelación y estacionariedad."""
    diagnostics = {'acf': {}, 'stationarity': {}}
    
    for hour in range(0, 24, 6):
        hour_data = df[df['hour'] == hour]['NOX'].dropna()
        if len(hour_data) > 50:
            # ACF
            try:
                acf_vals = acf(hour_data, nlags=20, alpha=0.05)
                diagnostics['acf'][f'Hora {hour}'] = acf_vals
            except:
                pass
            
            # Test ADF
            try:
                adf_result = adfuller(hour_data)
                diagnostics['stationarity'][f'Hora {hour}'] = {
                    'ADF Statistic': adf_result[0],
                    'p-value': adf_result[1],
                    'Estacionaria': adf_result[1] < 0.05
                }
            except:
                pass
    
    return diagnostics

# Función MANOVA
def run_manova(analysis_data):
    """Ejecuta análisis MANOVA."""
    try:
        # Variables disponibles
        h_cols = [col for col in analysis_data.columns if col.startswith("h_")]
        x_vars = ["after", "PRS", "RH", "TOUT", "WSR"]
        
        if len(h_cols) < 12:  # Necesitamos al menos la mitad de las horas
            return {"error": "Datos insuficientes para MANOVA", "success": False}
        
        # Crear fórmula
        formula = f"{' + '.join(h_cols)} ~ {' + '.join(x_vars)}"
        
        # Ejecutar MANOVA
        manova_model = MANOVA.from_formula(formula, data=analysis_data)
        manova_results = manova_model.mv_test()

        # Calcular cambios por hora
        hour_changes = {}
        for h_col in h_cols:
            hour_num = int(h_col.split("_")[1])
            before_mean = analysis_data.loc[analysis_data["after"] == 0, h_col].mean()
            after_mean = analysis_data.loc[analysis_data["after"] == 1, h_col].mean()
            
            if pd.notna(before_mean) and before_mean != 0:
                pct_change = ((after_mean - before_mean) / before_mean) * 100
            else:
                pct_change = np.nan
            
            hour_changes[hour_num] = {
                "before_mean": before_mean,
                "after_mean": after_mean,
                "absolute_change": after_mean - before_mean,
                "percent_change": pct_change
            }

        return {
            "manova_results": manova_results, 
            "hour_changes": hour_changes, 
            "success": True
        }
        
    except Exception as e:
        return {"error": str(e), "success": False}

# Función ITS
def run_its_analysis(df):
    """Ejecuta análisis Interrupted Time Series."""
    its_results = {}
    
    for hour in range(0, 24, 2):
        hour_data = df[df["hour"] == hour].copy()
        if len(hour_data) < 100:
            continue
            
        try:
            # Preparar variables
            X_vars = ["time_numeric", "after", "PRS", "RH", "TOUT", "WSR"]
            hour_data["after_time"] = hour_data["after"] * hour_data["time_numeric"]
            X_vars.append("after_time")
            
            # Regresión
            X = hour_data[X_vars].fillna(method='ffill')
            X = sm.add_constant(X)
            y = hour_data["NOX"]
            
            model = sm.OLS(y, X).fit()

            # Calcular métricas
            baseline = hour_data.loc[hour_data["after"] == 0, "NOX"].mean()
            imm_change = model.params.get("after", 0.0)
            trend_change = model.params.get("after_time", 0.0)
            
            if pd.notna(baseline) and baseline != 0:
                pct_change = (imm_change / baseline) * 100
            else:
                pct_change = np.nan

            its_results[hour] = {
                "immediate_change": imm_change,
                "trend_change": trend_change,
                "percent_change": pct_change,
                "p_value_immediate": model.pvalues.get("after", 1.0),
                "p_value_trend": model.pvalues.get("after_time", 1.0),
                "r_squared": model.rsquared
            }
            
        except Exception as e:
            its_results[hour] = {"error": str(e)}
    
    return its_results

# Función para generar plan de acción
def generate_action_plan(manova_results, its_results, station):
    """Genera plan de acción dinámico."""
    if not manova_results.get('success', False):
        return "⚠️ No se pudo generar plan de acción debido a errores en el análisis."
    
    hour_changes = manova_results['hour_changes']
    valid_changes = [h['percent_change'] for h in hour_changes.values() if pd.notna(h['percent_change'])]
    
    if not valid_changes:
        return "⚠️ No hay cambios válidos para analizar."
    
    avg_change = np.mean(valid_changes)
    significant_reductions = sum(1 for pct in valid_changes if pct < -5)
    significant_increases = sum(1 for pct in valid_changes if pct > 5)
    
    plan = f"## 📋 Plan de Acción para Estación {station}\n\n"
    
    if avg_change < -10:
        plan += f"""
        <div class="success-card">
        <h3>✅ ÉXITO SIGNIFICATIVO</h3>
        <p><strong>Cambio promedio en NOX: {avg_change:.1f}%</strong></p>
        <h4>Acciones Recomendadas:</h4>
        <ul>
            <li>🎯 <strong>Expandir programa de verificación vehicular</strong> a más municipios</li>
            <li>📈 <strong>Intensificar inspecciones</strong> en horarios pico identificados</li>
            <li>🚗 <strong>Implementar incentivos</strong> para vehículos de bajas emisiones</li>
            <li>📊 <strong>Mantener monitoreo continuo</strong> para sostenibilidad</li>
        </ul>
        </div>
        """
    elif avg_change < -5:
        plan += f"""
        <div class="warning-card">
        <h3>⚠️ MEJORA MODERADA</h3>
        <p><strong>Cambio promedio en NOX: {avg_change:.1f}%</strong></p>
        <h4>Acciones Recomendadas:</h4>
        <ul>
            <li>🔧 <strong>Reforzar verificación vehicular</strong> con controles más estrictos</li>
            <li>🚌 <strong>Promover transporte público</strong> en horarios críticos</li>
            <li>🏭 <strong>Revisar fuentes industriales</strong> adicionales de NOX</li>
            <li>📍 <strong>Implementar zonas de bajas emisiones</strong> temporales</li>
        </ul>
        </div>
        """
    else:
        plan += f"""
        <div class="danger-card">
        <h3>🚨 ALERTA - INTERVENCIÓN INSUFICIENTE</h3>
        <p><strong>Cambio promedio en NOX: {avg_change:.1f}%</strong></p>
        <h4>Acciones Urgentes:</h4>
        <ul>
            <li>⚡ <strong>Revisión inmediata</strong> de la efectividad del programa</li>
            <li>🔍 <strong>Auditoría completa</strong> de centros de verificación</li>
            <li>🚫 <strong>Implementar restricciones vehiculares</strong> temporales</li>
            <li>🏗️ <strong>Evaluar fuentes adicionales</strong> de emisiones NOX</li>
            <li>📋 <strong>Diseñar estrategia integral</strong> de calidad del aire</li>
        </ul>
        </div>
        """
    
    plan += f"\n\n### 🕐 Análisis por Horarios\n"
    plan += f"- **Horas con reducción significativa (>5%):** {significant_reductions}/{len(valid_changes)}\n"
    plan += f"- **Horas con aumento preocupante (>5%):** {significant_increases}/{len(valid_changes)}\n"
    
    if significant_reductions > len(valid_changes) // 2:
        plan += "- ✅ **Patrón positivo**: La mayoría de horas muestran mejora\n"
    elif significant_increases > len(valid_changes) // 4:
        plan += "- ⚠️ **Patrón preocupante**: Múltiples horas con aumentos\n"
    
    return plan

# Interfaz principal
def main():
    st.title("🌍 Análisis de NOX en Nuevo León")
    st.markdown("### Evaluación del impacto de la verificación vehicular (Abril 2023)")
    
    st.sidebar.header("⚙️ Configuración")
    
    # Cargar datos
    try:
        with st.spinner("Cargando datos..."):
            data_dict = load_real_data()
        
        if not data_dict:
            st.error("❌ No se pudieron cargar los datos")
            return
            
    except Exception as e:
        st.error(f"❌ Error cargando datos: {str(e)}")
        return
    
    # Selector de estación
    stations = list(data_dict.keys())
    station = st.sidebar.selectbox("📍 Seleccionar Estación:", stations)
    
    # Filtro de fechas
    min_date = pd.Timestamp('2022-06-01')
    max_date = pd.Timestamp('2024-06-01')
    date_range = st.sidebar.date_input(
        "📅 Rango de Fechas:",
        value=(min_date.date(), max_date.date()),
        min_value=min_date.date(),
        max_value=max_date.date()
    )
    
    # Preparar datos
    try:
        with st.spinner("Preparando datos para análisis..."):
            df_hourly, analysis_data = prepare_data(data_dict, station)
            
            # Filtrar por fechas
            mask = (df_hourly['date'] >= pd.Timestamp(date_range[0])) & (df_hourly['date'] <= pd.Timestamp(date_range[1]))
            df_hourly = df_hourly[mask]
            
            if len(df_hourly) == 0:
                st.error("❌ No hay datos en el rango de fechas seleccionado")
                return
                
        st.sidebar.success(f"✅ {len(df_hourly)} registros cargados")
        
    except Exception as e:
        st.error(f"❌ Error preparando datos: {str(e)}")
        return
    
    # Tabs principales
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Resumen", "🔍 Diagnósticos", "📈 MANOVA", "⏱️ ITS", "📋 Plan de Acción"
    ])
    
    with tab1:
        st.header("📊 Resumen Ejecutivo")
        
        # Métricas clave
        col1, col2, col3, col4 = st.columns(4)
        
        before_data = df_hourly[df_hourly['after'] == 0]['NOX']
        after_data = df_hourly[df_hourly['after'] == 1]['NOX']
        
        if len(before_data) > 0 and len(after_data) > 0:
            avg_before = before_data.mean()
            avg_after = after_data.mean()
            pct_change = ((avg_after - avg_before) / avg_before) * 100
            
            col1.metric("🏭 NOX Antes", f"{avg_before:.1f} µg/m³")
            col2.metric("🏭 NOX Después", f"{avg_after:.1f} µg/m³")
            col3.metric("📉 Cambio", f"{pct_change:.1f}%", delta=f"{avg_after-avg_before:.1f}")
            col4.metric("📅 Días Analizados", len(df_hourly['date_only'].unique()))
        
        # Gráfico de tendencia
        st.subheader("📈 Tendencia Temporal de NOX")
        
        daily_nox = df_hourly.groupby('date_only').agg({
            'NOX': 'mean',
            'after': 'first'
        }).reset_index()

        # Corregir el formato de 'date_only'
        daily_nox['date_only'] = pd.to_datetime(daily_nox['date_only']).dt.normalize()

        fig = px.line(
            daily_nox, 
            x='date_only', 
            y='NOX',
            color='after',
            title=f"Concentración Diaria de NOX - Estación {station}",
            labels={'NOX': 'NOX (µg/m³)', 'date_only': 'Fecha'}
        )

        from datetime import date

        fig.add_vline(
            x=date(2023, 4, 5),  # Usar datetime.date
            line_dash="dash",
            annotation_text="Evento",
            annotation_position="top left"
        )
        print(daily_nox.head())
        print(daily_nox.dtypes) 

        st.plotly_chart(fig, use_container_width=True)
                
        # Perfiles horarios
        st.subheader("🕐 Perfiles Horarios Promedio")
        
        hourly_profiles = df_hourly.groupby(['hour', 'after'])['NOX'].mean().reset_index()
        
        fig2 = px.line(
            hourly_profiles,
            x='hour',
            y='NOX',
            color='after',
            title="Perfil Horario de NOX (Antes vs Después)",
            labels={'NOX': 'NOX (µg/m³)', 'hour': 'Hora del Día'}
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    with tab2:
        st.header("🔍 Diagnósticos Estadísticos")
        
        with st.spinner("Ejecutando diagnósticos..."):
            diagnostics = run_diagnostics(df_hourly)
        
        # ACF Plots
        st.subheader("📊 Autocorrelación (ACF)")
        
        acf_data = diagnostics['acf']
        if acf_data:
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            axes = axes.ravel()
            
            for i, (hour, acf_vals) in enumerate(list(acf_data.items())[:4]):
                if len(acf_vals) >= 2 and len(acf_vals[0]) > 1:
                    axes[i].plot(acf_vals[0], 'b-', alpha=0.7)
                    if len(acf_vals) > 1 and acf_vals[1] is not None:
                        axes[i].fill_between(range(len(acf_vals[0])), 
                                           acf_vals[1][:, 0], 
                                           acf_vals[1][:, 1], 
                                           alpha=0.3, color='gray')
                    axes[i].set_title(f'{hour}')
                    axes[i].set_xlabel('Lag')
                    axes[i].set_ylabel('ACF')
                    axes[i].grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("ℹ️ No hay datos suficientes para generar gráficos ACF")
        
        # Tests de estacionariedad
        st.subheader("🔄 Tests de Estacionariedad (ADF)")
        
        stationarity_data = diagnostics['stationarity']
        if stationarity_data:
            stationarity_df = pd.DataFrame(stationarity_data).T
            st.dataframe(stationarity_df.style.format({
                'ADF Statistic': '{:.3f}',
                'p-value': '{:.3f}'
            }))
        else:
            st.info("ℹ️ No hay datos suficientes para tests de estacionariedad")
    
    with tab3:
        st.header("📈 Análisis MANOVA")
        st.markdown("*Análisis multivariado de perfiles diarios horarios*")
        
        with st.spinner("Ejecutando MANOVA..."):
            manova_results = run_manova(analysis_data)
        
        if manova_results['success']:
            # Resultados MANOVA
            st.subheader("🎯 Resultados MANOVA")
            
            manova_stats = manova_results['manova_results']
            st.text(str(manova_stats))
            
            # Cambios por hora
            st.subheader("⏰ Cambios por Hora del Día")
            
            hour_changes = manova_results['hour_changes']
            changes_df = pd.DataFrame(hour_changes).T
            changes_df.index.name = 'Hora'
            
            # Filtrar valores válidos
            changes_df = changes_df.dropna(subset=['percent_change'])
            
            if not changes_df.empty:
                st.dataframe(changes_df.style.format({
                    'before_mean': '{:.1f}',
                    'after_mean': '{:.1f}',
                    'absolute_change': '{:.1f}',
                    'percent_change': '{:.1f}%'
                }).background_gradient(subset=['percent_change'], cmap='RdYlGn_r'))
                
                # Gráfico de cambios porcentuales
                fig = px.bar(
                    x=changes_df.index,
                    y=changes_df['percent_change'],
                    title="Cambio Porcentual de NOX por Hora",
                    labels={'x': 'Hora del Día', 'y': 'Cambio (%)'},
                    color=changes_df['percent_change'],
                    color_continuous_scale='RdYlGn_r'
                )
                fig.add_hline(y=0, line_dash="dash", line_color="black")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠️ No hay cambios válidos para mostrar")
            
        else:
            st.error(f"❌ Error en MANOVA: {manova_results.get('error', 'Error desconocido')}")
    
    with tab4:
        st.header("⏱️ Análisis Interrupted Time Series")
        st.markdown("*Análisis de series de tiempo interrumpidas por hora*")
        
        with st.spinner("Ejecutando análisis ITS..."):
            its_results = run_its_analysis(df_hourly)
        
        if its_results:
            # Filtrar resultados válidos
            valid_results = {k: v for k, v in its_results.items() if 'error' not in v}
            
            if valid_results:
                st.subheader("📊 Resultados ITS por Hora")
                
                its_df = pd.DataFrame(valid_results).T
                its_df = its_df.dropna(subset=['percent_change'])
                
                if not its_df.empty:
                    st.dataframe(its_df.style.format({
                        'immediate_change': '{:.2f}',
                        'trend_change': '{:.4f}',
                        'percent_change': '{:.1f}%',
                        'p_value_immediate': '{:.3f}',
                        'p_value_trend': '{:.3f}',
                        'r_squared': '{:.3f}'
                    }).background_gradient(subset=['percent_change'], cmap='RdYlGn_r'))
                    
                    # Gráfico de cambios inmediatos
                    fig = make_subplots(
                        rows=2, cols=1,
                        subplot_titles=['Cambio Inmediato (%)', 'Significancia (p-value)']
                    )
                    
                    fig.add_trace(
                        go.Bar(x=its_df.index, y=its_df['percent_change'], name='% Cambio'),
                        row=1, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(x=its_df.index, y=its_df['p_value_immediate'], 
                                 mode='lines+markers', name='p-value'),
                        row=2, col=1
                    )
                    
                    fig.add_hline(y=0.05, line_dash="dash", line_color="red", row=2, col=1)
                    fig.update_layout(height=600, title_text="Análisis ITS por Hora")
                    st.plotly_chart(fig, use_container_width=True)
                
                else:
                    st.warning("⚠️ No se pudieron calcular resultados ITS válidos")
            else:
                st.warning("⚠️ No hay resultados ITS válidos para mostrar")
        else:
            st.error("❌ Error ejecutando análisis ITS")
    
    with tab5:
        st.header("📋 Plan de Acción Dinámico")
        
        # Generar plan basado en resultados
        if 'manova_results' in locals() and 'its_results' in locals():
            action_plan = generate_action_plan(manova_results, its_results, station)
            st.markdown(action_plan, unsafe_allow_html=True)
            
            # Métricas adicionales para el plan
            st.subheader("📊 Métricas de Soporte")
            
            col1, col2, col3 = st.columns(3)
            
            if manova_results.get('success', False):
                hour_changes = manova_results['hour_changes']
                valid_changes = {k: v for k, v in hour_changes.items() if pd.notna(v.get('percent_change'))}
                
                if valid_changes:
                    # Horas con mejora significativa
                    improved_hours = sum(1 for h in valid_changes.values() if h['percent_change'] < -5)
                    col1.metric("🕐 Horas Mejoradas", f"{improved_hours}/{len(valid_changes)}")
                    
                    # Cambio en horas pico (7-9, 18-20)
                    peak_hours = [7, 8, 18, 19]
                    peak_changes = [valid_changes[h]['percent_change'] for h in peak_hours if h in valid_changes]
                    avg_peak_change = np.mean(peak_changes) if peak_changes else 0
                    col2.metric("🚗 Cambio Horas Pico", f"{avg_peak_change:.1f}%")
                    
                    # Efectividad general
                    effectiveness = "Alta" if avg_peak_change < -10 else "Media" if avg_peak_change < -5 else "Baja"
                    col3.metric("⭐ Efectividad", effectiveness)
        
        else:
            st.info("ℹ️ Ejecuta los análisis en las pestañas anteriores para generar el plan de acción.")

if __name__ == "__main__":
    main()