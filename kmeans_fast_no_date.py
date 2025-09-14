#!/usr/bin/env python3
# model_assumptions_checker.py
# Evaluación de supuestos y comparación de modelos: GMM (BIC/AIC) vs KMeans (+HDBSCAN opcional)

import argparse
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from pathlib import Path
from datetime import datetime

from scipy.stats import shapiro, skew, kurtosis

from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

# HDBSCAN es opcional
try:
    import hdbscan
    HAS_HDBSCAN = True
except Exception:
    HAS_HDBSCAN = False


# --------------------------
# Utilidades
# --------------------------
WIN_MAP = [
    ("morning_peak", lambda h: (h >= 6)  & (h < 10)),
    ("midday",       lambda h: (h >= 10) & (h < 16)),
    ("evening_peak", lambda h: (h >= 16) & (h < 20)),
    ("night",        lambda h: (h >= 20) | (h < 6)),
]

FEATURES_DEFAULT = ['NOX','CO','PM10','PM2.5','O3','NO','NO2','SO2']


def make_windows(df):
    d = df.copy()
    d['hour'] = pd.to_datetime(d['date']).dt.hour
    conds = [f(d['hour']) for _, f in WIN_MAP]
    names = [n for n, _ in WIN_MAP]
    d['time_window'] = np.select(conds, names, default='night')
    return d


def drop_weekends(df):
    d = df.copy()
    d['date'] = pd.to_datetime(d['date'])
    before = len(d)
    d = d[d['date'].dt.dayofweek < 5].copy()
    after = len(d)
    return d, before, after


def safe_shapiro(x):
    # Shapiro recomienda n<=5000; muestreamos si hace falta
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return np.nan
    if len(x) > 5000:
        rng = np.random.default_rng(42)
        x = rng.choice(x, size=5000, replace=False)
    try:
        stat, p = shapiro(x)
        return p
    except Exception:
        return np.nan


def transform_and_scale(X):
    # Yeo-Johnson maneja ceros/negativos; luego StandardScaler
    pt = PowerTransformer(method="yeo-johnson", standardize=False)
    X_t = pt.fit_transform(X)
    ss = StandardScaler()
    X_ts = ss.fit_transform(X_t)
    return X_ts, pt, ss


def cluster_quality(X, labels):
    # Métricas de separación/cohesión
    uniq = np.unique(labels)
    if len(uniq) <= 1 or len(uniq) >= len(X):
        return {"silhouette": np.nan, "dbi": np.nan}
    return {
        "silhouette": float(silhouette_score(X, labels)),
        "dbi": float(davies_bouldin_score(X, labels)),
    }


def gaussianity_checks(X, labels, feat_names, max_clusters_report=12):
    """
    Para cada clúster y feature:
      - Shapiro p-value (>=0.05 sugiere normalidad)
      - Skewness (cerca de 0) y Excess Kurtosis (cerca de 0)
    """
    rows = []
    K = len(np.unique(labels))
    if K > max_clusters_report:
        # Evitar CSVs gigantes: muestreamos clusters si son demasiados
        uniq = np.unique(labels)
        rng = np.random.default_rng(0)
        uniq = rng.choice(uniq, size=max_clusters_report, replace=False)
    else:
        uniq = np.unique(labels)

    for k in uniq:
        mask = labels == k
        if mask.sum() < 10:
            continue
        Xk = X[mask]
        for j, f in enumerate(feat_names):
            col = Xk[:, j]
            p_sh = safe_shapiro(col)
            sk = float(skew(col, bias=False))
            exk = float(kurtosis(col, fisher=True, bias=False))  # excess kurtosis
            rows.append({
                "cluster": int(k),
                "feature": f,
                "n": int(mask.sum()),
                "shapiro_p": p_sh,
                "skew": sk,
                "excess_kurtosis": exk,
                "normal_like_flag": int(
                    (p_sh >= 0.05 if not np.isnan(p_sh) else False) and (abs(sk) < 1.0) and (abs(exk) < 2.0)
                )
            })
    df = pd.DataFrame(rows)
    if len(df):
        # % de (feature,cluster) con comportamiento ~normal
        pct_normal_like = 100 * df["normal_like_flag"].mean()
    else:
        pct_normal_like = np.nan
    return df, pct_normal_like


def run_block(name, X_raw, feat_names, outdir, do_k_range=True):
    """
    Ejecuta:
      - Transformación (YJ + StandardScaler)
      - GMM (k=1..10, 4 cov types) -> mejor por BIC y por AIC
      - Métricas de calidad
      - Normalidad por clúster
      - KMeans baseline con k*=k_BIC
      - HDBSCAN (si disponible)
    """
    Path(outdir).mkdir(parents=True, exist_ok=True)

    # Transformación
    X, pt, ss = transform_and_scale(X_raw)

    # --------------------------------
    # GMM: selección por BIC/AIC
    # --------------------------------
    k_list = list(range(1, 11)) if do_k_range else [1, 2, 3, 4, 5, 6]
    cov_types = ["full", "diag", "tied", "spherical"]

    rows = []
    best_bic = (np.inf, None, None)  # (bic, k, cov)
    best_aic = (np.inf, None, None)

    for cov in cov_types:
        for k in k_list:
            try:
                gmm = GaussianMixture(
                    n_components=k,
                    covariance_type=cov,
                    n_init=3,
                    random_state=42,
                    reg_covar=1e-6
                ).fit(X)
                bic = gmm.bic(X)
                aic = gmm.aic(X)
                rows.append({"k": k, "covariance": cov, "bic": bic, "aic": aic})
                if bic < best_bic[0]:
                    best_bic = (bic, k, cov)
                if aic < best_aic[0]:
                    best_aic = (aic, k, cov)
            except Exception as e:
                rows.append({"k": k, "covariance": cov, "bic": np.nan, "aic": np.nan})

    gmm_grid = pd.DataFrame(rows)
    gmm_grid.to_csv(os.path.join(outdir, f"grid_gmm_{name}.csv"), index=False)

    # Mejor por BIC
    _, k_bic, cov_bic = best_bic
    # FallBack: si todo NaN
    if k_bic is None:
        print(f"[{name}] No se pudo ajustar GMM (BIC).")
        return

    gmm_bic = GaussianMixture(
        n_components=int(k_bic),
        covariance_type=cov_bic,
        n_init=5,
        random_state=42,
        reg_covar=1e-6
    ).fit(X)
    labels_gmm = gmm_bic.predict(X)
    qual_gmm = cluster_quality(X, labels_gmm)

    norm_df, pct_norm = gaussianity_checks(X, labels_gmm, feat_names)
    norm_path = os.path.join(outdir, f"normality_{name}_gmm.csv")
    norm_df.to_csv(norm_path, index=False)

    # --------------------------------
    # KMeans baseline con k = k_bic
    # --------------------------------
    km = KMeans(n_clusters=int(k_bic), random_state=42, n_init=10)
    labels_km = km.fit_predict(X)
    qual_km = cluster_quality(X, labels_km)

    # --------------------------------
    # HDBSCAN opcional
    # --------------------------------
    qual_hdb = {"silhouette": np.nan, "dbi": np.nan}
    n_hdb = 0
    if HAS_HDBSCAN:
        try:
            hdb = hdbscan.HDBSCAN(min_cluster_size=max(20, X.shape[1]*4), min_samples=None).fit(X)
            labels_hdb = hdb.labels_
            # Los -1 son ruido; si todos son -1 no hay clusters
            if np.any(labels_hdb >= 0):
                qual_hdb = cluster_quality(X, labels_hdb)
                n_hdb = int(np.sum(labels_hdb >= 0))
        except Exception:
            pass

    # --------------------------------
    # Resumen
    # --------------------------------
    summary = {
        "block": name,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "gmm_best_k_bic": int(k_bic),
        "gmm_covariance": cov_bic,
        "gmm_silhouette": qual_gmm["silhouette"],
        "gmm_dbi": qual_gmm["dbi"],
        "%normal_like_cells": None if np.isnan(pct_norm) else float(pct_norm),

        "kmeans_k": int(k_bic),
        "kmeans_silhouette": qual_km["silhouette"],
        "kmeans_dbi": qual_km["dbi"],

        "hdbscan_silhouette": qual_hdb["silhouette"],
        "hdbscan_dbi": qual_hdb["dbi"],
        "hdbscan_core_count": n_hdb,
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Assumptions & Model Checker (GMM BIC/AIC vs KMeans/HDBSCAN)")
    parser.add_argument("--excel", type=str, required=True, help="Ruta al Excel (múltiples hojas por estación)")
    parser.add_argument("--out", type=str, default="assumption_outputs", help="Carpeta de salida")
    parser.add_argument("--features", type=str, nargs="*", default=FEATURES_DEFAULT, help="Features a usar")
    parser.add_argument("--by-window", action="store_true", help="Evaluar por ventana horaria")
    args = parser.parse_args()

    outdir = args.out
    Path(outdir).mkdir(parents=True, exist_ok=True)

    # --------------------------
    # Cargar Excel multi-hoja
    # --------------------------
    print("== Iniciando ==")
    print(f"Excel: {args.excel}")
    xls = pd.read_excel(args.excel, sheet_name=None)
    frames = []
    for station, df in xls.items():
        df = df.copy()
        df["station"] = station
        # Normalizar fecha
        if "date" in df.columns:
            if np.issubdtype(df["date"].dtype, np.number):
                df["date"] = pd.to_datetime(df["date"], unit="s", errors="coerce")
            else:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    data = data.dropna(subset=["date"]).copy()

    # Quitar fines de semana
    data, before, after = drop_weekends(data)
    print(f"Registros antes: {before:,} | después (sin fines de semana): {after:,}")

    # Ventanas
    data = make_windows(data)

    # Selección de features
    feats = [f for f in args.features if f in data.columns]
    if not feats:
        raise SystemExit("No se encontraron las columnas de features solicitadas en el Excel.")
    data = data.dropna(subset=feats).copy()

    summaries = []

    if args.by_window:
        for win_name, _ in WIN_MAP:
            dwin = data[data["time_window"] == win_name]
            if len(dwin) < 200:
                print(f"[{win_name}] Muy pocos registros ({len(dwin)}). Saltando.")
                continue
            X_raw = dwin[feats].to_numpy(dtype=float)
            print(f"[{win_name}] n={len(X_raw)} | features={feats}")
            summ = run_block(win_name, X_raw, feats, outdir, do_k_range=True)
            if summ:
                summaries.append(summ)
    else:
        X_raw = data[feats].to_numpy(dtype=float)
        print(f"[ALL] n={len(X_raw)} | features={feats}")
        summ = run_block("ALL", X_raw, feats, outdir, do_k_range=True)
        if summ:
            summaries.append(summ)

    if summaries:
        df_sum = pd.DataFrame(summaries)
        df_sum.to_csv(os.path.join(outdir, "model_assumptions_summary.csv"), index=False)
        print("\n== RESUMEN ==")
        print(df_sum.to_string(index=False))
        print(f"\n✅ Archivos guardados en: {outdir}/")
        print(" - grid_gmm_*.csv (BIC/AIC por k y covarianza)")
        print(" - normality_*_gmm.csv (p-values Shapiro, skew y kurtosis por clúster/feature)")
        print(" - model_assumptions_summary.csv (resumen por bloque)")
    else:
        print("No se pudieron generar resúmenes (¿muy pocos datos tras filtros?).")


if __name__ == "__main__":
    main()
