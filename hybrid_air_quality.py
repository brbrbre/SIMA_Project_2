#!/usr/bin/env python3
# hierarchical_air_quality.py
# Dendrogramas + clustering jerárquico (Ward) + resúmenes por clúster
# Lee Excel multi-hoja (cada hoja = estación), igual que antes.

import argparse, os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

# Preproceso y modelos
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.cluster import AgglomerativeClustering

# Dendrograma
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist

# Plots
import matplotlib.pyplot as plt

# --------------------------
# Config
# --------------------------
FEATURES_DEFAULT = ['NOX','CO','PM10','PM2.5','O3','NO','NO2','SO2']
WIN_MAP = [
    ("morning_peak", lambda h: (h >= 6)  & (h < 10)),
    ("midday",       lambda h: (h >= 10) & (h < 16)),
    ("evening_peak", lambda h: (h >= 16) & (h < 20)),
    ("night",        lambda h: (h >= 20) | (h < 6)),
]

# --------------------------
# Helpers
# --------------------------
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
    print(f"Registros antes: {before:,} | después (sin fines): {after:,}")
    return d

def transform_and_scale(X):
    # Yeo-Johnson (maneja ceros/negativos) + StandardScaler
    pt = PowerTransformer(method="yeo-johnson", standardize=False)
    X_t = pt.fit_transform(X)
    ss = StandardScaler()
    X_ts = ss.fit_transform(X_t)
    return X_ts, pt, ss

def build_and_save_dendrogram(X_std, title, outpath, max_samples=4000, random_state=42):
    # Muestra para que el dendrograma sea legible (O(n^2))
    n = X_std.shape[0]
    if n > max_samples:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(n, size=max_samples, replace=False)
        X_plot = X_std[idx]
        note = f"(muestra {max_samples} de {n})"
    else:
        X_plot = X_std
        note = f"(n={n})"

    # Linkage Ward (usa distancias euclidianas sobre X estandarizado)
    Z = linkage(X_plot, method='ward', optimal_ordering=True)

    plt.figure(figsize=(12, 6))
    dendrogram(Z, no_labels=True, color_threshold=None)
    plt.title(f"Dendrograma — {title} {note}")
    plt.ylabel("Distancia (Ward)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def summarize_clusters(df_block, feats, labels, out_csv):
    d = df_block.copy()
    d["cluster"] = labels
    # Percentiles + mediana por clúster
    def _agg(g):
        res = {}
        for f in feats:
            vals = g[f].dropna().to_numpy()
            if len(vals)==0:
                res[(f,"p25")] = np.nan
                res[(f,"median")] = np.nan
                res[(f,"p75")] = np.nan
            else:
                res[(f,"p25")] = float(np.percentile(vals, 25))
                res[(f,"median")] = float(np.percentile(vals, 50))
                res[(f,"p75")] = float(np.percentile(vals, 75))
        res[("count","count")] = len(g)
        return pd.Series(res)

    summary = d.groupby("cluster", as_index=True).apply(_agg)
    # Aplana columnas
    summary.columns = [f"{a}_{b}" for a,b in summary.columns]
    summary = summary.sort_index()
    summary.to_csv(out_csv)
    return d, summary

def run_block(name, df_block, feats, outdir, kmin=2, kmax=8, sample_for_dendro=4000):
    Path(outdir).mkdir(parents=True, exist_ok=True)

    dfb = df_block.dropna(subset=feats).copy()
    if len(dfb) < max(200, len(feats)*30):
        print(f"[{name}] Muy pocos registros ({len(dfb)}). Saltando.")
        return

    X_raw = dfb[feats].to_numpy(dtype=float)

    # Transformación + escala
    X_std, pt, ss = transform_and_scale(X_raw)

    # Dendrograma (sobre muestra)
    dendro_png = os.path.join(outdir, f"dendrogram_{name}.png")
    build_and_save_dendrogram(X_std, name, dendro_png, max_samples=sample_for_dendro)
    print(f"[{name}] Dendrograma guardado en: {dendro_png}")

    # Clustering jerárquico para varios k y resúmenes
    for k in range(kmin, kmax+1):
        agg = AgglomerativeClustering(
            n_clusters=k,
            linkage="ward",
            metric="euclidean"
        )

        labels = agg.fit_predict(X_std)

        # Guardar asignaciones + resumen
        assign_csv = os.path.join(outdir, f"agg_assignments_{name}_k{k}.csv")
        summ_csv   = os.path.join(outdir, f"cluster_summary_{name}_k{k}.csv")

        assigned, summary = summarize_clusters(dfb, feats, labels, summ_csv)

        # Guardar asignaciones con columnas clave para que puedas clasificar luego
        out_cols = (["date","station","time_window"] if "time_window" in assigned.columns
                    else ["date","station"])
        out_cols = [c for c in out_cols if c in assigned.columns]
        out_cols = out_cols + feats + ["cluster"]
        assigned[out_cols].to_csv(assign_csv, index=False)

        print(f"[{name}] k={k} -> asignaciones: {assign_csv} | resumen: {summ_csv}")

# --------------------------
# Main
# --------------------------
def main():
    ap = argparse.ArgumentParser(description="Dendrogramas + Clustering jerárquico (Ward) + resúmenes por clúster")
    ap.add_argument("--excel", required=True, help="Ruta al Excel (múltiples hojas por estación)")
    ap.add_argument("--out", default="hier_outputs", help="Carpeta de salida")
    ap.add_argument("--features", nargs="*", default=FEATURES_DEFAULT, help="Features a usar")
    ap.add_argument("--by-window", action="store_true", help="Separar por ventana horaria (solo como filtro)")
    ap.add_argument("--skip-weekends", action="store_true", help="No filtrar fines (úsalo si ya los quitaste antes)")
    ap.add_argument("--kmin", type=int, default=2, help="k mínimo (Agglomerative)")
    ap.add_argument("--kmax", type=int, default=8, help="k máximo (Agglomerative)")
    ap.add_argument("--dendro-sample", type=int, default=4000, help="Muestra máx. para dendrograma")
    args = ap.parse_args()

    outdir = args.out
    Path(outdir).mkdir(parents=True, exist_ok=True)

    # -------- Cargar Excel multi-hoja (igual que antes) --------
    print("=== Jerárquico: dendrogramas + resúmenes ===")
    print(f"Excel: {args.excel}")
    xls = pd.read_excel(args.excel, sheet_name=None)  # <-- mismo estilo de carga
    frames = []
    for station, df in xls.items():
        d = df.copy()
        d["station"] = station
        # Normalizar fecha
        if "date" in d.columns:
            if np.issubdtype(d["date"].dtype, np.number):
                d["date"] = pd.to_datetime(d["date"], unit="s", errors="coerce")
            else:
                d["date"] = pd.to_datetime(d["date"], errors="coerce")
        frames.append(d)

    data = pd.concat(frames, ignore_index=True)
    data = data.dropna(subset=["date"]).copy()

    # -------- Fines de semana --------
    if not args.skip_weekends:
        data = drop_weekends(data)

    # -------- Ventanas (para filtrar) --------
    data = make_windows(data)

    # -------- Features --------
    feats = [f for f in args.features if f in data.columns]
    if not feats:
        raise SystemExit("No se encontraron las columnas de features solicitadas en el Excel.")
    data = data.dropna(subset=feats).copy()

    # -------- Ejecutar en bloque (por ventana o global) --------
    if args.by_window:
        for win_name, _ in WIN_MAP:
            dwin = data[data["time_window"] == win_name].copy()
            if len(dwin) == 0:
                continue
            print(f"[{win_name}] n={len(dwin)} | feats={feats}")
            run_block(
                name=win_name,
                df_block=dwin,
                feats=feats,
                outdir=outdir,
                kmin=args.kmin,
                kmax=args.kmax,
                sample_for_dendro=args.dendro_sample,
            )
    else:
        print(f"[ALL] n={len(data)} | feats={feats}")
        run_block(
            name="ALL",
            df_block=data,
            feats=feats,
            outdir=outdir,
            kmin=args.kmin,
            kmax=args.kmax,
            sample_for_dendro=args.dendro_sample,
        )

    print("\n✅ Listo. Revisa la carpeta de salida:")
    print(f"   {outdir}/")
    print("   - dendrogram_*.png (elige dónde cortar)")
    print("   - agg_assignments_*_k{k}.csv (etiquetas por fila)")
    print("   - cluster_summary_*_k{k}.csv (p25/mediana/p75 por contaminante)")

if __name__ == "__main__":
    main()
