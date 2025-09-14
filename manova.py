#!/usr/bin/env python3
# manova_air_quality.py
# MANOVA por ventana horaria + (opcional) Box's M + ANOVA univariado con eta^2 y posthocs

import argparse, os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from pathlib import Path

# MANOVA / ANOVA
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.multivariate.manova import MANOVA
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Opcional: Box's M
try:
    import pingouin as pg
    HAS_PINGOUIN = True
except Exception:
    HAS_PINGOUIN = False

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

def make_windows(df):
    d = df.copy()
    d['date'] = pd.to_datetime(d['date'], errors='coerce')
    d = d.dropna(subset=['date'])
    d['hour'] = d['date'].dt.hour
    conds = [f(d['hour']) for _, f in WIN_MAP]
    names = [n for n, _ in WIN_MAP]
    d['time_window'] = np.select(conds, names, default='night')
    return d

def drop_weekends(df):
    d = df.copy()
    d['date'] = pd.to_datetime(d['date'], errors='coerce')
    d = d.dropna(subset=['date'])
    before = len(d)
    d = d[d['date'].dt.dayofweek < 5].copy()
    after = len(d)
    print(f"Registros antes: {before:,} | después (sin fines): {after:,}")
    return d

def safe_feature_names(df, feats):
    """
    Renombra columnas para fórmulas (patsy): PM2.5 -> PM25, etc.
    Devuelve (df_renamed, mapping)
    """
    mapping = {}
    for f in feats:
        sf = f.replace('.', '').replace(' ', '_')
        mapping[f] = sf
    d = df.rename(columns=mapping).copy()
    safe_feats = [mapping[f] for f in feats]
    return d, mapping, safe_feats

def manova_by_window(df, feats, outdir):
    """
    MANOVA: (NOX + CO + ... + SO2) ~ C(time_window)
    Exporta tablas: manova_multivariate.csv, anova_univariate.csv, tukey_<var>.csv (por variable)
    y box_m.csv si pingouin está disponible.
    """
    Path(outdir).mkdir(parents=True, exist_ok=True)

    # Variables limpias para la fórmula
    d, mapping, safe_feats = safe_feature_names(df, feats)

    # Asegura factor categórico
    d = d.dropna(subset=safe_feats + ['time_window']).copy()
    d['time_window'] = d['time_window'].astype('category')

    # ---------- MANOVA ----------
    # Construye la parte izquierda (dependientes)
    lhs = " + ".join(safe_feats)
    formula = f"{lhs} ~ C(time_window)"
    print(f"\nMANOVA fórmula: {formula}")

    mv = MANOVA.from_formula(formula, data=d)
    mv_res = mv.mv_test()

    # Serializar resultados multivariados (Wilks, Pillai, etc.)
    # mv_res.summary() es texto; extraemos tablas clave:
    with open(os.path.join(outdir, "manova_multivariate.txt"), "w") as f:
        f.write(str(mv_res))

    print("→ MANOVA multivariado guardado en manova_multivariate.txt")

    # ---------- Box's M (opcional) ----------
    if HAS_PINGOUIN:
        try:
            X = d[safe_feats]
            g = d['time_window']
            boxm = pg.box_m(X, g)
            boxm.to_csv(os.path.join(outdir, "box_m.csv"), index=False)
            print("→ Box's M guardado en box_m.csv")
        except Exception as e:
            print(f"(Aviso) Box's M falló: {e}")
    else:
        print("(Nota) Instala pingouin para Box's M: pip3 install pingouin")

    # ---------- ANOVA univariado + eta^2 parcial ----------
    rows_anova = []
    for v in safe_feats:
        model = smf.ols(f"{v} ~ C(time_window)", data=d).fit()
        aov = anova_lm(model, typ=2)  # SS entre (factor) y residual
        # eta^2 parcial = SS_factor / (SS_factor + SS_error)
        ss_factor = aov.loc['C(time_window)', 'sum_sq']
        ss_error  = aov.loc['Residual', 'sum_sq']
        eta2p = ss_factor / (ss_factor + ss_error)

        # Guardamos la tabla ANOVA (solo fila factor + residuales)
        aov_out = aov.reset_index().rename(columns={'index':'term'})
        aov_out['variable'] = v
        aov_out['eta2_partial'] = eta2p
        rows_anova.append(aov_out)

        # Tukey HSD por variable (posthoc entre ventanas)
        try:
            thsd = pairwise_tukeyhsd(d[v].values, d['time_window'].values)
            # Convert a DataFrame
            tukey_df = pd.DataFrame(data=thsd.summary().data[1:], columns=thsd.summary().data[0])
            tukey_df.to_csv(os.path.join(outdir, f"tukey_{v}.csv"), index=False)
        except Exception:
            pass

    anova_df = pd.concat(rows_anova, ignore_index=True)
    # Reemplaza con nombres originales donde aplique (para lectura del reporte)
    inv_map = {v:k for k,v in mapping.items()}
    anova_df['variable'] = anova_df['variable'].map(inv_map).fillna(anova_df['variable'])

    anova_df.to_csv(os.path.join(outdir, "anova_univariate.csv"), index=False)
    print("→ ANOVA univariado + eta^2 parcial guardado en anova_univariate.csv")

    # ---------- Resumen breve a consola ----------
    # Extrae líneas clave de MANOVA (Wilks y Pillai)
    try:
        txt = open(os.path.join(outdir, "manova_multivariate.txt"), "r").read()
        lines = [ln for ln in txt.splitlines() if "Wilks" in ln or "Pillai" in ln]
        print("\nResumen (Wilks & Pillai):")
        for ln in lines:
            print("  ", ln.strip())
    except Exception:
        pass

def main():
    ap = argparse.ArgumentParser(description="MANOVA por ventana horaria para contaminantes")
    ap.add_argument("--excel", required=True, help="Ruta al Excel (múltiples hojas por estación)")
    ap.add_argument("--out", default="manova_outputs", help="Carpeta de salida")
    ap.add_argument("--features", nargs="*", default=FEATURES_DEFAULT, help="Variables dependientes")
    ap.add_argument("--skip-weekends", action="store_true", help="Quitar fines de semana")
    args = ap.parse_args()

    outdir = args.out
    Path(outdir).mkdir(parents=True, exist_ok=True)

    print("=== MANOVA aire Monterrey ===")
    print(f"Excel: {args.excel}")

    # Cargar Excel multi-hoja
    xls = pd.read_excel(args.excel, sheet_name=None)
    frames = []
    for station, df in xls.items():
        d = df.copy()
        d["station"] = station
        # Normaliza fecha
        if "date" in d.columns:
            if np.issubdtype(d["date"].dtype, np.number):
                d["date"] = pd.to_datetime(d["date"], unit="s", errors="coerce")
            else:
                d["date"] = pd.to_datetime(d["date"], errors="coerce")
        frames.append(d)

    data = pd.concat(frames, ignore_index=True)
    data = data.dropna(subset=["date"]).copy()

    if args.skip_weekends:
        data = drop_weekends(data)

    # Ventanas
    data = make_windows(data)

    # Features presentes
    feats = [f for f in args.features if f in data.columns]
    if not feats:
        raise SystemExit("No se encontraron las columnas de features solicitadas en el Excel.")

    # Filtra filas válidas
    data = data.dropna(subset=feats + ['time_window']).copy()

    # Ejecuta MANOVA
    manova_by_window(data, feats, outdir)

    print("\n✅ Listo. Revisa la carpeta:")
    print(f"   {outdir}/")
    print("   - manova_multivariate.txt   (Wilks, Pillai, Hotelling-Lawley, Roy)")
    print("   - anova_univariate.csv      (por variable, incluye eta^2 parcial)")
    print("   - tukey_<variable>.csv      (posthoc entre ventanas)")
    if HAS_PINGOUIN:
        print("   - box_m.csv                 (homogeneidad de covarianzas, si disponible)")

if __name__ == "__main__":
    main()
