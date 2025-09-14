#!/usr/bin/env python3
# describe_fixed_cuts.py
# Lee agg_assignments_*_k{K}.csv ya generados y describe el comportamiento
# de cada clúster sin volver a correr el modelo.

import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

FEATURES = ['NOX','CO','PM10','PM2.5','O3','NO','NO2','SO2']

# Cortes solicitados:
K_FOR_WINDOW = {
    'night': 4,
    'evening_peak': 4,
    'midday': 5,
    'morning_peak': 5,
}

WINDOWS = ['morning_peak', 'midday', 'evening_peak', 'night']


def load_assignments_for_window(outdir: str, window: str, k: int) -> pd.DataFrame:
    """
    Carga el archivo de asignaciones ya generado por hierarchical_air_quality.py:
      hier_outputs/agg_assignments_{window}_k{k}.csv
    """
    path = os.path.join(outdir, f"agg_assignments_{window}_k{k}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No encontré {path}. Corre antes hierarchical_air_quality.py para ese window/k."
        )
    df = pd.read_csv(path)
    # Asegurar columnas esperadas
    feats = [f for f in FEATURES if f in df.columns]
    if 'cluster' not in df.columns:
        raise ValueError(f"El archivo {path} no contiene columna 'cluster'.")
    if not feats:
        raise ValueError(f"El archivo {path} no contiene ninguna de las FEATURES esperadas.")
    return df[['time_window','cluster'] + feats if 'time_window' in df.columns else ['cluster'] + feats]


def build_percentile_mapper(df: pd.DataFrame, feats):
    """
    Crea funciones para mapear un valor a percentil (0..100) por feature,
    usando TODOS los datos del window (robusto y no re-entrena nada).
    """
    qs_map = {}
    for f in feats:
        vals = df[f].dropna().to_numpy()
        if len(vals) == 0:
            qs_map[f] = None
        else:
            qs_map[f] = np.quantile(vals, np.linspace(0, 1, 101))

    def to_pct(val_dict):
        out = {}
        for f in feats:
            v = val_dict.get(f, np.nan)
            qs = qs_map.get(f)
            if qs is None or pd.isna(v):
                out[f] = np.nan
            else:
                idx = np.searchsorted(qs, v, side='right') - 1
                out[f] = int(np.clip(idx, 0, 100))
        return out

    return to_pct


def semantic_from_percentiles(pcts: dict) -> str:
    """Reglas simples para etiqueta semántica del clúster."""
    high_nox = (pcts.get('NOX',0) >= 75) or (pcts.get('CO',0) >= 75)
    high_o3  = (pcts.get('O3',0)  >= 75) and (pcts.get('NOX',100) <= 25)
    high_pm  = (pcts.get('PM10',0) >= 75) or (pcts.get('PM2.5',0) >= 75)
    low_all  = all(pcts.get(x,50) <= 25 for x in ['NOX','CO','PM10','PM2.5','O3'])
    if high_nox: return "Tráfico alto"
    if high_o3:  return "Ozono fotoquímico/industrias"
    if high_pm:  return "Partículas"
    if low_all:  return "Bajo/Normal"
    return "Mixto"


def word_from_pct(p: float) -> str:
    if np.isnan(p): return "—"
    if p >= 75: return "alto"
    if p <= 25: return "bajo"
    return "medio"


def describe_window(outdir: str, window: str, k: int, save: bool = True) -> pd.DataFrame:
    """
    Lee asignaciones para (window, k), calcula percentiles globales del window,
    resume por clúster y devuelve una tabla con:
      cluster | n | etiqueta_semántica | descripción | {feature_median, feature_pct}
    """
    df = load_assignments_for_window(outdir, window, k)
    feats = [f for f in FEATURES if f in df.columns]

    # Mapeador de percentiles globales por feature (en el window)
    pct_map = build_percentile_mapper(df, feats)

    rows = []
    for c, g in df.groupby('cluster'):
        n = len(g)
        meds = {f: float(np.median(g[f].dropna())) if g[f].notna().any() else np.nan for f in feats}
        pcts = pct_map(meds)
        sem  = semantic_from_percentiles(pcts)

        # descripción corta: alto/medio/bajo por feature clave
        desc_bits = [f"{f}: {word_from_pct(pcts.get(f, np.nan))}" for f in feats]
        desc = ", ".join(desc_bits)

        row = {
            "window": window,
            "k": k,
            "cluster": int(c) if str(c).isdigit() else c,
            "n": int(n),
            "semantic_label": sem,
            "description": desc,
        }
        # adjuntar medianas y percentiles por si quieres tabular
        for f in feats:
            row[f+"_median"] = meds[f]
            row[f+"_pct"]    = pcts.get(f, np.nan)
        rows.append(row)

    out = pd.DataFrame(rows).sort_values("cluster")

    if save:
        Path(outdir).mkdir(parents=True, exist_ok=True)
        csv_path = os.path.join(outdir, f"behavior_summary_{window}_k{k}.csv")
        txt_path = os.path.join(outdir, f"behavior_descriptions_{window}_k{k}.txt")
        out.to_csv(csv_path, index=False)

        # TXT compacto, legible
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"Comportamiento por clúster — {window} (k={k})\n")
            f.write("="*60 + "\n\n")
            for _, r in out.iterrows():
                f.write(f"- Cluster {r['cluster']} (n={r['n']}): {r['semantic_label']}\n")
                f.write(f"  {r['description']}\n\n")
        print(f"[{window}] Guardado:\n  - {csv_path}\n  - {txt_path}")

    return out


def main():
    ap = argparse.ArgumentParser(description="Describe comportamiento de clústeres fijos por ventana sin reentrenar.")
    ap.add_argument("--out", default="hier_outputs", help="Carpeta donde están los agg_assignments_*.csv")
    ap.add_argument("--windows", nargs="*", default=WINDOWS, help="Ventanas a procesar")
    args = ap.parse_args()

    for w in args.windows:
        if w not in K_FOR_WINDOW:
            print(f"Ventana desconocida '{w}', la salto.")
            continue
        k = K_FOR_WINDOW[w]
        try:
            describe_window(args.out, w, k, save=True)
        except Exception as e:
            print(f"[{w}] No se pudo describir (¿falta el CSV de asignaciones k={k}?): {e}")

    print("\n✅ Listo. Revisa los archivos behavior_* en la carpeta indicada.")


if __name__ == "__main__":
    main()
