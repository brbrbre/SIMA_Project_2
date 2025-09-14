# kmeans_no_fecha.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# === 1. Cargar datos ===
df = pd.read_excel("Bases_Datos/f24_clean.xlsx")

# Tomamos solo contaminantes
features = ["NOX","CO","PM10","PM2.5","O3"]
X = df[features].dropna().values

# === 2. Transformación robusta ===
pt = PowerTransformer(method="yeo-johnson", standardize=False)
X_pt = pt.fit_transform(X)
sc = StandardScaler()
X_t = sc.fit_transform(X_pt)

# === 3. Probar varios k ===
sil_scores, inertias = [], []
K_range = range(2, 10)

for k in K_range:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X_t)
    sil = silhouette_score(X_t, labels)
    sil_scores.append(sil)
    inertias.append(km.inertia_)

# Mejor K por silhouette
best_idx = np.argmax(sil_scores)
best_k = K_range[best_idx]
print(f"Mejor k según Silhouette: {best_k} con {sil_scores[best_idx]:.3f}")

# === 4. Ajustar modelo final ===
km = KMeans(n_clusters=best_k, n_init=10, random_state=42)
labels = km.fit_predict(X_t)
df["cluster"] = labels

# === 5. Visualización (primeras 2 variables transformadas) ===
plt.figure(figsize=(7,5))
plt.scatter(X_t[:,0], X_t[:,1], c=labels, cmap="tab10", s=10)
plt.title(f"KMeans con {best_k} clústeres (Silhouette={sil_scores[best_idx]:.2f})")
plt.xlabel("Componente 1 (transformado)")
plt.ylabel("Componente 2 (transformado)")
plt.colorbar(label="Cluster")
plt.tight_layout()
plt.show()

# === 6. Guardar resultados ===
df.to_csv("clusters_kmeans_no_fecha.csv", index=False)
print("Clusters guardados en clusters_kmeans_no_fecha.csv")
