import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors

BASE_DIR = r"c:\Users\souli\OneDrive\Documents\GitHub\CapstoneProject-Logistic-S.p.a"
input_path = os.path.join(BASE_DIR, 'source', 'dataset_eda.csv')
df = pd.read_csv(input_path)

M_COLS = [f'M-{i}' for i in range(35, -1, -1)]

# Feature engineering

df_feat = df.copy()
df_feat['log_revenue'] = np.log1p(df_feat['Revenue'])
df_feat['log_employees'] = np.log1p(df_feat['Employees'])

recent_cols = [f'M-{i}' for i in range(6)]
old_cols = [f'M-{i}' for i in range(35, 29, -1)]
last12_cols = [f'M-{i}' for i in range(12)]

total_vol = df_feat[M_COLS].sum(axis=1)
recent_vol = df_feat[recent_cols].sum(axis=1)
old_vol = df_feat[old_cols].sum(axis=1)
last12_vol = df_feat[last12_cols].sum(axis=1)

df_feat['log_total_vol'] = np.log1p(total_vol)
df_feat['log_recent_vol'] = np.log1p(recent_vol)
df_feat['log_old_vol'] = np.log1p(old_vol)
df_feat['volume_trend'] = recent_vol / (old_vol + 1)
trend_cap = df_feat['volume_trend'].quantile(0.99)
df_feat['volume_trend'] = df_feat['volume_trend'].clip(upper=trend_cap)

monthly_matrix = df_feat[M_COLS].values.astype(float)
monthly_mean = monthly_matrix.mean(axis=1)
monthly_std = monthly_matrix.std(axis=1)
df_feat['vol_cv'] = np.where(monthly_mean > 0, monthly_std / monthly_mean, 0)
df_feat['active_months'] = (monthly_matrix > 0).sum(axis=1)
df_feat['peak_month_share'] = np.where(total_vol > 0, monthly_matrix.max(axis=1) / total_vol, 0)
df_feat['recency_score'] = np.where(total_vol > 0, last12_vol / total_vol, 0)

FEATURE_COLS = ['log_revenue', 'log_employees', 'log_total_vol', 'log_recent_vol', 'log_old_vol', 'volume_trend', 'vol_cv', 'active_months', 'peak_month_share', 'recency_score']

X = df_feat[FEATURE_COLS].copy()
mask_valid = X.notna().all(axis=1)
X = X[mask_valid].reset_index(drop=True)
df_clean = df_feat[mask_valid].reset_index(drop=True)
print(f'Valid rows: {len(X):,} / {len(df_feat):,}')

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print('\n=== K-Means parameter comparison ===')
rows = []
for k in range(2, 12):
    km = KMeans(n_clusters=k, init='k-means++', n_init=20, max_iter=500, random_state=42)
    labels = km.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels, sample_size=min(5000, len(labels)), random_state=42)
    dbi = davies_bouldin_score(X_scaled, labels)
    ch = calinski_harabasz_score(X_scaled, labels)
    sizes = pd.Series(labels).value_counts().sort_index().tolist()
    rows.append((k, sil, dbi, ch, sizes))
    print(f'K={k:2d}  silhouette={sil:.4f}  dbi={dbi:.4f}  ch={ch:.1f}  sizes={sizes}')

best_k_sil = max(rows, key=lambda x: x[1])[0]
print(f'\nBest K by silhouette: {best_k_sil}')
print('K=5 metrics:')
row5 = next(r for r in rows if r[0] == 5)
print(f'  silhouette={row5[1]:.4f}  dbi={row5[2]:.4f}  ch={row5[3]:.1f}  sizes={row5[4]}')

# DBSCAN comparison
pca_full = PCA(random_state=42).fit(X_scaled)
cumvar = np.cumsum(pca_full.explained_variance_ratio_)
n_components_90 = int(np.argmax(cumvar >= 0.90)) + 1
nd = min(n_components_90, 6)
X_nd = PCA(n_components=nd, random_state=42).fit_transform(X_scaled)
print(f'\nPCA components to 90% variance: {n_components_90}; using {nd} for DBSCAN')

MIN_SAMPLES = max(5, int(np.log(len(X_nd))))

nbrs = NearestNeighbors(n_neighbors=MIN_SAMPLES, algorithm='auto', n_jobs=-1).fit(X_nd)
distances, _ = nbrs.kneighbors(X_nd)
k_distances = np.sort(distances[:, -1])[::-1]
grad = np.gradient(k_distances)
knee_idx = np.argmax(np.abs(np.gradient(grad)))
eps_suggested = float(k_distances[knee_idx])
print(f'Suggested eps by k-distance knee: {eps_suggested:.4f}  (min_samples={MIN_SAMPLES})')

print('\n=== DBSCAN grid search ===')
results = []
eps_values = np.round(np.linspace(eps_suggested * 0.5, eps_suggested * 1.5, 8), 4)
min_samples_vals = [5, MIN_SAMPLES, MIN_SAMPLES * 2]
for eps_v in eps_values:
    for ms in min_samples_vals:
        db = DBSCAN(eps=eps_v, min_samples=ms, n_jobs=-1)
        labels = db.fit_predict(X_nd)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = int((labels == -1).sum())
        noise_pct = n_noise / len(labels) * 100
        sil = None
        if n_clusters >= 2 and noise_pct < 50:
            mask_core = labels != -1
            if mask_core.sum() > 1:
                sil = silhouette_score(X_nd[mask_core], labels[mask_core], sample_size=min(5000, mask_core.sum()), random_state=42)
        results.append({'eps': eps_v, 'min_samples': ms, 'n_clusters': n_clusters, 'noise_pct': noise_pct, 'silhouette': sil})
        print(f'eps={eps_v:.4f}  ms={ms:3d}  -> clusters={n_clusters:3d}  noise={noise_pct:5.1f}%  sil={sil if sil is None else f"{sil:.4f}"}')

resdf = pd.DataFrame(results)
valid = resdf[resdf['silhouette'].notna()]
print('\n--- Top DBSCAN configs by silhouette ---')
print(valid.sort_values(['silhouette', 'noise_pct'], ascending=[False, True]).head(10).to_string(index=False))
print('\n--- Best configs with noise < 30% ---')
print(valid[valid['noise_pct'] < 30].sort_values(['silhouette', 'noise_pct'], ascending=[False, True]).head(10).to_string(index=False))
