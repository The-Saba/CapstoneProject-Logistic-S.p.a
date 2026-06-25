import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

BASE_DIR = r"c:\Users\souli\OneDrive\Documents\GitHub\CapstoneProject-Logistic-S.p.a"
input_path = os.path.join(BASE_DIR, 'source', 'dataset_eda.csv')

df = pd.read_csv(input_path)
M_COLS = [f'M-{i}' for i in range(35, -1, -1)]

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
X_scaled = StandardScaler().fit_transform(X)

pca_full = PCA(random_state=42).fit(X_scaled)
cumvar = np.cumsum(pca_full.explained_variance_ratio_)
n_components_90 = int(np.argmax(cumvar >= 0.90)) + 1
nd = min(n_components_90, 6)
X_nd = PCA(n_components=nd, random_state=42).fit_transform(X_scaled)
X_2d = PCA(n_components=2, random_state=42).fit_transform(X_scaled)
print(f'valid rows: {len(X)} / {len(df_feat)}')
print(f'PCA 2D shape: {X_2d.shape}, PCA {nd}D shape: {X_nd.shape}')
print(f'n_components_90={n_components_90}, min_samples_5={max(5, int(np.log(len(X_nd))))}')

candidates = []
for space_name, data in [('X_nd', X_nd), ('X_2d', X_2d)]:
    print(f'\n--- DBSCAN search on {space_name} ---')
    for eps in np.round(np.linspace(0.1, 3.0, 30), 4):
        for ms in [5, max(5, int(np.log(len(data)))), max(10, int(np.log(len(data))) * 2)]:
            db = DBSCAN(eps=eps, min_samples=ms, n_jobs=-1)
            labels = db.fit_predict(data)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = int((labels == -1).sum())
            noise_pct = n_noise / len(labels) * 100
            if n_clusters >= 2 and noise_pct < 80:
                candidates.append((space_name, eps, ms, n_clusters, noise_pct))
                print(f'eps={eps:5.2f} ms={ms:2d} clusters={n_clusters:3d} noise={noise_pct:5.1f}%')
    if not any(c[0] == space_name for c in candidates):
        print(f'No viable DBSCAN clusters found for {space_name}')

print('\n--- Summary of candidate DBSCAN settings ---')
for c in candidates:
    print(c)
