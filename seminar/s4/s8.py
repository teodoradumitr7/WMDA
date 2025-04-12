import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

#Visual Summary & Report
# 1. Load and scale the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target   # Not used for clustering, but sometimes nice for reference

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Apply three clustering algorithms
kmeans = KMeans(n_clusters=3, random_state=42).fit(X_scaled)
dbscan = DBSCAN(eps=0.5, min_samples=5).fit(X_scaled)
agg = AgglomerativeClustering(n_clusters=3, linkage='ward').fit(X_scaled)

# 3. Reduce to 2D with PCA for visualization
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# 4. Create a combined DataFrame for plotting & reporting
df_final = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
df_final['KMeans'] = kmeans.labels_
df_final['DBSCAN'] = dbscan.labels_
df_final['Agglo'] = agg.labels_

# 5. Plot side-by-side scatter plots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# KMeans Plot
axes[0].scatter(df_final['PC1'], df_final['PC2'], c=df_final['KMeans'], cmap='viridis', s=50)
axes[0].set_title('KMeans Clustering')
axes[0].set_xlabel('PC1')
axes[0].set_ylabel('PC2')

# DBSCAN Plot
axes[1].scatter(df_final['PC1'], df_final['PC2'], c=df_final['DBSCAN'], cmap='viridis', s=50)
axes[1].set_title('DBSCAN Clustering')
axes[1].set_xlabel('PC1')
axes[1].set_ylabel('PC2')

# Agglomerative Plot
axes[2].scatter(df_final['PC1'], df_final['PC2'], c=df_final['Agglo'], cmap='viridis', s=50)
axes[2].set_title('Agglomerative Clustering')
axes[2].set_xlabel('PC1')
axes[2].set_ylabel('PC2')

plt.tight_layout()
plt.show()

# 6. Print a short cluster distribution report
print("=== Cluster Counts ===")
print("\nKMeans:")
print(pd.Series(df_final['KMeans']).value_counts())

print("\nDBSCAN:")
print(pd.Series(df_final['DBSCAN']).value_counts())

print("\nAgglomerative Clustering:")
print(pd.Series(df_final['Agglo']).value_counts())
