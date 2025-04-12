import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score

#Evaluating Clusters with Silhouette Scores
# 1. Load and scale the Iris dataset
iris = load_iris()
X = iris.data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Fit each clustering method
kmeans = KMeans(n_clusters=3, random_state=42).fit(X_scaled)
dbscan = DBSCAN(eps=0.5, min_samples=5).fit(X_scaled)
agg = AgglomerativeClustering(n_clusters=3, linkage='ward').fit(X_scaled)

# 3. Get the cluster labels from each method
kmeans_labels = kmeans.labels_
dbscan_labels = dbscan.labels_
agg_labels = agg.labels_

# 4. Compute silhouette scores (check for >1 cluster to avoid errors)
kmeans_score = silhouette_score(X_scaled, kmeans_labels)

if len(set(dbscan_labels)) > 1:
    dbscan_score = silhouette_score(X_scaled, dbscan_labels)
else:
    dbscan_score = "Only 1 cluster"

agg_score = silhouette_score(X_scaled, agg_labels)

# 5. Print the scores
print(f"KMeans Silhouette Score: {kmeans_score:.3f}")
print(f"DBSCAN Silhouette Score: {dbscan_score}")
print(f"Agglomerative Clustering Silhouette Score: {agg_score:.3f}")
