import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import numpy as np

# K-Means Clustering
# 1. Load and simulate df_scaled (replace this with your actual preprocessed data if available)
iris = load_iris()
df_scaled = pd.DataFrame(iris.data, columns=iris.feature_names)

# 2. Instantiate K-Means with a chosen number of clusters, say 3
kmeans = KMeans(n_clusters=3, random_state=42)

# 3. Fit the model to the data
kmeans.fit(df_scaled)

# 4. Extract cluster labels
cluster_labels = kmeans.labels_

# 5. (Optional) Add the cluster labels to the DataFrame
df_scaled['cluster'] = cluster_labels

# 6. Print a sample of the DataFrame with cluster labels
print(df_scaled.head())

# 7. Optional quick visualization (scatter plot using two features)
plt.figure(figsize=(8, 5))
plt.scatter(df_scaled.iloc[:, 0], df_scaled.iloc[:, 1], c=cluster_labels, cmap='viridis')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title('K-Means Clustering (k=3)')
plt.colorbar(label='Cluster')
plt.show()
