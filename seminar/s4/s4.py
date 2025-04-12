import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

#Agglomerative Clustering & Dendrogram
# 1. Load and scale the Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# 2. Perform Agglomerative Clustering
agg_clustering = AgglomerativeClustering(n_clusters=3)
agg_labels = agg_clustering.fit_predict(df_scaled)

# 3. Add the cluster labels to the DataFrame
df_scaled['agg_cluster'] = agg_labels

# 4. Print a quick summary of how many points were assigned to each cluster
print("Cluster label counts:")
print(pd.Series(agg_labels).value_counts())

# 5. Create a linkage matrix for plotting a dendrogram
# (using Ward's method for linkage on the scaled features only)
linked = linkage(df_scaled.iloc[:, :-1], method='ward')

# 6. Plot the dendrogram
plt.figure(figsize=(10, 6))
dendrogram(linked,
           orientation='top',
           distance_sort='descending',
           show_leaf_counts=True)
plt.title('Hierarchical Clustering Dendrogram (Ward Linkage)')
plt.xlabel('Sample index')
plt.ylabel('Distance')
plt.show()
