import pandas as pd
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import numpy as np

#DBSCAN Clustering
# 1. Load and simulate df_scaled (replace this with your actual preprocessed data if available)
iris = load_iris()
df_scaled = pd.DataFrame(iris.data, columns=iris.feature_names)

# 2. Instantiate DBSCAN with chosen parameters
# eps is the neighborhood radius, min_samples is the minimum number of points for a dense region
dbscan = DBSCAN(eps=0.5, min_samples=5)

# 3. Fit the model to the data
dbscan.fit(df_scaled)

# 4. Extract cluster labels
dbscan_labels = dbscan.labels_

# 5. Identify outliers (DBSCAN labels outliers as -1)
outliers = np.sum(dbscan_labels == -1)
print(f"Nr outliers: {outliers}")

# 6. (Optional) Add the labels to the DataFrame
df_scaled['dbscan_cluster'] = dbscan_labels

# 7. Print the cluster label counts
print("\nCluster label counts:")
print(pd.Series(dbscan_labels).value_counts())

# 8. Optional quick visualization (scatter plot using two features)
plt.figure(figsize=(8, 5))
plt.scatter(df_scaled.iloc[:, 0], df_scaled.iloc[:, 1], c=dbscan_labels, cmap='viridis')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title('DBSCAN Clustering')
plt.colorbar(label='Cluster')
plt.show()
