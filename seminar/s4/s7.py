import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

#Anomaly Detection
# 1. Generate synthetic "normal" data
np.random.seed(42)
normal_data = np.random.normal(loc=50, scale=10, size=(200, 2))  # 200 points around mean=50

# 2. Generate synthetic "anomalous" data
outliers = np.array([[100, 100], [10, 90], [90, 10], [120, 40], [40, 120]])

# 3. Combine the datasets
X = np.vstack((normal_data, outliers))

# 4. Apply DBSCAN
dbscan = DBSCAN(eps=8, min_samples=5)
labels = dbscan.fit_predict(X)

# 5. Identify outliers (DBSCAN labels them as -1)
outlier_indices = np.where(labels == -1)[0]
print(f"Number of detected outliers: {len(outlier_indices)}")
print(f"Outlier indices: {outlier_indices}")

# 6. Visualization
plt.figure(figsize=(8, 6))
# Normal points and clusters
plt.scatter(X[labels != -1, 0], X[labels != -1, 1], c=labels[labels != -1], cmap='viridis', label='Clustered Points')
# Outliers
plt.scatter(X[labels == -1, 0], X[labels == -1, 1], c='red', marker='x', s=100, label='Outliers')
plt.title('DBSCAN Anomaly Detection')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

# 7. Reporting
num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print(f"Total clusters formed (excluding outliers): {num_clusters}")
print(f"Total outliers detected: {len(outlier_indices)}")
