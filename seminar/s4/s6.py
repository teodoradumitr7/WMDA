import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

#Customer Segmentation Use Case
# 1. Generate synthetic customer data
np.random.seed(42)
num_customers = 50

df_customers = pd.DataFrame({
    'purchase_frequency': np.random.randint(1, 15, num_customers),
    'average_spent': np.random.randint(10, 500, num_customers),
    'loyalty_score': np.random.randint(1, 6, num_customers)
})

print("=== Raw Customer Data (first 5 rows) ===")
print(df_customers.head(), "\n")

# 2. Scale the data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df_customers)

# 3. K-Means clustering (let's pick 3 customer segments)
kmeans = KMeans(n_clusters=3, random_state=42)
cluster_labels = kmeans.fit_predict(scaled_features)

# 4. Add cluster labels to the DataFrame
df_customers['cluster'] = cluster_labels

# 5. Inspect each segment
print("=== Cluster Counts ===")
print(df_customers['cluster'].value_counts(), "\n")

print("=== Cluster-wise Averages ===")
print(df_customers.groupby('cluster').mean())

# 6. (Optional) Quick interpretation
print("\n=== Quick Interpretation ===")
for cluster in df_customers['cluster'].unique():
    avg_purchase = df_customers[df_customers['cluster'] == cluster]['purchase_frequency'].mean()
    avg_spent = df_customers[df_customers['cluster'] == cluster]['average_spent'].mean()
    avg_loyalty = df_customers[df_customers['cluster'] == cluster]['loyalty_score'].mean()
    print(f"Cluster {cluster}: Avg Purchases = {avg_purchase:.1f}, Avg Spent = ${avg_spent:.2f}, Avg Loyalty = {avg_loyalty:.1f}")

