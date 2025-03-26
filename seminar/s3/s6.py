'''Ilustrarea modului în care k-Nearest Neighbors poate fi folosit pentru regresie, prin media valorilor țintă ale vecinilor.'''
## Exercise 6 (10 minutes): kNN for Regression
'''Ilustrează cum se face regresia în kNN prin media valorilor numerice ale vecinilor.
Reiterează importanța scalării caracteristicilor în modelele bazate pe distanță.
Link la rezolvare'''

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 1. Create a synthetic dataset
np.random.seed(42)
num_samples = 30

# Let's generate two features (e.g., Feature1, Feature2) and a target
X = np.random.rand(num_samples, 2) * 10
# Define a "true" relationship for the target: y = 3*X1 + 2*X2 + noise
y = 3 * X[:, 0] + 2 * X[:, 1] + np.random.normal(0, 5, size=num_samples)

# Convert to a DataFrame for clarity
df = pd.DataFrame(X, columns=["Feature1", "Feature2"])
df["Target"] = y

# 2. Separate features and target
X = df[["Feature1", "Feature2"]]
y = df["Target"]

# 3. Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 4. Feature scaling (recommended for distance-based methods like kNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Create and train the kNN Regressor
#    We'll start with n_neighbors=3 (can try different values)
knn_reg = KNeighborsRegressor(n_neighbors=3)
knn_reg.fit(X_train, y_train)

# 6. Evaluate on the test set
y_pred_test = knn_reg.predict(X_test)
print("Test Set Predictions:", y_pred_test)
print("True Values:", y_test.values)

r2 = r2_score(y_test, y_pred_test)
mse = mean_squared_error(y_test, y_pred_test)
mae = mean_absolute_error(y_test, y_pred_test)

print("Evaluation Metrics for kNN Regressor:")
print(f"R²: {r2:.3f}")
print(f"Mean Squared Error (MSE): {mse:.3f}")
print(f"Mean Absolute Error (MAE): {mae:.3f}")

# 7. (Optional) Explore the effect of different k values
#    You can loop over various values of k and compare performance.
print("\n\nDifferent values")
knn_reg = KNeighborsRegressor(n_neighbors=6)
knn_reg.fit(X_train, y_train)

y_pred_test = knn_reg.predict(X_test)
print("Test Set Predictions:", y_pred_test)
print("True Values:", y_test.values)

r2 = r2_score(y_test, y_pred_test)
mse = mean_squared_error(y_test, y_pred_test)
mae = mean_absolute_error(y_test, y_pred_test)

print("Evaluation Metrics for kNN Regressor:")
print(f"R²: {r2:.3f}")
print(f"Mean Squared Error (MSE): {mse:.3f}")
print(f"Mean Absolute Error (MAE): {mae:.3f}")
