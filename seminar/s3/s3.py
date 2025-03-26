'''Antrenarea unui Decision Tree Regressor și vizualizarea sau interpretarea modului în care face diviziunea'''
'''Ilustrează o abordare neliniară, neparametrică a regresiei.
Reiterează cât de importante sunt hiperparametrii (precum max_depth) pentru a evita supraadaptarea'''
## Exercise 3 (10 minutes): Regression Trees
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from sklearn import tree

# 1. Create a synthetic dataset with multiple features
np.random.seed(42)
num_samples = 30
X = np.random.rand(num_samples, 3) * 10  # e.g., three numeric features

# Define a "true" relationship for the target
true_y = 2 * X[:, 0] + 0.5 * (X[:, 1]**2) - 3 * X[:, 2]
noise = np.random.normal(0, 5, size=num_samples)  # Add some noise
y = true_y + noise

# Convert to a pandas DataFrame for familiarity
df = pd.DataFrame(X, columns=["Feature1", "Feature2", "Feature3"])
df["Target"] = y

# 2. Separate features and target
X = df[["Feature1", "Feature2", "Feature3"]].values
y = df["Target"].values

# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Create and train the Decision Tree Regressor
# tune `max_depth` to avoid overfitting => e.g., max_depth=3
model = DecisionTreeRegressor(max_depth=3)
model.fit(X_train, y_train)

# 5. Evaluate on the test set
y_pred = model.predict(X_test)

# Evaluation metrics
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# Display evaluation metrics
print("\nModel Evaluation:")
print(f"R-squared: {r2:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")

# 6. Optional: Inspect feature importances
print("\nFeature Importances:")
print(f"Feature1 importance: {model.feature_importances_[0]:.2f}")
print(f"Feature2 importance: {model.feature_importances_[1]:.2f}")
print(f"Feature3 importance: {model.feature_importances_[2]:.2f}")

# 7. Optional: Visualize the tree
plt.figure(figsize=(15, 10))
tree.plot_tree(model, filled=True, feature_names=["Feature1", "Feature2", "Feature3"], fontsize=10)
plt.title("Decision Tree Regressor Visualization")
plt.show()
