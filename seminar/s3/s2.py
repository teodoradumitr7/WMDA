## Exercise 2 (10 minutes): Polynomial Regression
'''Extinderea modelului liniar pentru a surprinde relații neliniare
Folosiți PolynomialFeatures (de exemplu, degree=2 sau degree=3) pe una sau mai multe caracteristici numerice (de ex. sqft) pentru a introduce neliniaritate.'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 1. Create a synthetic non-linear dataset
np.random.seed(42)
num_samples = 30  # Number of data points

# Single feature for clarity (e.g., 'sqft' or just X)
X = np.linspace(0, 10, num_samples).reshape(-1, 1)

# True relationship: y = 2 * X^2 - 3 * X + noise
y_true = 2 * (X**2) - 3 * X  # Non-linear relationship
noise = np.random.normal(0, 3, size=num_samples)
y = y_true.flatten() + noise

# Ensure that X and y have the same length
print("Length of X:", len(X))
print("Length of y:", len(y))

# Convert to DataFrame
df = pd.DataFrame({"Feature": X.flatten(), "Target": y})

# 2. Separate features and target
X = df[["Feature"]].values
y = df["Target"].values

# 3. Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Transform features to polynomial (degree=2 for illustration)
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# 5. Create and train a Linear Regression model on the polynomial features
model = LinearRegression()
model.fit(X_train_poly, y_train)

# 6. Use the model to predict on the test set
y_pred = model.predict(X_test_poly)

# 7. Evaluate the model
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("\nModel Evaluation:")
print(f"R-squared: {r2:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")

# 8. Optional: Plot to visualize the fit
# Generate a smooth curve for plotting
X_range = np.linspace(0, 10, 100).reshape(-1, 1)
X_range_poly = poly.transform(X_range)
y_range = model.predict(X_range_poly)

# Plot the data and the polynomial regression curve
plt.scatter(X, y, color='blue', label='Data points')
plt.plot(X_range, y_range, color='red', label='Polynomial Regression (degree=2)')
plt.xlabel('Feature (X)')
plt.ylabel('Target (y)')
plt.legend()
plt.title('Polynomial Regression Fit')
plt.show()