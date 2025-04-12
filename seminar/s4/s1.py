import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

#Load & Preprocess Your Dataset
# 1. Load the Iris dataset from scikit-learn
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# 2. Introduce some artificial missing values (optional, for demonstration)
df.iloc[5:10, 2] = np.nan 

# 3. Handle missing values
imputer = SimpleImputer(strategy='mean')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# 4. Scale the data
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_imputed), columns=df.columns)

# 5. Check the results
print("Primele 5 randuri:")
print(df_scaled.head())

# 6. (Optional) Print summary stats to confirm preprocessing
print("\nStatistici")
print(df_scaled.describe())
