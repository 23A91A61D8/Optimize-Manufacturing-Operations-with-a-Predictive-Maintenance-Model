import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# ------------------------------
# 1. Load Dataset
# ------------------------------
df = pd.read_csv("predictive_maintenance_project/ai4i2020.csv")
print("Dataset Loaded Successfully!")
print(df.head())
print(df.info())

# ------------------------------
# 2. Separate Target & Features
# ------------------------------
X = df.drop("Machine failure", axis=1)
y = df["Machine failure"]

# ------------------------------
# 3. Column Types
# ------------------------------
categorical_cols = ["Product ID", "Type"]
numerical_cols = [col for col in X.columns if col not in categorical_cols]

# ------------------------------
# 4. Encoding + Scaling
# ------------------------------
ct = ColumnTransformer([
    ("onehot", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ("scale", StandardScaler(), numerical_cols)
])

X_processed = ct.fit_transform(X)

print("Preprocessing Complete! Encoded + Scaled shape:", X_processed.shape)

# ------------------------------
# 5. Class Imbalance Weight
# ------------------------------
positive = sum(y == 1)
negative = sum(y == 0)
scale_pos_weight = negative / positive

print("Positive:", positive, "Negative:", negative, "scale_pos_weight:", scale_pos_weight)

# ------------------------------
# 6. TimeSeriesSplit (for training)
# ------------------------------
tscv = TimeSeriesSplit(n_splits=5)

print("TimeSeriesSplit Ready!")
