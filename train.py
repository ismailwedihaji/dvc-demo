import pandas as pd
import numpy as np
import pickle
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load dataset
df = pd.read_csv("data/sp500.csv")

# Ensure Price column is numeric (remove commas if present)
df["Price"] = pd.to_numeric(df["Price"].astype(str).str.replace(",", "", regex=True))

# Target (y) and features (X = just an index timeline here)
y = df["Price"].values
X = np.arange(len(y)).reshape(-1, 1)

# Split data (train = older data, test = newer data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Polynomial regression
degree = 2  # 👈 change this value if you want to test other degrees
poly = PolynomialFeatures(degree=degree)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

model = LinearRegression()
model.fit(X_train_poly, y_train)

# Predict
y_pred = model.predict(X_test_poly)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump((model, poly), f)

# Save metrics
metrics = {
    "MAE": float(mean_absolute_error(y_test, y_pred)),
    "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
    "degree": degree
}
with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print(" Training complete. Metrics saved to metrics.json")
