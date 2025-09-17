import pandas as pd
import numpy as np
import pickle
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


df = pd.read_csv("data/sp500.csv")


df["Price"] = pd.to_numeric(df["Price"].astype(str).str.replace(",", "", regex=True))

y = df["Price"].values
X = np.arange(len(y)).reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)


degree = 4
poly = PolynomialFeatures(degree=degree)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

model = LinearRegression()
model.fit(X_train_poly, y_train)


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
