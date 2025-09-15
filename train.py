import json, joblib, pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("data/trips.csv")
X = df[["km","minutes","tempC"]]; y = df["price"]
Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.4, random_state=42)

m = LinearRegression().fit(Xtr, ytr)
mae = float(mean_absolute_error(yva, m.predict(Xva)))

joblib.dump(m, "model.pkl")
with open("metrics.json", "w") as f:
    json.dump({"val_mae": mae}, f, indent=2)

print(f"Trained. VAL_MAE={mae:.2f}")
