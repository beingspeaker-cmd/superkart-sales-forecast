import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib, json, os

df = pd.read_csv("data/SuperKart.csv")
df.fillna(df.mean(numeric_only=True), inplace=True)
df.fillna("Unknown", inplace=True)

cat_cols = df.select_dtypes(include=["object"]).columns
for c in cat_cols:
    le = LabelEncoder()
    df[c] = le.fit_transform(df[c])

X = df.drop("Product_Store_Sales_Total", axis=1)
y = df["Product_Store_Sales_Total"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)

os.makedirs("artifacts", exist_ok=True)
joblib.dump(model, "artifacts/best_model.pkl")

metrics = {"mae": mae, "r2": r2}
with open("artifacts/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("✅ Training complete! Model and metrics saved in artifacts/")
