import pandas as pd
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv(r"C:\Users\srava\OneDrive\Desktop\churn_project\Telco-Customer-Churn.csv")

# Rename churn column safely
churn_column = [col for col in df.columns if "churn" in col.lower()][0]
df.rename(columns={churn_column:"Churn"}, inplace=True)

# Convert target variable
df["Churn"] = df["Churn"].map({"Yes":1,"No":0})

# Drop ID column
if "customerID" in df.columns:
    df.drop("customerID", axis=1, inplace=True)

# Convert numeric columns
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.fillna(df.median(numeric_only=True), inplace=True)

# Encode categorical features
df = pd.get_dummies(df, drop_first=True)

# Split dataset
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Train model
model = xgb.XGBClassifier(n_estimators=100)
model.fit(X, y)

# Save model + features
joblib.dump(model, "churn_model.pkl")
joblib.dump(X.columns, "model_features.pkl")

print("✅ Model training completed")