import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
df = pd.read_csv("C:/Users/baava/Downloads/ecommerce_transactions_2023.csv")

# Create sentiment labels
df['Sentiment'] = df['ReviewScore'].apply(lambda x: 1 if x >= 4 else (0 if x <= 2 else None))
df.dropna(subset=['Sentiment'], inplace=True)
df['Sentiment'] = df['Sentiment'].astype(int)

# Prepare features and target
X = df.drop(columns=["TransactionID", "UserID", "ProductID", "PurchaseDate", "ReviewScore", "Sentiment"])
y = df["Sentiment"]
X = pd.get_dummies(X)

# Scale numeric values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model and scaler
joblib.dump(model, "sentiment_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(X.columns.tolist(), "feature_columns.pkl")

print("✅ Model trained and saved as 'sentiment_model.pkl'")
