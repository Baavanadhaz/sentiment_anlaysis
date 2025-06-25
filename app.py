from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load model, scaler, and columns
model = joblib.load("sentiment_model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("feature_columns.pkl")

category_list = ["Books", "Clothing", "Electronics", "Home & Kitchen", "Toys"]
payment_list = ["Credit Card", "Debit Card", "Gift Card", "PayPal"]

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    # Extract and build input row
    row = {
        "Quantity": int(data["quantity"]),
        "PriceUSD": float(data["price"])
    }

    for cat in category_list:
        row[f"Category_{cat}"] = 1 if cat == data["category"] else 0
    for pay in payment_list:
        row[f"PaymentMethod_{pay}"] = 1 if pay == data["payment"] else 0

    # Create DataFrame and align columns
    df = pd.DataFrame([row])
    for col in columns:
        if col not in df.columns:
            df[col] = 0
    df = df[columns]

    # Scale input
    X_scaled = scaler.transform(df)

    # Predict
    pred = model.predict(X_scaled)[0]
    sentiment = "Positive ✅" if pred == 1 else "Negative ❌"

    return jsonify({"sentiment": sentiment})

if __name__ == "__main__":
    app.run(debug=True)
