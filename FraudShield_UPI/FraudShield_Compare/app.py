import joblib
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__, static_folder="static", template_folder="templates")


# Load models
dt = joblib.load("model/dt_model.pkl")
svm = joblib.load("model/svm_model.pkl")
rf = joblib.load("model/rf_model.pkl")
svm_scaler = joblib.load("model/svm_scaler.pkl")

xgb = joblib.load("model/fraudshield_xgboost.pkl")
pca = joblib.load("model/pca_transform.pkl")
xgb_scaler = joblib.load("model/scaler.pkl")

@app.route("/", methods=["GET", "POST"])
def compare():
    results = None

    if request.method == "POST":
        amount = float(request.form["amount"])
        frequency = int(request.form["frequency"])
        location = int(request.form["location_change"])
        device = int(request.form["device_change"])
        merchant = float(request.form["merchant_risk"])

        data = np.array([[amount, 12, frequency, location, device, merchant, 1]])

        # Decision Tree
        dt_pred = dt.predict(data)[0]

        # SVM
        svm_scaled = svm_scaler.transform(data)
        svm_pred = svm.predict(svm_scaled)[0]

        # Random Forest
        rf_pred = rf.predict(data)[0]

        # XGBoost (SMOTE + PCA pipeline)
        xgb_scaled = xgb_scaler.transform(data)
        xgb_reduced = pca.transform(xgb_scaled)
        xgb_pred = xgb.predict(xgb_reduced)[0]

        results = {
            "Decision Tree": "Fraud" if dt_pred == 1 else "Valid",
            "SVM": "Fraud" if svm_pred == 1 else "Valid",
            "Random Forest": "Fraud" if rf_pred == 1 else "Valid",
            "XGBoost (Proposed)": "Fraud" if xgb_pred == 1 else "Valid",
        }

    return render_template("compare.html", results=results)

if __name__ == "__main__":
    app.run(debug=True, port=5001)
