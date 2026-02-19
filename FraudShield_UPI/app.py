import joblib
import numpy as np
import sqlite3
from flask import Flask, render_template, request, redirect, url_for, session
from flask_login import (
    LoginManager, UserMixin,
    login_user, login_required, logout_user
)
from werkzeug.security import generate_password_hash, check_password_hash

# =====================================================
# LOAD ML MODELS
# =====================================================
model = joblib.load("model/fraudshield_xgboost.pkl")
pca = joblib.load("model/pca_transform.pkl")
scaler = joblib.load("model/scaler.pkl")
# =====================================================
# LOAD COMPARE MODELS
# =====================================================
dt = joblib.load("FraudShield_Compare/model/dt_model.pkl")
svm = joblib.load("FraudShield_Compare/model/svm_model.pkl")
rf = joblib.load("FraudShield_Compare/model/rf_model.pkl")
svm_scaler = joblib.load("FraudShield_Compare/model/svm_scaler.pkl")

# =====================================================
# FLASK APP SETUP
# =====================================================
app = Flask(__name__)
app.secret_key = "fraudshield-secret-key"

# =====================================================
# LOGIN MANAGER
# =====================================================
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

DB_NAME = "users.db"

# =====================================================
# USER CLASS
# =====================================================
class User(UserMixin):
    def __init__(self, id, username):
        self.id = id
        self.username = username

# =====================================================
# DATABASE INITIALIZATION
# =====================================================
def init_db():
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            amount REAL,
            frequency INTEGER,
            location_change INTEGER,
            device_change INTEGER,
            merchant_risk REAL,
            result TEXT
        )
        """)

        conn.commit()

init_db()

# =====================================================
# LOAD USER
# =====================================================
@login_manager.user_loader
def load_user(user_id):
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, username FROM users WHERE id = ?",
            (user_id,)
        )
        user = cursor.fetchone()

    if user:
        return User(user[0], user[1])
    return None

# =====================================================
# LOGIN ROUTE
# =====================================================
@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        with sqlite3.connect(DB_NAME) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, password FROM users WHERE username = ?",
                (username,)
            )
            user = cursor.fetchone()

        if user and check_password_hash(user[1], password):
            login_user(User(user[0], username))
            return redirect(url_for("dashboard"))

    return render_template("login.html")

# =====================================================
# REGISTER ROUTE
# =====================================================
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        hashed_password = generate_password_hash(password)

        try:
            with sqlite3.connect(DB_NAME, timeout=10) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO users (username, password) VALUES (?, ?)",
                    (username, hashed_password)
                )
                conn.commit()
        except sqlite3.IntegrityError:
            return "Username already exists"

        return redirect(url_for("login"))

    return render_template("register.html")

# =====================================================
# DASHBOARD
# =====================================================
@app.route("/dashboard")
@login_required
def dashboard():
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM transactions")
        total_transactions = cursor.fetchone()[0] or 0

        cursor.execute("SELECT COUNT(*) FROM transactions WHERE result='Fraud'")
        fraud_count = cursor.fetchone()[0] or 0

        cursor.execute("SELECT COUNT(*) FROM transactions WHERE result='Valid'")
        valid_count = cursor.fetchone()[0] or 0

        cursor.execute("""
            SELECT amount, result
            FROM transactions
            ORDER BY id DESC
            LIMIT 5
        """)
        recent_transactions = cursor.fetchall() or []

        cursor.execute("""
            SELECT amount
            FROM transactions
            ORDER BY id DESC
            LIMIT 6
        """)
        amounts = [row[0] for row in cursor.fetchall()]
        amounts.reverse()

    last_result = session.pop("last_result", None)
    play_alert = (last_result == "Fraud")

    return render_template(
        "dashboard.html",
        total=total_transactions,
        frauds=fraud_count,
        valids=valid_count,
        recent=recent_transactions,
        amounts=amounts,
        play_alert=play_alert
    )

# =====================================================
# TRANSACTIONS PAGE
# =====================================================
@app.route("/transactions")
@login_required
def transactions():
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT amount, result
            FROM transactions
            ORDER BY id DESC
        """)
        records = cursor.fetchall()

    return render_template("transactions.html", records=records)

# =====================================================
# FRAUD ALERTS
# =====================================================
@app.route("/fraud-alerts")
@login_required
def fraud_alerts():
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT amount
            FROM transactions
            WHERE result='Fraud'
            ORDER BY id DESC
        """)
        frauds = cursor.fetchall()

    return render_template("fraud_alerts.html", frauds=frauds)
# =====================================================
# COMPARE MODELS PAGE
# =====================================================
@app.route("/compare", methods=["GET", "POST"])
@login_required
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

        # XGBoost (Already loaded in main app)
        scaled = scaler.transform(data)
        reduced = pca.transform(scaled)
        xgb_pred = model.predict(reduced)[0]

        results = {
            "Decision Tree": "Fraud" if dt_pred == 1 else "Valid",
            "SVM": "Fraud" if svm_pred == 1 else "Valid",
            "Random Forest": "Fraud" if rf_pred == 1 else "Valid",
            "XGBoost (Proposed)": "Fraud" if xgb_pred == 1 else "Valid",
        }

    return render_template("compare.html", results=results)

# =====================================================
# ANALYZE TRANSACTION
# =====================================================
@app.route("/analyze", methods=["POST"])
@login_required
def analyze():
    amount = float(request.form["amount"])
    frequency = int(request.form["frequency"])
    location = int(request.form["location_change"])
    device = int(request.form["device_change"])
    merchant = float(request.form["merchant_risk"])

    data = np.array([[amount, 12, frequency, location, device, merchant, 1]])

    scaled = scaler.transform(data)
    reduced = pca.transform(scaled)
    prediction = model.predict(reduced)[0]

    result = "Fraud" if prediction == 1 else "Valid"
    session["last_result"] = result

    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO transactions
            (amount, frequency, location_change, device_change, merchant_risk, result)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (amount, frequency, location, device, merchant, result))
        conn.commit()

    return render_template(
    "result.html",
    amount=amount,
    frequency=frequency,
    location=location,
    device=device,
    merchant=merchant,
    result=result
)


# =====================================================
# LOGOUT
# =====================================================
@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))

# =====================================================
# RUN SERVER
# =====================================================
if __name__ == "__main__":
    app.run(debug=True)
