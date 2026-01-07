import joblib
import numpy as np
import sqlite3

from flask import session
from flask import Flask, render_template, request, redirect, url_for
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user
from werkzeug.security import generate_password_hash, check_password_hash

# =====================================================
# LOAD ML MODELS (ONCE AT STARTUP)
# =====================================================
model = joblib.load("model/fraudshield_xgboost.pkl")
pca = joblib.load("model/pca_transform.pkl")
scaler = joblib.load("model/scaler.pkl")

# =====================================================
# FLASK APP SETUP
# =====================================================
app = Flask(__name__)
app.secret_key = "fraudshield-secret-key"

# =====================================================
# LOGIN MANAGER SETUP
# =====================================================
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

# =====================================================
# USER CLASS
# =====================================================
class User(UserMixin):
    def __init__(self, id, username):
        self.id = id
        self.username = username

# =====================================================
# LOAD USER FROM DATABASE
# =====================================================
@login_manager.user_loader
def load_user(user_id):
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute("SELECT id, username FROM users WHERE id = ?", (user_id,))
    user = cursor.fetchone()
    conn.close()

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

        conn = sqlite3.connect("users.db")
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, password FROM users WHERE username = ?",
            (username,)
        )
        user = cursor.fetchone()
        conn.close()

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
        password = generate_password_hash(request.form["password"])

        conn = sqlite3.connect("users.db")
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO users (username, password) VALUES (?, ?)",
            (username, password)
        )
        conn.commit()
        conn.close()

        return redirect(url_for("login"))

    return render_template("register.html")

# =====================================================
# DASHBOARD ROUTE (PROTECTED + DATA)
# =====================================================
@app.route("/dashboard")
@login_required
def dashboard():
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()

    # Total transactions
    cursor.execute("SELECT COUNT(*) FROM transactions")
    total_transactions = cursor.fetchone()[0] or 0

    # Fraud count
    cursor.execute("SELECT COUNT(*) FROM transactions WHERE result='Fraud'")
    fraud_count = cursor.fetchone()[0] or 0

    # Valid count
    cursor.execute("SELECT COUNT(*) FROM transactions WHERE result='Valid'")
    valid_count = cursor.fetchone()[0] or 0

    # Recent transactions
    cursor.execute("""
        SELECT amount, result
        FROM transactions
        ORDER BY id DESC
        LIMIT 5
    """)
    recent_transactions = cursor.fetchall() or []

    # Amounts for bar chart
    cursor.execute("""
        SELECT amount
        FROM transactions
        ORDER BY id DESC
        LIMIT 6
    """)
    amounts = [row[0] for row in cursor.fetchall()]
    amounts.reverse()   # safer than [::-1]

    conn.close()

    last_result = session.pop('last_result', None)
    play_alert = (last_result == "Fraud")
    play_alert=play_alert



    return render_template(
    "dashboard.html",
    total=total_transactions,
    frauds=fraud_count,
    valids=valid_count,
    recent=recent_transactions,
    amounts=amounts,
    play_alert=play_alert
)


@app.route("/transactions")
@login_required
def transactions():
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()

    cursor.execute("""
        SELECT amount, result
        FROM transactions
        ORDER BY id DESC
    """)
    records = cursor.fetchall()
    conn.close()

    return render_template("transactions.html", records=records)

@app.route("/fraud-alerts")
@login_required
def fraud_alerts():
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()

    cursor.execute("""
        SELECT amount
        FROM transactions
        WHERE result='Fraud'
        ORDER BY id DESC
    """)
    frauds = cursor.fetchall()
    conn.close()

    return render_template("fraud_alerts.html", frauds=frauds)

# =====================================================
# ANALYZE TRANSACTION (ML PREDICTION)
# =====================================================
@app.route("/analyze", methods=["POST"])
@login_required
def analyze():
    amount = float(request.form["amount"])
    frequency = int(request.form["frequency"])
    location = int(request.form["location_change"])
    device = int(request.form["device_change"])
    merchant = float(request.form["merchant_risk"])

    # IMPORTANT: Feature order must match training
    data = np.array([[amount, 12, frequency, location, device, merchant, 1]])

    scaled = scaler.transform(data)
    reduced = pca.transform(scaled)
    prediction = model.predict(reduced)[0]

    result = "Fraud" if prediction == 1 else "Valid"
    session['last_result']=result

    # Store transaction in DB
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO transactions
        (amount, frequency, location_change, device_change, merchant_risk, result)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (amount, frequency, location, device, merchant, result))
    conn.commit()
    conn.close()

    return redirect(url_for("dashboard"))

# =====================================================
# LOGOUT ROUTE
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
