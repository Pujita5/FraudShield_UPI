
> **FraudShield UPI** – A machine learning–driven UPI fraud detection system using SMOTE, PCA, and XGBoost with a real-time web dashboard for fraud analytics and alerts.

---

# 📄 README.md 

```markdown
# 🚨 FraudShield UPI  
### Machine Learning–Driven Fraud Detection System for UPI Transactions

FraudShield UPI is a web-based fraud detection system that uses machine learning to identify suspicious UPI transactions. The project demonstrates how modern AI techniques can be applied to financial transaction data to improve fraud detection accuracy and provide real-time analytics through an interactive dashboard.

---

## 🎯 Project Objective

The primary objective of this project is to:
- Detect fraudulent UPI transactions using machine learning
- Handle highly imbalanced transaction data
- Compare traditional ML algorithms with an advanced proposed model
- Provide a real-time, user-friendly web interface for analysis and visualization

---

## 🧠 Machine Learning Approach

### Baseline Models (Comparative Study)
- Decision Tree
- Support Vector Machine (SVM)
- Random Forest  
**Best observed accuracy:** 85.9%

### Proposed Model
- SMOTE (Synthetic Minority Over-sampling Technique)
- Feature Scaling
- Principal Component Analysis (PCA)
- XGBoost Classifier  

**Achieved Accuracy:** ~99.4%

The proposed system significantly improves fraud detection performance by addressing class imbalance and complex feature interactions.

---

## 🏗️ System Architecture

```

User Input (Web UI)
↓
Feature Preprocessing
↓
Scaler → PCA → XGBoost Model
↓
Fraud / Valid Prediction
↓
Dashboard Analytics & Alerts

```

---

## 💻 Tech Stack

### Backend
- Python
- Flask
- SQLite

### Machine Learning
- Scikit-learn
- XGBoost
- SMOTE (imbalanced-learn)

### Frontend
- HTML
- CSS
- JavaScript
- Chart.js

---

## 📊 Key Features

- Secure login and authentication
- Real-time fraud prediction
- Interactive dashboard with analytics
- Fraud vs Valid transaction visualization
- Transaction history and fraud alerts
- Scalable ML pipeline design

---

## 📁 Project Structure

```

FraudShield_UPI/
│
├── app.py
├── users.db
├── model/
│   ├── fraudshield_xgboost.pkl
│   ├── scaler.pkl
│   └── pca_transform.pkl
│
├── static/
│   ├── dashboard.css
│
├── templates/
│   ├── login.html
│   ├── register.html
│   ├── dashboard.html
│   ├── transactions.html
│   └── fraud_alerts.html
│
└── README.md

````

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/FraudShield-UPI.git
cd FraudShield-UPI
````

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Application

```bash
python app.py
```

### 4️⃣ Open in Browser

```
http://127.0.0.1:5000
```

---

## 🧪 Dataset

* Synthetic UPI transaction dataset
* Features include:

  * Transaction Amount
  * Transaction Frequency
  * Location Change
  * Device Change
  * Merchant Risk Score
* Highly imbalanced fraud vs non-fraud data

---

## 📈 Performance Metrics

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

Special emphasis is given to **fraud recall**, as missing fraud cases can lead to financial loss.

---

## 🚧 Limitations

* Uses synthetic data instead of real UPI transaction data
* No real-time NPCI or bank integration
* Prototype-level deployment

---

## 🔮 Future Enhancements

* Real-time transaction streaming
* Deep learning models (LSTM / Autoencoders)
* User-level data isolation
* SMS / Email fraud alerts
* Explainable AI (SHAP) integration

---

## 🌍 Real-World Applications

* Banking and UPI platforms
* FinTech payment gateways
* E-commerce fraud monitoring
* Risk analytics dashboards
* Cybersecurity monitoring systems

---

## 🏫 Academic Relevance

This project is suitable for:

* Final Year Engineering Projects
* Research Demonstrations
* Machine Learning Case Studies
* Fraud Analytics Prototypes

---

## 📜 License

This project is for academic and educational purposes only.


