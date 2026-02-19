import sqlite3

conn = sqlite3.connect("users.db")
cursor = conn.cursor()

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
conn.close()

print("✅ Transactions table ready")
