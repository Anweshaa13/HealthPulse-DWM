import hashlib
import hmac
import os
import json
import sqlite3
from datetime import datetime
from .warehouse import get_connection, DB_PATH


def _hash_password(password: str) -> str:
    """Password hashing using PBKDF2-HMAC-SHA256 with a per-password salt."""
    salt = os.urandom(16)
    iterations = 310_000
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"pbkdf2_sha256${iterations}${salt.hex()}${digest.hex()}"


def _verify_password(password: str, stored_hash: str) -> bool:
    """Verify modern PBKDF2 hashes and transparently support legacy SHA-256 hashes."""
    if stored_hash.startswith("pbkdf2_sha256$"):
        try:
            _, iterations, salt_hex, digest_hex = stored_hash.split("$", 3)
            expected = bytes.fromhex(digest_hex)
            actual = hashlib.pbkdf2_hmac(
                "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), int(iterations)
            )
            return hmac.compare_digest(actual, expected)
        except (ValueError, TypeError):
            return False
    # Legacy accounts created by the previous project version.
    legacy = hashlib.sha256(password.encode("utf-8")).hexdigest()
    return hmac.compare_digest(legacy, stored_hash)


def init_auth_schema():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS App_Users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            full_name TEXT NOT NULL,
            email TEXT,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('admin','user')),
            patient_id TEXT,
            active INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS User_Assessments (
            assessment_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            patient_id TEXT,
            assessment_date TEXT NOT NULL,
            age INTEGER,
            gender TEXT,
            smoking TEXT,
            exercise_level TEXT,
            systolic_bp REAL,
            diastolic_bp REAL,
            bmi REAL,
            fasting_glucose REAL,
            hba1c REAL,
            total_cholesterol REAL,
            serum_creatinine REAL,
            length_of_stay INTEGER,
            hypertension INTEGER,
            diabetes INTEGER,
            heart_disease INTEGER,
            chronic_kidney_disease INTEGER,
            risk_score REAL,
            estimated_cost REAL,
            prediction_json TEXT,
            FOREIGN KEY(user_id) REFERENCES App_Users(user_id)
        );
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_user_assessments_user ON User_Assessments(user_id);")
    # Bootstrap only the administrator account if it does not already exist.
    cur.execute("SELECT COUNT(*) FROM App_Users WHERE username = 'admin'")
    if cur.fetchone()[0] == 0:
        cur.execute(
            "INSERT INTO App_Users (username, full_name, email, password_hash, role) VALUES (?, ?, ?, ?, ?)",
            ("admin", "HealthPulse Administrator", "admin@healthpulse.local", _hash_password("admin123"), "admin")
        )
    conn.commit()
    conn.close()


def authenticate(username: str, password: str):
    init_auth_schema()
    conn = get_connection()
    row = conn.execute(
        "SELECT user_id, username, full_name, email, password_hash, role, patient_id, active FROM App_Users WHERE username = ? AND active = 1",
        (username.strip(),)
    ).fetchone()
    conn.close()
    if not row:
        return None
    keys = ["user_id", "username", "full_name", "email", "password_hash", "role", "patient_id", "active"]
    record = dict(zip(keys, row))
    if not _verify_password(password, record["password_hash"]):
        return None
    # Upgrade legacy SHA-256 passwords after a successful login.
    if not record["password_hash"].startswith("pbkdf2_sha256$"):
        conn = get_connection()
        conn.execute("UPDATE App_Users SET password_hash = ? WHERE user_id = ?", (_hash_password(password), record["user_id"]))
        conn.commit()
        conn.close()
    record.pop("password_hash", None)
    return record


def create_user(username, full_name, email, password, patient_id=None, role="user"):
    init_auth_schema()
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO App_Users (username, full_name, email, password_hash, role, patient_id) VALUES (?, ?, ?, ?, ?, ?)",
            (username.strip(), full_name.strip(), email.strip(), _hash_password(password), role, patient_id.strip() if patient_id else None)
        )
        conn.commit()
        return True, "User created successfully."
    except sqlite3.IntegrityError:
        return False, "Username already exists."
    finally:
        conn.close()


def get_user(user_id):
    init_auth_schema()
    conn = get_connection()
    row = conn.execute(
        "SELECT user_id, username, full_name, email, role, patient_id, active, created_at FROM App_Users WHERE user_id = ?",
        (int(user_id),)
    ).fetchone()
    conn.close()
    if not row:
        return None
    keys = ["user_id", "username", "full_name", "email", "role", "patient_id", "active", "created_at"]
    return dict(zip(keys, row))


def update_user_profile(user_id, full_name, email, patient_id=None):
    init_auth_schema()
    conn = get_connection()
    conn.execute(
        "UPDATE App_Users SET full_name = ?, email = ?, patient_id = ? WHERE user_id = ?",
        (full_name.strip(), email.strip(), patient_id.strip() if patient_id else None, int(user_id))
    )
    conn.commit()
    conn.close()


def change_password(user_id, current_password, new_password):
    init_auth_schema()
    conn = get_connection()
    row = conn.execute("SELECT password_hash FROM App_Users WHERE user_id = ?", (int(user_id),)).fetchone()
    if not row or not _verify_password(current_password, row[0]):
        conn.close()
        return False, "Current password is incorrect."
    conn.execute("UPDATE App_Users SET password_hash = ? WHERE user_id = ?", (_hash_password(new_password), int(user_id)))
    conn.commit()
    conn.close()
    return True, "Password changed successfully."


def list_users():
    init_auth_schema()
    conn = get_connection()
    df = __import__("pandas").read_sql_query(
        "SELECT user_id, username, full_name, email, role, patient_id, active, created_at FROM App_Users ORDER BY user_id DESC",
        conn
    )
    conn.close()
    return df


def set_user_active(user_id, active):
    init_auth_schema()
    conn = get_connection()
    conn.execute("UPDATE App_Users SET active = ? WHERE user_id = ?", (1 if active else 0, int(user_id)))
    conn.commit()
    conn.close()


def save_assessment(user_id, patient_id, values, predictions, risk_score, estimated_cost):
    init_auth_schema()
    conn = get_connection()
    conn.execute("""
        INSERT INTO User_Assessments (
            user_id, patient_id, assessment_date, age, gender, smoking, exercise_level,
            systolic_bp, diastolic_bp, bmi, fasting_glucose, hba1c, total_cholesterol,
            serum_creatinine, length_of_stay, hypertension, diabetes, heart_disease,
            chronic_kidney_disease, risk_score, estimated_cost, prediction_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        user_id, patient_id, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        values.get("age"), values.get("gender"), values.get("smoking"), values.get("exercise_level"),
        values.get("systolic_bp"), values.get("diastolic_bp"), values.get("bmi"), values.get("fasting_glucose"),
        values.get("hba1c"), values.get("total_cholesterol"), values.get("serum_creatinine"), values.get("length_of_stay"),
        values.get("hypertension", 0), values.get("diabetes", 0), values.get("heart_disease", 0),
        values.get("chronic_kidney_disease", 0), risk_score, estimated_cost, json.dumps(predictions)
    ))
    conn.commit()
    assessment_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.close()
    return assessment_id


def get_user_assessments(user_id):
    init_auth_schema()
    conn = get_connection()
    df = __import__("pandas").read_sql_query(
        "SELECT * FROM User_Assessments WHERE user_id = ? ORDER BY assessment_id DESC",
        conn, params=[user_id]
    )
    conn.close()
    return df


def get_all_assessments():
    init_auth_schema()
    conn = get_connection()
    df = __import__("pandas").read_sql_query("""
        SELECT a.*, u.username, u.full_name
        FROM User_Assessments a
        JOIN App_Users u ON a.user_id = u.user_id
        ORDER BY a.assessment_id DESC
    """, conn)
    conn.close()
    return df
