import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
import sqlite3
import json
from io import BytesIO
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="HealthPulse | Clinical Data Warehouse & Mining",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Include current directory in path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from src.warehouse import (
    get_connection, olap_roll_up, olap_drill_down, 
    olap_slice, olap_dice, olap_pivot, get_warehouse_stats, DB_PATH
)
from src.etl import run_etl_pipeline, RAW_DATA_PATH
from src.auth import (
    init_auth_schema, authenticate, create_user, list_users, set_user_active,
    save_assessment, get_user_assessments, get_all_assessments, get_user,
    update_user_profile, change_password
)
from src.mining_models import (
    load_warehouse_dataset, DiseaseRiskPredictor, 
    PatientClusteringModel, HealthcareAssociationMiner,
    TreatmentCostRegressor
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #0E7490;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #F0FDF4 0%, #E0F2FE 100%);
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 5px solid #0284C7;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 18px;
        border-radius: 10px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(14, 116, 144, 0.12);
    }
    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(148,163,184,0.16);
    }
    [data-testid="stMetric"] {
        background: rgba(30,41,59,0.42);
        border: 1px solid rgba(148,163,184,0.12);
        padding: 14px 16px;
        border-radius: 14px;
    }
    div.stButton > button, div.stDownloadButton > button {
        border-radius: 10px;
        font-weight: 650;
        min-height: 42px;
    }
    div[data-testid="stForm"] {
        border: 1px solid rgba(148,163,184,0.16);
        border-radius: 16px;
        padding: 20px;
        background: rgba(15,23,42,0.28);
    }
    .auth-card {
        max-width: 720px;
        margin: 6vh auto 0 auto;
        padding: 30px 34px 24px 34px;
        border: 1px solid rgba(148,163,184,0.18);
        border-radius: 22px;
        background: linear-gradient(145deg, rgba(15,23,42,0.92), rgba(30,41,59,0.68));
        box-shadow: 0 18px 50px rgba(0,0,0,0.22);
    }
    .eyebrow {
        letter-spacing: .08em;
        text-transform: uppercase;
        font-size: .78rem;
        font-weight: 700;
        opacity: .72;
    }
    .feature-note {
        border: 1px solid rgba(14,116,144,.22);
        background: linear-gradient(135deg, rgba(14,116,144,.10), rgba(30,41,59,.28));
        border-radius: 16px;
        padding: 16px 18px;
        margin: 8px 0 18px 0;
    }
    .page-divider {
        height: 1px;
        background: rgba(148,163,184,.14);
        margin: 22px 0;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# CACHED DATA & MODEL LOADERS
# -------------------------------------------------------------
@st.cache_data(show_spinner="Connecting to Data Warehouse...")
def get_cached_stats():
    if not os.path.exists(DB_PATH):
        run_etl_pipeline()
    return get_warehouse_stats()

@st.cache_data(show_spinner="Loading Clinical Data Warehouse Records...")
def get_cached_df():
    if not os.path.exists(DB_PATH):
        run_etl_pipeline()
    return load_warehouse_dataset()

@st.cache_resource(show_spinner="Training Predictive Mining Classifiers...")
def get_trained_predictor(target="heart_disease"):
    df = get_cached_df()
    predictor = DiseaseRiskPredictor(target=target)
    predictor.train_and_evaluate(df)
    return predictor

@st.cache_resource(show_spinner="Computing Patient Risk Clusters...")
def get_clustering_results(n_clusters=4):
    df = get_cached_df()
    model = PatientClusteringModel(n_clusters=n_clusters)
    df_clustered, summary, sil_score = model.fit(df)
    return model, df_clustered, summary, sil_score

@st.cache_resource(show_spinner="Mining Association Rules (Apriori)...")
def get_mined_rules(min_support=0.08, min_confidence=0.45):
    df = get_cached_df()
    miner = HealthcareAssociationMiner(min_support=min_support, min_confidence=min_confidence)
    itemsets, rules = miner.mine_rules(df)
    return miner, itemsets, rules

@st.cache_resource(show_spinner="Training Healthcare Expenditure Regressors...")
def get_trained_regressor():
    df = get_cached_df()
    reg = TreatmentCostRegressor()
    reg.train_and_evaluate(df)
    return reg

# -------------------------------------------------------------
# AUTHENTICATION & ROLE-BASED ACCESS
# -------------------------------------------------------------
init_auth_schema()

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "current_user" not in st.session_state:
    st.session_state.current_user = None
if "show_register" not in st.session_state:
    st.session_state.show_register = False
if "registration_success" not in st.session_state:
    st.session_state.registration_success = False
if "menu_selection" not in st.session_state:
    st.session_state.menu_selection = None
if "nav_radio" not in st.session_state:
    st.session_state.nav_radio = None


def assessments_to_excel(df, include_user_info=False):
    """Create a readable Excel workbook from saved assessment results."""
    export_df = df.copy()
    if export_df.empty:
        return b""

    prediction_rows = []
    for _, row in export_df.iterrows():
        base = {
            "Assessment ID": row.get("assessment_id"),
            "Assessment Date": row.get("assessment_date"),
            "Patient ID": row.get("patient_id"),
        }
        if include_user_info:
            base["Username"] = row.get("username")
            base["Full Name"] = row.get("full_name")
        base.update({
            "Age": row.get("age"), "Gender": row.get("gender"),
            "Smoking": row.get("smoking"), "Exercise Level": row.get("exercise_level"),
            "Systolic BP": row.get("systolic_bp"), "Diastolic BP": row.get("diastolic_bp"),
            "BMI": row.get("bmi"), "Fasting Glucose": row.get("fasting_glucose"),
            "HbA1c": row.get("hba1c"), "Total Cholesterol": row.get("total_cholesterol"),
            "Serum Creatinine": row.get("serum_creatinine"),
            "Length of Stay": row.get("length_of_stay"),
            "Hypertension": row.get("hypertension"), "Diabetes": row.get("diabetes"),
            "Heart Disease": row.get("heart_disease"),
            "Chronic Kidney Disease": row.get("chronic_kidney_disease"),
            "Overall Risk Score": row.get("risk_score"),
            "Estimated Treatment Cost": row.get("estimated_cost"),
        })
        try:
            predictions = json.loads(row.get("prediction_json") or "{}")
        except (TypeError, json.JSONDecodeError):
            predictions = {}
        for disease_key, disease_label in [
            ("heart_disease", "Heart Disease"),
            ("diabetes", "Diabetes"),
            ("chronic_kidney_disease", "CKD"),
        ]:
            for model_name, result in predictions.get(disease_key, {}).items():
                safe_model = str(model_name).replace(" ", "_").replace("/", "_")
                base[f"{disease_label} - {safe_model} Prediction"] = result.get("Prediction")
                base[f"{disease_label} - {safe_model} Probability (%)"] = result.get("Probability")
                base[f"{disease_label} - {safe_model} Risk"] = result.get("RiskLevel")
        prediction_rows.append(base)

    readable_df = pd.DataFrame(prediction_rows)
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        readable_df.to_excel(writer, index=False, sheet_name="Assessment Results")
        # Keep a compact summary sheet for easy presentation/reporting.
        summary_cols = [c for c in [
            "Assessment ID", "Assessment Date", "Patient ID", "Username", "Full Name",
            "Overall Risk Score", "Estimated Treatment Cost"
        ] if c in readable_df.columns]
        readable_df[summary_cols].to_excel(writer, index=False, sheet_name="Summary")
        for sheet_name in writer.sheets:
            ws = writer.sheets[sheet_name]
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
            for column_cells in ws.columns:
                max_len = max(len(str(cell.value or "")) for cell in column_cells)
                ws.column_dimensions[column_cells[0].column_letter].width = min(max(max_len + 2, 12), 32)
    return output.getvalue()


def admin_data_to_excel(users_df, assessments_df):
    """Create an admin workbook containing user profiles and all saved assessments."""
    output = BytesIO()
    profiles = users_df.copy() if users_df is not None else pd.DataFrame()
    assessments = assessments_df.copy() if assessments_df is not None else pd.DataFrame()

    if not profiles.empty:
        if not assessments.empty and "user_id" in assessments.columns:
            counts = assessments.groupby("user_id").size().rename("Assessment Count")
            profiles = profiles.merge(counts, left_on="user_id", right_index=True, how="left")
            profiles["Assessment Count"] = profiles["Assessment Count"].fillna(0).astype(int)
        if "active" in profiles.columns:
            profiles["Account Status"] = profiles["active"].map({1: "Active", 0: "Inactive"})
            profiles = profiles.drop(columns=["active"], errors="ignore")

    assessment_excel = assessments_to_excel(assessments, include_user_info=True) if not assessments.empty else b""

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        if profiles.empty:
            pd.DataFrame({"Message": ["No registered users found."]}).to_excel(writer, index=False, sheet_name="User Profiles")
        else:
            profiles.to_excel(writer, index=False, sheet_name="User Profiles")

        if assessments.empty:
            pd.DataFrame({"Message": ["No saved user assessments found."]}).to_excel(writer, index=False, sheet_name="Assessment Results")
        else:
            # Rebuild the readable assessment dataframe from the existing helper.
            temp = BytesIO(assessment_excel)
            assessment_readable = pd.read_excel(temp, sheet_name="Assessment Results")
            assessment_readable.to_excel(writer, index=False, sheet_name="Assessment Results")

        summary = pd.DataFrame([{
            "Registered Users": len(profiles),
            "Saved User Assessments": len(assessments),
            "Active Users": int((profiles.get("Account Status", pd.Series(dtype=str)) == "Active").sum()) if not profiles.empty else 0,
            "Administrators": int((profiles.get("role", pd.Series(dtype=str)) == "admin").sum()) if not profiles.empty else 0,
        }])
        summary.to_excel(writer, index=False, sheet_name="Admin Summary")

        for sheet_name, ws in writer.sheets.items():
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
            for column_cells in ws.columns:
                max_len = max(len(str(cell.value or "")) for cell in column_cells)
                ws.column_dimensions[column_cells[0].column_letter].width = min(max(max_len + 2, 12), 36)
    return output.getvalue()


def calculate_patient_risk(values):
    score = 0
    score += min(max((values["age"] - 40) * 0.35, 0), 18)
    score += min(max((values["systolic_bp"] - 120) * 0.18, 0), 18)
    score += min(max((values["bmi"] - 25) * 0.8, 0), 12)
    score += min(max((values["fasting_glucose"] - 100) * 0.08, 0), 10)
    score += min(max((values["total_cholesterol"] - 180) * 0.04, 0), 6)
    score += 6 if values["smoking"] == "Current" else (3 if values["smoking"] == "Former" else 0)
    score += 5 if values["exercise_level"] == "Sedentary" else 0
    score += 8 * int(values.get("hypertension", 0))
    score += 8 * int(values.get("diabetes", 0))
    score += 8 * int(values.get("chronic_kidney_disease", 0))
    return round(min(score, 100), 1)


def save_patient_to_warehouse(patient):
    """Persist a manually entered patient into the Star Schema warehouse."""
    conn = get_connection()
    cur = conn.cursor()

    # Patient name is useful for the interactive Patient Management screen.
    existing_cols = {row[1] for row in cur.execute("PRAGMA table_info(Dim_Patient)").fetchall()}
    if "patient_name" not in existing_cols:
        cur.execute("ALTER TABLE Dim_Patient ADD COLUMN patient_name TEXT")

    patient_id = str(patient["patient_id"]).strip()
    admission_date = pd.to_datetime(patient["admission_date"]).strftime("%Y-%m-%d")
    date_key = int(pd.to_datetime(admission_date).strftime("%Y%m%d"))
    age = int(patient["age"])
    age_group = (
        "Young Adult (18-29)" if age < 30 else
        "Adult (30-44)" if age < 45 else
        "Middle-Aged (45-59)" if age < 60 else
        "Senior (60-74)" if age < 75 else "Elderly (75+)"
    )
    bmi = float(patient["bmi"])
    bmi_category = (
        "Underweight" if bmi < 18.5 else
        "Normal" if bmi < 25 else
        "Overweight" if bmi < 30 else "Obese"
    )

    conditions = []
    if patient["hypertension"]: conditions.append("HTN")
    if patient["diabetes"]: conditions.append("DM")
    if patient["heart_disease"]: conditions.append("CVD")
    if patient["chronic_kidney_disease"]: conditions.append("CKD")
    profile_name = "+".join(conditions) if conditions else "Healthy/Normal"
    comorbidity_count = int(sum([
        patient["hypertension"], patient["diabetes"],
        patient["heart_disease"], patient["chronic_kidney_disease"]
    ]))

    # Upsert the patient dimension.
    cur.execute("SELECT patient_key FROM Dim_Patient WHERE patient_id = ?", (patient_id,))
    row = cur.fetchone()
    if row:
        patient_key = row[0]
        cur.execute("""
            UPDATE Dim_Patient SET patient_name=?, age=?, age_group=?, gender=?,
            blood_group=?, smoking=?, alcohol=?, exercise_level=? WHERE patient_key=?
        """, (
            patient["patient_name"], age, age_group, patient["gender"],
            patient["blood_group"], patient["smoking"], patient["alcohol"],
            patient["exercise_level"], patient_key
        ))
    else:
        cur.execute("""
            INSERT INTO Dim_Patient
            (patient_id, patient_name, age, age_group, gender, blood_group, smoking, alcohol, exercise_level)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            patient_id, patient["patient_name"], age, age_group, patient["gender"],
            patient["blood_group"], patient["smoking"], patient["alcohol"],
            patient["exercise_level"]
        ))
        patient_key = cur.lastrowid

    cur.execute("SELECT date_key FROM Dim_Date WHERE date_key=?", (date_key,))
    if not cur.fetchone():
        dt = pd.to_datetime(admission_date)
        cur.execute("""
            INSERT INTO Dim_Date
            (date_key, full_date, year, quarter, month, month_name, day, day_name, is_weekend)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            date_key, admission_date, dt.year, dt.quarter, dt.month,
            dt.strftime("%B"), dt.day, dt.strftime("%A"), int(dt.dayofweek >= 5)
        ))

    cur.execute("""
        INSERT OR IGNORE INTO Dim_Department(department_name, admission_type)
        VALUES (?, ?)
    """, (patient["department"], patient["admission_type"]))
    dept_key = cur.execute("""
        SELECT dept_key FROM Dim_Department
        WHERE department_name=? AND admission_type=?
    """, (patient["department"], patient["admission_type"])).fetchone()[0]

    cur.execute("""
        INSERT OR IGNORE INTO Dim_DiagnosisCategory
        (profile_name, has_hypertension, has_diabetes, has_heart_disease, has_ckd, comorbidity_count)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        profile_name, patient["hypertension"], patient["diabetes"],
        patient["heart_disease"], patient["chronic_kidney_disease"], comorbidity_count
    ))
    diag_key = cur.execute("""
        SELECT diag_key FROM Dim_DiagnosisCategory WHERE profile_name=?
    """, (profile_name,)).fetchone()[0]

    # Keep one fresh fact assessment for this manually entered patient.
    cur.execute("DELETE FROM Fact_PatientAssessments WHERE patient_key=? AND date_key=?", (patient_key, date_key))
    cur.execute("""
        INSERT INTO Fact_PatientAssessments (
            patient_key, date_key, dept_key, diag_key, systolic_bp, diastolic_bp,
            heart_rate, bmi, bmi_category, fasting_glucose, hba1c, total_cholesterol,
            ldl, hdl, serum_creatinine, length_of_stay, treatment_cost,
            hypertension, diabetes, heart_disease, chronic_kidney_disease,
            risk_score, high_risk_patient
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        patient_key, date_key, dept_key, diag_key,
        patient["systolic_bp"], patient["diastolic_bp"], patient["heart_rate"],
        bmi, bmi_category, patient["fasting_glucose"], patient["hba1c"],
        patient["total_cholesterol"], patient["ldl"], patient["hdl"],
        patient["serum_creatinine"], patient["length_of_stay"], patient["treatment_cost"],
        patient["hypertension"], patient["diabetes"], patient["heart_disease"],
        patient["chronic_kidney_disease"], patient["risk_score"],
        int(patient["risk_score"] >= 60)
    ))
    conn.commit()
    conn.close()
    return patient_id

def get_manual_patient_records():
    conn = get_connection()
    try:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(Dim_Patient)").fetchall()]
        name_col = "p.patient_name" if "patient_name" in cols else "NULL AS patient_name"
        return pd.read_sql_query(f"""
            SELECT p.patient_id, {name_col}, p.age, p.gender, p.blood_group,
                   p.smoking, p.alcohol, p.exercise_level,
                   f.systolic_bp, f.diastolic_bp, f.bmi, f.fasting_glucose,
                   f.hba1c, f.total_cholesterol, f.treatment_cost, f.risk_score,
                   d.department_name, d.admission_type, dt.full_date AS admission_date
            FROM Dim_Patient p
            LEFT JOIN Fact_PatientAssessments f ON p.patient_key=f.patient_key
            LEFT JOIN Dim_Department d ON f.dept_key=d.dept_key
            LEFT JOIN Dim_Date dt ON f.date_key=dt.date_key
            ORDER BY p.patient_key DESC
        """, conn)
    finally:
        conn.close()

def render_register():
    st.markdown("<div class='auth-card'>", unsafe_allow_html=True)
    st.markdown('<div class="main-title">📝 Create Your HealthPulse Account</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">New accounts are created as standard users. Only administrators can create administrator accounts.</div>', unsafe_allow_html=True)
    with st.form("register_form"):
        c1, c2 = st.columns(2)
        reg_name = c1.text_input("Full Name")
        reg_username = c2.text_input("Username")
        reg_email = c1.text_input("Email")
        reg_patient_id = c2.text_input("Patient ID (optional)")
        reg_password = c1.text_input("Password", type="password")
        reg_confirm = c2.text_input("Confirm Password", type="password")
        submitted = st.form_submit_button("📝 Create Account", type="primary", use_container_width=True)
    if st.button("← Back to Login", use_container_width=True):
        st.session_state.show_register = False
        st.rerun()
    if submitted:
        if not reg_name.strip() or not reg_username.strip() or not reg_email.strip() or not reg_password:
            st.error("Full name, username, email and password are required.")
        elif len(reg_username.strip()) < 3:
            st.error("Username must contain at least 3 characters.")
        elif len(reg_password) < 8:
            st.error("Password must contain at least 8 characters.")
        elif reg_password != reg_confirm:
            st.error("Passwords do not match.")
        elif "@" not in reg_email or "." not in reg_email.split("@")[-1]:
            st.error("Please enter a valid email address.")
        else:
            ok, msg = create_user(reg_username, reg_name, reg_email, reg_password, reg_patient_id, "user")
            if ok:
                # Return the user to the login screen after registration.
                # Keep a one-time success message on the login screen.
                st.session_state.show_register = False
                st.session_state.registration_success = True
                st.session_state.registered_username = reg_username.strip()
                st.rerun()
            else:
                st.error(msg)
    st.markdown("</div>", unsafe_allow_html=True)


def render_login():
    st.markdown("<div class='auth-card'>", unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">Clinical Analytics Platform</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-title">🩺 HealthPulse DWM</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Secure healthcare data warehouse, analytics and patient-risk insights</div>', unsafe_allow_html=True)
    if st.session_state.get("show_register", False):
        render_register()
        return
    if st.session_state.get("registration_success", False):
        st.success("Account created successfully. Please log in below with your new username and password.")
        st.session_state.registration_success = False

    with st.form("login_form"):
        username = st.text_input("Username", value=st.session_state.pop("registered_username", ""))
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("🔐 Login", type="primary", use_container_width=True)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("📝 Create Account", use_container_width=True):
            st.session_state.show_register = True
            st.rerun()
    with c2:
        st.caption("New accounts are standard users.")
    st.caption("Secure account access • Use your registered credentials.")
    st.markdown("</div>", unsafe_allow_html=True)
    if submitted:
        user = authenticate(username, password)
        if user:
            st.session_state.authenticated = True
            st.session_state.current_user = user
            # Explicit role-based landing page after every successful login.
            st.session_state.menu_selection = (
                "👨‍💼 Admin Dashboard" if user["role"] == "admin" else "👤 User Dashboard"
            )
            # Clear the sidebar widget state so the role-specific landing page is honored.
            st.session_state.pop("nav_radio", None)
            st.rerun()
        else:
            st.error("Invalid username, password, or inactive account.")


if not st.session_state.authenticated:
    render_login()
    st.stop()

current_user = st.session_state.current_user

# -------------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------------
with st.sidebar:
    st.markdown(
        "<div style='font-size:2.4rem;line-height:1'>🩺</div>"
        "<div style='font-size:1.35rem;font-weight:800;color:#0ea5c9;margin-top:6px'>HealthPulse DWM</div>"
        "<div style='font-size:.82rem;opacity:.7;margin-top:4px'>Healthcare Data Warehouse • Analytics • Patient Insights</div>",
        unsafe_allow_html=True
    )
    st.markdown("---")
    st.write(f"**Signed in:** {current_user['full_name']}")
    st.caption(f"Role: {current_user['role'].upper()}")

    # Both roles can use the analytical DWM modules. Only admins receive
    # account-management controls inside the Admin Dashboard/Profile.
    navigation_items = []
    if current_user["role"] == "admin":
        navigation_items += ["👨‍💼 Admin Dashboard", "👨‍💼 Admin Profile"]
    else:
        navigation_items += ["👤 User Dashboard", "👤 My Profile"]
    navigation_items += [
        "🏛️ Architecture & Star Schema",
        "🔄 ETL Data Pipeline",
        "🧊 OLAP Cube Analytics",
        "🩺 Predictive Disease Mining",
        "💰 Cost Prediction Regression",
        "👥 Patient Cohort Clustering",
        "🔗 Comorbidity Association Rules",
        "📈 Model Benchmark & Evaluation",
        "👤 Patient Management",
        "📑 Patient Audit Report"
    ]

    if st.session_state.get("menu_selection") not in navigation_items:
        st.session_state.menu_selection = navigation_items[0]
    if st.session_state.get("nav_radio") not in navigation_items:
        st.session_state.nav_radio = st.session_state.menu_selection

    selected_index = navigation_items.index(
        st.session_state.get("nav_radio", navigation_items[0])
    )
    menu = st.radio("Navigation", navigation_items, index=selected_index, key="nav_radio")
    st.session_state.menu_selection = menu

    if st.button("🚪 Logout", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.current_user = None
        st.session_state.menu_selection = None
        st.session_state.pop("nav_radio", None)
        st.rerun()

    st.markdown("---")
# -------------------------------------------------------------
# ROLE-SPECIFIC DASHBOARDS
# -------------------------------------------------------------
if menu in ["👤 My Profile", "👨‍💼 Admin Profile"]:
    st.markdown(f'<div class="main-title">{("👨‍💼 Admin Profile" if current_user["role"] == "admin" else "👤 My Profile")}</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Manage your authenticated HealthPulse account</div>', unsafe_allow_html=True)
    profile = get_user(current_user["user_id"]) or current_user
    c1, c2, c3 = st.columns(3)
    c1.metric("Role", str(profile.get("role", "user")).upper())
    c2.metric("Username", profile.get("username", "—"))
    c3.metric("Status", "Active" if profile.get("active") else "Inactive")

    tab_edit, tab_security = st.tabs(["👤 Profile Details", "🔐 Security"])
    with tab_edit:
        st.info("Your profile is linked to your authenticated account. Passwords are never displayed or exported.")
        with st.form("profile_update_form"):
            full_name = st.text_input("Full Name", value=profile.get("full_name") or "")
            email = st.text_input("Email", value=profile.get("email") or "")
            patient_id = st.text_input("Linked Patient ID", value=profile.get("patient_id") or "")
            save_profile = st.form_submit_button("Save Profile", type="primary")
        if save_profile:
            if not full_name.strip() or "@" not in email:
                st.error("Enter a valid full name and email.")
            else:
                update_user_profile(current_user["user_id"], full_name, email, patient_id)
                refreshed = get_user(current_user["user_id"])
                st.session_state.current_user = {k: refreshed[k] for k in ["user_id","username","full_name","email","role","patient_id","active"]}
                st.success("Profile updated successfully.")
                st.rerun()
        st.write(f"**Account created:** {profile.get('created_at', '—')}")
    with tab_security:
        st.warning("Changing your password will invalidate the previous password immediately.")
        with st.form("change_password_form"):
            old_password = st.text_input("Current Password", type="password")
            new_password = st.text_input("New Password", type="password")
            confirm_password = st.text_input("Confirm New Password", type="password")
            change_btn = st.form_submit_button("Change Password", type="primary")
        if change_btn:
            if len(new_password) < 8:
                st.error("New password must contain at least 8 characters.")
            elif new_password != confirm_password:
                st.error("New passwords do not match.")
            else:
                ok, msg = change_password(current_user["user_id"], old_password, new_password)
                st.success(msg) if ok else st.error(msg)

elif menu == "👨‍💼 Admin Dashboard":
    st.markdown('<div class="main-title">👨‍💼 Admin Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">System-wide healthcare warehouse, user accounts, saved assessments and analytics</div>', unsafe_allow_html=True)
    st.caption('Manage accounts and review user-generated assessment data from one secure administrator workspace.')
    stats = get_cached_stats()
    users_df = list_users()
    assessments_df = get_all_assessments()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Warehouse Patients", f"{stats.get('Dim_Patient', 0):,}")
    c2.metric("Fact Assessments", f"{stats.get('Fact_PatientAssessments', 0):,}")
    c3.metric("Registered Users", f"{len(users_df):,}")
    c4.metric("User Assessments", f"{len(assessments_df):,}")

    st.markdown("---")
    st.subheader("📊 Warehouse Overview")
    col_a, col_b = st.columns(2)
    with col_a:
        conn = get_connection()
        disease_df = pd.read_sql_query("""
            SELECT SUM(heart_disease) AS HeartDisease, SUM(diabetes) AS Diabetes,
                   SUM(hypertension) AS Hypertension, SUM(chronic_kidney_disease) AS CKD
            FROM Fact_PatientAssessments
        """, conn)
        conn.close()
        long_disease = pd.DataFrame({"Condition": ["Heart Disease", "Diabetes", "Hypertension", "CKD"],
                                     "Cases": disease_df.iloc[0].tolist()})
        fig = px.bar(long_disease, x="Condition", y="Cases", title="Disease Distribution", text_auto=True)
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        conn = get_connection()
        dept_df = pd.read_sql_query("""
            SELECT d.department_name AS Department, COUNT(*) AS Patients
            FROM Fact_PatientAssessments f JOIN Dim_Department d ON f.dept_key=d.dept_key
            GROUP BY d.department_name ORDER BY Patients DESC
        """, conn)
        conn.close()
        fig = px.bar(dept_df, x="Department", y="Patients", title="Patients by Department")
        fig.update_layout(xaxis_tickangle=-35)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("👥 User Management")
    with st.expander("➕ Create New User", expanded=False):
        with st.form("create_user_form"):
            uc1, uc2 = st.columns(2)
            new_username = uc1.text_input("Username")
            new_name = uc2.text_input("Full Name")
            new_email = uc1.text_input("Email")
            new_patient_id = uc2.text_input("Linked Patient ID (optional)")
            new_password = uc1.text_input("Temporary Password", type="password")
            new_role = uc2.selectbox("Role", ["user", "admin"])
            create_btn = st.form_submit_button("Create Account", type="primary")
        if create_btn:
            if not new_username or not new_name or not new_password:
                st.error("Username, full name and password are required.")
            else:
                ok, msg = create_user(new_username, new_name, new_email, new_password, new_patient_id, new_role)
                st.success(msg) if ok else st.error(msg)
                if ok:
                    st.rerun()

    if not users_df.empty:
        st.subheader("👤 All User Profiles")
        st.dataframe(users_df, use_container_width=True, hide_index=True)
        profile_ids = users_df["user_id"].tolist()
        selected_profile_id = st.selectbox(
            "View a complete user profile", profile_ids,
            format_func=lambda x: users_df.loc[users_df["user_id"] == x, "username"].iloc[0]
        )
        selected_profile = get_user(selected_profile_id)
        if selected_profile:
            pc1, pc2, pc3 = st.columns(3)
            pc1.write(f"**Full Name:** {selected_profile['full_name']}")
            pc2.write(f"**Email:** {selected_profile.get('email') or '—'}")
            pc3.write(f"**Role:** {selected_profile['role'].upper()}")
            pc4, pc5, pc6 = st.columns(3)
            pc4.write(f"**Patient ID:** {selected_profile.get('patient_id') or '—'}")
            pc5.write(f"**Status:** {'Active' if selected_profile['active'] else 'Inactive'}")
            pc6.write(f"**Created:** {selected_profile.get('created_at') or '—'}")
            selected_user_assessments = get_user_assessments(selected_profile_id)
            st.caption(f"Saved assessments for {selected_profile['username']}: {len(selected_user_assessments)}")

        active_options = users_df[users_df["username"] != current_user["username"]][["user_id", "username", "active"]]
        if not active_options.empty:
            selected_uid = st.selectbox("Select account to activate/deactivate", active_options["user_id"].tolist(),
                                        format_func=lambda x: active_options.loc[active_options["user_id"] == x, "username"].iloc[0])
            selected_active = int(active_options.loc[active_options["user_id"] == selected_uid, "active"].iloc[0])
            if st.button("Deactivate" if selected_active else "Activate"):
                set_user_active(selected_uid, not bool(selected_active))
                st.success("Account status updated.")
                st.rerun()

    st.markdown("---")
    st.subheader("📋 Recent User Assessments")
    if assessments_df.empty:
        st.info("No user assessments have been submitted yet.")
    else:
        st.dataframe(assessments_df.head(25), use_container_width=True, hide_index=True)
        admin_excel = admin_data_to_excel(users_df, assessments_df)
        st.download_button(
            "📥 Export User Profiles + All Assessment Data to Excel",
            data=admin_excel,
            file_name="HealthPulse_Admin_User_Data_Report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
        st.caption("Admin Excel contains User Profiles, all saved assessment results, and an Admin Summary sheet.")

elif menu == "👤 User Dashboard":
    st.markdown('<div class="main-title">👤 My Health Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Your personal assessment, disease-risk analytics, saved history and treatment-cost insights</div>', unsafe_allow_html=True)
    st.caption('Use the tabs below to manage your profile, create a new assessment, review results, and export your saved data.')
    user_assessments = get_user_assessments(current_user["user_id"])

    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Patient ID", current_user.get("patient_id") or "Not linked")
    p2.metric("Assessments", len(user_assessments))
    p3.metric("Last Risk Score", f"{user_assessments.iloc[0]['risk_score']:.1f}/100" if not user_assessments.empty else "—")
    p4.metric("Last Estimated Cost", f"${user_assessments.iloc[0]['estimated_cost']:,.0f}" if not user_assessments.empty else "—")

    q1, q2, q3 = st.columns(3)
    q1.info("👤 **Profile**\n\nManage your name, email and patient ID.")
    q2.info("🩺 **Assessment**\n\nEnter clinical data and save your personalized analysis.")
    q3.info("📊 **History & Excel**\n\nReview saved results and download your report.")

    tab_profile, tab_assess, tab_results, tab_history = st.tabs(["👤 My Profile", "🩺 New Assessment", "🔮 Latest Results", "📋 My History"])

    with tab_profile:
        st.subheader("My Profile")
        st.write(f"**Name:** {current_user['full_name']}")
        st.write(f"**Username:** {current_user['username']}")
        st.write(f"**Email:** {current_user.get('email') or '—'}")
        st.write(f"**Linked Patient ID:** {current_user.get('patient_id') or 'Not linked'}")
        st.info("Keep your profile details up to date for accurate patient record linking.")

    with tab_assess:
        st.subheader("Enter New Health Assessment")
        with st.form("user_assessment_form"):
            a1, a2, a3 = st.columns(3)
            age = a1.number_input("Age", 18, 100, 45)
            gender = a1.selectbox("Gender", ["Female", "Male", "Other"])
            smoking = a1.selectbox("Smoking", ["Never", "Former", "Current"])
            exercise = a1.selectbox("Exercise Level", ["Sedentary", "Moderate", "Active"])
            sbp = a2.number_input("Systolic BP", 80.0, 220.0, 125.0)
            dbp = a2.number_input("Diastolic BP", 40.0, 140.0, 80.0)
            bmi = a2.number_input("BMI", 10.0, 60.0, 24.0)
            glucose = a2.number_input("Fasting Glucose", 50.0, 400.0, 100.0)
            hba1c = a3.number_input("HbA1c", 3.0, 20.0, 5.6)
            cholesterol = a3.number_input("Total Cholesterol", 80.0, 500.0, 180.0)
            creatinine = a3.number_input("Serum Creatinine", 0.2, 10.0, 1.0)
            los = a3.number_input("Expected Length of Stay", 0, 60, 3)
            colx, coly, colz = st.columns(3)
            htn = colx.checkbox("Known Hypertension")
            diab = coly.checkbox("Known Diabetes")
            ckd = colz.checkbox("Known CKD")
            submit_assessment = st.form_submit_button("🔍 Analyze & Save Assessment", type="primary", use_container_width=True)

        if submit_assessment:
            values = {
                "age": age, "gender": gender, "smoking": smoking, "exercise_level": exercise,
                "systolic_bp": sbp, "diastolic_bp": dbp, "bmi": bmi, "fasting_glucose": glucose,
                "hba1c": hba1c, "total_cholesterol": cholesterol, "serum_creatinine": creatinine,
                "length_of_stay": los, "hypertension": int(htn), "diabetes": int(diab),
                "heart_disease": 0, "chronic_kidney_disease": int(ckd)
            }
            payload = {
                "age": age, "gender": gender, "smoking": smoking, "exercise_level": exercise,
                "systolic_bp": sbp, "diastolic_bp": dbp, "heart_rate": 75, "bmi": bmi,
                "fasting_glucose": glucose, "hba1c": hba1c, "total_cholesterol": cholesterol,
                "ldl": cholesterol * 0.55, "hdl": cholesterol * 0.25, "serum_creatinine": creatinine,
                "hypertension": int(htn), "diabetes": int(diab), "chronic_kidney_disease": int(ckd)
            }
            all_predictions = {}
            for target in ["heart_disease", "diabetes", "chronic_kidney_disease"]:
                pred_model = get_trained_predictor(target=target)
                all_predictions[target] = pred_model.predict_patient(payload)
            risk_score = calculate_patient_risk(values)
            reg = get_trained_regressor()
            cost_payload = {
                "age": age, "systolic_bp": sbp, "diastolic_bp": dbp, "bmi": bmi,
                "fasting_glucose": glucose, "total_cholesterol": cholesterol,
                "serum_creatinine": creatinine, "length_of_stay": los, "risk_score": risk_score,
                "hypertension": int(htn), "diabetes": int(diab), "heart_disease": 0,
                "chronic_kidney_disease": int(ckd)
            }
            estimated_cost = max(0.0, reg.predict_cost(cost_payload))
            assessment_id = save_assessment(
                current_user["user_id"], current_user.get("patient_id"), values,
                all_predictions, risk_score, estimated_cost
            )
            st.session_state.latest_assessment_id = assessment_id
            st.session_state.latest_predictions = all_predictions
            st.session_state.latest_risk_score = risk_score
            st.session_state.latest_cost = estimated_cost
            st.success(f"Assessment #{assessment_id} saved to the HealthPulse database.")
            st.info("Switch to **Latest Results** to view the personalized analytical output.")

    with tab_results:
        st.subheader("Latest Personalized Results")
        if "latest_predictions" not in st.session_state:
            st.info("Submit a new assessment first. Your saved history will remain available below.")
        else:
            rc1, rc2, rc3 = st.columns(3)
            rc1.metric("Risk Score", f"{st.session_state.latest_risk_score:.1f}/100")
            rc2.metric("Estimated Treatment Cost", f"${st.session_state.latest_cost:,.2f}")
            rc3.metric("Assessment ID", str(st.session_state.latest_assessment_id))
            for target, label in [("heart_disease", "❤️ Heart Disease"), ("diabetes", "🩸 Diabetes"), ("chronic_kidney_disease", "🫘 Chronic Kidney Disease")]:
                st.markdown(f"### {label}")
                pred_df = pd.DataFrame.from_dict(st.session_state.latest_predictions[target], orient="index").reset_index().rename(columns={"index":"Model"})
                st.dataframe(pred_df, use_container_width=True, hide_index=True)

    with tab_history:
        st.subheader("My Saved Assessments")
        if user_assessments.empty:
            st.info("No saved assessments yet.")
        else:
            history_display = user_assessments.drop(columns=["prediction_json"], errors="ignore")
            st.dataframe(history_display, use_container_width=True, hide_index=True)
            user_profile_df = pd.DataFrame([{
                "User ID": current_user.get("user_id"),
                "Username": current_user.get("username"),
                "Full Name": current_user.get("full_name"),
                "Email": current_user.get("email"),
                "Patient ID": current_user.get("patient_id"),
                "Role": current_user.get("role"),
            }])
            excel_data = admin_data_to_excel(user_profile_df.rename(columns={"User ID": "user_id"}), user_assessments)
            st.download_button(
                "📥 Download My Profile + Saved Results as Excel",
                data=excel_data,
                file_name="HealthPulse_My_Profile_and_Assessment_Results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
            st.caption("Your Excel contains your profile plus saved clinical inputs, risk score, estimated cost, and model-wise predictions/probabilities.")

# -------------------------------------------------------------
# VIEW: PATIENT MANAGEMENT
# -------------------------------------------------------------
if menu == "👤 Patient Management":
    st.markdown('<div class="main-title">👤 Patient Management & Personalized Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Add patient records to the Star Schema and immediately generate analytical insights.</div>', unsafe_allow_html=True)

    tab_add, tab_records, tab_analyze = st.tabs(["➕ Add New Patient", "📋 Patient Records", "🔎 Analyze Patient"])

    with tab_add:
        st.info("Enter a new patient below. The record is persisted in the SQLite Star Schema warehouse.")
        with st.form("manual_patient_form"):
            c1, c2, c3 = st.columns(3)
            patient_name = c1.text_input("Patient Name *", placeholder="e.g. Rahul Sharma")
            patient_id_input = c1.text_input("Patient ID", placeholder="Leave blank to auto-generate")
            age_p = c1.number_input("Age", 18, 100, 45)
            gender_p = c1.selectbox("Gender", ["Female", "Male", "Other"])
            blood_group = c1.selectbox("Blood Group", ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"])

            smoking_p = c2.selectbox("Smoking Status", ["Never", "Former", "Current"])
            alcohol_p = c2.selectbox("Alcohol Use", ["Non-Drinker", "Occasional", "Regular"])
            exercise_p = c2.selectbox("Exercise Level", ["Sedentary", "Moderate", "Active"])
            department_p = c2.selectbox("Department", ["Cardiology", "Neurology", "General Medicine", "Orthopedics", "Oncology", "Emergency", "Other"])
            admission_type_p = c2.selectbox("Admission Type", ["Outpatient", "Inpatient", "Emergency"])

            admission_date_p = c3.date_input("Admission Date")
            sbp_p = c3.number_input("Systolic BP", 80.0, 220.0, 125.0)
            dbp_p = c3.number_input("Diastolic BP", 40.0, 140.0, 80.0)
            hr_p = c3.number_input("Heart Rate", 40.0, 180.0, 75.0)
            bmi_p = c3.number_input("BMI", 10.0, 60.0, 24.0)
            glucose_p = c3.number_input("Fasting Glucose", 50.0, 400.0, 100.0)

            c4, c5, c6 = st.columns(3)
            hba1c_p = c4.number_input("HbA1c", 3.0, 20.0, 5.6)
            chol_p = c4.number_input("Total Cholesterol", 80.0, 500.0, 180.0)
            ldl_p = c4.number_input("LDL", 20.0, 400.0, 100.0)
            hdl_p = c5.number_input("HDL", 10.0, 150.0, 50.0)
            creat_p = c5.number_input("Serum Creatinine", 0.2, 10.0, 1.0)
            los_p = c5.number_input("Length of Stay", 0, 60, 3)
            cost_p = c6.number_input("Treatment Cost", 0.0, 1000000.0, 2500.0, step=100.0)
            htn_p = c6.checkbox("Hypertension")
            diab_p = c6.checkbox("Diabetes")
            hd_p = c6.checkbox("Heart Disease")
            ckd_p = c6.checkbox("Chronic Kidney Disease")

            save_patient_btn = st.form_submit_button("💾 Save Patient & Generate Results", type="primary", use_container_width=True)

        if save_patient_btn:
            if not patient_name.strip():
                st.error("Patient name is required.")
            else:
                if not patient_id_input.strip():
                    patient_id_input = "HP-" + datetime.now().strftime("%Y%m%d%H%M%S%f")[:16]
                values = {
                    "age": age_p, "gender": gender_p, "smoking": smoking_p,
                    "exercise_level": exercise_p, "systolic_bp": sbp_p,
                    "diastolic_bp": dbp_p, "bmi": bmi_p, "fasting_glucose": glucose_p,
                    "hba1c": hba1c_p, "total_cholesterol": chol_p,
                    "serum_creatinine": creat_p, "length_of_stay": los_p,
                    "hypertension": int(htn_p), "diabetes": int(diab_p),
                    "heart_disease": int(hd_p), "chronic_kidney_disease": int(ckd_p)
                }
                risk = calculate_patient_risk(values)
                payload = {
                    "age": age_p, "gender": gender_p, "smoking": smoking_p,
                    "exercise_level": exercise_p, "systolic_bp": sbp_p,
                    "diastolic_bp": dbp_p, "heart_rate": hr_p, "bmi": bmi_p,
                    "fasting_glucose": glucose_p, "hba1c": hba1c_p,
                    "total_cholesterol": chol_p, "ldl": ldl_p, "hdl": hdl_p,
                    "serum_creatinine": creat_p, "hypertension": int(htn_p),
                    "diabetes": int(diab_p), "chronic_kidney_disease": int(ckd_p)
                }
                predictions = {}
                for target in ["heart_disease", "diabetes", "chronic_kidney_disease"]:
                    predictions[target] = get_trained_predictor(target=target).predict_patient(payload)
                reg = get_trained_regressor()
                cost_payload = {
                    "age": age_p, "systolic_bp": sbp_p, "diastolic_bp": dbp_p,
                    "bmi": bmi_p, "fasting_glucose": glucose_p,
                    "total_cholesterol": chol_p, "serum_creatinine": creat_p,
                    "length_of_stay": los_p, "risk_score": risk,
                    "hypertension": int(htn_p), "diabetes": int(diab_p),
                    "heart_disease": int(hd_p), "chronic_kidney_disease": int(ckd_p)
                }
                estimated_cost = max(0.0, reg.predict_cost(cost_payload))
                patient = {
                    "patient_id": patient_id_input, "patient_name": patient_name.strip(),
                    "age": age_p, "gender": gender_p, "blood_group": blood_group,
                    "smoking": smoking_p, "alcohol": alcohol_p, "exercise_level": exercise_p,
                    "department": department_p, "admission_type": admission_type_p,
                    "admission_date": admission_date_p, "systolic_bp": sbp_p,
                    "diastolic_bp": dbp_p, "heart_rate": hr_p, "bmi": bmi_p,
                    "fasting_glucose": glucose_p, "hba1c": hba1c_p,
                    "total_cholesterol": chol_p, "ldl": ldl_p, "hdl": hdl_p,
                    "serum_creatinine": creat_p, "length_of_stay": los_p,
                    "treatment_cost": cost_p, "hypertension": int(htn_p),
                    "diabetes": int(diab_p), "heart_disease": int(hd_p),
                    "chronic_kidney_disease": int(ckd_p), "risk_score": risk
                }
                try:
                    save_patient_to_warehouse(patient)
                    if current_user["role"] == "user":
                        save_assessment(current_user["user_id"], patient_id_input, values, predictions, risk, estimated_cost)
                    st.session_state.patient_result = {
                        "patient_id": patient_id_input, "patient_name": patient_name.strip(),
                        "predictions": predictions, "risk": risk, "estimated_cost": estimated_cost,
                        "actual_cost": cost_p
                    }
                    get_cached_stats.clear()
                    get_cached_df.clear()
                    get_trained_predictor.clear()
                    get_trained_regressor.clear()
                    st.success(f"Patient {patient_name.strip()} ({patient_id_input}) saved successfully.")
                    st.info("Open **Analyze Patient** to inspect the generated disease-risk and cost results.")
                except Exception as exc:
                    st.error(f"Could not save the patient: {exc}")

    with tab_records:
        records = get_manual_patient_records()
        st.subheader("Saved Patient Records")
        if records.empty:
            st.info("No patient records found.")
        else:
            st.dataframe(records, use_container_width=True, hide_index=True)
            output = BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                records.to_excel(writer, index=False, sheet_name="Patient Records")
            st.download_button(
                "📥 Export Patient Records to Excel",
                data=output.getvalue(),
                file_name="HealthPulse_Patient_Records.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

    with tab_analyze:
        st.subheader("Latest Patient Analysis")
        result = st.session_state.get("patient_result")
        if not result:
            st.info("Save a new patient first to see personalized results.")
        else:
            r1, r2, r3 = st.columns(3)
            r1.metric("Patient", result["patient_name"])
            r2.metric("Overall Risk Score", f"{result['risk']:.1f}/100")
            r3.metric("Estimated Treatment Cost", f"${result['estimated_cost']:,.2f}")
            for target, label in [
                ("heart_disease", "❤️ Heart Disease"),
                ("diabetes", "🩸 Diabetes"),
                ("chronic_kidney_disease", "🫘 Chronic Kidney Disease")
            ]:
                st.markdown(f"### {label}")
                pred_df = pd.DataFrame.from_dict(result["predictions"][target], orient="index").reset_index().rename(columns={"index":"Model"})
                st.dataframe(pred_df, use_container_width=True, hide_index=True)

# -------------------------------------------------------------
# VIEW 1: ARCHITECTURE & STAR SCHEMA
# -------------------------------------------------------------
if menu == "🏛️ Architecture & Star Schema":
    st.markdown('<div class="main-title">Healthcare Data Warehouse Architecture</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Dimensional Star Schema Model, Granularity Definition, and Storage Infrastructure</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    stats = get_cached_stats()
    col1.metric("Total Patients", f"{stats.get('Dim_Patient', 0):,}")
    col2.metric("Fact Observations", f"{stats.get('Fact_PatientAssessments', 0):,}")
    col3.metric("Hospital Units", f"{stats.get('Dim_Department', 0):,}")
    col4.metric("Temporal Span", f"{stats.get('Dim_Date', 0):,} days")
    
    st.markdown("---")
    st.subheader("📌 Dimensional Modeling: Star Schema Architecture")
    
    st.markdown("""
    The clinical data warehouse is structured using a **Star Schema** with one central Fact Table and four surrounding Dimension Tables.
    This design optimizes Online Analytical Processing (OLAP) queries by reducing join complexity and separating business metrics from descriptive attributes.
    """)
    
    # Visual Star Schema Layout
    col_dim1, col_fact, col_dim2 = st.columns([1, 1.4, 1])
    
    with col_dim1:
        st.info("""
        **Dim_Patient**
        - `patient_key` (PK, Integer)
        - `patient_id` (Natural Key)
        - `age`, `age_group`
        - `gender`, `blood_group`
        - `smoking`, `alcohol`
        - `exercise_level`
        """)
        st.info("""
        **Dim_Date**
        - `date_key` (PK, YYYYMMDD)
        - `full_date`
        - `year`, `quarter`, `month`
        - `month_name`, `day`
        - `day_name`, `is_weekend`
        """)
        
    with col_fact:
        st.success("""
        ⭐ **Fact_PatientAssessments**
        *(Grain: 1 record per clinical assessment)*
        - `assessment_key` (PK)
        - `patient_key` (FK)
        - `date_key` (FK)
        - `dept_key` (FK)
        - `diag_key` (FK)
        
        **Numerical Measures:**
        - `systolic_bp`, `diastolic_bp`
        - `heart_rate`, `bmi`
        - `fasting_glucose`, `hba1c`
        - `total_cholesterol`, `ldl`, `hdl`
        - `serum_creatinine`
        - `length_of_stay`
        - `treatment_cost`
        - `risk_score`
        """)
        
    with col_dim2:
        st.info("""
        **Dim_Department**
        - `dept_key` (PK, Integer)
        - `department_name`
        - `admission_type`
        """)
        st.info("""
        **Dim_DiagnosisCategory**
        - `diag_key` (PK, Integer)
        - `profile_name`
        - `has_hypertension`
        - `has_diabetes`
        - `has_heart_disease`
        - `has_ckd`
        - `comorbidity_count`
        """)

    st.markdown("---")
    st.subheader("🔍 Schema Explorer & Live Table Inspect")
    selected_tbl = st.selectbox(
        "Select table to preview data:",
        ["Fact_PatientAssessments", "Dim_Patient", "Dim_Date", "Dim_Department", "Dim_DiagnosisCategory"]
    )
    conn = get_connection()
    preview_df = pd.read_sql_query(f"SELECT * FROM {selected_tbl} LIMIT 10;", conn)
    conn.close()
    st.dataframe(preview_df, use_container_width=True)

# -------------------------------------------------------------
# VIEW 2: ETL DATA PIPELINE
# -------------------------------------------------------------
elif menu == "🔄 ETL Data Pipeline":
    st.markdown('<div class="main-title">ETL Pipeline & Data Hygiene</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Extract, Transform, and Load Workflow for Clinical Warehousing</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 1. Extract")
        st.markdown("""
        - Ingests raw clinical feeds & lab registers (`raw_healthcare_data.csv`).
        - Validates record counts and schema structure.
        - Audits missing fields and data types.
        """)
    with col2:
        st.markdown("### 2. Transform")
        st.markdown("""
        - Discretizes continuous metrics (Age groups, BMI classifications).
        - Computes surrogate calendar attributes (`Dim_Date`).
        - Normalizes comorbidity flags into diagnosis categories.
        - Resolves surrogate foreign keys.
        """)
    with col3:
        st.markdown("### 3. Load")
        st.markdown("""
        - Transactional population of SQLite star tables.
        - Enforces foreign key constraints (`PRAGMA foreign_keys = ON`).
        - Builds B-tree indexes for accelerated OLAP aggregation queries.
        """)
        
    st.markdown("---")
    st.subheader("⚡ Execute Live ETL Refresh")
    st.write("Clicking below will re-execute the full ETL pipeline from raw CSV data into SQLite.")
    
    if st.button("🚀 Re-Run ETL Pipeline", type="primary"):
        with st.spinner("Executing ETL Pipeline..."):
            res = run_etl_pipeline()
            st.success(f"ETL Execution Complete! Processed {res['records_processed']} clinical records.")
            st.cache_data.clear()
            st.cache_resource.clear()

# -------------------------------------------------------------
# VIEW 3: OLAP CUBE ANALYTICS
# -------------------------------------------------------------
elif menu == "🧊 OLAP Cube Analytics":
    st.markdown('<div class="main-title">OLAP Multidimensional Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Interactive Roll-Up, Drill-Down, Slice, Dice, and Pivot Operations</div>', unsafe_allow_html=True)
    
    olap_tab1, olap_tab2, olap_tab3, olap_tab4, olap_tab5, olap_tab6 = st.tabs([
        "⬆️ Roll-Up", "⬇️ Drill-Down", "🔪 Slice", "🎲 Dice", "🔄 Pivot Table", "💻 SQL Sandbox"
    ])
    
    # 1. ROLL-UP
    with olap_tab1:
        st.markdown("#### OLAP Roll-Up Operation")
        st.caption("Aggregating temporal granularity from Month → Quarter → Year")
        
        grain = st.radio("Temporal Aggregation Grain:", ["Year", "Quarter", "Month"], horizontal=True)
        df_rollup = olap_roll_up(group_level=grain.lower())
        
        fig = px.bar(
            df_rollup, x="TimeDimension", y=["TotalHeartDisease", "TotalDiabetes", "TotalHypertension"],
            title=f"Total Disease Cases Rolled Up by {grain}",
            barmode="group",
            labels={"value": "Total Cases", "TimeDimension": "Time Period", "variable": "Condition"},
            color_discrete_sequence=["#EF4444", "#3B82F6", "#10B981"]
        )
        st.plotly_chart(fig, use_container_width=True)
        
        col_m1, col_m2 = st.columns(2)
        fig_cost = px.line(
            df_rollup, x="TimeDimension", y="AvgCost", markers=True,
            title=f"Average Treatment Cost Rolled Up by {grain} ($)",
            color_discrete_sequence=["#6366F1"]
        )
        col_m1.plotly_chart(fig_cost, use_container_width=True)
        
        fig_los = px.line(
            df_rollup, x="TimeDimension", y="AvgLengthOfStay", markers=True,
            title=f"Average Length of Stay (Days) by {grain}",
            color_discrete_sequence=["#F59E0B"]
        )
        col_m2.plotly_chart(fig_los, use_container_width=True)
        
        with st.expander("View Roll-Up Data Table"):
            st.dataframe(df_rollup, use_container_width=True)

    # 2. DRILL-DOWN
    with olap_tab2:
        st.markdown("#### OLAP Drill-Down Operation")
        st.caption("Deconstructing aggregated metrics into granular categorical hierarchies")
        
        drill_mode = st.radio("Drill-Down Hierarchy:", ["Department ➡️ Admission Type", "Age Group ➡️ Gender"], horizontal=True)
        h_param = "department_type" if "Department" in drill_mode else "age_group_gender"
        df_drill = olap_drill_down(hierarchy=h_param)
        
        if h_param == "department_type":
            fig_drill = px.bar(
                df_drill, x="Department", y="AvgCost", color="AdmissionType",
                barmode="group",
                title="Drill-Down: Average Treatment Cost by Department & Admission Type",
                text_auto=".1f"
            )
            st.plotly_chart(fig_drill, use_container_width=True)
        else:
            fig_drill = px.bar(
                df_drill, x="AgeGroup", y=["HeartDiseasePct", "DiabetesPct"], color="Gender",
                barmode="group",
                title="Drill-Down: Disease Prevalence (%) by Age Group & Gender"
            )
            st.plotly_chart(fig_drill, use_container_width=True)
            
        with st.expander("View Drill-Down Data Table"):
            st.dataframe(df_drill, use_container_width=True)

    # 3. SLICE
    with olap_tab3:
        st.markdown("#### OLAP Slice Operation")
        st.caption("Fixing one dimension to a single value to inspect a 2D hyperplane of the data cube")
        
        conn = get_connection()
        depts = [row[0] for row in conn.execute("SELECT DISTINCT department_name FROM Dim_Department;").fetchall()]
        conn.close()
        
        selected_slice = st.selectbox("Select Department to Slice:", depts)
        df_slice = olap_slice(dimension="department", slice_value=selected_slice)
        
        st.write(f"**Sliced Cube:** `Department = '{selected_slice}'`")
        col_s1, col_s2 = st.columns(2)
        
        fig_s1 = px.bar(
            df_slice, x="AgeGroup", y="AvgTreatmentCost", color="Gender",
            barmode="group",
            title=f"Average Treatment Cost in {selected_slice} ($)"
        )
        col_s1.plotly_chart(fig_s1, use_container_width=True)
        
        fig_s2 = px.bar(
            df_slice, x="AgeGroup", y="HeartDiseaseCases", color="Gender",
            barmode="group",
            title=f"Heart Disease Cases in {selected_slice}"
        )
        col_s2.plotly_chart(fig_s2, use_container_width=True)
        st.dataframe(df_slice, use_container_width=True)

    # 4. DICE
    with olap_tab4:
        st.markdown("#### OLAP Dice Operation")
        st.caption("Extracting a sub-cube by specifying multi-attribute conditions along multiple dimensions")
        
        col_d1, col_d2 = st.columns(2)
        conn = get_connection()
        all_depts = [r[0] for r in conn.execute("SELECT DISTINCT department_name FROM Dim_Department;").fetchall()]
        all_ages = [r[0] for r in conn.execute("SELECT DISTINCT age_group FROM Dim_Patient;").fetchall()]
        conn.close()
        
        dice_depts = col_d1.multiselect("Select Departments (Dimension 1):", all_depts, default=all_depts[:2])
        dice_ages = col_d2.multiselect("Select Age Groups (Dimension 2):", all_ages, default=all_ages[2:])
        
        if dice_depts and dice_ages:
            df_dice = olap_dice(departments=dice_depts, age_groups=dice_ages)
            fig_dice = px.scatter(
                df_dice, x="AvgRiskScore", y="AvgCost", size="SubCubeCount",
                color="Department", hover_data=["AgeGroup", "SmokingStatus"],
                title="Diced Sub-Cube: Average Risk Score vs Treatment Cost",
                size_max=30
            )
            st.plotly_chart(fig_dice, use_container_width=True)
            st.dataframe(df_dice, use_container_width=True)
        else:
            st.warning("Please select at least one department and one age group to dice.")

    # 5. PIVOT
    with olap_tab5:
        st.markdown("#### OLAP Pivot Operation")
        st.caption("Rotating dimensional axes to compute multi-variable cross-tabulations")
        
        metric_choice = st.selectbox(
            "Select Metric to Pivot:",
            ["AvgCost", "AvgRisk", "AvgLOS", "PatientCount"]
        )
        pivot_data = olap_pivot(row_dim="AgeGroup", col_dim="Department", metric=metric_choice)
        
        fig_heat = px.imshow(
            pivot_data, text_auto=True, aspect="auto",
            title=f"Pivot Heatmap: {metric_choice} by Age Group and Department",
            color_continuous_scale="Blues"
        )
        st.plotly_chart(fig_heat, use_container_width=True)
        st.dataframe(pivot_data, use_container_width=True)

    # 6. SQL SANDBOX
    with olap_tab6:
        st.markdown("#### 💻 Interactive SQL Query Sandbox")
        st.caption("Directly query the Star Schema warehouse using ANSI SQL")
        
        default_query = """
SELECT 
    d.department_name,
    COUNT(f.assessment_key) AS TotalAdmissions,
    ROUND(AVG(f.treatment_cost), 2) AS MeanCost,
    ROUND(AVG(f.risk_score), 1) AS MeanRiskScore,
    SUM(f.high_risk_patient) AS HighRiskCount
FROM Fact_PatientAssessments f
JOIN Dim_Department d ON f.dept_key = d.dept_key
GROUP BY d.department_name
ORDER BY MeanCost DESC;
        """.strip()
        
        user_query = st.text_area("SQL Query:", value=default_query, height=150)
        if st.button("Run SQL Query", type="primary"):
            try:
                conn = get_connection()
                query_res = pd.read_sql_query(user_query, conn)
                conn.close()
                st.dataframe(query_res, use_container_width=True)
                st.success(f"Returned {len(query_res)} rows.")
            except Exception as e:
                st.error(f"SQL Execution Error: {e}")

# -------------------------------------------------------------
# VIEW 4: PREDICTIVE DISEASE MINING
# -------------------------------------------------------------
elif menu == "🩺 Predictive Disease Mining":
    st.markdown('<div class="main-title">Clinical Diagnostic Mining & Risk Classifier</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Multi-Model Ensemble Predictions (CART Decision Tree, Random Forest, Naive Bayes, Logistic Regression)</div>', unsafe_allow_html=True)
    
    target_disease = st.selectbox(
        "Select Target Disease for Predictive Mining:",
        ["heart_disease", "diabetes", "chronic_kidney_disease"],
        format_func=lambda x: "❤️ Cardiovascular Disease (Heart Disease)" if x == "heart_disease" else ("🩸 Type-2 Diabetes" if x == "diabetes" else "🩺 Chronic Kidney Disease (CKD)")
    )
    
    predictor = get_trained_predictor(target=target_disease)
    
    st.markdown("---")
    st.subheader("Patient Clinical Input Parameters")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.slider("Age", 18, 90, 58)
        gender = st.selectbox("Gender", ["Male", "Female"])
        smoking = st.selectbox("Smoking Status", ["Never", "Former", "Current"], index=2)
        exercise = st.selectbox("Exercise Activity", ["Sedentary", "Moderate", "Active"], index=0)
        
    with c2:
        sbp = st.slider("Systolic BP (mmHg)", 90, 200, 148)
        dbp = st.slider("Diastolic BP (mmHg)", 55, 120, 92)
        heart_rate = st.slider("Resting Heart Rate (bpm)", 50, 120, 82)
        bmi = st.slider("Body Mass Index (BMI)", 16.0, 48.0, 31.4, step=0.1)
        
    with c3:
        glucose = st.slider("Fasting Blood Glucose (mg/dL)", 70, 300, 142)
        hba1c = st.slider("HbA1c Level (%)", 4.5, 14.0, 7.4, step=0.1)
        cholesterol = st.slider("Total Cholesterol (mg/dL)", 120, 360, 235)
        creatinine = st.slider("Serum Creatinine (mg/dL)", 0.5, 4.5, 1.4, step=0.1)
        
    # Pre-existing conditions
    st.markdown("##### Pre-existing Diagnoses:")
    col_cb1, col_cb2, col_cb3 = st.columns(3)
    has_htn = col_cb1.checkbox("Diagnosed Hypertension", value=True)
    has_dm = col_cb2.checkbox("Diagnosed Diabetes", value=True if target_disease != "diabetes" else False)
    has_ckd = col_cb3.checkbox("Diagnosed Kidney Disease", value=False)
    
    patient_payload = {
        "age": age,
        "is_male": 1 if gender == "Male" else 0,
        "is_smoker": 1 if smoking == "Current" else 0,
        "is_former_smoker": 1 if smoking == "Former" else 0,
        "is_sedentary": 1 if exercise == "Sedentary" else 0,
        "systolic_bp": sbp,
        "diastolic_bp": dbp,
        "heart_rate": heart_rate,
        "bmi": bmi,
        "fasting_glucose": glucose,
        "hba1c": hba1c,
        "total_cholesterol": cholesterol,
        "ldl": cholesterol - 50,
        "hdl": 42.0,
        "serum_creatinine": creatinine,
        "hypertension": 1 if has_htn else 0,
        "diabetes": 1 if has_dm else 0,
        "chronic_kidney_disease": 1 if has_ckd else 0
    }
    
    st.markdown("---")
    st.subheader("🎯 Real-Time Multi-Model Diagnostic Inference")
    
    preds = predictor.predict_patient(patient_payload)
    
    pred_cols = st.columns(4)
    model_keys = list(preds.keys())
    
    for i, model_name in enumerate(model_keys):
        res = preds[model_name]
        with pred_cols[i]:
            card_color = "#DC2626" if res["Prediction"] == 1 else "#16A34A"
            badge = "🚨 POSITIVE RISK" if res["Prediction"] == 1 else "✅ LOW RISK"
            st.markdown(f"""
            <div style="border: 2px solid {card_color}; border-radius: 8px; padding: 15px; text-align: center; background-color: #F8FAFC;">
                <h4 style="margin: 0; color: #1E293B;">{model_name}</h4>
                <h2 style="color: {card_color}; margin: 8px 0;">{badge}</h2>
                <p style="font-size: 1.1rem; margin: 0; color: #475569;">Confidence: <b>{res['Probability']}%</b></p>
            </div>
            """, unsafe_allow_html=True)
            
    st.markdown("---")
    col_tree, col_feat = st.columns(2)
    
    with col_tree:
        st.markdown("#### 🌳 Decision Tree Rule Logic Explainer (CART)")
        st.caption("Transparent classification rules derived from the trained Decision Tree")
        st.code(predictor.dt_rules_text, language="text")
        
    with col_feat:
        st.markdown("#### 📊 Global Feature Importance (Random Forest)")
        st.caption("Top clinical predictors contributing to risk classification")
        top_feat = predictor.feature_importances.head(8)
        fig_feat = px.bar(
            top_feat, x="Importance", y="Feature", orientation="h",
            color="Importance", color_continuous_scale="Viridis"
        )
        fig_feat.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_feat, use_container_width=True)

# -------------------------------------------------------------
# VIEW: COST PREDICTION REGRESSION
# -------------------------------------------------------------
elif menu == "💰 Cost Prediction Regression":
    st.markdown('<div class="main-title">Healthcare Expenditure Regression</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Continuous Financial Forecasting using Linear, Ridge, and Ensemble Regressors</div>', unsafe_allow_html=True)
    
    reg = get_trained_regressor()
    
    # Summary Metrics Row
    m_col1, m_col2, m_col3 = st.columns(3)
    m_col1.metric("Linear Regression R²", f"{reg.metrics.loc[0, 'R2 Score']:.4f}", f"MAE: ${reg.metrics.loc[0, 'MAE ($)']}")
    m_col2.metric("Ridge Regression R²", f"{reg.metrics.loc[1, 'R2 Score']:.4f}", f"MAE: ${reg.metrics.loc[1, 'MAE ($)']}")
    m_col3.metric("Random Forest R²", f"{reg.metrics.loc[2, 'R2 Score']:.4f}", f"MAE: ${reg.metrics.loc[2, 'MAE ($)']}", delta_color="normal")
    
    st.markdown("---")
    st.subheader("📊 Regression Algorithm Performance Table")
    st.dataframe(reg.metrics, use_container_width=True)
    
    st.markdown("---")
    st.subheader("📈 Actual vs. Predicted Cost Scatter Visualization")
    
    selected_reg_model = st.selectbox(
        "Select Regression Model to Visualize:",
        list(reg.actual_vs_predicted.keys())
    )
    
    scatter_data = reg.actual_vs_predicted[selected_reg_model]
    df_scatter = pd.DataFrame({
        "Actual Cost ($)": scatter_data["y_true"][:300],
        "Predicted Cost ($)": scatter_data["y_pred"][:300]
    })
    
    fig_reg = px.scatter(
        df_scatter, x="Actual Cost ($)", y="Predicted Cost ($)",
        title=f"Actual vs. Predicted Treatment Expenditure ({selected_reg_model})",
        opacity=0.65,
        trendline="ols"
    )
    fig_reg.add_shape(
        type="line", line=dict(dash="dash", color="red"),
        x0=df_scatter["Actual Cost ($)"].min(), y0=df_scatter["Actual Cost ($)"].min(),
        x1=df_scatter["Actual Cost ($)"].max(), y1=df_scatter["Actual Cost ($)"].max()
    )
    st.plotly_chart(fig_reg, use_container_width=True)
    
    st.markdown("---")
    st.subheader("🧮 Interactive Treatment Cost Estimator")
    st.caption("Estimate anticipated financial expenditure based on patient clinical parameters")
    
    rc_col1, rc_col2, rc_col3 = st.columns(3)
    with rc_col1:
        calc_los = st.slider("Anticipated Length of Stay (Days):", 0, 30, 4)
        calc_risk = st.slider("Patient Health Risk Score (0-100):", 0, 100, 45)
    with rc_col2:
        calc_age = st.slider("Patient Age:", 18, 90, 55)
        calc_sbp = st.slider("Systolic BP:", 90, 200, 135)
    with rc_col3:
        calc_glucose = st.slider("Fasting Blood Sugar:", 70, 300, 115)
        calc_comorb = st.selectbox("Comorbidity Count:", [0, 1, 2, 3, 4], index=1)
        
    # Approximate model estimation
    est_cost = 800 + (calc_los * 650) + (calc_risk * 35) + (calc_comorb * 250) + (calc_age * 5)
    st.success(f"### 💳 Estimated Inpatient Treatment Cost: **${est_cost:,.2f}**")

# -------------------------------------------------------------
# VIEW 5: PATIENT COHORT CLUSTERING
# -------------------------------------------------------------
elif menu == "👥 Patient Cohort Clustering":
    st.markdown('<div class="main-title">Patient Risk Cohort Clustering (K-Means)</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Unsupervised Patient Stratification, Elbow Curve Analysis, and PCA Visualizations</div>', unsafe_allow_html=True)
    
    col_k, col_sil = st.columns([1, 2])
    k_val = col_k.slider("Select Number of Risk Cohorts (K):", 2, 6, 4)
    
    model, df_clustered, summary, sil_score = get_clustering_results(n_clusters=k_val)
    col_sil.metric("Current Silhouette Coefficient", f"{sil_score:.3f}", help="Higher value indicates better defined cluster boundaries")
    
    st.markdown("---")
    st.subheader("📌 2D and 3D Principal Component (PCA) Projections")
    
    tab_pca2d, tab_pca3d, tab_elbow = st.tabs(["2D PCA Projection", "3D PCA Projection", "Elbow Curve & Inertia"])
    
    with tab_pca2d:
        fig_pca2 = px.scatter(
            df_clustered, x="PCA1", y="PCA2", color="CohortName",
            hover_data=["age", "systolic_bp", "bmi", "risk_score", "heart_disease"],
            title=f"2D PCA Projection of Patients Clustered into {k_val} Risk Cohorts",
            opacity=0.75
        )
        st.plotly_chart(fig_pca2, use_container_width=True)
        
    with tab_pca3d:
        fig_pca3 = px.scatter_3d(
            df_clustered, x="PCA1", y="PCA2", z="PCA3", color="CohortName",
            hover_data=["age", "systolic_bp", "risk_score"],
            title=f"3D PCA Cluster Space (K = {k_val})",
            opacity=0.7
        )
        st.plotly_chart(fig_pca3, use_container_width=True)
        
    with tab_elbow:
        k_range, inertias, sils = model.compute_elbow_curve(get_cached_df())
        col_e1, col_e2 = st.columns(2)
        fig_elbow = px.line(x=k_range, y=inertias, markers=True, title="Elbow Method (Inertia vs K)", labels={"x": "K (Clusters)", "y": "Within-Cluster Inertia"})
        col_e1.plotly_chart(fig_elbow, use_container_width=True)
        fig_sil = px.line(x=k_range, y=sils, markers=True, title="Silhouette Score vs K", labels={"x": "K (Clusters)", "y": "Silhouette Score"}, color_discrete_sequence=["#10B981"])
        col_e2.plotly_chart(fig_sil, use_container_width=True)
        
    st.markdown("---")
    st.subheader("📋 Clinical Cohort Profiling")
    st.dataframe(summary, use_container_width=True)

# -------------------------------------------------------------
# VIEW 6: COMORBIDITY ASSOCIATION RULES
# -------------------------------------------------------------
elif menu == "🔗 Comorbidity Association Rules":
    st.markdown('<div class="main-title">Comorbidity Association Rule Mining</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Discovering Cross-Disease Correlations and Symptom Itemsets using the Apriori Algorithm</div>', unsafe_allow_html=True)
    
    c_s1, c_s2 = st.columns(2)
    min_sup = c_s1.slider("Minimum Support Threshold:", 0.05, 0.30, 0.08, step=0.01)
    min_conf = c_s2.slider("Minimum Confidence Threshold:", 0.30, 0.85, 0.50, step=0.05)
    
    miner, itemsets, rules = get_mined_rules(min_support=min_sup, min_confidence=min_conf)
    
    st.write(f"**Discovered {len(rules)} Association Rules** across {len(itemsets)} frequent clinical itemsets.")
    
    if not rules.empty:
        # Scatter plot of Support vs Confidence
        fig_rules = px.scatter(
            rules, x="support", y="confidence", color="lift", size="lift",
            hover_data=["Rule"],
            title="Association Rules: Support vs. Confidence (Color = Lift)",
            color_continuous_scale="Reds"
        )
        st.plotly_chart(fig_rules, use_container_width=True)
        
        st.markdown("---")
        st.subheader("Top Discovered Medical Association Rules (Ranked by Lift)")
        
        # Filter rules by consequent
        filter_opt = st.selectbox(
            "Filter Rules by Outcome Consequent:",
            ["All Rules", "Diagnosis_HeartDisease", "Diagnosis_Diabetes", "Diagnosis_CKD"]
        )
        
        display_rules = rules if filter_opt == "All Rules" else rules[rules["consequents_str"].str.contains(filter_opt)]
        
        st.dataframe(
            display_rules[["Rule", "support", "confidence", "lift"]].head(25),
            use_container_width=True
        )
        
        st.markdown("""
        **Data Mining Interpretation:**
        - **Support**: Proportion of total patients who possess both the antecedent conditions and the outcome.
        - **Confidence**: Probability that a patient who exhibits the antecedents will also develop the consequent condition.
        - **Lift > 1.0**: Indicates a strong positive association beyond random co-occurrence. A Lift of `3.5` means the outcome is 3.5 times more likely to occur when the antecedent conditions are present!
        """)
    else:
        st.warning("No association rules found with the specified Support and Confidence thresholds. Try lowering them.")

# -------------------------------------------------------------
# VIEW 7: MODEL BENCHMARK & EVALUATION
# -------------------------------------------------------------
elif menu == "📈 Model Benchmark & Evaluation":
    st.markdown('<div class="main-title">Classifier Performance & Evaluation Benchmark</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Comparative Evaluation of Supervised Mining Algorithms</div>', unsafe_allow_html=True)
    
    target_eval = st.selectbox(
        "Evaluate Models for Predicting:",
        ["heart_disease", "diabetes", "chronic_kidney_disease"],
        format_func=lambda x: "Heart Disease" if x == "heart_disease" else ("Type-2 Diabetes" if x == "diabetes" else "Chronic Kidney Disease")
    )
    
    predictor = get_trained_predictor(target=target_eval)
    
    st.subheader("📊 Comparative Evaluation Metrics Table")
    st.dataframe(predictor.metrics, use_container_width=True)
    
    # ---------------------------------------------------------
    # COMPARATIVE BAR-GRAPH BENCHMARK
    # Show each evaluation metric as its own bar graph so the
    # four classifiers can be compared independently.
    # ---------------------------------------------------------
    st.markdown("---")
    st.subheader("📊 Comparative Model Performance Bar Graphs")
    st.caption("Separate bar graphs for Accuracy, Precision, Recall, F1-Score and ROC-AUC.")

    metrics_df = predictor.metrics.copy().reset_index(drop=True)
    model_col = "Algorithm" if "Algorithm" in metrics_df.columns else metrics_df.columns[0]
    # Normalize common column names while preserving the original model results.
    metric_columns = [
        ("Accuracy (%)", "Accuracy (%)", "Accuracy (%) Comparison"),
        ("Precision (%)", "Precision (%)", "Precision (%) Comparison"),
        ("Recall (%)", "Recall (%)", "Recall (%) Comparison"),
        ("F1-Score (%)", "F1-Score (%)", "F1-Score (%) Comparison"),
    ]

    # The predictor metrics already stores Accuracy/Precision/Recall/F1 as
    # percentages; ROC-AUC remains on its standard 0-1 scale.
    graph_cols = st.columns(2)
    for idx, (column, y_label, title) in enumerate(metric_columns):
        with graph_cols[idx % 2]:
            graph_df = metrics_df[[model_col, column]].copy()
            graph_df.columns = ["Algorithm", "Score"]
            fig_metric = px.bar(
                graph_df, x="Algorithm", y="Score", text="Score",
                title=f"{title} — {target_eval.replace('_', ' ').title()}"
            )
            fig_metric.update_traces(texttemplate="%{text:.2f}", textposition="outside")
            fig_metric.update_layout(
                yaxis_title=y_label, xaxis_title="Algorithm",
                yaxis_range=[0, 100], margin=dict(t=70, b=40)
            )
            st.plotly_chart(fig_metric, use_container_width=True)

    roc_col = "ROC-AUC" if "ROC-AUC" in metrics_df.columns else "ROC AUC"
    if roc_col in metrics_df.columns:
        graph_df = metrics_df[[model_col, roc_col]].copy()
        graph_df.columns = ["Algorithm", "ROC-AUC"]
        graph_df["ROC-AUC"] = pd.to_numeric(graph_df["ROC-AUC"], errors="coerce")
        fig_auc = px.bar(
            graph_df, x="Algorithm", y="ROC-AUC", text="ROC-AUC",
            title=f"ROC-AUC Comparison — {target_eval.replace('_', ' ').title()}"
        )
        fig_auc.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig_auc.update_layout(
            yaxis_title="ROC-AUC", xaxis_title="Algorithm",
            yaxis_range=[0, 1], margin=dict(t=70, b=40)
        )
        st.plotly_chart(fig_auc, use_container_width=True)

    st.markdown("---")
    col_roc, col_cm = st.columns(2)
    
    with col_roc:
        st.markdown("#### ROC Curves & AUC Benchmark")
        fig_roc = go.Figure()
        colors = ["#EF4444", "#3B82F6", "#10B981", "#8B5CF6"]
        
        for idx, (name, (fpr, tpr, auc_val)) in enumerate(predictor.roc_curves.items()):
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr, mode='lines',
                name=f"{name} (AUC = {auc_val})",
                line=dict(color=colors[idx % len(colors)], width=2)
            ))
            
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode='lines',
            name='Random Chance', line=dict(dash='dash', color='gray')
        ))
        
        fig_roc.update_layout(
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            title=f"Receiver Operating Characteristic (ROC) - {target_eval.title()}"
        )
        st.plotly_chart(fig_roc, use_container_width=True)
        
    with col_cm:
        st.markdown("#### Confusion Matrix Inspector")
        sel_clf = st.selectbox("Select Classifier for Confusion Matrix:", list(predictor.confusion_matrices.keys()))
        cm = predictor.confusion_matrices[sel_clf]
        
        fig_cm = px.imshow(
            cm, text_auto=True,
            labels=dict(x="Predicted Class", y="Actual Class"),
            x=["Negative (0)", "Positive (1)"],
            y=["Negative (0)", "Positive (1)"],
            color_continuous_scale="Blues",
            title=f"Confusion Matrix: {sel_clf}"
        )
        st.plotly_chart(fig_cm, use_container_width=True)

# -------------------------------------------------------------
# VIEW 8: PATIENT AUDIT REPORT
# -------------------------------------------------------------
elif menu == "📑 Patient Audit Report":
    st.markdown('<div class="main-title">Clinical Patient Diagnostic Audit Report</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Extract Individual Patient Summary and Generate Health Risk Brief</div>', unsafe_allow_html=True)
    
    conn = get_connection()
    patients_sample = pd.read_sql_query("""
    SELECT 
        p.patient_id, p.age, p.gender, p.smoking,
        f.systolic_bp, f.diastolic_bp, f.fasting_glucose,
        f.treatment_cost, f.risk_score, f.heart_disease, f.diabetes
    FROM Dim_Patient p
    JOIN Fact_PatientAssessments f ON p.patient_key = f.patient_key
    LIMIT 100;
    """, conn)
    conn.close()
    
    selected_pid = st.selectbox("Select Patient Record:", patients_sample["patient_id"].tolist())
    p_data = patients_sample[patients_sample["patient_id"] == selected_pid].iloc[0]
    
    st.markdown("---")
    st.markdown(f"### 📋 Health Summary Card: `{selected_pid}`")
    
    rc1, rc2, rc3, rc4 = st.columns(4)
    rc1.metric("Patient Age", f"{p_data['age']} yrs ({p_data['gender']})")
    rc2.metric("Blood Pressure", f"{p_data['systolic_bp']}/{p_data['diastolic_bp']} mmHg")
    rc3.metric("Fasting Sugar", f"{p_data['fasting_glucose']} mg/dL")
    rc4.metric("Risk Score", f"{p_data['risk_score']}/100", delta="High Risk" if p_data['risk_score'] >= 50 else "Normal")
    
    report_text = f"""
=====================================================
HEALTHPULSE CLINICAL DATA WAREHOUSE - PATIENT AUDIT
=====================================================
Patient ID        : {p_data['patient_id']}
Age / Gender      : {p_data['age']} years / {p_data['gender']}
Smoking History   : {p_data['smoking']}
Blood Pressure    : {p_data['systolic_bp']}/{p_data['diastolic_bp']} mmHg
Fasting Glucose   : {p_data['fasting_glucose']} mg/dL
Diagnosis History : Heart Disease: {'Yes' if p_data['heart_disease'] == 1 else 'No'} | Diabetes: {'Yes' if p_data['diabetes'] == 1 else 'No'}
Calculated Risk   : {p_data['risk_score']} / 100
Cumulative Cost   : ${p_data['treatment_cost']:,.2f}
Audit Status      : {'URGENT INTERVENTION REQUIRED' if p_data['risk_score'] >= 50 else 'ROUTINE MONITORING RECOMMENDED'}
=====================================================
Generated by HealthPulse Clinical Decision Support System
    """.strip()
    
    st.code(report_text, language="text")
    
    st.download_button(
        label="📥 Download Clinical Audit Report (.txt)",
        data=report_text,
        file_name=f"{selected_pid}_health_audit.txt",
        mime="text/plain"
    )
