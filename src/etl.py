import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys

# Ensure local imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from warehouse import get_connection, init_star_schema

RAW_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw", "raw_healthcare_data.csv")

def categorize_age(age):
    if age < 30:
        return "Young Adult (18-29)"
    elif age < 45:
        return "Adult (30-44)"
    elif age < 60:
        return "Middle-Aged (45-59)"
    elif age < 75:
        return "Senior (60-74)"
    else:
        return "Elderly (75+)"

def categorize_bmi(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25.0:
        return "Normal"
    elif bmi < 30.0:
        return "Overweight"
    else:
        return "Obese"

def run_etl_pipeline(csv_path=None, db_path=None):
    """
    Executes the End-to-End ETL (Extract, Transform, Load) Pipeline:
    1. EXTRACT: Reads raw clinical transactional data.
    2. TRANSFORM: Data hygiene, categorical discretization, dimension lookups, surrogate key generation.
    3. LOAD: Populates Star Schema tables (Dimensions first, then Fact table).
    """
    csv_file = csv_path or RAW_DATA_PATH
    print(f"--- [ETL Step 1: EXTRACT] Reading from {csv_file} ---")
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"Raw data file not found at: {csv_file}")
    
    df_raw = pd.read_csv(csv_file)
    initial_rows = len(df_raw)
    print(f"Extracted {initial_rows} records.")
    
    print("--- [ETL Step 2: TRANSFORM] Cleaning and Enriching Data ---")
    # Data Cleaning
    df = df_raw.dropna().copy()
    
    # Feature Engineering / Discretization
    df["AgeGroup"] = df["Age"].apply(categorize_age)
    df["BMICategory"] = df["BMI"].apply(categorize_bmi)
    
    # Parse dates
    df["ParsedDate"] = pd.to_datetime(df["AdmissionDate"])
    df["DateKey"] = df["ParsedDate"].dt.strftime("%Y%m%d").astype(int)
    
    # Diagnosis Profile string for categorization
    def get_diag_profile(row):
        conds = []
        if row["Hypertension"] == 1: conds.append("HTN")
        if row["Diabetes"] == 1: conds.append("DM")
        if row["HeartDisease"] == 1: conds.append("CVD")
        if row["ChronicKidneyDisease"] == 1: conds.append("CKD")
        return "+".join(conds) if conds else "Healthy/Normal"
    
    df["DiagnosisProfile"] = df.apply(get_diag_profile, axis=1)
    df["ComorbidityCount"] = df[["Hypertension", "Diabetes", "HeartDisease", "ChronicKidneyDisease"]].sum(axis=1)

    print("--- [ETL Step 3: LOAD] Initializing Database & Loading Star Schema ---")
    init_star_schema(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()
    
    # Clear existing tables for idempotent reload
    cursor.execute("DELETE FROM Fact_PatientAssessments;")
    cursor.execute("DELETE FROM Dim_Patient;")
    cursor.execute("DELETE FROM Dim_Date;")
    cursor.execute("DELETE FROM Dim_Department;")
    cursor.execute("DELETE FROM Dim_DiagnosisCategory;")
    conn.commit()
    
    # 3a. Load Dim_Patient
    dim_patient_df = df[[
        "PatientID", "Age", "AgeGroup", "Gender", "BloodGroup", "Smoking", "Alcohol", "ExerciseLevel"
    ]].drop_duplicates(subset=["PatientID"]).rename(columns={
        "PatientID": "patient_id",
        "Age": "age",
        "AgeGroup": "age_group",
        "Gender": "gender",
        "BloodGroup": "blood_group",
        "Smoking": "smoking",
        "Alcohol": "alcohol",
        "ExerciseLevel": "exercise_level"
    })
    
    dim_patient_df.to_sql("Dim_Patient", conn, if_exists="append", index=False)
    print(f"Loaded {len(dim_patient_df)} rows into Dim_Patient.")
    
    # 3b. Load Dim_Date
    dates_unique = df["ParsedDate"].drop_duplicates()
    dim_date_df = pd.DataFrame({
        "date_key": dates_unique.dt.strftime("%Y%m%d").astype(int),
        "full_date": dates_unique.dt.strftime("%Y-%m-%d"),
        "year": dates_unique.dt.year,
        "quarter": dates_unique.dt.quarter,
        "month": dates_unique.dt.month,
        "month_name": dates_unique.dt.strftime("%B"),
        "day": dates_unique.dt.day,
        "day_name": dates_unique.dt.strftime("%A"),
        "is_weekend": dates_unique.dt.dayofweek.apply(lambda x: 1 if x >= 5 else 0)
    }).sort_values("date_key")
    
    dim_date_df.to_sql("Dim_Date", conn, if_exists="append", index=False)
    print(f"Loaded {len(dim_date_df)} rows into Dim_Date.")
    
    # 3c. Load Dim_Department
    dim_dept_df = df[["Department", "AdmissionType"]].drop_duplicates().rename(
        columns={"Department": "department_name", "AdmissionType": "admission_type"}
    )
    dim_dept_df.to_sql("Dim_Department", conn, if_exists="append", index=False)
    print(f"Loaded {len(dim_dept_df)} rows into Dim_Department.")
    
    # 3d. Load Dim_DiagnosisCategory
    dim_diag_df = df[["DiagnosisProfile", "Hypertension", "Diabetes", "HeartDisease", "ChronicKidneyDisease", "ComorbidityCount"]].drop_duplicates(
        subset=["DiagnosisProfile"]
    ).rename(columns={
        "DiagnosisProfile": "profile_name",
        "Hypertension": "has_hypertension",
        "Diabetes": "has_diabetes",
        "HeartDisease": "has_heart_disease",
        "ChronicKidneyDisease": "has_ckd",
        "ComorbidityCount": "comorbidity_count"
    })
    dim_diag_df.to_sql("Dim_DiagnosisCategory", conn, if_exists="append", index=False)
    print(f"Loaded {len(dim_diag_df)} rows into Dim_DiagnosisCategory.")
    
    # 3e. Load Fact_PatientAssessments with Key Resolution
    # Read generated keys from SQLite
    patient_keys = pd.read_sql("SELECT patient_key, patient_id FROM Dim_Patient;", conn)
    dept_keys = pd.read_sql("SELECT dept_key, department_name, admission_type FROM Dim_Department;", conn)
    diag_keys = pd.read_sql("SELECT diag_key, profile_name FROM Dim_DiagnosisCategory;", conn)
    
    # Join keys into main dataframe
    fact_df = df.merge(patient_keys, left_on="PatientID", right_on="patient_id", how="left")
    fact_df = fact_df.merge(dept_keys, left_on=["Department", "AdmissionType"], right_on=["department_name", "admission_type"], how="left")
    fact_df = fact_df.merge(diag_keys, left_on="DiagnosisProfile", right_on="profile_name", how="left")
    
    fact_records = pd.DataFrame({
        "patient_key": fact_df["patient_key"],
        "date_key": fact_df["DateKey"],
        "dept_key": fact_df["dept_key"],
        "diag_key": fact_df["diag_key"],
        "systolic_bp": fact_df["SystolicBP"],
        "diastolic_bp": fact_df["DiastolicBP"],
        "heart_rate": fact_df["HeartRate"],
        "bmi": fact_df["BMI"],
        "bmi_category": fact_df["BMICategory"],
        "fasting_glucose": fact_df["FastingGlucose"],
        "hba1c": fact_df["HbA1c"],
        "total_cholesterol": fact_df["TotalCholesterol"],
        "ldl": fact_df["LDL"],
        "hdl": fact_df["HDL"],
        "serum_creatinine": fact_df["SerumCreatinine"],
        "length_of_stay": fact_df["LengthOfStay"],
        "treatment_cost": fact_df["TreatmentCost"],
        "hypertension": fact_df["Hypertension"],
        "diabetes": fact_df["Diabetes"],
        "heart_disease": fact_df["HeartDisease"],
        "chronic_kidney_disease": fact_df["ChronicKidneyDisease"],
        "risk_score": fact_df["RiskScore"],
        "high_risk_patient": fact_df["HighRiskPatient"]
    })
    
    fact_records.to_sql("Fact_PatientAssessments", conn, if_exists="append", index=False)
    print(f"Loaded {len(fact_records)} rows into Fact_PatientAssessments.")
    
    conn.commit()
    conn.close()
    print("=== ETL Pipeline Finished Successfully! ===")
    return {
        "status": "SUCCESS",
        "records_processed": len(fact_records),
        "patients": len(dim_patient_df),
        "dates": len(dim_date_df),
        "departments": len(dim_dept_df),
        "diagnoses": len(dim_diag_df)
    }

if __name__ == "__main__":
    run_etl_pipeline()
