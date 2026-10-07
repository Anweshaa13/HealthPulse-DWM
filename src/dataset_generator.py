import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_healthcare_dataset(n_samples=3500, random_state=42):
    """
    Generates a medically realistic synthetic patient dataset with correlated
    demographics, vital signs, lab metrics, and clinical outcomes.
    """
    np.random.seed(random_state)
    
    # 1. Demographics
    patient_ids = [f"PAT-{10000 + i}" for i in range(n_samples)]
    ages = np.random.normal(loc=52, scale=17, size=n_samples).clip(18, 90).astype(int)
    genders = np.random.choice(["Male", "Female"], size=n_samples, p=[0.49, 0.51])
    blood_groups = np.random.choice(["A+", "A-", "B+", "B-", "O+", "O-", "AB+", "AB-"], size=n_samples, 
                                    p=[0.30, 0.05, 0.25, 0.05, 0.25, 0.04, 0.04, 0.02])
    
    # Lifestyle factors
    smoking_prob = np.where(genders == "Male", 0.28, 0.15)
    smoking = [np.random.choice(["Never", "Former", "Current"], p=[1 - p - 0.1, 0.1, p]) for p in smoking_prob]
    alcohol = np.random.choice(["Non-Drinker", "Moderate", "Heavy"], size=n_samples, p=[0.55, 0.35, 0.10])
    exercise = np.random.choice(["Sedentary", "Moderate", "Active"], size=n_samples, p=[0.40, 0.45, 0.15])
    
    # Socioeconomic / Department
    departments = np.random.choice(
        ["Cardiology", "Endocrinology", "Nephrology", "General Medicine", "Emergency"], 
        size=n_samples, 
        p=[0.25, 0.25, 0.15, 0.25, 0.10]
    )
    admission_types = np.random.choice(["Outpatient", "Inpatient", "Emergency"], size=n_samples, p=[0.50, 0.35, 0.15])
    
    # Date generation over past 2 years (2025-2026)
    start_date = datetime(2025, 1, 1)
    admission_dates = [start_date + timedelta(days=int(d)) for d in np.random.uniform(0, 600, n_samples)]
    
    # 2. Vitals & Labs with clinical correlation logic
    # Base BMI correlated with age and exercise
    bmi_base = 24.0 + (ages - 18) * 0.08
    bmi_lifestyle_adj = np.where(exercise == "Sedentary", 3.5, np.where(exercise == "Active", -2.0, 0.5))
    bmi = np.random.normal(loc=bmi_base + bmi_lifestyle_adj, scale=4.0).clip(16.0, 48.0).round(1)
    
    # Blood Pressure (Systolic & Diastolic) correlated with age, BMI, and smoking
    sbp_mean = 110 + (ages * 0.35) + (bmi - 22) * 0.65 + np.where(pd.Series(smoking) == "Current", 6.0, 0.0)
    systolic_bp = np.random.normal(loc=sbp_mean, scale=12.0).clip(90, 210).astype(int)
    diastolic_bp = (systolic_bp * 0.62 + np.random.normal(0, 5, n_samples)).clip(55, 125).astype(int)
    
    # Glucose and HbA1c correlated with BMI and Age
    glucose_mean = 85 + (bmi - 20) * 1.8 + (ages - 20) * 0.4
    fasting_glucose = np.random.normal(loc=glucose_mean, scale=28.0).clip(65, 320).round(1)
    hba1c = (fasting_glucose * 0.028 + 2.2 + np.random.normal(0, 0.4, n_samples)).clip(4.2, 14.5).round(2)
    
    # Total Cholesterol and Lipids
    chol_mean = 160 + (ages * 0.55) + (bmi - 20) * 1.1 + np.where(pd.Series(smoking) == "Current", 15.0, 0.0)
    total_cholesterol = np.random.normal(loc=chol_mean, scale=32.0).clip(110, 360).round(1)
    hdl = np.random.normal(loc=55 - (bmi - 22) * 0.35, scale=10.0).clip(25, 95).round(1)
    ldl = (total_cholesterol - hdl - np.random.normal(25, 6, n_samples)).clip(45, 260).round(1)
    
    # Kidney markers (Serum Creatinine & eGFR)
    creatinine_base = np.where(genders == "Male", 0.95, 0.80) + (ages * 0.005)
    serum_creatinine = np.random.normal(loc=creatinine_base, scale=0.35).clip(0.4, 5.5).round(2)
    
    # Heart rate
    heart_rate = np.random.normal(loc=74 + (bmi - 24) * 0.2, scale=9.0).clip(50, 125).astype(int)
    
    # 3. Clinical Outcomes & Diseases (Data Mining Targets)
    # Hypertension: SBP >= 140 or DBP >= 90
    htn_prob = 1 / (1 + np.exp(-((systolic_bp - 135) / 10 + (diastolic_bp - 85) / 8)))
    has_hypertension = (np.random.rand(n_samples) < htn_prob).astype(int)
    
    # Diabetes: Fasting glucose >= 126 or HbA1c >= 6.5
    dm_prob = 1 / (1 + np.exp(-((fasting_glucose - 120) / 18 + (hba1c - 6.2) / 0.7)))
    has_diabetes = (np.random.rand(n_samples) < dm_prob).astype(int)
    
    # Chronic Kidney Disease: Serum Creatinine >= 1.4 or Age > 65 + Diabetes
    ckd_prob = 1 / (1 + np.exp(-((serum_creatinine - 1.3) / 0.3 + has_hypertension * 0.5 + has_diabetes * 0.6)))
    has_ckd = (np.random.rand(n_samples) < ckd_prob).astype(int)
    
    # Heart Disease (Cardiovascular Event Risk)
    cad_score = (
        (ages - 45) * 0.04 +
        (systolic_bp - 120) * 0.02 +
        (total_cholesterol - 190) * 0.01 +
        (has_diabetes * 1.1) +
        (has_hypertension * 0.9) +
        (np.where(pd.Series(smoking) == "Current", 1.2, 0.0)) +
        (np.where(genders == "Male", 0.4, 0.0))
    )
    cad_prob = 1 / (1 + np.exp(-(cad_score - 1.5)))
    has_heart_disease = (np.random.rand(n_samples) < cad_prob).astype(int)
    
    # Overall High-Risk Cohort flag
    risk_score = (
        has_heart_disease * 35 +
        has_diabetes * 25 +
        has_hypertension * 20 +
        has_ckd * 20 +
        (ages > 65) * 10 +
        (bmi > 30) * 10
    ).clip(0, 100)
    high_risk_flag = (risk_score >= 50).astype(int)
    
    # Length of Stay and Cost
    los = np.where(
        admission_types == "Outpatient", 0,
        np.random.poisson(lam=2.5 + high_risk_flag * 3.5 + has_heart_disease * 2.0).clip(1, 28)
    )
    base_cost = np.where(admission_types == "Outpatient", 150, 800)
    cost = (base_cost + los * 650 + risk_score * 35 + np.random.normal(200, 100, n_samples)).clip(100, 25000).round(2)
    
    df = pd.DataFrame({
        "PatientID": patient_ids,
        "Age": ages,
        "Gender": genders,
        "BloodGroup": blood_groups,
        "Smoking": smoking,
        "Alcohol": alcohol,
        "ExerciseLevel": exercise,
        "Department": departments,
        "AdmissionType": admission_types,
        "AdmissionDate": [d.strftime("%Y-%m-%d") for d in admission_dates],
        "SystolicBP": systolic_bp,
        "DiastolicBP": diastolic_bp,
        "HeartRate": heart_rate,
        "BMI": bmi,
        "FastingGlucose": fasting_glucose,
        "HbA1c": hba1c,
        "TotalCholesterol": total_cholesterol,
        "LDL": ldl,
        "HDL": hdl,
        "SerumCreatinine": serum_creatinine,
        "LengthOfStay": los,
        "TreatmentCost": cost,
        "Hypertension": has_hypertension,
        "Diabetes": has_diabetes,
        "HeartDisease": has_heart_disease,
        "ChronicKidneyDisease": has_ckd,
        "RiskScore": risk_score,
        "HighRiskPatient": high_risk_flag
    })
    
    return df

if __name__ == "__main__":
    import os
    raw_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    df = generate_healthcare_dataset(n_samples=3500)
    out_path = os.path.join(raw_dir, "raw_healthcare_data.csv")
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} patient records saved to {out_path}")
