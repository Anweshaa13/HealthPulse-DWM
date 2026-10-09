import sqlite3
import pandas as pd
import os

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "warehouse", "healthcare_dw.db")

def get_connection(db_path=None):
    """Establishes a connection to the SQLite Data Warehouse."""
    path = db_path or DB_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def init_star_schema(db_path=None):
    """
    Initializes the DWM Star Schema in SQLite.
    Consists of 4 Dimension Tables and 1 Central Fact Table.
    """
    conn = get_connection(db_path)
    cursor = conn.cursor()
    
    # 1. Dim_Patient
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS Dim_Patient (
        patient_key INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id TEXT UNIQUE,
        age INTEGER,
        age_group TEXT,
        gender TEXT,
        blood_group TEXT,
        smoking TEXT,
        alcohol TEXT,
        exercise_level TEXT
    );
    """)
    
    # 2. Dim_Date
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS Dim_Date (
        date_key INTEGER PRIMARY KEY, -- Format: YYYYMMDD
        full_date TEXT UNIQUE,
        year INTEGER,
        quarter INTEGER,
        month INTEGER,
        month_name TEXT,
        day INTEGER,
        day_name TEXT,
        is_weekend INTEGER
    );
    """)
    
    # 3. Dim_Department
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS Dim_Department (
        dept_key INTEGER PRIMARY KEY AUTOINCREMENT,
        department_name TEXT,
        admission_type TEXT,
        UNIQUE(department_name, admission_type)
    );
    """)
    
    # 4. Dim_DiagnosisCategory
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS Dim_DiagnosisCategory (
        diag_key INTEGER PRIMARY KEY AUTOINCREMENT,
        profile_name TEXT UNIQUE,
        has_hypertension INTEGER,
        has_diabetes INTEGER,
        has_heart_disease INTEGER,
        has_ckd INTEGER,
        comorbidity_count INTEGER
    );
    """)
    
    # 5. Fact_PatientAssessments (Central Fact Table)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS Fact_PatientAssessments (
        assessment_key INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_key INTEGER,
        date_key INTEGER,
        dept_key INTEGER,
        diag_key INTEGER,
        systolic_bp INTEGER,
        diastolic_bp INTEGER,
        heart_rate INTEGER,
        bmi REAL,
        bmi_category TEXT,
        fasting_glucose REAL,
        hba1c REAL,
        total_cholesterol REAL,
        ldl REAL,
        hdl REAL,
        serum_creatinine REAL,
        length_of_stay INTEGER,
        treatment_cost REAL,
        hypertension INTEGER,
        diabetes INTEGER,
        heart_disease INTEGER,
        chronic_kidney_disease INTEGER,
        risk_score INTEGER,
        high_risk_patient INTEGER,
        FOREIGN KEY (patient_key) REFERENCES Dim_Patient(patient_key),
        FOREIGN KEY (date_key) REFERENCES Dim_Date(date_key),
        FOREIGN KEY (dept_key) REFERENCES Dim_Department(dept_key),
        FOREIGN KEY (diag_key) REFERENCES Dim_DiagnosisCategory(diag_key)
    );
    """)
    
    # Indexes for OLAP query performance
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_fact_date ON Fact_PatientAssessments(date_key);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_fact_dept ON Fact_PatientAssessments(dept_key);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_fact_patient ON Fact_PatientAssessments(patient_key);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_fact_diag ON Fact_PatientAssessments(diag_key);")
    
    conn.commit()
    conn.close()
    print("Star Schema initialized successfully.")

# ==========================================
# OLAP CUBE OPERATIONS
# ==========================================

def olap_roll_up(group_level="year", db_path=None):
    """
    OLAP Roll-Up Operation:
    Aggregates metrics from fine grain to coarser grain (e.g. Month -> Quarter -> Year).
    """
    conn = get_connection(db_path)
    if group_level == "year":
        query = """
        SELECT 
            d.year AS TimeDimension,
            COUNT(f.assessment_key) AS TotalPatients,
            ROUND(AVG(f.treatment_cost), 2) AS AvgCost,
            ROUND(AVG(f.length_of_stay), 2) AS AvgLengthOfStay,
            ROUND(AVG(f.risk_score), 2) AS AvgRiskScore,
            SUM(f.heart_disease) AS TotalHeartDisease,
            SUM(f.diabetes) AS TotalDiabetes,
            SUM(f.hypertension) AS TotalHypertension
        FROM Fact_PatientAssessments f
        JOIN Dim_Date d ON f.date_key = d.date_key
        GROUP BY d.year
        ORDER BY d.year;
        """
    elif group_level == "quarter":
        query = """
        SELECT 
            (d.year || '-Q' || d.quarter) AS TimeDimension,
            COUNT(f.assessment_key) AS TotalPatients,
            ROUND(AVG(f.treatment_cost), 2) AS AvgCost,
            ROUND(AVG(f.length_of_stay), 2) AS AvgLengthOfStay,
            ROUND(AVG(f.risk_score), 2) AS AvgRiskScore,
            SUM(f.heart_disease) AS TotalHeartDisease,
            SUM(f.diabetes) AS TotalDiabetes,
            SUM(f.hypertension) AS TotalHypertension
        FROM Fact_PatientAssessments f
        JOIN Dim_Date d ON f.date_key = d.date_key
        GROUP BY d.year, d.quarter
        ORDER BY d.year, d.quarter;
        """
    else:  # month
        query = """
        SELECT 
            (d.year || '-' || printf('%02d', d.month) || ' (' || d.month_name || ')') AS TimeDimension,
            COUNT(f.assessment_key) AS TotalPatients,
            ROUND(AVG(f.treatment_cost), 2) AS AvgCost,
            ROUND(AVG(f.length_of_stay), 2) AS AvgLengthOfStay,
            ROUND(AVG(f.risk_score), 2) AS AvgRiskScore,
            SUM(f.heart_disease) AS TotalHeartDisease,
            SUM(f.diabetes) AS TotalDiabetes,
            SUM(f.hypertension) AS TotalHypertension
        FROM Fact_PatientAssessments f
        JOIN Dim_Date d ON f.date_key = d.date_key
        GROUP BY d.year, d.month, d.month_name
        ORDER BY d.year, d.month;
        """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def olap_drill_down(hierarchy="department_type", db_path=None):
    """
    OLAP Drill-Down Operation:
    Deconstructs high-level summary into granular details (e.g. Department -> Admission Type).
    """
    conn = get_connection(db_path)
    if hierarchy == "department_type":
        query = """
        SELECT 
            dep.department_name AS Department,
            dep.admission_type AS AdmissionType,
            COUNT(f.assessment_key) AS PatientCount,
            ROUND(AVG(f.treatment_cost), 2) AS AvgCost,
            ROUND(AVG(f.risk_score), 2) AS AvgRiskScore,
            ROUND(AVG(f.length_of_stay), 1) AS AvgLengthOfStay,
            ROUND(100.0 * SUM(f.high_risk_patient) / COUNT(f.assessment_key), 1) AS HighRiskPct
        FROM Fact_PatientAssessments f
        JOIN Dim_Department dep ON f.dept_key = dep.dept_key
        GROUP BY dep.department_name, dep.admission_type
        ORDER BY dep.department_name, dep.admission_type;
        """
    else:  # age_group_gender
        query = """
        SELECT 
            p.age_group AS AgeGroup,
            p.gender AS Gender,
            COUNT(f.assessment_key) AS PatientCount,
            ROUND(AVG(f.treatment_cost), 2) AS AvgCost,
            ROUND(AVG(f.fasting_glucose), 1) AS AvgGlucose,
            ROUND(AVG(f.systolic_bp), 1) AS AvgSystolicBP,
            ROUND(100.0 * SUM(f.heart_disease) / COUNT(f.assessment_key), 1) AS HeartDiseasePct,
            ROUND(100.0 * SUM(f.diabetes) / COUNT(f.assessment_key), 1) AS DiabetesPct
        FROM Fact_PatientAssessments f
        JOIN Dim_Patient p ON f.patient_key = p.patient_key
        GROUP BY p.age_group, p.gender
        ORDER BY p.age_group, p.gender;
        """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def olap_slice(dimension="department", slice_value="Cardiology", db_path=None):
    """
    OLAP Slice Operation:
    Filters the data cube along ONE dimension (e.g. Department = 'Cardiology').
    """
    conn = get_connection(db_path)
    query = """
    SELECT 
        p.age_group AS AgeGroup,
        p.gender AS Gender,
        COUNT(f.assessment_key) AS SlicedCount,
        ROUND(AVG(f.treatment_cost), 2) AS AvgTreatmentCost,
        ROUND(AVG(f.systolic_bp), 1) AS AvgBP,
        ROUND(AVG(f.total_cholesterol), 1) AS AvgCholesterol,
        SUM(f.heart_disease) AS HeartDiseaseCases,
        SUM(f.diabetes) AS DiabetesCases
    FROM Fact_PatientAssessments f
    JOIN Dim_Patient p ON f.patient_key = p.patient_key
    JOIN Dim_Department dep ON f.dept_key = dep.dept_key
    WHERE dep.department_name = ?
    GROUP BY p.age_group, p.gender
    ORDER BY p.age_group, p.gender;
    """
    df = pd.read_sql_query(query, conn, params=[slice_value])
    conn.close()
    return df

def olap_dice(departments=None, age_groups=None, db_path=None):
    """
    OLAP Dice Operation:
    Extracts a sub-cube by selecting multiple values along TWO or MORE dimensions.
    """
    conn = get_connection(db_path)
    departments = departments or ["Cardiology", "Endocrinology"]
    age_groups = age_groups or ["Middle-Aged (45-59)", "Senior (60-74)", "Elderly (75+)"]
    
    dept_placeholders = ",".join(["?"] * len(departments))
    age_placeholders = ",".join(["?"] * len(age_groups))
    
    query = f"""
    SELECT 
        dep.department_name AS Department,
        p.age_group AS AgeGroup,
        p.smoking AS SmokingStatus,
        COUNT(f.assessment_key) AS SubCubeCount,
        ROUND(AVG(f.risk_score), 2) AS AvgRiskScore,
        ROUND(AVG(f.treatment_cost), 2) AS AvgCost,
        SUM(f.high_risk_patient) AS HighRiskCount
    FROM Fact_PatientAssessments f
    JOIN Dim_Patient p ON f.patient_key = p.patient_key
    JOIN Dim_Department dep ON f.dept_key = dep.dept_key
    WHERE dep.department_name IN ({dept_placeholders})
      AND p.age_group IN ({age_placeholders})
    GROUP BY dep.department_name, p.age_group, p.smoking
    ORDER BY dep.department_name, p.age_group;
    """
    params = departments + age_groups
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df

def olap_pivot(row_dim="AgeGroup", col_dim="Department", metric="AvgCost", db_path=None):
    """
    OLAP Pivot Operation:
    Rotates dimensions to form a cross-tabulated matrix.
    """
    conn = get_connection(db_path)
    query = """
    SELECT 
        p.age_group AS AgeGroup,
        dep.department_name AS Department,
        ROUND(AVG(f.treatment_cost), 2) AS AvgCost,
        ROUND(AVG(f.risk_score), 2) AS AvgRisk,
        ROUND(AVG(f.length_of_stay), 1) AS AvgLOS,
        COUNT(f.assessment_key) AS PatientCount
    FROM Fact_PatientAssessments f
    JOIN Dim_Patient p ON f.patient_key = p.patient_key
    JOIN Dim_Department dep ON f.dept_key = dep.dept_key
    GROUP BY p.age_group, dep.department_name;
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    pivot_df = df.pivot(index="AgeGroup", columns="Department", values=metric)
    return pivot_df

def get_warehouse_stats(db_path=None):
    """Returns row counts and metadata for each warehouse table."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    stats = {}
    for tbl in ["Dim_Patient", "Dim_Date", "Dim_Department", "Dim_DiagnosisCategory", "Fact_PatientAssessments"]:
        cursor.execute(f"SELECT COUNT(*) FROM {tbl};")
        stats[tbl] = cursor.fetchone()[0]
    conn.close()
    return stats
