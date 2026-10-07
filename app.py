import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
import sqlite3

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
    olap_slice, olap_dice, olap_pivot, get_warehouse_stats, DB_PATH,
    add_patient_record, save_patient_predictions, get_patient_records
)
from src.etl import run_etl_pipeline, RAW_DATA_PATH
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
        padding: 8px 16px;
        border-radius: 6px;
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
# SIDEBAR
# -------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/hospital-3.png", width=64)
    st.title("HealthPulse DWM")
    st.caption("Clinical Data Warehouse & Data Mining System")
    
    st.markdown("---")
    menu = st.radio(
        "Navigation",
        [
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
    )
    
    st.markdown("---")
    st.markdown("### 🗄️ Warehouse Status")
    try:
        stats = get_cached_stats()
        st.success(f"**Connected**: `healthcare_dw.db`")
        st.write(f"• **Fact Assessments**: `{stats.get('Fact_PatientAssessments', 0):,}`")
        st.write(f"• **Dim Patients**: `{stats.get('Dim_Patient', 0):,}`")
        st.write(f"• **Dim Dates**: `{stats.get('Dim_Date', 0):,}`")
        st.write(f"• **Dim Departments**: `{stats.get('Dim_Department', 0):,}`")
    except Exception as e:
        st.error(f"DB Error: {e}")
        
    

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

    st.markdown("#### 📊 Comparative Model Performance Bar Graph")
    metric_options=["Accuracy (%)","Precision (%)","Recall (%)","F1-Score (%)","ROC-AUC"]
    selected_metric=st.selectbox("Select metric for visual comparison:",metric_options,key=f"benchmark_metric_{target_eval}")
    bar_df=predictor.metrics[["Algorithm",selected_metric]].copy()
    fig_bar=px.bar(bar_df,x="Algorithm",y=selected_metric,text=selected_metric,title=f"{selected_metric} Comparison — {target_eval.replace('_',' ').title()}")
    fig_bar.update_traces(texttemplate="%{text}",textposition="outside")
    fig_bar.update_layout(yaxis_range=[0,100] if selected_metric!="ROC-AUC" else [0,1])
    st.plotly_chart(fig_bar,use_container_width=True)
    
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
# VIEW 8: PATIENT MANAGEMENT
# -------------------------------------------------------------
elif menu == "👤 Patient Management":
    st.markdown('<div class="main-title">Patient Management & Personalized Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Add patients to the Star Schema database and immediately generate disease, cost, and risk insights.</div>', unsafe_allow_html=True)
    tab_add,tab_records,tab_analyze=st.tabs(["➕ Add New Patient","🗂️ Patient Records","🔎 Analyze Patient"])

    with tab_add:
        st.info("Enter a new patient below. The record is saved permanently in the SQLite Star Schema warehouse.")
        with st.form("new_patient_form"):
            st.markdown("#### 👤 Patient & Admission Details")
            c1,c2,c3=st.columns(3)
            with c1:
                patient_name=st.text_input("Patient Name *",placeholder="e.g. Rahul Sharma")
                patient_id=st.text_input("Patient ID (optional)",placeholder="Leave blank to auto-generate")
                age=st.number_input("Age",18,90,45)
                gender=st.selectbox("Gender",["Male","Female"])
            with c2:
                blood_group=st.selectbox("Blood Group",["A+","A-","B+","B-","O+","O-","AB+","AB-"])
                smoking=st.selectbox("Smoking Status",["Never","Former","Current"])
                alcohol=st.selectbox("Alcohol Use",["Non-Drinker","Moderate","Heavy"])
                exercise=st.selectbox("Exercise Level",["Sedentary","Moderate","Active"])
            with c3:
                department=st.selectbox("Department",["Cardiology","Endocrinology","Nephrology","General Medicine","Emergency"])
                admission_type=st.selectbox("Admission Type",["Outpatient","Inpatient","Emergency"])
                admission_date=st.date_input("Admission Date")

            st.markdown("#### 🩺 Clinical Measurements")
            c1,c2,c3,c4=st.columns(4)
            with c1:
                systolic_bp=st.number_input("Systolic BP",90,210,135); diastolic_bp=st.number_input("Diastolic BP",55,125,85); heart_rate=st.number_input("Heart Rate",50,125,75)
            with c2:
                bmi=st.number_input("BMI",16.0,48.0,24.5,step=0.1); fasting_glucose=st.number_input("Fasting Glucose",65.0,320.0,105.0,step=0.1); hba1c=st.number_input("HbA1c (%)",4.2,14.5,5.7,step=0.1)
            with c3:
                total_cholesterol=st.number_input("Total Cholesterol",110.0,360.0,190.0,step=0.1); ldl=st.number_input("LDL",45.0,260.0,115.0,step=0.1); hdl=st.number_input("HDL",25.0,95.0,55.0,step=0.1)
            with c4:
                serum_creatinine=st.number_input("Serum Creatinine",0.4,5.5,0.9,step=0.01); length_of_stay=st.number_input("Length of Stay (days)",0,28,2); treatment_cost=st.number_input("Recorded Treatment Cost ($)",0.0,25000.0,1500.0,step=50.0)

            st.markdown("#### 🧬 Known Diagnosis Information")
            st.caption("These are recorded diagnoses, not the model prediction. Leave unchecked when not known.")
            d1,d2,d3,d4=st.columns(4)
            hypertension=d1.checkbox("Hypertension"); diabetes=d2.checkbox("Diabetes"); heart_disease=d3.checkbox("Heart Disease"); chronic_kidney_disease=d4.checkbox("Chronic Kidney Disease")
            submitted=st.form_submit_button("💾 Save Patient & Generate Results",type="primary",use_container_width=True)

        if submitted:
            if not patient_name.strip():
                st.error("Please enter the patient's name.")
            else:
                risk_score=int(int(heart_disease)*35+int(diabetes)*25+int(hypertension)*20+int(chronic_kidney_disease)*20+int(age>65)*10+int(bmi>30)*10)
                record={"patient_id":patient_id,"patient_name":patient_name.strip(),"age":age,"gender":gender,"blood_group":blood_group,"smoking":smoking,"alcohol":alcohol,"exercise_level":exercise,"department":department,"admission_type":admission_type,"admission_date":admission_date.strftime("%Y-%m-%d"),"systolic_bp":systolic_bp,"diastolic_bp":diastolic_bp,"heart_rate":heart_rate,"bmi":bmi,"fasting_glucose":fasting_glucose,"hba1c":hba1c,"total_cholesterol":total_cholesterol,"ldl":ldl,"hdl":hdl,"serum_creatinine":serum_creatinine,"length_of_stay":length_of_stay,"treatment_cost":treatment_cost,"hypertension":int(hypertension),"diabetes":int(diabetes),"heart_disease":int(heart_disease),"chronic_kidney_disease":int(chronic_kidney_disease),"risk_score":risk_score,"high_risk_patient":int(risk_score>=50)}
                payload={"age":age,"is_male":int(gender=="Male"),"is_smoker":int(smoking=="Current"),"is_former_smoker":int(smoking=="Former"),"is_sedentary":int(exercise=="Sedentary"),"systolic_bp":systolic_bp,"diastolic_bp":diastolic_bp,"heart_rate":heart_rate,"bmi":bmi,"fasting_glucose":fasting_glucose,"hba1c":hba1c,"total_cholesterol":total_cholesterol,"ldl":ldl,"hdl":hdl,"serum_creatinine":serum_creatinine,"hypertension":int(hypertension),"diabetes":int(diabetes),"chronic_kidney_disease":int(chronic_kidney_disease)}
                try:
                    predictions={}
                    for target in ["heart_disease","diabetes","chronic_kidney_disease"]:
                        predictions[target]=get_trained_predictor(target=target).predict_patient(payload)
                    reg=get_trained_regressor()
                    cost_preds=reg.predict_cost({"age":age,"systolic_bp":systolic_bp,"diastolic_bp":diastolic_bp,"bmi":bmi,"fasting_glucose":fasting_glucose,"total_cholesterol":total_cholesterol,"serum_creatinine":serum_creatinine,"length_of_stay":length_of_stay,"risk_score":risk_score,"hypertension":int(hypertension),"diabetes":int(diabetes),"heart_disease":int(heart_disease),"chronic_kidney_disease":int(chronic_kidney_disease)})
                    saved_id=add_patient_record(record); save_patient_predictions(saved_id,predictions)
                    get_cached_stats.clear(); get_cached_df.clear(); get_trained_predictor.clear(); get_clustering_results.clear(); get_mined_rules.clear(); get_trained_regressor.clear()
                    st.success(f"✅ Patient **{patient_name}** saved successfully with Patient ID **{saved_id}**.")
                    st.markdown("### 🔮 Personalized Disease Prediction")
                    for target,models in predictions.items():
                        label={"heart_disease":"❤️ Heart Disease","diabetes":"🩸 Diabetes","chronic_kidney_disease":"🩺 Chronic Kidney Disease"}[target]
                        st.markdown(f"#### {label}"); cols=st.columns(4)
                        for i,(model_name,res) in enumerate(models.items()):
                            cols[i].metric(model_name,"Positive Risk" if res["Prediction"] else "Low Risk",f"{res['Probability']}% confidence")
                    st.markdown("### 💰 Personalized Treatment Cost Prediction")
                    st.dataframe(pd.DataFrame([{"Regression Model":k,"Predicted Cost ($)":v} for k,v in cost_preds.items()]),use_container_width=True)
                    st.success(f"Random Forest estimate: **${cost_preds['Random Forest Regressor']:,.2f}**")
                    st.markdown(f"### 🧭 Patient Risk Summary\n**Calculated clinical risk score:** `{risk_score}/100` — **{'HIGH RISK' if risk_score>=50 else 'LOW / MODERATE RISK'}**")
                    st.caption("Predictions are for academic/data-mining demonstration and are not a medical diagnosis.")
                except sqlite3.IntegrityError as e: st.error(f"Could not save the patient. The Patient ID may already exist. Details: {e}")
                except Exception as e: st.error(f"Patient processing failed: {e}")

    with tab_records:
        st.markdown("#### 🗂️ Patients Currently Stored in the Warehouse")
        records=get_patient_records()
        if records.empty: st.warning("No patient records found.")
        else:
            st.dataframe(records,use_container_width=True,height=520)
            st.download_button("📥 Download Patient Records CSV",records.to_csv(index=False).encode("utf-8"),"healthpulse_patient_records.csv","text/csv")

    with tab_analyze:
        records=get_patient_records()
        if records.empty: st.info("Add a patient first.")
        else:
            opts={f"{r.PatientID} — {r.PatientName or 'Unnamed'}":r.PatientID for r in records.itertuples()}; selected=st.selectbox("Select patient to analyze:",list(opts)); pid=opts[selected]; p=records[records.PatientID==pid].iloc[0]
            st.markdown(f"### 🔎 Analysis for {p['PatientName'] or p['PatientID']}")
            a1,a2,a3,a4=st.columns(4); a1.metric("Age",f"{int(p['Age'])} years"); a2.metric("BMI",f"{p['BMI']:.1f}"); a3.metric("Risk Score",f"{int(p['RiskScore'])}/100"); a4.metric("Treatment Cost",f"${p['TreatmentCost']:,.2f}")
            payload={"age":p["Age"],"is_male":int(p["Gender"]=="Male"),"is_smoker":int(p["Smoking"]=="Current"),"is_former_smoker":int(p["Smoking"]=="Former"),"is_sedentary":int(p["ExerciseLevel"]=="Sedentary"),"systolic_bp":p["SystolicBP"],"diastolic_bp":p["DiastolicBP"],"heart_rate":p["HeartRate"],"bmi":p["BMI"],"fasting_glucose":p["FastingGlucose"],"hba1c":p["HbA1c"],"total_cholesterol":p["TotalCholesterol"],"ldl":p["LDL"],"hdl":p["HDL"],"serum_creatinine":p["SerumCreatinine"],"hypertension":p["Hypertension"],"diabetes":p["Diabetes"],"chronic_kidney_disease":p["ChronicKidneyDisease"]}
            st.markdown("#### 🔮 Disease Risk")
            for target in ["heart_disease","diabetes","chronic_kidney_disease"]:
                pred=get_trained_predictor(target=target).predict_patient(payload); best=max(pred.items(),key=lambda x:x[1]["Probability"]); label={"heart_disease":"Heart Disease","diabetes":"Diabetes","chronic_kidney_disease":"Chronic Kidney Disease"}[target]; st.write(f"**{label}:** {'Positive Risk' if best[1]['Prediction'] else 'Low Risk'} — {best[1]['Probability']}% ({best[0]})")
            reg2=get_trained_regressor(); cp=reg2.predict_cost({"age":p["Age"],"systolic_bp":p["SystolicBP"],"diastolic_bp":p["DiastolicBP"],"bmi":p["BMI"],"fasting_glucose":p["FastingGlucose"],"total_cholesterol":p["TotalCholesterol"],"serum_creatinine":p["SerumCreatinine"],"length_of_stay":p["LengthOfStay"],"risk_score":p["RiskScore"],"hypertension":p["Hypertension"],"diabetes":p["Diabetes"],"heart_disease":p["HeartDisease"],"chronic_kidney_disease":p["ChronicKidneyDisease"]}); st.markdown("#### 💰 Cost Prediction"); st.dataframe(pd.DataFrame([{"Model":k,"Predicted Cost ($)":v} for k,v in cp.items()]),use_container_width=True)

# -------------------------------------------------------------
# VIEW 9: PATIENT AUDIT REPORT
# -------------------------------------------------------------
elif menu == "📑 Patient Audit Report":
    st.markdown('<div class="main-title">Clinical Patient Diagnostic Audit Report</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Extract Individual Patient Summary and Generate Health Risk Brief</div>', unsafe_allow_html=True)
    
    conn = get_connection()
    patients_sample = pd.read_sql_query("""
    SELECT 
        p.patient_id, COALESCE(p.patient_name, '') AS patient_name, p.age, p.gender, p.smoking,
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
    if p_data['patient_name']:
        st.caption(f"Patient Name: **{p_data['patient_name']}**")
    
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
Patient Name      : {p_data['patient_name']}
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
