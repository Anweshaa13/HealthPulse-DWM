import os
import sys
import sqlite3
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from mlxtend.frequent_patterns import apriori, association_rules

# Path to warehouse DB
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "warehouse", "healthcare_dw.db")

def load_warehouse_dataset(db_path=None):
    """Fetches denormalized analytics dataset directly from the Star Schema warehouse."""
    path = db_path or DB_PATH
    conn = sqlite3.connect(path)
    query = """
    SELECT 
        p.age, p.gender, p.smoking, p.alcohol, p.exercise_level,
        f.systolic_bp, f.diastolic_bp, f.heart_rate, f.bmi,
        f.fasting_glucose, f.hba1c, f.total_cholesterol, f.ldl, f.hdl,
        f.serum_creatinine, f.length_of_stay, f.treatment_cost, f.hypertension, f.diabetes,
        f.heart_disease, f.chronic_kidney_disease, f.risk_score, f.high_risk_patient
    FROM Fact_PatientAssessments f
    JOIN Dim_Patient p ON f.patient_key = p.patient_key;
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

# ==========================================
# 1. CLASSIFICATION MINING MODELS
# ==========================================

class DiseaseRiskPredictor:
    def __init__(self, target="heart_disease"):
        self.target = target
        self.models = {}
        self.metrics = {}
        self.feature_names = []
        self.scaler = StandardScaler()
        self.dt_rules_text = ""
        self.feature_importances = {}
        self.roc_curves = {}
        self.confusion_matrices = {}

    def prepare_data(self, df):
        # Feature encoding
        X = pd.DataFrame()
        X["age"] = df["age"]
        X["is_male"] = (df["gender"] == "Male").astype(int)
        X["is_smoker"] = (df["smoking"] == "Current").astype(int)
        X["is_former_smoker"] = (df["smoking"] == "Former").astype(int)
        X["is_sedentary"] = (df["exercise_level"] == "Sedentary").astype(int)
        X["systolic_bp"] = df["systolic_bp"]
        X["diastolic_bp"] = df["diastolic_bp"]
        X["heart_rate"] = df["heart_rate"]
        X["bmi"] = df["bmi"]
        X["fasting_glucose"] = df["fasting_glucose"]
        X["hba1c"] = df["hba1c"]
        X["total_cholesterol"] = df["total_cholesterol"]
        X["ldl"] = df["ldl"]
        X["hdl"] = df["hdl"]
        X["serum_creatinine"] = df["serum_creatinine"]
        
        # Clinical comorbidities as features (if not the target itself)
        if self.target != "hypertension":
            X["hypertension"] = df["hypertension"]
        if self.target != "diabetes":
            X["diabetes"] = df["diabetes"]
        if self.target != "chronic_kidney_disease":
            X["chronic_kidney_disease"] = df["chronic_kidney_disease"]
            
        y = df[self.target]
        self.feature_names = list(X.columns)
        return X, y

    def train_and_evaluate(self, df):
        X, y = self.prepare_data(df)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        classifiers = {
            "Decision Tree (CART)": DecisionTreeClassifier(max_depth=5, min_samples_leaf=15, random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42),
            "Gaussian Naive Bayes": GaussianNB(),
            "Logistic Regression": LogisticRegression(max_iter=500, random_state=42)
        }
        
        results = []
        for name, clf in classifiers.items():
            # Fit model
            if name in ["Logistic Regression", "Gaussian Naive Bayes"]:
                clf.fit(X_train_scaled, y_train)
                y_pred = clf.predict(X_test_scaled)
                y_proba = clf.predict_proba(X_test_scaled)[:, 1]
            else:
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                y_proba = clf.predict_proba(X_test)[:, 1]
                
            self.models[name] = clf
            
            # Compute evaluation metrics
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            auc = roc_auc_score(y_test, y_proba)
            
            # ROC curve data
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            self.roc_curves[name] = (fpr.tolist(), tpr.tolist(), round(auc, 3))
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            self.confusion_matrices[name] = cm
            
            results.append({
                "Algorithm": name,
                "Accuracy (%)": round(acc * 100, 2),
                "Precision (%)": round(prec * 100, 2),
                "Recall (%)": round(rec * 100, 2),
                "F1-Score (%)": round(f1 * 100, 2),
                "ROC-AUC": round(auc, 3)
            })
            
        self.metrics = pd.DataFrame(results)
        
        # Decision Tree readable rules
        dt_model = self.models["Decision Tree (CART)"]
        self.dt_rules_text = export_text(dt_model, feature_names=self.feature_names, max_depth=3)
        
        # Feature importances from Random Forest
        rf_model = self.models["Random Forest"]
        self.feature_importances = pd.DataFrame({
            "Feature": self.feature_names,
            "Importance": rf_model.feature_importances_
        }).sort_values("Importance", ascending=False)
        
        return self.metrics

    def predict_patient(self, input_dict):
        """
        Takes patient clinical attributes and generates multi-model risk predictions.
        """
        row_dict = {}
        for feat in self.feature_names:
            row_dict[feat] = input_dict.get(feat, 0)
            
        sample_df = pd.DataFrame([row_dict])
        sample_scaled = self.scaler.transform(sample_df)
        
        predictions = {}
        for name, clf in self.models.items():
            if name in ["Logistic Regression", "Gaussian Naive Bayes"]:
                prob = clf.predict_proba(sample_scaled)[0][1]
                pred = clf.predict(sample_scaled)[0]
            else:
                prob = clf.predict_proba(sample_df)[0][1]
                pred = clf.predict(sample_df)[0]
            predictions[name] = {
                "Prediction": int(pred),
                "Probability": round(float(prob) * 100, 1),
                "RiskLevel": "High Risk" if prob >= 0.50 else "Moderate/Low Risk"
            }
        return predictions

# ==========================================
# 2. PATIENT COHORT CLUSTERING (K-MEANS)
# ==========================================

class PatientClusteringModel:
    def __init__(self, n_clusters=4):
        self.n_clusters = n_clusters
        self.kmeans = None
        self.scaler = StandardScaler()
        self.cluster_features = [
            "age", "systolic_bp", "bmi", "fasting_glucose", 
            "total_cholesterol", "serum_creatinine", "risk_score"
        ]
        self.pca_2d = None
        self.pca_3d = None

    def fit(self, df):
        X = df[self.cluster_features].copy()
        X_scaled = self.scaler.fit_transform(X)
        
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        clusters = self.kmeans.fit_predict(X_scaled)
        
        # Silhouette Score
        sil_score = silhouette_score(X_scaled, clusters)
        
        # PCA for 2D & 3D visualization
        self.pca_2d = PCA(n_components=2, random_state=42)
        coords_2d = self.pca_2d.fit_transform(X_scaled)
        
        self.pca_3d = PCA(n_components=3, random_state=42)
        coords_3d = self.pca_3d.fit_transform(X_scaled)
        
        df_clustered = df.copy()
        df_clustered["Cluster"] = [f"Cluster {c}" for c in clusters]
        df_clustered["Cluster_ID"] = clusters
        df_clustered["PCA1"] = coords_2d[:, 0]
        df_clustered["PCA2"] = coords_2d[:, 1]
        df_clustered["PCA3"] = coords_3d[:, 2]
        
        # Compute cluster centroids profile
        cluster_summary = df_clustered.groupby("Cluster")[self.cluster_features].mean().round(1)
        cluster_summary["PatientCount"] = df_clustered.groupby("Cluster").size()
        cluster_summary["HeartDiseasePct"] = (df_clustered.groupby("Cluster")["heart_disease"].mean() * 100).round(1)
        cluster_summary["DiabetesPct"] = (df_clustered.groupby("Cluster")["diabetes"].mean() * 100).round(1)
        
        # Assign clinical cohort titles based on average risk score
        sorted_clusters = cluster_summary.sort_values("risk_score").index.tolist()
        cohort_labels = {
            sorted_clusters[0]: "Cohort A: Low Risk / Healthy Vital Profile",
            sorted_clusters[1]: "Cohort B: Mild Pre-Hypertensive / Metabolic Watch",
            sorted_clusters[2]: "Cohort C: Moderate Cardiovascular Risk",
            sorted_clusters[3]: "Cohort D: High-Risk Multimorbid Cohort"
        }
        df_clustered["CohortName"] = df_clustered["Cluster"].map(cohort_labels)
        cluster_summary["CohortProfile"] = [cohort_labels.get(c, c) for c in cluster_summary.index]
        
        return df_clustered, cluster_summary, sil_score

    def compute_elbow_curve(self, df, k_range=range(2, 8)):
        X = df[self.cluster_features].copy()
        X_scaled = self.scaler.fit_transform(X)
        inertias = []
        silhouettes = []
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(X_scaled)
            inertias.append(km.inertia_)
            silhouettes.append(silhouette_score(X_scaled, km.labels_))
        return list(k_range), inertias, silhouettes

# ==========================================
# 3. ASSOCIATION RULE MINING (APRIORI)
# ==========================================

class HealthcareAssociationMiner:
    def __init__(self, min_support=0.08, min_confidence=0.45):
        self.min_support = min_support
        self.min_confidence = min_confidence

    def discretize_records(self, df):
        """Converts continuous clinical parameters into discrete boolean medical facts."""
        basket = pd.DataFrame()
        basket["Elderly(Age>=60)"] = (df["age"] >= 60).astype(int)
        basket["Male"] = (df["gender"] == "Male").astype(int)
        basket["CurrentSmoker"] = (df["smoking"] == "Current").astype(int)
        basket["Sedentary"] = (df["exercise_level"] == "Sedentary").astype(int)
        basket["Hypertension(BP>=140)"] = (df["systolic_bp"] >= 140).astype(int)
        basket["HighFastingSugar(>=126)"] = (df["fasting_glucose"] >= 126).astype(int)
        basket["HighHbA1c(>=6.5)"] = (df["hba1c"] >= 6.5).astype(int)
        basket["HighCholesterol(>=200)"] = (df["total_cholesterol"] >= 200).astype(int)
        basket["Obese(BMI>=30)"] = (df["bmi"] >= 30).astype(int)
        basket["ElevatedCreatinine(>=1.3)"] = (df["serum_creatinine"] >= 1.3).astype(int)
        basket["Diagnosis_HeartDisease"] = df["heart_disease"].astype(int)
        basket["Diagnosis_Diabetes"] = df["diabetes"].astype(int)
        basket["Diagnosis_CKD"] = df["chronic_kidney_disease"].astype(int)
        return basket.astype(bool)

    def mine_rules(self, df):
        basket = self.discretize_records(df)
        frequent_itemsets = apriori(basket, min_support=self.min_support, use_colnames=True)
        
        if frequent_itemsets.empty:
            return pd.DataFrame(), pd.DataFrame()
            
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=self.min_confidence)
        
        # Clean rules presentation
        rules["antecedents_str"] = rules["antecedents"].apply(lambda s: ", ".join(list(s)))
        rules["consequents_str"] = rules["consequents"].apply(lambda s: ", ".join(list(s)))
        rules["Rule"] = rules["antecedents_str"] + "  ===>  " + rules["consequents_str"]
        rules["support"] = rules["support"].round(3)
        rules["confidence"] = rules["confidence"].round(3)
        rules["lift"] = rules["lift"].round(3)
        
        # Sort by lift descending
        rules = rules.sort_values(by="lift", ascending=False).reset_index(drop=True)
        return frequent_itemsets, rules

# ==========================================
# 4. REGRESSION MODEL (HEALTHCARE EXPENDITURE)
# ==========================================

class TreatmentCostRegressor:
    def __init__(self):
        self.models = {}
        self.metrics = {}
        self.scaler = StandardScaler()
        self.feature_names = [
            "age", "systolic_bp", "diastolic_bp", "bmi", "fasting_glucose",
            "total_cholesterol", "serum_creatinine", "length_of_stay",
            "risk_score", "hypertension", "diabetes", "heart_disease", "chronic_kidney_disease"
        ]
        self.actual_vs_predicted = {}

    def train_and_evaluate(self, df):
        X = df[self.feature_names].copy()
        y = df["treatment_cost"].copy()
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        regressors = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(alpha=1.0),
            "Random Forest Regressor": RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
        }
        
        results = []
        for name, reg in regressors.items():
            if name in ["Linear Regression", "Ridge Regression"]:
                reg.fit(X_train_scaled, y_train)
                y_pred = reg.predict(X_test_scaled)
            else:
                reg.fit(X_train, y_train)
                y_pred = reg.predict(X_test)
                
            self.models[name] = reg
            
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            self.actual_vs_predicted[name] = {
                "y_true": y_test.values.tolist(),
                "y_pred": y_pred.tolist()
            }
            
            results.append({
                "Algorithm": name,
                "MAE ($)": round(mae, 2),
                "RMSE ($)": round(rmse, 2),
                "R2 Score": round(r2, 4)
            })
            
        self.metrics = pd.DataFrame(results)
        return self.metrics

    def predict_cost(self, input_dict):
        """Predict treatment cost for one new patient using all trained regressors."""
        row={feature:input_dict.get(feature,0) for feature in self.feature_names}
        sample=pd.DataFrame([row]); scaled=self.scaler.transform(sample)
        out={}
        for name,reg in self.models.items():
            value=reg.predict(scaled if name in ["Linear Regression","Ridge Regression"] else sample)[0]
            out[name]=round(float(max(0,value)),2)
        return out

