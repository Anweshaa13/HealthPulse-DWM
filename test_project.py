import os
import sys
import unittest
import pandas as pd
import sqlite3

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.warehouse import (
    get_connection, olap_roll_up, olap_drill_down, 
    olap_slice, olap_dice, olap_pivot, get_warehouse_stats, DB_PATH
)
from src.etl import run_etl_pipeline, RAW_DATA_PATH
from src.mining_models import (
    load_warehouse_dataset, DiseaseRiskPredictor, 
    PatientClusteringModel, HealthcareAssociationMiner
)

class TestDWMHealthcareProject(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Ensure database is generated and loaded
        if not os.path.exists(DB_PATH):
            print("Database not found, running ETL pipeline...")
            run_etl_pipeline()

    def test_warehouse_tables_exist(self):
        """Verify all 5 tables in the Star Schema exist and have data."""
        stats = get_warehouse_stats()
        self.assertGreater(stats["Dim_Patient"], 1000, "Dim_Patient should have > 1000 records")
        self.assertGreater(stats["Dim_Date"], 100, "Dim_Date should have > 100 records")
        self.assertGreater(stats["Dim_Department"], 5, "Dim_Department should have records")
        self.assertGreater(stats["Dim_DiagnosisCategory"], 5, "Dim_DiagnosisCategory should have records")
        self.assertGreater(stats["Fact_PatientAssessments"], 1000, "Fact_PatientAssessments should have > 1000 records")
        print("[PASS] Star Schema tables verified with proper row counts.")
    def test_olap_roll_up(self):
        """Test OLAP Roll-up across Year, Quarter, and Month."""
        for grain in ["year", "quarter", "month"]:
            df = olap_roll_up(group_level=grain)
            self.assertFalse(df.empty, f"Roll-up by {grain} returned empty result")
            self.assertIn("TotalPatients", df.columns)
            self.assertIn("AvgCost", df.columns)
        print("[PASS] OLAP Roll-Up operation verified.")

    def test_olap_drill_down(self):
        """Test OLAP Drill-down operations."""
        df_dept = olap_drill_down(hierarchy="department_type")
        self.assertFalse(df_dept.empty)
        self.assertIn("Department", df_dept.columns)
        self.assertIn("AdmissionType", df_dept.columns)
        
        df_age = olap_drill_down(hierarchy="age_group_gender")
        self.assertFalse(df_age.empty)
        self.assertIn("AgeGroup", df_age.columns)
        print("[PASS] OLAP Drill-Down operations verified.")

    def test_olap_slice_and_dice(self):
        """Test OLAP Slice and Dice operations."""
        df_slice = olap_slice(slice_value="Cardiology")
        self.assertFalse(df_slice.empty)
        
        df_dice = olap_dice(departments=["Cardiology", "Nephrology"], age_groups=["Middle-Aged (45-59)", "Senior (60-74)"])
        self.assertFalse(df_dice.empty)
        self.assertTrue(all(d in ["Cardiology", "Nephrology"] for d in df_dice["Department"].unique()))
        print("[PASS] OLAP Slice and Dice operations verified.")

    def test_olap_pivot(self):
        """Test OLAP Pivot operation."""
        pivot_df = olap_pivot(row_dim="AgeGroup", col_dim="Department", metric="AvgCost")
        self.assertFalse(pivot_df.empty)
        self.assertGreater(pivot_df.shape[0], 0)
        self.assertGreater(pivot_df.shape[1], 0)
        print("[PASS] OLAP Pivot operation verified.")

    def test_predictive_mining_classifiers(self):
        """Test training and inference of all 4 classification algorithms."""
        df = load_warehouse_dataset()
        predictor = DiseaseRiskPredictor(target="heart_disease")
        metrics = predictor.train_and_evaluate(df)
        
        self.assertEqual(len(metrics), 4, "Should evaluate 4 classification algorithms")
        for _, row in metrics.iterrows():
            self.assertGreater(row["Accuracy (%)"], 60.0, f"{row['Algorithm']} accuracy below threshold")
            self.assertGreater(row["ROC-AUC"], 0.65, f"{row['Algorithm']} ROC-AUC below threshold")
            
        # Test inference
        patient_sample = {
            "age": 62, "is_male": 1, "is_smoker": 1, "is_former_smoker": 0, "is_sedentary": 1,
            "systolic_bp": 155, "diastolic_bp": 95, "heart_rate": 85, "bmi": 32.5,
            "fasting_glucose": 160.0, "hba1c": 8.1, "total_cholesterol": 240.0, "ldl": 150.0,
            "hdl": 38.0, "serum_creatinine": 1.5, "hypertension": 1, "diabetes": 1, "chronic_kidney_disease": 0
        }
        preds = predictor.predict_patient(patient_sample)
        self.assertEqual(len(preds), 4)
        for model_name, res in preds.items():
            self.assertIn(res["Prediction"], [0, 1])
            self.assertTrue(0.0 <= res["Probability"] <= 100.0)
        print("[PASS] Classification models and inference verified.")

    def test_patient_clustering_kmeans(self):
        """Test K-Means clustering and PCA projections."""
        df = load_warehouse_dataset()
        clusterer = PatientClusteringModel(n_clusters=4)
        df_clustered, summary, sil_score = clusterer.fit(df)
        
        self.assertEqual(len(df_clustered), len(df))
        self.assertEqual(len(summary), 4)
        self.assertIn("PCA1", df_clustered.columns)
        self.assertIn("PCA2", df_clustered.columns)
        self.assertIn("PCA3", df_clustered.columns)
        self.assertTrue(-1.0 <= sil_score <= 1.0)
        print("[PASS] K-Means clustering and PCA verified.")

    def test_association_rule_mining_apriori(self):
        """Test Apriori itemset mining and comorbidity rule extraction."""
        df = load_warehouse_dataset()
        miner = HealthcareAssociationMiner(min_support=0.08, min_confidence=0.45)
        itemsets, rules = miner.mine_rules(df)
        
        self.assertFalse(itemsets.empty, "Frequent itemsets should not be empty")
        self.assertFalse(rules.empty, "Association rules should not be empty")
        self.assertIn("lift", rules.columns)
        self.assertIn("confidence", rules.columns)
        self.assertGreater(rules["lift"].max(), 1.0, "Top lift should be > 1.0")
        print("[PASS] Association Rule Mining (Apriori) verified.")

    def test_treatment_cost_regression(self):
        """Test Healthcare Expenditure Regressors (Linear, Ridge, Random Forest)."""
        from src.mining_models import TreatmentCostRegressor
        df = load_warehouse_dataset()
        reg = TreatmentCostRegressor()
        metrics = reg.train_and_evaluate(df)
        
        self.assertEqual(len(metrics), 3, "Should evaluate 3 regression algorithms")
        for _, row in metrics.iterrows():
            self.assertGreater(row["R2 Score"], 0.90, f"{row['Algorithm']} R2 score below threshold")
            self.assertLess(row["MAE ($)"], 300.0, f"{row['Algorithm']} MAE above threshold")
        print("[PASS] Treatment Cost Regression verified.")

if __name__ == "__main__":
    unittest.main()

