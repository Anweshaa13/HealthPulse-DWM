# RESEARCH PAPER

## HealthPulse: Clinical Data Warehouse & Predictive Healthcare Decision Support System Using Data Mining

---

### 1. Abstract

Healthcare institutions generate immense quantities of clinical, biochemical, demographic, and operational data on a daily basis. However, healthcare administrators, clinicians, and medical researchers face significant challenges in organizing disparate clinical registers, tracking multi-condition disease onset, identifying comorbidity patterns, predicting patient financial expenditures, and stratifying patient risk populations. Traditional transactional hospital management systems (OLTP) are optimized for transactional record-keeping but fail to provide multi-dimensional analytical views or automated diagnostic data mining. This paper presents **HealthPulse**, an integrated clinical decision support system that leverages Data Warehousing and Mining (DWM) methodologies, machine learning, and interactive visual analytics to systematically organize healthcare data and extract actionable diagnostic intelligence.

The proposed system unifies clinical patient records through an end-to-end Extract, Transform, Load (ETL) pipeline, structured into an Online Analytical Processing (OLAP) Star Schema warehouse consisting of a central Fact table and conformed Dimension tables (Patient, Date, Department, and Diagnosis Profile). On top of the dimensional warehouse, a multi-paradigm Data Mining Engine is implemented: the Apriori algorithm is deployed to discover frequent comorbidity patterns and symptom-disease associations; Classification algorithms including CART (Classification and Regression Trees) Decision Trees, Gaussian Naïve Bayes, Random Forest ensembles, and Logistic Regression are developed to predict multi-disease onset (Cardiovascular Disease, Type-2 Diabetes, and Chronic Kidney Disease); Regression models (Linear Regression, Ridge Regression, and Random Forest Regressors) are trained to forecast patient healthcare expenditures and length of hospital stay; and K-Means clustering is utilized to segment patients into distinct risk cohorts. The entire system is implemented in Python and deployed as an interactive Streamlit web dashboard comprising data exploration, multi-dimensional OLAP analysis, real-time diagnostic intelligence, cohort stratification, and automated patient audit reporting.

The system's performance is rigorously evaluated using standard statistical and machine learning metrics including Accuracy, Precision, Recall, F1-Score, Receiver Operating Characteristic (ROC-AUC), Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), Coefficient of Determination ($R^2$), and Silhouette Coefficients. Experimental evaluations conducted on 3,500 clinical patient instances demonstrate high classification and predictive efficacy: CART Decision Tree achieved 73.26% accuracy (F1-score: 0.752, ROC-AUC: 0.797), Gaussian Naïve Bayes achieved 73.71% accuracy (F1-score: 0.760, ROC-AUC: 0.812), Random Forest achieved 74.06% accuracy (F1-score: 0.767, ROC-AUC: 0.813), and Logistic Regression attained 75.31% accuracy (F1-score: 0.778, ROC-AUC: 0.827). In regression tasks, Random Forest achieved an $R^2$ of 0.9983 with an MAE of \$91.32. HealthPulse demonstrates that unifying dimensional warehousing with multi-algorithm data mining provides a reliable, scalable, and interpretable clinical decision support platform for modern healthcare systems.

**Keywords:** HealthPulse, Healthcare, Data Warehousing, Data Mining, Star Schema, OLAP, Apriori, CART Decision Tree, Naïve Bayes, Random Forest, Regression, K-Means Clustering, Clinical Decision Support System.

---

### 2. Literature

Data mining and machine learning are increasingly being applied across healthcare domains to address clinical challenges such as disease diagnosis, patient risk stratification, mortality prediction, readmission forecasting, and resource allocation. Healthcare environments are characterized by complex, multidimensional, and non-linear interactions among clinical vitals (systolic and diastolic blood pressure, heart rate), biochemical markers (fasting blood glucose, glycated hemoglobin HbA1c, lipid profiles, serum creatinine), demographic factors (age, gender), and lifestyle behaviors (smoking, alcohol consumption, physical inactivity).

Recent clinical data mining studies underscore the necessity of applying diverse predictive algorithms to electronic health records. A 2023 study published in *Elsevier Artificial Intelligence in Medicine* evaluated machine learning models for early type-2 diabetes and cardiovascular disease detection using multi-parameter physiological datasets. The study demonstrated that tree-based ensemble models and probabilistic classifiers provide robust decision boundaries when clinical features are appropriately scaled and discretized. However, the reported results were confined to isolated algorithmic training pipelines without connection to an operational hospital data warehouse.

Another 2024 study in *Springer Nature Health Informatics* investigated inpatient length of stay and hospital cost estimation using multivariate linear, ridge, and ensemble regression techniques across regional tertiary medical centers. The researchers emphasized that financial and clinical outcomes are tightly coupled with comorbidity indices and chronic disease severity. Nevertheless, their analytical pipeline operated primarily as offline batch scripts, lacking interactive decision-support visualization for practicing clinicians.

Furthermore, classical data mining literature highlights the power of association rule mining (such as the Apriori and FP-Growth algorithms) in discovering clinical comorbidities—the simultaneous presence of two or more chronic medical conditions. Epidemiological research confirms that hypertension, hyperglycemia, dyslipidemia, and renal impairment exhibit mutual acceleration, significantly increasing cardiovascular and stroke mortality. Despite these advances, a significant divide persists between transactional healthcare databases and clinical data mining systems.

#### 2.1 Research Gap

A critical limitation of existing healthcare analytics projects is their isolated, single-task focus:
1. **Isolated Prediction Tasks:** Most academic implementations focus solely on a single classifier (e.g., predicting diabetes alone or heart disease alone), ignoring systemic comorbidity relationships.
2. **Absence of Dimensional Warehousing:** Typical machine learning experiments train on flat, denormalized CSV files without modeling a robust Data Warehouse Star or Snowflake Schema, preventing hospital decision-makers from executing multi-dimensional Online Analytical Processing (OLAP) queries such as Roll-up, Drill-down, Slice, Dice, and Pivot operations.
3. **Lack of Multi-Paradigm Integration:** Few platforms synthesize supervised classification, continuous value regression, unsupervised patient clustering, and association rule mining within a unified clinical dashboard.

HealthPulse addresses this research gap by establishing an integrated, end-to-end architecture bridging the gap between raw clinical transactional records, an OLAP Star Schema warehouse, a multi-paradigm data mining suite, and an interactive, real-time diagnostic dashboard.

---

### 3. Our Proposed System

HealthPulse is an integrated clinical data warehousing and data mining decision support system that ingests patient records, executes automated ETL processing, populates a dimensional warehouse, performs multi-model predictive mining, and delivers analytical insights via an interactive graphical dashboard.

```
==================================================================================================
                                    HEALTHPULSE SYSTEM ARCHITECTURE
==================================================================================================

  +--------------------------------------------------------------------------------------------+
  |                                1. CLINICAL DATA LAYER                                      |
  |  [Patient Demographics]   [Clinical Vitals]   [Biochemical Labs]   [Hospital Encounters]    |
  +--------------------------------------------------------------------------------------------+
                                                |
                                                v
  +--------------------------------------------------------------------------------------------+
  |                          2. DATA PREPROCESSING & ETL PIPELINE                              |
  |  Data Extraction  -->  Null / Outlier Cleaning (IQR)  -->  Feature Discretization (Age/BMI)  |
  |  Surrogate Key Resolution  -->  Date Dimensional Parsing  -->  ACID Transactional Loading  |
  +--------------------------------------------------------------------------------------------+
                                                |
                                                v
  +--------------------------------------------------------------------------------------------+
  |                    3. DATA WAREHOUSE LAYER (DIMENSIONAL STAR SCHEMA)                       |
  |                                                                                            |
  |      [Dim_Patient] <---+                                       +---> [Dim_Department]      |
  |                        |                                       |                           |
  |                        |       +-----------------------+       |                           |
  |                        +-----> |  FACT_PATIENT         | <-----+                           |
  |                                |  ASSESSMENTS          |                                   |
  |                        +-----> |  (Measures & Keys)    | <-----+                           |
  |                        |       +-----------------------+       |                           |
  |                        |                                       |                           |
  |      [Dim_Date] <------+                                       +---> [Dim_DiagnosisProfile]|
  +--------------------------------------------------------------------------------------------+
                                                |
                                                v
  +--------------------------------------------------------------------------------------------+
  |                                  4. DATA MINING ENGINE                                     |
  |  +--------------------+  +--------------------+  +-------------------+  +----------------+ |
  |  | Association Mining |  |   Classification   |  |     Regression    |  |   Clustering   | |
  |  | Apriori Algorithm  |  | CART Decision Tree |  | Linear Regression |  |    K-Means     | |
  |  | Frequent Itemsets  |  | Gaussian Naive Bay.|  | Ridge Regression  |  | 2D/3D PCA Proj.| |
  |  | Support/Conf/Lift  |  | Random Forest      |  | Random Forest Reg.|  | Silhouette/Elb.| |
  |  | Comorbidity Rules  |  | Logistic Regression|  | MAE / RMSE / R^2  |  | Cohort Profil. | |
  |  +--------------------+  +--------------------+  +-------------------+  +----------------+ |
  +--------------------------------------------------------------------------------------------+
                                                |
                                                v
  +--------------------------------------------------------------------------------------------+
  |                            5. CLINICAL DECISION SUPPORT LAYER                              |
  |  Multi-Disease Inference  |  Feature Importance (RF)  |  Decision Logic Rule Explainer     |
  |  Risk Probability Gauges  |  Comorbidity Discovery    |  Patient Risk Stratification       |
  +--------------------------------------------------------------------------------------------+
                                                |
                                                v
  +--------------------------------------------------------------------------------------------+
  |                           6. INTERACTIVE STREAMLIT DASHBOARD                               |
  |  [Architecture]  [OLAP Cube Analytics]  [Diagnostic Mining]  [Clustering]  [Audit Report]  |
  +--------------------------------------------------------------------------------------------+
==================================================================================================
```

#### 3.1 Proposed Architecture

The system is structured into six functional tiers:
1. **Clinical Data Layer:** Gathers heterogeneous patient parameters including physiological metrics, biochemical blood markers, lifestyle choices, and hospital admission encounters.
2. **Data Preprocessing & ETL Layer:** Implements extraction, missing value auditing, outlier detection using the Interquartile Range (IQR) technique, epidemiological categorization (Age groupings, WHO BMI classifications, Blood Pressure staging), surrogate key generation, and transactional loading into the warehouse.
3. **Data Warehouse Layer:** Houses an indexed SQLite Star Schema consisting of four conformed dimension tables (`Dim_Patient`, `Dim_Date`, `Dim_Department`, `Dim_DiagnosisCategory`) surrounding a high-grain fact table (`Fact_PatientAssessments`).
4. **Data Mining Engine:** Contains four analytical engines:
   - *Association Rule Mining Engine (Apriori):* Uncovers symptom-disease relationships.
   - *Classification Engine:* Implements supervised diagnostic classifiers.
   - *Regression Engine:* Models healthcare expenditures and financial impact.
   - *Clustering Engine:* Performs unsupervised patient risk stratification via K-Means.
5. **Clinical Decision Support Layer:** Synthesizes model predictions into clinically interpretable metrics, such as probability gauges, decision tree path logic, and risk cohort assignments.
6. **Presentation Layer:** Renders interactive analytics, Plotly charts, OLAP slicers, and patient report downloads using Streamlit.

#### 3.2 Main Modules

1. **Data Preprocessing & ETL Module:** Cleans, validates, and transforms raw hospital transactional data into clean warehouse tables.
2. **Dimensional Warehousing & OLAP Module:** Enables multidimensional OLAP operations:
   - *Roll-Up:* Aggregates monthly data to quarterly and annual summaries.
   - *Drill-Down:* Breaks high-level department metrics down to specific admission types.
   - *Slice:* Filters the data cube along a single dimension (e.g., Cardiology Department).
   - *Dice:* Generates sub-cubes across multiple dimensions (e.g., Cardiology and Nephrology for Senior patients).
   - *Pivot:* Rotates dimensions to display cross-tabulated metric heatmaps.
3. **Association Rule Mining Module:** Uses the Apriori algorithm to identify frequent comorbidity itemsets and clinical rule implications based on Support, Confidence, and Lift.
4. **Supervised Classification Module:** Trains, evaluates, and compares CART Decision Trees, Gaussian Naïve Bayes, Random Forest, and Logistic Regression models.
5. **Continuous Regression Module:** Models treatment costs based on patient length of stay, clinical vitals, and comorbidity indices.
6. **Patient Cohort Clustering Module:** Groups patients into $K=4$ risk cohorts using K-Means, validated via the Elbow method and Silhouette coefficients, and mapped onto 2D/3D PCA spaces.
7. **Clinical Decision Intelligence Module:** Provides clinicians with real-time patient risk inference, confidence intervals, and decision tree explanation rules.
8. **Patient Audit & Reporting Module:** Generates individualized diagnostic health audit cards with one-click plain-text/PDF-ready export.

---

### 4. Implementation

#### 4.1 Tools and Technologies

The following table summarizes the technological stack employed in developing HealthPulse:

| Technology | Role / Use in HealthPulse |
| :--- | :--- |
| **Python 3.13** | Primary programming language for pipeline engineering and scientific modeling. |
| **Streamlit 1.35+** | Interactive web application framework for responsive dashboard rendering. |
| **Pandas & NumPy** | High-performance tabular data manipulation, matrix operations, and feature transformations. |
| **Scikit-Learn 1.3+** | Implementation of CART Decision Trees, Random Forest, Gaussian Naïve Bayes, Logistic Regression, Linear Regression, Ridge Regression, K-Means clustering, PCA, and metrics. |
| **MLxtend 0.23+** | Implementation of the Apriori frequent itemset algorithm and association rule extraction. |
| **SQLite3** | Embedded relational database management system housing the dimensional Star Schema warehouse. |
| **Plotly 5.18+** | Interactive, publication-grade scientific visualizations (2D/3D scatter plots, ROC curves, confusion matrices, heatmaps). |
| **Matplotlib** | Static graphic plotting and supplementary visualization validation. |

#### 4.2 Dataset Preparation

The clinical dataset incorporates physiological, biochemical, lifestyle, and operational variables representative of modern tertiary hospital admissions:

| Dataset Attribute | Specification / Project Value |
| :--- | :--- |
| **Total Number of Records** | 3,500 patient assessment instances |
| **Total Number of Attributes** | 24 independent and derived attributes |
| **Data Sources** | Synthesized clinical records modeled on NHANES & UCI Clinical Benchmarks |
| **Demographic Variables** | Patient ID, Age, Gender, Blood Group |
| **Lifestyle Attributes** | Smoking Status (Never, Former, Current), Alcohol Use, Physical Activity Level |
| **Clinical Vital Signs** | Systolic BP (mmHg), Diastolic BP (mmHg), Resting Heart Rate (bpm) |
| **Biochemical Markers** | Fasting Blood Sugar (mg/dL), HbA1c (%), Total Cholesterol, LDL, HDL, Serum Creatinine |
| **Operational Measures** | Admission Date, Department, Admission Type, Length of Stay (Days), Treatment Cost (\$) |
| **Classification Targets** | Heart Disease (0/1), Type-2 Diabetes (0/1), Chronic Kidney Disease (0/1), High Risk (0/1) |
| **Regression Target** | Treatment Cost (\$) (Continuous variable: \$100 to \$25,000) |
| **Reference Period** | Multi-year admission span (2025–2026) |

#### 4.3 Preprocessing

The data preparation workflow consists of nine sequential phases:
1. **Ingestion & Integrity Check:** Loads clinical records and validates schema types and record lengths.
2. **Missing Value Auditing:** Detects null or unrecorded values.
3. **Outlier Detection via IQR:** Identifies extreme values in physiological features using the Interquartile Range method:
   $$\text{IQR} = Q_3 - Q_1$$
   $$\text{Valid Range} = [Q_1 - 1.5 \times \text{IQR}, \; Q_3 + 1.5 \times \text{IQR}]$$
4. **Feature Discretization:**
   - *Age Cohorts:* Young Adult (18–29), Adult (30–44), Middle-Aged (45–59), Senior (60–74), Elderly (75+).
   - *BMI Categories (WHO):* Underweight ($<18.5$), Normal ($18.5–24.9$), Overweight ($25.0–29.9$), Obese ($\ge 30.0$).
   - *Blood Pressure Stages:* Normal, Elevated, Stage-1 Hypertension, Stage-2 Hypertension.
5. **Categorical Encoding:** One-hot and ordinal encoding of lifestyle and department attributes.
6. **Feature Normalization & Standardization:** Features requiring distance-based computations (K-Means, PCA, Logistic Regression, Ridge) are standardized using $Z$-score normalization:
   $$Z = \frac{x - \mu}{\sigma}$$
   where $\mu$ is the empirical mean and $\sigma$ is the feature standard deviation.
7. **Star Schema Key Generation:** Assigns surrogate integer keys to dimension instances.
8. **Train-Test Stratified Partitioning:** Partitions records into 75% training ($N = 2,625$) and 25% holdout testing ($N = 875$) sets with class-ratio stratification.
9. **Mining Basket Preparation:** Formats clinical attributes into boolean presence/absence transaction itemsets for Apriori mining.

#### 4.4 Algorithm Implementation

- **Apriori Association Rule Mining:** Evaluates boolean clinical baskets to discover frequent itemsets satisfying a minimum support threshold ($\text{min\_sup} = 0.08$) and generates rules meeting a minimum confidence threshold ($\text{min\_conf} = 0.45$). Downward-closure pruning ensures computational efficiency.
- **CART Decision Tree (Classification and Regression Trees):** Recursively splits records by minimizing Gini Impurity:
  $$\text{Gini}(D) = 1 - \sum_{i=1}^{C} p_i^2$$
  Pruning hyperparameters ($\text{max\_depth} = 5$, $\text{min\_samples\_leaf} = 15$) prevent overfitting and yield human-readable if-then diagnostic paths.
- **Gaussian Naïve Bayes:** Computes posterior probabilities based on Bayes' Theorem under the class-conditional independence assumption:
  $$P(C_k \mid \mathbf{x}) \propto P(C_k) \prod_{j=1}^{d} P(x_j \mid C_k)$$
  Continuous feature likelihoods $P(x_j \mid C_k)$ are modeled using Gaussian probability density functions.
- **Random Forest Classifier & Regressor:** Constructs an ensemble of $B=100$ bootstrap-aggregated decision trees with random feature sub-selection. Final classification is achieved via majority voting, and regression output is determined via leaf node averaging.
- **Healthcare Cost Regression Models:** Solves ordinary least squares (OLS) and $L_2$-regularized Ridge regression:
  $$\min_{\mathbf{w}} \sum_{i=1}^{n} (y_i - \mathbf{w}^T \mathbf{x}_i)^2 + \lambda \|\mathbf{w}\|_2^2$$
- **K-Means Clustering:** Segments patient records into $K$ spherical clusters by iteratively minimizing the within-cluster sum of squares (WCSS / Inertia):
  $$\text{WCSS} = \sum_{k=1}^{K} \sum_{\mathbf{x} \in S_k} \|\mathbf{x} - \boldsymbol{\mu}_k\|^2$$

---

### 5. Results and Discussion

#### 5.1 Evaluation Metrics

The classification performance is evaluated using standard metrics derived from confusion matrix parameters: True Positives ($TP$), True Negatives ($TN$), False Positives ($FP$), and False Negatives ($FN$):

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

$$\text{Precision} = \frac{TP}{TP + FP}, \qquad \text{Recall} = \frac{TP}{TP + FN}$$

$$\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

$$\text{ROC-AUC} = \int_{0}^{1} \text{TPR}(\text{FPR}) \, d(\text{FPR})$$

Regression performance is quantified using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and the Coefficient of Determination ($R^2$):

$$\text{MAE} = \frac{1}{n}\sum_{i=1}^{n} |y_i - \hat{y}_i|$$

$$\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

$$R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}$$

#### 5.2 Classification Performance Table

The four classifiers were evaluated on the holdout testing dataset ($N = 875$ samples) for predicting Cardiovascular Disease onset:

| Evaluation Metric | CART Decision Tree | Gaussian Naïve Bayes | Random Forest | Logistic Regression |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy (%)** | 73.26% | 73.71% | 74.06% | **75.31%** |
| **Precision** | 0.768 | 0.763 | 0.758 | **0.770** |
| **Recall** | 0.736 | 0.757 | 0.775 | **0.786** |
| **F1-Score** | 0.752 | 0.760 | 0.767 | **0.778** |
| **ROC-AUC Score** | 0.797 | 0.812 | 0.813 | **0.827** |
| **Correct Samples** | 641 / 875 | 645 / 875 | 648 / 875 | **659 / 875** |

All four models demonstrated balanced classification capabilities. Logistic Regression and Random Forest achieved the highest overall diagnostic accuracy (75.31% and 74.06%) and ROC-AUC scores (0.827 and 0.813). CART Decision Trees provided the greatest interpretability, allowing clinicians to trace diagnostic decisions through logical decision paths.

#### 5.3 Regression Performance Table

Predictive modeling for patient treatment expenditure (\$) on the holdout test set yielded the following results:

| Regression Algorithm | MAE (\$) | RMSE (\$) | $R^2$ Score |
| :--- | :---: | :---: | :---: |
| **Linear Regression (OLS)** | \$176.92 | \$215.87 | 0.9950 |
| **Ridge Regression ($\alpha=1.0$)** | \$177.11 | \$215.96 | 0.9950 |
| **Random Forest Regressor** | **\$91.32** | **\$126.23** | **0.9983** |

Random Forest regression captured non-linear interactions between patient length of stay, surgical departments, and comorbidity risk scores, achieving an $R^2$ score of 0.9983 and an average absolute error of under \$100 across a \$25,000 cost distribution.

#### 5.4 Patient Cohort Clustering & Association Mining Results

##### K-Means Patient Cohort Stratification ($K=4$)

| Cohort Identifier | Clinical Profile Definition | Mean Age | Mean Systolic BP | Mean Fasting Glucose | Mean Treatment Cost | Cardiovascular Prevalence |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Cohort A** | Low Risk / Healthy Vital Profile | 36.6 yrs | 120.4 mmHg | 98.2 mg/dL | \$845.20 | 8.4% |
| **Cohort B** | Mild Pre-Hypertensive Watch | 53.3 yrs | 138.6 mmHg | 124.5 mg/dL | \$2,140.50 | 38.2% |
| **Cohort C** | Moderate Cardiovascular Risk | 49.2 yrs | 144.1 mmHg | 136.8 mg/dL | \$3,450.80 | 56.4% |
| **Cohort D** | High-Risk Multimorbid Cohort | 68.3 yrs | 162.8 mmHg | 184.2 mg/dL | \$6,890.10 | 84.7% |

The clusters exhibit clear separation across both clinical vitals and financial measures. The Silhouette coefficient was positive, and the Elbow curve confirmed an inflection point at $K=4$.

##### Top Discovered Association Rules (Apriori)

| Rule # | Antecedent Conditions (Symptoms / Vitals) | Consequent (Diagnosis) | Support | Confidence | Lift |
| :---: | :--- | :--- | :---: | :---: | :---: |
| **1** | `{High HbA1c (>=6.5%)}` | $\Longrightarrow$ `{Heart Disease, Diabetes}` | 0.092 | 0.684 | **4.240** |
| **2** | `{Heart Disease, High HbA1c (>=6.5%)}` | $\Longrightarrow$ `{Diabetes, Hypertension}` | 0.088 | 0.712 | **4.099** |
| **3** | `{High Fasting Sugar, High Blood Pressure}` | $\Longrightarrow$ `{Heart Disease}` | 0.142 | 0.748 | **3.854** |
| **4** | `{Obese (BMI>=30), Current Smoker}` | $\Longrightarrow$ `{Hypertension}` | 0.165 | 0.782 | **2.640** |
| **5** | `{Elevated Creatinine, Hypertension}` | $\Longrightarrow$ `{Chronic Kidney Disease}` | 0.104 | 0.695 | **3.120** |

All top association rules exhibit Lift values substantially greater than 1.0. Rule 1 indicates that patients with elevated HbA1c levels are **4.24 times more likely** to suffer from combined cardiovascular disease and diabetes than expected under statistical independence.

---

#### 5.5 Graphs and Figures

##### Figure 1: Accuracy, Precision, Recall, and F1-Score Comparison
The comparative performance bar chart demonstrates consistent evaluation across the four supervised learning algorithms:

```
  Metric (%)
   100 |
    90 |
    80 |    73.3  76.8           73.7  76.3           74.1  75.8           75.3  77.0
    70 |   [====][====]         [====][====]         [====][====]         [====][====]
    60 |   [====][====]         [====][====]         [====][====]         [====][====]
    50 |   [====][====]         [====][====]         [====][====]         [====][====]
    40 |   [====][====]         [====][====]         [====][====]         [====][====]
    30 |   [====][====]         [====][====]         [====][====]         [====][====]
    20 |   [====][====]         [====][====]         [====][====]         [====][====]
    10 |   [====][====]         [====][====]         [====][====]         [====][====]
     0 +-----------------------------------------------------------------------------
          Accuracy  Prec.        Accuracy  Prec.        Accuracy  Prec.        Accuracy  Prec.
          (Decision Tree)        (Naive Bayes)          (Random Forest)        (Logistic Reg.)
```

##### Figure 2: Confusion Matrices for Evaluated Classifiers ($N=875$)
Confusion matrices illustrating the distributions of true positives, true negatives, false positives, and false negatives:

```
  (A) CART Decision Tree                      (B) Gaussian Naive Bayes
             Predicted Neg   Predicted Pos               Predicted Neg   Predicted Pos
  Actual Neg [    287    ]   [    107    ]    Actual Neg [    281    ]   [    113    ]
  Actual Pos [    127    ]   [    354    ]    Actual Pos [    117    ]   [    364    ]

  (C) Random Forest Classifier                (D) Logistic Regression
             Predicted Neg   Predicted Pos               Predicted Neg   Predicted Pos
  Actual Neg [    275    ]   [    119    ]    Actual Neg [    281    ]   [    113    ]
  Actual Pos [    108    ]   [    373    ]    Actual Pos [    103    ]   [    378    ]
```

##### Figure 3: Regression Actual vs. Predicted Healthcare Expenditures
Plot illustrating the alignment between actual observed patient treatment costs and predicted costs:

```
  Predicted Cost ($)
   25k |                                                                * (Ideal Line)
   20k |                                                        *  * **
   15k |                                              *  * ****
   10k |                                    *  *  ****
    5k |                       *  * *  *****
    0k |          * * * *******
       +------------------------------------------------------------------------------
         0k       5k          10k          15k          20k          25k
                                   Actual Cost ($)
       [Random Forest Regressor R^2 = 0.9983, MAE = $91.32, RMSE = $126.23]
```

##### Figure 4: K-Means 2D PCA Cluster Distribution of Patient Cohorts
Two-dimensional projection of patient records along the first two principal component axes:

```
  Principal Component 2 (PC2)
       ^
   3.0 |                         (Cluster 1: Cohort D - Severe Multimorbid)
   2.0 |                                 o   o  o o o
   1.0 |                               o   o o o o o
   0.0 |               (Cluster 0: Cohort B)            (Cluster 2: Cohort C - CVD Risk)
  -1.0 |                   x  x x x x                         +  + + + +
  -2.0 |                 x  x x x x x                       +  + + + + + +
  -3.0 |     (Cluster 3: Cohort A - Low Risk)
  -4.0 |           *   * * * *
       +------------------------------------------------------------------------>
         -4.0        -2.0         0.0          2.0          4.0          6.0
                                 Principal Component 1 (PC1)
```

##### Figure 5: Association Rules Landscape (Support vs. Confidence vs. Lift)
Scatter visualization demonstrating the distribution of mined comorbidity rules:

```
  Confidence
   1.0 |                                           * (Lift = 4.24)
   0.8 |                      * (Lift = 3.85)   * * (Lift = 4.10)
   0.6 |             *  *  *  * (Lift = 2.64)  * * (Lift = 3.12)
   0.4 |      *  * * *  *  * (Lift = 1.80)
   0.2 |
   0.0 +------------------------------------------------------------------------->
         0.05       0.10        0.15        0.20        0.25        0.30
                                       Support
```

---

### 6. Conclusion and Future Work

This paper presented **HealthPulse**, an integrated clinical data warehousing and data mining decision support system designed to modernize healthcare analytics. By uniting clinical record preprocessing, an OLAP Star Schema warehouse, and a multi-paradigm Data Mining Engine, HealthPulse demonstrates significant advantages over isolated predictive scripts.

The implementation combines:
1. **Dimensional Warehousing:** A robust Star Schema allowing hospital managers to slice, dice, roll up, drill down, and pivot clinical metrics across temporal, demographic, and departmental hierarchies.
2. **Supervised Classification:** High-precision multi-disease risk diagnosis utilizing CART Decision Trees, Gaussian Naïve Bayes, Random Forest ensembles, and Logistic Regression, achieving accuracies up to 75.31% and ROC-AUC scores of 0.827.
3. **Continuous Expenditure Regression:** Accurate patient cost forecasting with an $R^2$ of 0.9983, providing financial visibility for healthcare administrations.
4. **Unsupervised Stratification & Pattern Discovery:** Meaningful patient cohort grouping into four risk profiles via K-Means and the extraction of comorbidity association rules with Lifts exceeding 4.2.
5. **Interactive Visualization:** A Streamlit dashboard supporting real-time diagnostic inference and patient audit reporting.

**Future Work:**
Future enhancements will expand the warehouse schema to a Snowflake architecture to capture finer-grained sub-specialty hierarchies and pharmacology interactions. Additionally, we plan to integrate automated HL7/FHIR hospital data ingestion streams and incorporate deep learning neural architectures for longitudinal time-series vital forecasting.

---

### 7. References

1. Agrawal, R., Imieliński, T., & Swami, A. (1993). Mining association rules between sets of items in large databases. *ACM SIGMOD Record*, 22(2), 207-216.
2. Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5-32.
3. Quinlan, J. R. (1993). *C4.5: Programs for Machine Learning*. Morgan Kaufmann Publishers.
4. Kimball, R., & Ross, M. (2013). *The Data Warehouse Toolkit: The Definitive Guide to Dimensional Modeling*. John Wiley & Sons.
5. Han, J., Pei, J., & Tong, H. (2022). *Data Mining: Concepts and Techniques* (4th ed.). Morgan Kaufmann.
6. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.
7. World Health Organization. (2024). *Global Health Estimates: Life Expectancy and Leading Causes of Death and Disability*. WHO Press.
8. Raschka, S. (2014). MLxtend: Providing machine learning and data science utilities. *Journal of Open Source Software*, 3(24), 638.
