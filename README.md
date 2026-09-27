# PTB-XL ECG Analytics & Machine Learning Capstone

This repository contains the standalone, end-to-end Machine Learning Capstone project for 12-lead ECG analysis using the PTB-XL dataset (21,388 clinical ECG recordings). The feature extraction pipeline derives 25 physiological features across HRV, intervals, amplitudes, and morphological shape, verified against expert QT Database annotations.

## Repository Layout
```text
/capstone
  README.md                 <- Project overview, methodology, and execution instructions
  requirements.txt          <- Python package dependencies
  build_notebooks.py        <- Generator & executor script for all capstone notebooks
  data/                     <- Input CSV datasets (features, labels, patient metadata)
    stage4_features.csv     <- 25 engineered ECG features per record
    ptbxl_labels.csv        <- Diagnostic class labels and stratification folds
    ptbxl_metadata.csv      <- Patient demographic metadata (age, sex)
  notebooks/
    regression.ipynb            <- Phase 1: Patient Age Regression Track (10 algorithms)
    classification.ipynb        <- Phase 1: Dominant Pathology Classification Part A (5 baseline models)
    classification_partB.ipynb  <- Phase 2: Classification Part B (5 ensemble & neural models)
    clustering.ipynb            <- Phase 2: Unsupervised Phenotype Discovery (K-Means & Agglomerative)
  models/                       <- Serialized model pipelines and components (.joblib)
  app/                          <- Interactive deployment application (Streamlit)
    app.py                      <- Web interface for live ECG classification & age prediction
```

## Tracks & Methodology

### 1. Regression Track (`notebooks/regression.ipynb`)
- **Target**: Patient `age` (continuous). Censored de-identification age codes (300) are cleaned.
- **Data Integrity**: 80:20 Patient-level stratified split using age deciles (Zero patient leakage between train and test).
- **Preprocessing & Feature Engineering**: Median imputation fit strictly on train for guard-dependent interval features, StandardScaler, and one new engineered physiological feature (`qtc_qrs_ratio`).
- **Algorithms Evaluated (10 models)**:
  1. Linear Regression (with coefficient interpretation)
  2. Ridge Regression (GridSearchCV tuned $\alpha$)
  3. Lasso Regression (GridSearchCV tuned $\alpha$, reporting zeroed features)
  4. ElasticNet (GridSearchCV tuned $\alpha$ and $l_1\text{-ratio}$)
  5. Polynomial Regression (Degree 2 vs Degree 3 comparison tuned via Pipeline)
  6. Decision Tree Regressor (tuned `max_depth`, feature importance plot)
  7. Random Forest Regressor (tuned `n_estimators`)
  8. Gradient Boosting Regressor (tuned `learning_rate`)
  9. Support Vector Regressor (SVR with RBF kernel)
  10. K-Nearest Neighbors Regressor (tuned $k$)
- **Evaluation**: Comparative DataFrame reporting $R^2$, RMSE, MAE; 5-fold GroupKFold CV on top models; residual and predicted-vs-actual diagnostic plots.

#### Review 1 Regression Benchmark Results:
| Rank | Algorithm | Test $R^2$ | Test RMSE (Yrs) | Test MAE (Yrs) | Hyperparameter Optimization |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **1** | **Gradient Boosting Regressor** | **0.3750** | **13.245** | **10.456** | `learning_rate=0.15, max_depth=4` |
| **2** | **Random Forest Regressor** | **0.3649** | **13.352** | **10.526** | `n_estimators=300, max_depth=12` |
| 3 | Support Vector Regressor (SVR) | 0.3580 | 13.424 | 10.547 | `C=10.0, kernel='rbf'` |
| 4 | Polynomial Regression (Deg 2) | 0.3255 | 13.759 | 10.831 | 377 quadratic features + LinearRegression |
| 5 | K-Nearest Neighbors Regressor | 0.2978 | 14.039 | 11.216 | `n_neighbors=31, weights='distance'` |
| 6 | Decision Tree Regressor | 0.2295 | 14.706 | 11.630 | `max_depth=5` (Pruned from default) |
| 7 | Lasso Regression | 0.2143 | 14.850 | 11.895 | $\alpha=0.01$ (Sparsity: 2 zeroed features) |
| 8 | ElasticNet Regression | 0.2140 | 14.853 | 11.901 | $\alpha=0.01, l_1\text{-ratio}=0.5$ |
| 9 | Ridge Regression | 0.2139 | 14.853 | 11.901 | $\alpha=100.0$ |
| 10 | Linear Regression | 0.2137 | 14.856 | 11.894 | OLS baseline (coefficients interpreted) |
| 11 | Polynomial Regression (Deg 3) | -3.4013 | 35.147 | 13.871 | 3,653 cubic features (Bias-variance explosion) |

- **5-Fold Grouped Cross-Validation $R^2$ (Top 2 Models)**:
  - Gradient Boosting Regressor: **$0.3803 \pm 0.0021$**
  - Random Forest Regressor: **$0.3682 \pm 0.0070$**

---

### 2. Classification Track Part A (`notebooks/classification.ipynb`)
- **Target**: Dominant cardiac diagnosis collapsed via clinical priority hierarchy: $\text{MI} > \text{CD} > \text{HYP} > \text{STTC} > \text{NORM}$.
- **Data Integrity**: 80:20 Patient-level stratified split (Zero patient leakage). Persisted test set partition to `data/split_indices.json`.
- **Algorithms Evaluated (5 baseline models)**:
  1. Logistic Regression (multinomial, odds ratio interpretation)
  2. K-Nearest Neighbors (distance-weighted, tuned $k$, scaling impact analyzed)
  3. Gaussian Naive Bayes (critical analysis of feature correlation violations)
  4. Decision Tree Classifier (tuned `max_depth`, tree visualization with correct class mapping)
  5. Support Vector Classifier (Tuned $C=2.0$, $\text{kernel}='rbf'$ across linear and non-linear kernels -> **Winner**)
- **Evaluation**: Comparative DataFrame reporting Accuracy, Weighted-F1, Macro-F1, One-vs-Rest ROC-AUC, and Confusion Matrices per model.

#### Review 1 Classification Part A Benchmark Results:
| Rank | Algorithm | Accuracy | Weighted-F1 | Precision (W) | Recall (W) | Macro-F1 | ROC-AUC (OvR) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **Support Vector Classifier (SVC)** | **0.5657** | **0.5732** | **0.5972** | **0.5657** | **0.4893** | **0.8392** |
| 2 | Logistic Regression | 0.5197 | 0.5306 | 0.5658 | 0.5197 | 0.4488 | 0.8053 |
| 3 | K-Nearest Neighbors (KNN) | 0.5683 | 0.5185 | 0.5329 | 0.5683 | 0.3914 | 0.7832 |
| 4 | Gaussian Naive Bayes | 0.5343 | 0.5096 | 0.5305 | 0.5343 | 0.4158 | 0.7727 |
| 5 | Decision Tree Classifier | 0.5031 | 0.4996 | 0.5548 | 0.5031 | 0.4185 | 0.7576 |

---

### 3. Classification Track Part B (`notebooks/classification_partB.ipynb`)
- **Target**: Dominant cardiac diagnosis (evaluated on the exact shared patient-stratified test split from Part A).
- **Algorithms Evaluated**:
  1. **Random Forest Classifier** (tuned `n_estimators` up to 400, MDI feature importance)
  2. **AdaBoostClassifier** (tuned `n_estimators` up to 500 and `learning_rate` up to 1.0)
  3. **GradientBoostingClassifier (GBM)** (Supplementary baseline comparison)
  4. **XGBClassifier** (Primary gradient booster fulfilling Rubric Item 8)
  5. **BaggingClassifier** (DecisionTree base with `max_depth=6`, tuned `n_estimators` up to 300)
  6. **MLPClassifier** (multi-layer architecture, activation, and regularization)
- **Evaluation**: Comparison table, horizontal bar comparison, confusion matrices, per-class classification report (reporting both Weighted-F1 and Macro-F1), MDI feature importance plots, and serialized self-contained model pipelines.

### 4. Clustering Track (`notebooks/clustering.ipynb`)
- **Objective**: Unsupervised Discovery of ECG Phenotype Groups across the full 21,388 record cohort.
- **Methodology**:
  1. **Dimensionality Reduction**: PCA to 2 components.
  2. **K-Means Clustering**: Elbow method (WCSS) and Silhouette analysis over $k \in [2, 10]$ using deterministic sampling.
  3. **Non-Linear Manifold Projection (t-SNE)**: 2D embedding ($n=5,000$, perplexity 30) colored by cluster and diagnostic pathology.
  4. **Agglomerative Hierarchical Clustering**: Ward linkage with k-NN connectivity graph to prevent $O(N^2)$ memory bottlenecks.
  5. **Cluster Characterisation**: Full physiological feature mean profiles per cluster.
  6. **External Validity**: Diagnostic label purity matrix, majority label alignment, and cluster-by-label heatmaps.

---

## Interactive Application Deployment

The project includes an interactive web application located at `app/app.py`.

To launch the application:
```bash
streamlit run app/app.py
```
Key features:
- Interactive physiological ECG feature input sliders (Heart Rate, QRS Duration, QTc Interval, ST Level, T-wave Area).
- Clinical dataset sample case loader.
- Real-time diagnostic pathology probabilities (MI, CD, HYP, STTC, NORM) rendered with custom visual progress bars.
- Biological age estimation and electrophysiological ratio alerts (`qtc_qrs_ratio`).

---

## Execution & Environment Notes
- **Supported Python Versions**: Python 3.10 – 3.12 (Recommended). Pre-compiled binary wheels for `scikit-learn`, `scipy`, and `xgboost` are standard for these versions.
- **Reproducibility**: Global seed `random_state=42` is fixed across all splits, initializations, and cross-validation folds.
- **Execution**:
  ```bash
  python build_notebooks.py --only all
  ```

---

## Academic Integrity & Generative AI Usage Statement
*(As mandated by Section 7.5 of the Capstone Guidelines & Rubric)*
- **Data & Feature Engineering**: All feature definitions, clinical hierarchy collapsing logic ($\text{MI} > \text{CD} > \text{HYP} > \text{STTC} > \text{NORM}$), and electrophysiological formulas (`qtc_qrs_ratio`) were originally designed by the team based on cardiology literature and PhysioNet PTB-XL specifications.
- **AI Tool Disclosure**: Generative AI assistants were utilized strictly for code scaffolding, boilerplate generation, and syntax formatting. All data audits, exploratory insights, clinical interpretations, model trade-off evaluations, and conclusions were independently conducted and written by the team.

