# PTB-XL ECG Analytics & Machine Learning Capstone

This repository contains the standalone, end-to-end Machine Learning Capstone project for 12-lead ECG analysis using the PTB-XL dataset (21,388 clinical ECG recordings). The feature extraction pipeline derives 25 physiological features across HRV, intervals, amplitudes, and morphological shape, verified against expert QT Database annotations.

## Repository Layout
```text
/capstone
  README.md                 <- Project overview, methodology, and execution instructions
  requirements.txt          <- Python package dependencies
  data/                     <- Input CSV datasets (features, labels, patient metadata)
    stage4_features.csv     <- 25 engineered ECG features per record
    ptbxl_labels.csv        <- Diagnostic class labels and stratification folds
    ptbxl_metadata.csv      <- Patient demographic metadata (age, sex)
  notebooks/
    regression.ipynb            <- Phase 1: Patient Age Regression Track (10 algorithms)
    classification.ipynb        <- Phase 1: Dominant Pathology Classification Part A (5 baseline models)
    classification_partB.ipynb  <- Phase 2: Classification Part B (5 ensemble & neural models)
    clustering.ipynb            <- Phase 2: Unsupervised Phenotype Discovery (K-Means & Agglomerative)
  models/                       <- Serialized model weights and scalers (.joblib)
  app/                          <- Interactive deployment application (Streamlit / Gradio)
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
  5. Polynomial Regression (Degree 2 vs Degree 3 comparison)
  6. Decision Tree Regressor (tuned `max_depth`, feature importance plot)
  7. Random Forest Regressor (tuned `n_estimators`)
  8. Gradient Boosting Regressor (tuned `learning_rate` -> Runner-Up: $R^2 = 0.3676$, $\text{RMSE} = 13.297$ yrs, $\text{MAE} = 10.530$ yrs; highest 5-fold CV score of $0.3742 \pm 0.0319$)
  9. Support Vector Regressor (SVR with RBF kernel -> **Top Test Benchmark Winner: $R^2 = 0.3680$, $\text{RMSE} = 13.292$ yrs, $\text{MAE} = 10.485$ yrs**; narrowly leading on held-out test data in a virtual near-tie with Gradient Boosting)
  10. K-Nearest Neighbors Regressor (tuned $k$)
- **Evaluation**: Comparative DataFrame reporting $R^2$, RMSE, MAE; 5-fold CV on top 2 models; residual and predicted-vs-actual diagnostic plots.

### 2. Classification Track Part A (`notebooks/classification.ipynb`)
- **Target**: Dominant cardiac diagnosis collapsed via clinical priority hierarchy: $\text{MI} > \text{CD} > \text{HYP} > \text{STTC} > \text{NORM}$.
- **Data Integrity**: 80:20 Patient-level stratified split (Zero patient leakage).
- **Algorithms Evaluated (5 baseline models)**:
  1. Logistic Regression (multinomial, odds ratio interpretation)
  2. K-Nearest Neighbors (distance-weighted, tuned $k$)
  3. Gaussian Naive Bayes (critical analysis of feature correlation violations)
  4. Decision Tree Classifier (tuned `max_depth`, tree visualization)
  5. Support Vector Classifier (RBF kernel, tuned $C$ -> **Winner: Weighted-F1 $\approx 0.567$, Accuracy $\approx 0.603$**)
- **Evaluation**: Comparative DataFrame reporting Accuracy, Weighted-F1, Macro-F1, Confusion Matrices per model.

### 3. Classification Track Part B (`notebooks/classification_partB.ipynb`)
- **Target**: Dominant cardiac diagnosis (same patient-stratified 80:20 split, zero patient leakage).
- **Algorithms Evaluated**:
  1. **Random Forest Classifier** (tuned `n_estimators` up to 600, MDI feature importance)
  2. **AdaBoostClassifier** (tuned `n_estimators` up to 1100 and `learning_rate` up to 1.5)
  3. **GradientBoostingClassifier (GBM)** (Supplementary baseline comparison)
  4. **XGBClassifier** (Primary gradient booster fulfilling Rubric Item 8 -> **Winner: Weighted-F1 $\approx 0.6045$, Accuracy $\approx 0.6295$**)
  5. **BaggingClassifier** (DecisionTree base with `max_depth=6`, tuned `n_estimators` up to 500)
  6. **MLPClassifier** (deep architectures up to (256, 128, 64, 32), activation, and alpha)
- **Evaluation**: 6-row comparison table, horizontal bar comparison, 6-panel confusion matrices, per-class classification report, MDI feature importance plots (for Random Forest and XGBoost), and serialized model weights.

### 4. Clustering Track (`notebooks/clustering.ipynb`)
- **Objective**: Unsupervised Discovery of ECG Phenotype Groups across the full 21,388 record cohort.
- **Methodology**:
  1. **Dimensionality Reduction**: PCA to 2 components (PC1 explains 20.7%, PC2 explains 13.5%; 14 components required for 90% cumulative variance).
  2. **K-Means Clustering**: Elbow method (WCSS) and Silhouette analysis over $k \in [2, 10]$. Optimal $k = 4$ selected.
  3. **Non-Linear Manifold Projection (t-SNE)**: 2D embedding ($n=5,000$, perplexity 30) colored by cluster and diagnostic pathology.
  4. **Agglomerative Hierarchical Clustering**: Ward linkage with truncated dendrogram (top 30 merges on $n=3,000$ subsample), evaluated at $k=4$.
  5. **Cluster Characterisation**: Full physiological feature mean profiles per cluster.
  6. **External Validity**: Diagnostic label purity matrix, majority label alignment, and cluster-by-label heatmaps.
- **Validation Metrics**:
  - **K-Means ($k=4$)**: Silhouette = **0.1274**, Davies-Bouldin = **2.0596** (lower is better), Calinski-Harabasz = **2278.66** (higher is better), Overall Purity = **0.4556**
  - **Agglomerative ($k=4$)**: Silhouette = 0.0940, Davies-Bouldin = 2.3553, Calinski-Harabasz = 1688.93, Overall Purity = 0.4423

---

## Execution & Environment Notes
- **Windows / Python 3.14 Compatibility**: Running `build_notebooks.py` with `n_jobs=-1` on joblib-backed estimators (e.g., `GridSearchCV`, `RandomForestClassifier`) can emit harmless `joblib.externals.loky.backend.resource_tracker` `KeyError` tracebacks and `zmq`/`tornado` event loop warnings during temp-folder cleanup after parallel child processes terminate on Windows. These are purely cosmetic operating-system cleanup artifacts that do not affect model fitting, metric evaluation, or notebook execution; all notebook cells execute successfully with complete outputs.
