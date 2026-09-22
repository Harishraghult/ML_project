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

### 2. Classification Track Part A (`notebooks/classification.ipynb`)
- **Target**: Dominant cardiac diagnosis collapsed via clinical priority hierarchy: $\text{MI} > \text{CD} > \text{HYP} > \text{STTC} > \text{NORM}$.
- **Data Integrity**: 80:20 Patient-level stratified split (Zero patient leakage). Persisted test set partition to `data/split_indices.json`.
- **Algorithms Evaluated (5 baseline models)**:
  1. Logistic Regression (multinomial, odds ratio interpretation)
  2. K-Nearest Neighbors (distance-weighted, tuned $k$)
  3. Gaussian Naive Bayes (critical analysis of feature correlation violations)
  4. Decision Tree Classifier (tuned `max_depth`, tree visualization with correct class name mapping)
  5. Support Vector Classifier (RBF kernel, tuned $C$ -> **Winner: Weighted-F1 $\approx 0.567$, Accuracy $\approx 0.558$**)
- **Evaluation**: Comparative DataFrame reporting Accuracy, Weighted-F1, Macro-F1, Confusion Matrices per model.

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

The project includes a Streamlit web application located at [`app/app.py`](file:///e:/ML_PROJECT/capstone/app/app.py).

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
- **Execution**:
  ```bash
  python build_notebooks.py --only all
  ```
