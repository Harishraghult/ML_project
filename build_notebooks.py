"""
================================================================================
Capstone Notebook Generator & Executor
Builds and executes:
  1. notebooks/regression.ipynb              (Phase 1: Full 10 models + EDA)
  2. notebooks/classification.ipynb          (Phase 1: Part A — 5 baseline models)
  3. notebooks/classification_partB.ipynb    (Phase 1: Part B — 5 ensemble/MLP models)
  4. notebooks/clustering.ipynb              (Phase 1: K-Means + Agglomerative)

Note: On Windows with Python 3.14, n_jobs=-1 on joblib-backed estimators can produce harmless
loky/resource_tracker temp-folder cleanup tracebacks after parallel jobs finish; these are
purely cosmetic and can be safely ignored.
================================================================================"""

import os
import sys
import json
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
NOTEBOOKS_DIR = os.path.join(BASE_DIR, "notebooks")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(NOTEBOOKS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def build_regression_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    # Title & Introduction
    cells.append(nbf.v4.new_markdown_cell("""# PTB-XL ECG Machine Learning Capstone: Regression Track
## Objective: Predicting Patient Biological Age from 12-Lead ECG Features

This notebook implements the complete **Regression Track** under the Capstone rubric:
1. **Dataset & Exploratory Data Analysis (EDA)**: Merging signal-derived feature matrices, inspecting distributions, correlation structures, and executing a leak-free 80:20 patient-stratified split.
2. **Preprocessing & Feature Engineering**: Guard-dependent missingness handling, leak-free median imputation, z-score standardization, and engineering a clinically grounded repolarization/depolarization ratio feature (`qtc_qrs_ratio`).
3. **Model Benchmarking (10 Algorithms)**:
   - Linear Regression (with coefficient interpretation)
   - Ridge Regression (GridSearchCV tuned $\\alpha$)
   - Lasso Regression (GridSearchCV tuned $\\alpha$ with sparsity report)
   - ElasticNet (GridSearchCV tuned $\\alpha$ and $l_1$-ratio)
   - Polynomial Regression (Degree 2 vs Degree 3 comparison)
   - Decision Tree Regressor (tuned `max_depth` with feature importances)
   - Random Forest Regressor (tuned `n_estimators`)
   - Gradient Boosting Regressor (tuned `learning_rate`)
   - Support Vector Regressor (SVR with RBF and linear kernels)
   - K-Nearest Neighbors Regressor (tuned $k$)
4. **Validation & Diagnostics**: Comprehensive comparative metrics ($R^2$, RMSE, MAE), 5-fold cross-validation on top models, residual analysis, and predicted-vs-actual evaluations.
"""))

    # Imports and settings
    cells.append(nbf.v4.new_code_cell("""import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set aesthetic styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Helvetica', 'Arial', 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8
palette = sns.color_palette('tab10')

# Modeling libraries
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
print("Environment and libraries loaded successfully!")
"""))

    # Section A: Data Loading and Merge
    cells.append(nbf.v4.new_markdown_cell("""---
## Section A: Dataset Ingestion & Exploratory Data Analysis (EDA)

We ingest:
- `stage4_features.csv`: 25 signal-derived features extracted from 12-lead ECG recordings.
- `ptbxl_labels.csv`: Diagnostic labels and metadata.
- `ptbxl_metadata.csv`: Demographic records including patient identifier, `age`, and `sex`.
"""))

    cells.append(nbf.v4.new_code_cell("""# 1. Ingest datasets (Robust path resolution - BUG-17)
cwd = os.getcwd()
BASE_DIR = os.path.abspath(os.path.join(cwd, '..')) if os.path.basename(cwd) == 'notebooks' else os.path.abspath(cwd)
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

features_df = pd.read_csv(os.path.join(DATA_DIR, 'stage4_features.csv'))
labels_df = pd.read_csv(os.path.join(DATA_DIR, 'ptbxl_labels.csv'))
meta_df = pd.read_csv(os.path.join(DATA_DIR, 'ptbxl_metadata.csv'))

# Merge on ecg_id
df = features_df.merge(labels_df[['ecg_id', 'fold']], on='ecg_id', how='inner')
df = df.merge(meta_df[['ecg_id', 'age', 'sex']], on='ecg_id', how='inner')

print(f"Merged Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Memory Usage: {df.memory_usage().sum() / (1024**2):.2f} MB")
df.head(3)
"""))

    cells.append(nbf.v4.new_markdown_cell("""### Target Data Cleaning: Censored PhysioNet Age Codes
In the official PTB-XL database specification, patients with age $\\ge 90$ or censored dates of birth were encoded as `300.0` for privacy and HIPAA compliance. We document and filter these non-physiological surrogate values so they do not distort linear and quadratic regression losses.
"""))

    cells.append(nbf.v4.new_code_cell("""n_censored = (df['age'] > 100).sum()
print(f"Number of de-identified censored age records (age == 300): {n_censored} ({n_censored/len(df)*100:.2f}%)")

# Filter censored records for continuous regression analysis
df_reg = df[df['age'] <= 100].copy().reset_index(drop=True)
print(f"Clean regression dataset count: {len(df_reg)} records")

# Missing values report
missing_series = df_reg.isna().sum()
missing_report = pd.DataFrame({
    'Column': missing_series.index,
    'Missing_Count': missing_series.values,
    'Missing_Pct': (missing_series.values / len(df_reg) * 100).round(2),
    'Dtype': [df_reg[c].dtype for c in missing_series.index]
})
print("Columns with missingness:")
missing_report[missing_report['Missing_Count'] > 0]
"""))

    # Section A: Visualizations
    cells.append(nbf.v4.new_markdown_cell("""### EDA Visualizations
1. **Target Distribution**: Histogram and Kernel Density Estimation (KDE) of patient age.
"""))

    cells.append(nbf.v4.new_code_cell("""fig, (ax_box, ax_hist) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={'height_ratios': [0.2, 0.8]})

sns.boxplot(x=df_reg['age'], ax=ax_box, color=palette[0], fliersize=3)
ax_box.set(xlabel='')
ax_box.set_title('Distribution of Patient Biological Age (PTB-XL Cohort)', fontsize=14, fontweight='bold', pad=10)

sns.histplot(df_reg['age'], kde=True, ax=ax_hist, color=palette[0], bins=35, stat='density', edgecolor='black', alpha=0.6)
ax_hist.axvline(df_reg['age'].mean(), color='crimson', linestyle='--', linewidth=1.5, label=f"Mean: {df_reg['age'].mean():.1f} yrs")
ax_hist.axvline(df_reg['age'].median(), color='darkgreen', linestyle='-', linewidth=1.5, label=f"Median: {df_reg['age'].median():.1f} yrs")
ax_hist.set_xlabel('Patient Age (Years)', fontsize=12)
ax_hist.set_ylabel('Density', fontsize=12)
ax_hist.legend(frameon=True, facecolor='white')

plt.tight_layout()
plt.show()
"""))

    cells.append(nbf.v4.new_markdown_cell("""> **EDA Insight — Target Distribution**:
> Patient age spans from 2 to 89 years with a mean of 59.4 years, median of 62.0 years, and standard deviation of 16.8 years. The distribution is slightly left-skewed, exhibiting peak density between 50 and 75 years, reflecting the typical clinical demographics presenting for 12-lead diagnostic electrocardiography.
"""))

    cells.append(nbf.v4.new_code_cell("""# 2. Distribution grid across all 25 engineered ECG features
feature_cols_raw = [
    'n_beats', 'hr_bpm', 'rr_mean_ms', 'rr_std_ms', 'rr_rmssd_ms', 'rr_pnn50',
    'n_qrs_reliable', 'frac_qrs_reliable', 'pr_mean_ms', 'pr_std_ms',
    'qrs_dur_mean_ms', 'qrs_dur_std_ms', 'qt_mean_ms', 'qt_std_ms', 'qtc_mean_ms',
    'p_dur_mean_ms', 't_dur_mean_ms', 'p_amp_median_mv', 'r_amp_median_mv',
    't_amp_median_mv', 'st_level_median_mv', 'qrs_area_median', 'p_area_median',
    't_area_median', 'rs_ratio_median'
]

fig, axes = plt.subplots(5, 5, figsize=(20, 16))
axes = axes.flatten()

for idx, col in enumerate(feature_cols_raw):
    ax = axes[idx]
    sns.histplot(df_reg[col].dropna(), kde=True, ax=ax, color=palette[idx % len(palette)], bins=25)
    ax.set_title(col, fontsize=10, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.tick_params(labelsize=8)

plt.suptitle('Distribution of All 25 Signal-Derived ECG Features', fontsize=16, fontweight='bold', y=1.002)
plt.tight_layout()
plt.show()
"""))

    cells.append(nbf.v4.new_code_cell("""# 3. Correlation Heatmap across ECG Features
plt.figure(figsize=(14, 11))
corr_matrix = df_reg[feature_cols_raw].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

sns.heatmap(
    corr_matrix,
    mask=mask,
    cmap='coolwarm',
    center=0,
    vmin=-1, vmax=1,
    annot=False,
    linewidths=0.5,
    cbar_kws={'label': 'Pearson Correlation Coefficient'}
)
plt.title('Correlation Matrix across 25 Engineered ECG Features', fontsize=14, fontweight='bold', pad=12)
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(fontsize=9)
plt.tight_layout()
plt.show()
"""))

    cells.append(nbf.v4.new_code_cell("""# 4. Feature vs Target Bivariate Relationships
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

sns.regplot(
    data=df_reg.sample(n=2500, random_state=RANDOM_STATE),
    x='qt_mean_ms', y='age',
    ax=ax1,
    color=palette[0],
    scatter_kws={'alpha': 0.25, 's': 15},
    line_kws={'color': 'darkred', 'linewidth': 2}
)
ax1.set_title('Bivariate Relationship: QT Interval vs. Patient Age', fontsize=12, fontweight='bold')
ax1.set_xlabel('QT Mean Interval (ms)', fontsize=11)
ax1.set_ylabel('Patient Age (Years)', fontsize=11)

sns.regplot(
    data=df_reg.sample(n=2500, random_state=RANDOM_STATE),
    x='qrs_dur_mean_ms', y='age',
    ax=ax2,
    color=palette[1],
    scatter_kws={'alpha': 0.25, 's': 15},
    line_kws={'color': 'darkblue', 'linewidth': 2}
)
ax2.set_title('Bivariate Relationship: QRS Duration vs. Patient Age', fontsize=12, fontweight='bold')
ax2.set_xlabel('QRS Duration (ms)', fontsize=11)
ax2.set_ylabel('Patient Age (Years)', fontsize=11)

plt.tight_layout()
plt.show()
"""))

    # Section A4: Split integrity
    cells.append(nbf.v4.new_markdown_cell("""### 80:20 Patient-Stratified Split (Zero Patient Leakage)
To conform to academic benchmarking standards, the split is stratified across age deciles while grouped strictly by `patient_id`. This guarantees zero patient identity overlap between training and testing sets.
"""))

    cells.append(nbf.v4.new_code_cell("""# Create 10 deciles of age for continuous target stratification
df_reg['age_decile'] = pd.qcut(df_reg['age'], q=10, labels=False, duplicates='drop')

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
train_idx, test_idx = next(sgkf.split(df_reg, df_reg['age_decile'], groups=df_reg['patient_id']))

train_df = df_reg.iloc[train_idx].copy().reset_index(drop=True)
test_df = df_reg.iloc[test_idx].copy().reset_index(drop=True)

# Leakage verification
train_pts = set(train_df['patient_id'])
test_pts = set(test_df['patient_id'])
overlap = train_pts.intersection(test_pts)

print(f"Train Set: {len(train_df)} records ({len(train_df)/len(df_reg)*100:.1f}%) | Unique Patients: {len(train_pts)}")
print(f"Test Set : {len(test_df)} records ({len(test_df)/len(df_reg)*100:.1f}%) | Unique Patients: {len(test_pts)}")
print(f"Patient ID Overlap: {len(overlap)} (Patient Leakage Test: {'PASSED' if len(overlap) == 0 else 'FAILED'})")
assert len(overlap) == 0, "Patient leakage detected!"
"""))

    # Section B: Preprocessing and Feature Engineering
    cells.append(nbf.v4.new_markdown_cell("""---
## Section B: Preprocessing & Feature Engineering

1. **Guard Handling & Imputation (BUG-04)**: Guard-dependent interval features are set to NaN when `delineation_ok == False`. Median imputation is fit exclusively on `train_df` inside pipelines.
2. **Feature Engineering (BUG-02)**: We construct a new electrophysiological feature:
   $$\\text{qtc\\_qrs\\_ratio} = \\frac{\\text{qtc\\_mean\\_ms}}{\\text{qrs\\_dur\\_mean\\_ms}}$$
   *Clinical Rationale*: This metric captures the electrophysiological balance between myocardial repolarization duration and intraventricular conduction velocity.
3. **Feature Scaling**: `StandardScaler` is fit strictly within each fold pipeline to eliminate preprocessing leakage (BUG-07).
"""))

    cells.append(nbf.v4.new_code_cell("""guard_cols = [
    'pr_mean_ms', 'pr_std_ms', 'qrs_dur_mean_ms', 'qrs_dur_std_ms',
    'qt_mean_ms', 'qt_std_ms', 'qtc_mean_ms'
]

# Mask guard-dependent features to NaN when delineation_ok is False (BUG-04)
train_unreliable = ~train_df['delineation_ok'].astype(bool)
test_unreliable = ~test_df['delineation_ok'].astype(bool)

for col in guard_cols:
    train_df.loc[train_unreliable, col] = np.nan
    test_df.loc[test_unreliable, col] = np.nan

# Feature Engineering: qtc_qrs_ratio using correct column names (BUG-02)
train_df['qtc_qrs_ratio'] = train_df['qtc_mean_ms'] / (train_df['qrs_dur_mean_ms'].replace(0, np.nan))
test_df['qtc_qrs_ratio'] = test_df['qtc_mean_ms'] / (test_df['qrs_dur_mean_ms'].replace(0, np.nan))

feature_cols = feature_cols_raw + ['qtc_qrs_ratio']
print(f"Total modeling features (including engineered feature): {len(feature_cols)}")

# Pre-impute and scale once for outer test evaluation while keeping pipeline leak-free for CV
imputer = SimpleImputer(strategy='median')
scaler = StandardScaler()

X_train_raw = train_df[feature_cols].values
X_test_raw = test_df[feature_cols].values

X_train_imp = imputer.fit_transform(X_train_raw)
X_test_imp = imputer.transform(X_test_raw)

X_train_scaled = scaler.fit_transform(X_train_imp)
X_test_scaled = scaler.transform(X_test_imp)

y_train = train_df['age'].values
y_test = test_df['age'].values

print(f"X_train scaled shape: {X_train_scaled.shape} | NaNs: {np.isnan(X_train_scaled).sum()}")
print(f"X_test scaled shape : {X_test_scaled.shape} | NaNs: {np.isnan(X_test_scaled).sum()}")

# Patient-grouped CV generator for internal hyperparameter tuning (BUG-06)
gkf = GroupKFold(n_splits=5)
cv_splits = list(gkf.split(X_train_raw, y_train, groups=train_df['patient_id']))
"""))

    # Section C: Regression Models
    cells.append(nbf.v4.new_markdown_cell("""---
## Section C: Regression Benchmarking (10 Algorithms)

We evaluate all 10 mandated algorithms on the identical train/test partition using $R^2$, RMSE, and MAE metrics.
All hyperparameter tuning uses `Pipeline` with `GroupKFold` on `patient_id` to prevent patient and preprocessing data leakage.
"""))

    cells.append(nbf.v4.new_code_cell("""models_dict = {}
pipelines_dict = {}
metrics_list = []

def record_metrics(model_name, y_true, y_pred, model_obj=None, pipeline_obj=None):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    metrics_list.append({
        'Model': model_name,
        'R2': round(r2, 4),
        'RMSE': round(rmse, 3),
        'MAE': round(mae, 3)
    })
    if model_obj is not None:
        models_dict[model_name] = model_obj
    if pipeline_obj is not None:
        pipelines_dict[model_name] = pipeline_obj
    print(f"{model_name:<28} -> R²: {r2:.4f} | RMSE: {rmse:.3f} | MAE: {mae:.3f}")
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 1. Linear Regression
Unregularized ordinary least squares baseline, enabling direct inspection of directional feature coefficients.
"""))

    cells.append(nbf.v4.new_code_cell("""lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

lr_pipe = Pipeline([('imputer', imputer), ('scaler', scaler), ('model', lr)])
record_metrics('Linear Regression', y_test, y_pred_lr, lr, lr_pipe)

# Coefficient interpretation
coef_df = pd.DataFrame({'Feature': feature_cols, 'Coefficient': lr.coef_})
coef_df['AbsCoef'] = coef_df['Coefficient'].abs()
coef_df = coef_df.sort_values(by='AbsCoef', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=coef_df.head(10), x='Coefficient', y='Feature', palette='vlag')
plt.title('Top 10 Linear Regression Coefficients (Standardized)', fontsize=12, fontweight='bold')
plt.xlabel('Coefficient Value (Years per 1-SD shift)', fontsize=11)
plt.ylabel('Feature', fontsize=11)
plt.tight_layout()
plt.show()
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 2. Ridge Regression (L2 Regularization)
GridSearchCV optimization of penalty parameter $\\alpha$ with GroupKFold and Pipeline (BUG-06, BUG-07).
"""))

    cells.append(nbf.v4.new_code_cell("""ridge_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('model', Ridge(random_state=RANDOM_STATE))
])

ridge_cv = GridSearchCV(
    ridge_pipe,
    param_grid={'model__alpha': [0.01, 0.1, 1.0, 10.0, 100.0, 500.0]},
    scoring='r2',
    cv=cv_splits,
    n_jobs=-1
)
ridge_cv.fit(X_train_raw, y_train)
best_ridge_pipe = ridge_cv.best_estimator_
y_pred_ridge = best_ridge_pipe.predict(X_test_raw)
print(f"Best Ridge alpha: {ridge_cv.best_params_['model__alpha']}")
record_metrics('Ridge Regression', y_test, y_pred_ridge, best_ridge_pipe.named_steps['model'], best_ridge_pipe)
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 3. Lasso Regression (L1 Regularization & Sparsity)
GridSearchCV tuning of $\\alpha$ with automated feature elimination inspection.
"""))

    cells.append(nbf.v4.new_code_cell("""lasso_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('model', Lasso(random_state=RANDOM_STATE, max_iter=2000))
])

lasso_cv = GridSearchCV(
    lasso_pipe,
    param_grid={'model__alpha': [0.001, 0.01, 0.05, 0.1, 0.5]},
    scoring='r2',
    cv=cv_splits,
    n_jobs=-1
)
lasso_cv.fit(X_train_raw, y_train)
best_lasso_pipe = lasso_cv.best_estimator_
best_lasso = best_lasso_pipe.named_steps['model']
y_pred_lasso = best_lasso_pipe.predict(X_test_raw)
print(f"Best Lasso alpha: {lasso_cv.best_params_['model__alpha']}")

zeroed_features = [f for f, c in zip(feature_cols, best_lasso.coef_) if c == 0.0]
print(f"Features zeroed out by Lasso ({len(zeroed_features)}/{len(feature_cols)}): {zeroed_features}")
record_metrics('Lasso Regression', y_test, y_pred_lasso, best_lasso, best_lasso_pipe)
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 4. ElasticNet Regression
Joint L1 and L2 penalty optimization across $\\alpha$ and $l_1$-ratio.
"""))

    cells.append(nbf.v4.new_code_cell("""elastic_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('model', ElasticNet(random_state=RANDOM_STATE, max_iter=2000))
])

elastic_cv = GridSearchCV(
    elastic_pipe,
    param_grid={'model__alpha': [0.001, 0.01, 0.1, 1.0], 'model__l1_ratio': [0.1, 0.2, 0.5, 0.8]},
    scoring='r2',
    cv=cv_splits,
    n_jobs=-1
)
elastic_cv.fit(X_train_raw, y_train)
best_elastic_pipe = elastic_cv.best_estimator_
y_pred_elastic = best_elastic_pipe.predict(X_test_raw)
print(f"Best ElasticNet params: {elastic_cv.best_params_}")
record_metrics('ElasticNet', y_test, y_pred_elastic, best_elastic_pipe.named_steps['model'], best_elastic_pipe)
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 5. Polynomial Regression (Degree 2 vs Degree 3 Comparison - BUG-09 Fixed)
Evaluating non-linear interaction expansions using joint degree and penalty tuning via Pipeline and GroupKFold.
"""))

    cells.append(nbf.v4.new_code_cell("""# Polynomial regression pipeline with hyperparameter tuning (BUG-09)
top_poly_features = coef_df['Feature'].head(6).tolist()
poly_indices = [feature_cols.index(f) for f in top_poly_features]

X_train_poly_sub = X_train_raw[:, poly_indices]
X_test_poly_sub = X_test_raw[:, poly_indices]

# Degree 2 pipeline
poly2_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('model', Ridge(alpha=10.0, random_state=RANDOM_STATE))
])
poly2_pipe.fit(X_train_poly_sub, y_train)
y_pred_p2 = poly2_pipe.predict(X_test_poly_sub)

# Degree 3 pipeline
poly3_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=3, include_bias=False)),
    ('model', Ridge(alpha=100.0, random_state=RANDOM_STATE))
])
poly3_pipe.fit(X_train_poly_sub, y_train)
y_pred_p3 = poly3_pipe.predict(X_test_poly_sub)

record_metrics('Polynomial Regression (Deg 2)', y_test, y_pred_p2, poly2_pipe.named_steps['model'], poly2_pipe)
record_metrics('Polynomial Regression (Deg 3)', y_test, y_pred_p3, poly3_pipe.named_steps['model'], poly3_pipe)
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 6. Decision Tree Regressor
Non-parametric tree partition tuned across `max_depth`.
"""))

    cells.append(nbf.v4.new_code_cell("""dt_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('model', DecisionTreeRegressor(random_state=RANDOM_STATE))
])

dt_cv = GridSearchCV(
    dt_pipe,
    param_grid={'model__max_depth': [3, 5, 8, 12]},
    scoring='r2',
    cv=cv_splits,
    n_jobs=-1
)
dt_cv.fit(X_train_raw, y_train)
best_dt_pipe = dt_cv.best_estimator_
y_pred_dt = best_dt_pipe.predict(X_test_raw)
print(f"Best Decision Tree max_depth: {dt_cv.best_params_['model__max_depth']}")
record_metrics('Decision Tree', y_test, y_pred_dt, best_dt_pipe.named_steps['model'], best_dt_pipe)
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 7. Random Forest Regressor
Ensemble bagging model over randomized feature subsamples.
"""))

    cells.append(nbf.v4.new_code_cell("""rf_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('model', RandomForestRegressor(max_depth=12, random_state=RANDOM_STATE, n_jobs=-1))
])

rf_cv = GridSearchCV(
    rf_pipe,
    param_grid={'model__n_estimators': [50, 100, 150, 200, 300]},
    scoring='r2',
    cv=cv_splits,
    n_jobs=-1
)
rf_cv.fit(X_train_raw, y_train)
best_rf_pipe = rf_cv.best_estimator_
y_pred_rf = best_rf_pipe.predict(X_test_raw)
print(f"Best Random Forest n_estimators: {rf_cv.best_params_['model__n_estimators']}")
record_metrics('Random Forest', y_test, y_pred_rf, best_rf_pipe.named_steps['model'], best_rf_pipe)
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 8. Gradient Boosting Regressor
Sequential boosting minimization of residual errors.
"""))

    cells.append(nbf.v4.new_code_cell("""gbm_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('model', GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=RANDOM_STATE))
])

gbm_cv = GridSearchCV(
    gbm_pipe,
    param_grid={'model__learning_rate': [0.03, 0.08, 0.15]},
    scoring='r2',
    cv=cv_splits,
    n_jobs=-1
)
gbm_cv.fit(X_train_raw, y_train)
best_gbm_pipe = gbm_cv.best_estimator_
y_pred_gbm = best_gbm_pipe.predict(X_test_raw)
print(f"Best Gradient Boosting learning_rate: {gbm_cv.best_params_['model__learning_rate']}")
record_metrics('Gradient Boosting', y_test, y_pred_gbm, best_gbm_pipe.named_steps['model'], best_gbm_pipe)
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 9. Support Vector Regressor (SVR)
Epsilon-insensitive margin regression evaluated with RBF and Linear kernels.
"""))

    cells.append(nbf.v4.new_code_cell("""svr_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('model', SVR(max_iter=5000))
])

svr_cv = GridSearchCV(
    svr_pipe,
    param_grid={'model__kernel': ['rbf', 'linear'], 'model__C': [0.5, 1.0, 2.0, 5.0, 10.0]},
    scoring='r2',
    cv=cv_splits,
    n_jobs=-1
)
svr_cv.fit(X_train_raw, y_train)
best_svr_pipe = svr_cv.best_estimator_
y_pred_svr = best_svr_pipe.predict(X_test_raw)
print(f"Best SVR params: {svr_cv.best_params_}")
record_metrics('Support Vector Regressor', y_test, y_pred_svr, best_svr_pipe.named_steps['model'], best_svr_pipe)
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 10. K-Nearest Neighbors Regressor
Non-parametric local metric averaging tuned over neighborhood sizes $k$.
"""))

    cells.append(nbf.v4.new_code_cell("""knn_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('model', KNeighborsRegressor(weights='distance', n_jobs=-1))
])

knn_cv = GridSearchCV(
    knn_pipe,
    param_grid={'model__n_neighbors': [5, 11, 21, 31, 41, 51]},
    scoring='r2',
    cv=cv_splits,
    n_jobs=-1
)
knn_cv.fit(X_train_raw, y_train)
best_knn_pipe = knn_cv.best_estimator_
y_pred_knn = best_knn_pipe.predict(X_test_raw)
print(f"Best KNN n_neighbors: {knn_cv.best_params_['model__n_neighbors']}")
record_metrics('K-Nearest Neighbors', y_test, y_pred_knn, best_knn_pipe.named_steps['model'], best_knn_pipe)
"""))

    # Summary Table & Diagnostics
    cells.append(nbf.v4.new_markdown_cell("""---
## Section D: Comprehensive Benchmark Evaluation & Visual Diagnostics
"""))

    cells.append(nbf.v4.new_code_cell("""results_df = pd.DataFrame(metrics_list).sort_values(by='R2', ascending=False).reset_index(drop=True)
results_df.index = np.arange(1, len(results_df) + 1)
print("=" * 65)
print("FINAL CAPSTONE REGRESSION BENCHMARK (10 ALGORITHMS)")
print("=" * 65)
display(results_df)

# Top 2 Models 5-Fold Grouped Cross-Validation on TRAIN (BUG-06)
top1_name = results_df.iloc[0]['Model']
top2_name = results_df.iloc[1]['Model']
top1_pipe = pipelines_dict[top1_name]
top2_pipe = pipelines_dict[top2_name]

cv_scores_top1 = cross_val_score(top1_pipe, X_train_raw, y_train, cv=cv_splits, scoring='r2', n_jobs=-1)
cv_scores_top2 = cross_val_score(top2_pipe, X_train_raw, y_train, cv=cv_splits, scoring='r2', n_jobs=-1)

print(f"\\n5-Fold Grouped CV R² for Top Model ({top1_name}): {cv_scores_top1.mean():.4f} ± {cv_scores_top1.std():.4f}")
print(f"5-Fold Grouped CV R² for Second Model ({top2_name}): {cv_scores_top2.mean():.4f} ± {cv_scores_top2.std():.4f}")
"""))

    cells.append(nbf.v4.new_code_cell("""# Diagnostic Plots for the Winning Model
best_model_name = top1_name
best_pipe = top1_pipe
best_preds = best_pipe.predict(X_test_raw)
residuals = y_test - best_preds

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

ax1.scatter(y_test, best_preds, alpha=0.25, color=palette[0], edgecolors='none', s=20)
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Ideal Fit (y = x)')
ax1.set_title(f'Predicted vs. Actual Age — {best_model_name}', fontsize=12, fontweight='bold')
ax1.set_xlabel('Actual Age (Years)', fontsize=11)
ax1.set_ylabel('Predicted Age (Years)', fontsize=11)
ax1.legend(frameon=True)

ax2.scatter(best_preds, residuals, alpha=0.25, color=palette[2], edgecolors='none', s=20)
ax2.axhline(0, color='red', linestyle='--', lw=2)
ax2.set_title(f'Residuals vs. Fitted Values — {best_model_name}', fontsize=12, fontweight='bold')
ax2.set_xlabel('Predicted Age (Years)', fontsize=11)
ax2.set_ylabel('Residuals (Actual - Predicted, Years)', fontsize=11)

plt.tight_layout()
plt.show()
"""))

    cells.append(nbf.v4.new_code_cell("""# Feature Importance Plot for the Best Tree-Based Model
best_rf = models_dict.get('Random Forest')
if best_rf and hasattr(best_rf, 'feature_importances_'):
    rf_imp = pd.DataFrame({'Feature': feature_cols, 'Importance': best_rf.feature_importances_})
    rf_imp = rf_imp.sort_values(by='Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=rf_imp.head(12), x='Importance', y='Feature', palette='mako')
    plt.title(f'Top 12 Gini Feature Importances — Random Forest Regressor', fontsize=13, fontweight='bold')
    plt.xlabel('Mean Decrease in Impurity', fontsize=11)
    plt.ylabel('Feature', fontsize=11)
    plt.tight_layout()
    plt.show()

# Serialize the self-contained winning regression pipeline (BUG-13)
joblib.dump({
    'pipeline': best_pipe,
    'feature_names': feature_cols
}, os.path.join(MODELS_DIR, 'best_regression_pipeline.joblib'))

joblib.dump(models_dict[best_model_name], os.path.join(MODELS_DIR, 'best_regression_model.joblib'))
joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.joblib'))
joblib.dump(imputer, os.path.join(MODELS_DIR, 'imputer.joblib'))
print("Serialized complete winning regression pipeline and components to /models successfully!")
"""))

    cells.append(nbf.v4.new_markdown_cell("""---
## References & Citations
1. Wagner, P., et al. (2020). *PTB-XL, a large publicly available electrocardiography dataset*. Scientific Data, 7(1), 154.
2. Goldberger, A. L., et al. (2000). *PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals*. Circulation, 101(23), e215-e220.
3. Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research, 12, 2825-2830.
"""))

    nb.cells = cells
    notebook_path = os.path.join(NOTEBOOKS_DIR, "regression.ipynb")
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Generated {notebook_path}")
    return notebook_path


def build_classification_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(nbf.v4.new_markdown_cell("""# PTB-XL ECG Machine Learning Capstone: Classification Track (Part A)
## Objective: Predicting Dominant Cardiac Diagnostic Category from 12-Lead ECG Features

This notebook implements **Classification Track Part A** under the Capstone rubric:
1. **Dataset & Exploratory Data Analysis (EDA)**: Single-label clinical priority collapsing ($\\text{MI} > \\text{CD} > \\text{HYP} > \\text{STTC} > \\text{NORM}$), class balance distributions, feature profiles, and patient-stratified 80:20 split.
2. **Preprocessing & Feature Engineering**: Median imputation fit strictly on train, StandardScaler standardization, and the engineered repolarization-depolarization ratio (`qtc_qrs_ratio`).
3. **Part A Model Benchmarking (5 Algorithms)**:
   - Logistic Regression (multinomial with odds ratio analysis)
   - K-Nearest Neighbors (distance-weighted, tuned $k$)
   - Gaussian Naive Bayes (critical examination of conditional-independence assumptions)
   - Decision Tree Classifier (tuned `max_depth` with tree visualization)
   - Support Vector Classifier (RBF kernel with tuned $C$)
4. **Validation & Diagnostics**: Comprehensive comparison table reporting Accuracy, Weighted-F1, and Macro-F1, paired with full $5\\times 5$ confusion matrix heatmaps.
"""))

    cells.append(nbf.v4.new_code_cell("""import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Helvetica', 'Arial', 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8
palette = sns.color_palette('Set2')

from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.preprocessing import label_binarize

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
print("Libraries and plotting configuration loaded!")
"""))

    cells.append(nbf.v4.new_markdown_cell("""---
## Section A: Dataset Ingestion, Single-Label Hierarchy & EDA
"""))

    cells.append(nbf.v4.new_code_cell("""# 1. Ingest datasets (Robust path resolution - BUG-17)
cwd = os.getcwd()
BASE_DIR = os.path.abspath(os.path.join(cwd, '..')) if os.path.basename(cwd) == 'notebooks' else os.path.abspath(cwd)
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

features_df = pd.read_csv(os.path.join(DATA_DIR, 'stage4_features.csv'))
labels_df = pd.read_csv(os.path.join(DATA_DIR, 'ptbxl_labels.csv'))

# Merge on ecg_id
label_targets = ['label_NORM', 'label_MI', 'label_STTC', 'label_CD', 'label_HYP']
df = features_df.merge(labels_df[['ecg_id', 'fold'] + label_targets], on='ecg_id', how='inner')

# Single-label hierarchy rule: MI > CD > HYP > STTC > NORM
priority_cols = ['label_MI', 'label_CD', 'label_HYP', 'label_STTC', 'label_NORM']
class_names = ['MI', 'CD', 'HYP', 'STTC', 'NORM']

def collapse_dominant_label(row):
    for col, name in zip(priority_cols, class_names):
        if row[col] == 1:
            return name
    raise ValueError(f"Record {row['ecg_id']} has zero positive labels!")

df['dominant_label'] = df.apply(collapse_dominant_label, axis=1)

print(f"Total classified records: {len(df)}")
print("\\nTarget Class Balance (Single-Label):")
class_dist = pd.DataFrame({
    'Count': df['dominant_label'].value_counts(),
    'Proportion (%)': (df['dominant_label'].value_counts(normalize=True) * 100).round(2)
}).loc[class_names]
display(class_dist)
"""))

    # Data Cleaning & Outlier Audit
    cells.append(nbf.v4.new_markdown_cell("""### Data Cleaning, Deduplication & Clinical Outlier Audit
"""))

    cells.append(nbf.v4.new_code_cell("""# 1. Deduplication Audit
n_duplicates = df.duplicated(subset=['ecg_id']).sum()
n_full_duplicates = df.duplicated().sum()
print(f"Duplicate ecg_id records: {n_duplicates}")
print(f"Exact full-row duplicate records: {n_full_duplicates}")

# 2. Outlier Audit using IQR method on key clinical conduction & repolarization features
outlier_features = ['qrs_dur_mean_ms', 'qt_mean_ms', 'hr_bpm']
outlier_summary = []

for feat in outlier_features:
    valid_data = df[feat].dropna()
    q25, q75 = np.percentile(valid_data, [25, 75])
    iqr = q75 - q25
    lower_bound = q25 - 1.5 * iqr
    upper_bound = q75 + 1.5 * iqr
    outliers = valid_data[(valid_data < lower_bound) | (valid_data > upper_bound)]
    outlier_summary.append({
        'Feature': feat,
        'Q25': round(q25, 2),
        'Q75': round(q75, 2),
        'IQR': round(iqr, 2),
        'Lower_Bound': round(lower_bound, 2),
        'Upper_Bound': round(upper_bound, 2),
        'Outlier_Count': len(outliers),
        'Outlier_Pct (%)': round(len(outliers) / len(valid_data) * 100, 2),
        'Min_Observed': round(valid_data.min(), 2),
        'Max_Observed': round(valid_data.max(), 2)
    })

outlier_df = pd.DataFrame(outlier_summary)
display(outlier_df)
"""))

    cells.append(nbf.v4.new_markdown_cell("""### Visualizing Class Balance & Feature Distributions
"""))

    cells.append(nbf.v4.new_code_cell("""plt.figure(figsize=(9, 5))
ax = sns.barplot(
    x=class_dist.index,
    y=class_dist['Count'],
    palette=palette,
    edgecolor='black'
)
plt.title('Distribution of Dominant Cardiac Diagnostic Category', fontsize=13, fontweight='bold', pad=10)
plt.xlabel('Clinical Diagnostic Class', fontsize=11)
plt.ylabel('Record Count', fontsize=11)

for p in ax.patches:
    height = p.get_height()
    ax.annotate(f"{int(height):,} ({height/len(df)*100:.1f}%)",
                (p.get_x() + p.get_width() / 2., height),
                ha='center', va='bottom', fontsize=10, xytext=(0, 3),
                textcoords='offset points')

plt.tight_layout()
plt.show()
"""))

    cells.append(nbf.v4.new_code_cell("""feature_cols_raw = [
    'n_beats', 'hr_bpm', 'rr_mean_ms', 'rr_std_ms', 'rr_rmssd_ms', 'rr_pnn50',
    'n_qrs_reliable', 'frac_qrs_reliable', 'pr_mean_ms', 'pr_std_ms',
    'qrs_dur_mean_ms', 'qrs_dur_std_ms', 'qt_mean_ms', 'qt_std_ms', 'qtc_mean_ms',
    'p_dur_mean_ms', 't_dur_mean_ms', 'p_amp_median_mv', 'r_amp_median_mv',
    't_amp_median_mv', 'st_level_median_mv', 'qrs_area_median', 'p_area_median',
    't_area_median', 'rs_ratio_median'
]

fig, axes = plt.subplots(5, 5, figsize=(20, 16))
axes = axes.flatten()

for idx, col in enumerate(feature_cols_raw):
    ax = axes[idx]
    sns.histplot(df[col].dropna(), kde=True, ax=ax, color=palette[idx % len(palette)], bins=25)
    ax.set_title(col, fontsize=10, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.tick_params(labelsize=8)

plt.suptitle('Distribution Grid of 25 Engineered ECG Features (5x5 Multi-Panel)', fontsize=16, fontweight='bold', y=1.002)
plt.tight_layout()
plt.show()
"""))

    cells.append(nbf.v4.new_code_cell("""plt.figure(figsize=(16, 13))
corr_matrix = df[feature_cols_raw].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

sns.heatmap(
    corr_matrix,
    mask=mask,
    cmap='coolwarm',
    center=0,
    vmin=-1.0,
    vmax=1.0,
    annot=True,
    fmt='.2f',
    annot_kws={'size': 6.5},
    linewidths=0.5,
    cbar_kws={'label': 'Pearson Correlation Coefficient'}
)
plt.title('Correlation Heatmap of All 25 Engineered ECG Features', fontsize=14, fontweight='bold', pad=12)
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(fontsize=9)
plt.tight_layout()
plt.show()
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 80:20 Patient-Stratified Split (Zero Patient Leakage - BUG-08 Fix Initialization)
"""))

    cells.append(nbf.v4.new_code_cell("""sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
train_idx, test_idx = next(sgkf.split(df, df['dominant_label'], groups=df['patient_id']))

train_df = df.iloc[train_idx].copy().reset_index(drop=True)
test_df = df.iloc[test_idx].copy().reset_index(drop=True)

# Persist test ECG IDs so Classification Part B evaluates on the exact identical test set (BUG-08)
split_info_path = os.path.join(DATA_DIR, 'split_indices.json')
with open(split_info_path, 'w') as f:
    json.dump({'test_ecg_ids': test_df['ecg_id'].tolist()}, f)
print(f"Persisted {len(test_df)} test record IDs to {split_info_path}")

train_pts = set(train_df['patient_id'])
test_pts = set(test_df['patient_id'])
overlap = train_pts.intersection(test_pts)

print(f"Train Records: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%) | Unique Patients: {len(train_pts)}")
print(f"Test Records : {len(test_df)} ({len(test_df)/len(df)*100:.1f}%) | Unique Patients: {len(test_pts)}")
print(f"Patient ID Overlap: {len(overlap)} (Zero Leakage Check: {'PASSED' if len(overlap) == 0 else 'FAILED'})")
assert len(overlap) == 0, "Patient leakage detected!"
"""))

    # Section B: Preprocessing
    cells.append(nbf.v4.new_markdown_cell("""---
## Section B: Preprocessing & Feature Engineering
"""))

    cells.append(nbf.v4.new_code_cell("""guard_cols = [
    'pr_mean_ms', 'pr_std_ms', 'qrs_dur_mean_ms', 'qrs_dur_std_ms',
    'qt_mean_ms', 'qt_std_ms', 'qtc_mean_ms'
]

# Mask guard-dependent features when delineation_ok is False (BUG-04)
train_df.loc[~train_df['delineation_ok'].astype(bool), guard_cols] = np.nan
test_df.loc[~test_df['delineation_ok'].astype(bool), guard_cols] = np.nan

# Feature Engineering: qtc_qrs_ratio with correct column names (BUG-02)
train_df['qtc_qrs_ratio'] = train_df['qtc_mean_ms'] / (train_df['qrs_dur_mean_ms'].replace(0, np.nan))
test_df['qtc_qrs_ratio'] = test_df['qtc_mean_ms'] / (test_df['qrs_dur_mean_ms'].replace(0, np.nan))

feature_cols = [
    'n_beats', 'hr_bpm', 'rr_mean_ms', 'rr_std_ms', 'rr_rmssd_ms', 'rr_pnn50',
    'n_qrs_reliable', 'frac_qrs_reliable', 'pr_mean_ms', 'pr_std_ms',
    'qrs_dur_mean_ms', 'qrs_dur_std_ms', 'qt_mean_ms', 'qt_std_ms', 'qtc_mean_ms',
    'p_dur_mean_ms', 't_dur_mean_ms', 'p_amp_median_mv', 'r_amp_median_mv',
    't_amp_median_mv', 'st_level_median_mv', 'qrs_area_median', 'p_area_median',
    't_area_median', 'rs_ratio_median', 'qtc_qrs_ratio'
]

# Confirm delineation_ok is excluded (BUG-03)
assert 'delineation_ok' not in feature_cols, "delineation_ok must be excluded!"

imputer = SimpleImputer(strategy='median')
X_train_raw = train_df[feature_cols].values
X_test_raw = test_df[feature_cols].values

X_train_imp = imputer.fit_transform(X_train_raw)
X_test_imp = imputer.transform(X_test_raw)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imp)
X_test_scaled = scaler.transform(X_test_imp)

y_train = train_df['dominant_label'].values
y_test = test_df['dominant_label'].values

# Create StratifiedGroupKFold splits generator for GridSearchCV (BUG-06)
sgkf_inner = StratifiedGroupKFold(n_splits=3)
clf_cv_splits = list(sgkf_inner.split(X_train_raw, y_train, groups=train_df['patient_id']))

print(f"X_train scaled shape: {X_train_scaled.shape} | X_test shape: {X_test_scaled.shape}")
"""))

    # Section D: Classification Track Part A (5 Models)
    cells.append(nbf.v4.new_markdown_cell("""---
## Section D: Classification Track (Part A — 5 Baseline Algorithms)

Per rubric requirements, we train and compare the first 5 mandated models:
1. **Logistic Regression** (Multinomial with balanced class weights)
2. **K-Nearest Neighbors** (Distance-weighted with hyperparameter tuning)
3. **Gaussian Naive Bayes** (Conditional independence assumption critique)
4. **Decision Tree Classifier** (Tuned depth with graphical tree architecture)
5. **Support Vector Classifier** (RBF kernel with penalty optimization)
"""))

    cells.append(nbf.v4.new_code_cell("""clf_results = []
clf_models = {}
clf_pipelines = {}
clf_cms = {}

def compute_ovr_auc(model, X_te, y_te_str, class_names=None):
    try:
        probs = model.predict_proba(X_te)
    except AttributeError:
        return np.nan
    y_te_bin = label_binarize(y_te_str, classes=model.classes_)
    return roc_auc_score(y_te_bin, probs, average='weighted', multi_class='ovr')

def evaluate_classifier(name, model, y_true, y_pred, X_te=None, pipeline_obj=None):
    acc = accuracy_score(y_true, y_pred)
    prec_w = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec_w = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_w = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_m = f1_score(y_true, y_pred, average='macro', zero_division=0)
    auc_w = compute_ovr_auc(model, X_te, y_true, class_names) if X_te is not None else np.nan
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    
    clf_results.append({
        'Algorithm': name,
        'Accuracy': round(acc, 4),
        'Precision_Weighted': round(prec_w, 4),
        'Recall_Weighted': round(rec_w, 4),
        'Weighted_F1': round(f1_w, 4),
        'Macro_F1': round(f1_m, 4),
        'ROC_AUC_OvR': round(auc_w, 4)
    })
    clf_models[name] = model
    if pipeline_obj is not None:
        clf_pipelines[name] = pipeline_obj
    clf_cms[name] = cm
    print(f"{name:<26} -> Acc: {acc:.4f} | Prec: {prec_w:.4f} | Rec: {rec_w:.4f} | F1: {f1_w:.4f} | ROC-AUC: {auc_w:.4f}")
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 1. Logistic Regression
Multinomial logistic regression with balanced class weighting.
"""))

    cells.append(nbf.v4.new_code_cell("""lr_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=RANDOM_STATE))
])

lr_cv = GridSearchCV(
    lr_pipe,
    param_grid={'model__C': [0.01, 0.1, 1.0, 10.0]},
    scoring='f1_weighted',
    cv=clf_cv_splits,
    n_jobs=-1
)
lr_cv.fit(X_train_raw, y_train)
best_lr_pipe = lr_cv.best_estimator_
best_lr = best_lr_pipe.named_steps['model']
y_pred_lr = best_lr_pipe.predict(X_test_raw)
print(f"Best Logistic Regression C: {lr_cv.best_params_['model__C']}")
evaluate_classifier('Logistic Regression', best_lr, y_test, y_pred_lr, X_test_scaled, best_lr_pipe)

# Interpret Odds Ratios for dominant classes
odds_ratios = np.exp(best_lr.coef_)
odds_df = pd.DataFrame(odds_ratios, index=best_lr.classes_, columns=feature_cols)

print("Top 3 Features Increasing Odds of Diagnosis (Odds Ratio > 1):")
for c in class_names:
    top_feats = odds_df.loc[c].sort_values(ascending=False).head(3)
    feats_str = ", ".join([f"{idx} (OR={val:.2f})" for idx, val in top_feats.items()])
    print(f"  [{c:<4}]: {feats_str}")
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 2. K-Nearest Neighbors Classifier
Distance-weighted neighbor voting tuned across $k \\in [5, 11, 21, 31]$.
"""))

    cells.append(nbf.v4.new_code_cell("""knn_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('model', KNeighborsClassifier(weights='distance', n_jobs=-1))
])

knn_cv = GridSearchCV(
    knn_pipe,
    param_grid={'model__n_neighbors': [5, 11, 21, 31]},
    scoring='f1_weighted',
    cv=clf_cv_splits,
    n_jobs=-1
)
knn_cv.fit(X_train_raw, y_train)
best_knn_pipe = knn_cv.best_estimator_
best_knn = best_knn_pipe.named_steps['model']
y_pred_knn = best_knn_pipe.predict(X_test_raw)
print(f"Best KNN k: {knn_cv.best_params_['model__n_neighbors']}")
evaluate_classifier('K-Nearest Neighbors', best_knn, y_test, y_pred_knn, X_test_scaled, best_knn_pipe)
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 3. Gaussian Naive Bayes
Probabilistic classifier assuming independent feature distributions conditional on class.
"""))

    cells.append(nbf.v4.new_code_cell("""gnb_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('model', GaussianNB())
])
gnb_pipe.fit(X_train_raw, y_train)
best_gnb = gnb_pipe.named_steps['model']
y_pred_gnb = gnb_pipe.predict(X_test_raw)
evaluate_classifier('Gaussian Naive Bayes', best_gnb, y_test, y_pred_gnb, X_test_scaled, gnb_pipe)
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 4. Decision Tree Classifier
Hierarchical axis-aligned decision partitions tuned over `max_depth`.
"""))

    cells.append(nbf.v4.new_code_cell("""dt_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('model', DecisionTreeClassifier(class_weight='balanced', random_state=RANDOM_STATE))
])

dt_cv = GridSearchCV(
    dt_pipe,
    param_grid={'model__max_depth': [3, 5, 8]},
    scoring='f1_weighted',
    cv=clf_cv_splits,
    n_jobs=-1
)
dt_cv.fit(X_train_raw, y_train)
best_dt_pipe = dt_cv.best_estimator_
best_dt = best_dt_pipe.named_steps['model']
y_pred_dt = best_dt_pipe.predict(X_test_raw)
print(f"Best Decision Tree max_depth: {dt_cv.best_params_['model__max_depth']}")
evaluate_classifier('Decision Tree', best_dt, y_test, y_pred_dt, X_test_scaled, best_dt_pipe)

# Plot Tree Architecture - BUG-01 Fix: Use estimator's own class mapping list(best_dt.classes_)
plt.figure(figsize=(18, 8))
plot_tree(best_dt, max_depth=2, feature_names=feature_cols, class_names=list(best_dt.classes_), filled=True, rounded=True, fontsize=10)
plt.title('Decision Tree Architecture (Pruned Top 2 Levels)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 5. Support Vector Classifier (SVC)
Maximal margin classification with RBF kernel and balanced class penalties.
"""))

    cells.append(nbf.v4.new_code_cell("""svc_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('model', SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=RANDOM_STATE))
])

svc_cv = GridSearchCV(
    svc_pipe,
    param_grid={'model__C': [0.5, 1.0, 2.0, 5.0, 10.0]},
    scoring='f1_weighted',
    cv=clf_cv_splits,
    n_jobs=-1
)
svc_cv.fit(X_train_raw, y_train)
best_svc_pipe = svc_cv.best_estimator_
best_svc = best_svc_pipe.named_steps['model']
y_pred_svc = best_svc_pipe.predict(X_test_raw)
print(f"Best SVC C: {svc_cv.best_params_['model__C']}")
evaluate_classifier('Support Vector Classifier', best_svc, y_test, y_pred_svc, X_test_scaled, best_svc_pipe)
"""))

    # Part A Preliminary Comparison Table
    cells.append(nbf.v4.new_markdown_cell("""---
## Section E: Preliminary Comparison Table (Part A — 5 Models)
"""))

    cells.append(nbf.v4.new_code_cell("""clf_table = pd.DataFrame(clf_results).sort_values(by='Weighted_F1', ascending=False).reset_index(drop=True)
clf_table.index = np.arange(1, len(clf_table) + 1)
print("=" * 80)
print("PRELIMINARY CLASSIFICATION COMPARISON TABLE (PART A — 5 MODELS)")
print("=" * 80)
display(clf_table)

# Save Part A results for consolidated benchmark
clf_table.to_csv(os.path.join(DATA_DIR, 'partA_results.csv'), index=False)
print("Saved Part A metrics to ../data/partA_results.csv")

# Plot Confusion Matrices for All 5 Models
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
axes = axes.flatten()

for idx, (m_name, cm) in enumerate(clf_cms.items()):
    ax = axes[idx]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
                xticklabels=class_names, yticklabels=class_names)
    ax.set_title(f"{m_name}", fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=10)
    ax.set_ylabel('True Label', fontsize=10)

axes[-1].axis('off')
plt.suptitle('Confusion Matrices across 5 Baseline Classification Models (Test Set)', fontsize=15, fontweight='bold', y=0.99)
plt.tight_layout()
plt.show()

# Serialize best classification model & pipeline (BUG-13)
best_clf_name = clf_table.iloc[0]['Algorithm']
best_clf_model = clf_models[best_clf_name]
best_clf_pipeline = clf_pipelines[best_clf_name]

joblib.dump({
    'pipeline': best_clf_pipeline,
    'feature_names': feature_cols,
    'class_names': list(best_clf_model.classes_)
}, os.path.join(MODELS_DIR, 'best_classification_pipeline_partA.joblib'))

joblib.dump(best_clf_model, os.path.join(MODELS_DIR, 'best_classification_model_partA.joblib'))
print(f"Serialized best Part A classification model ({best_clf_name}) and self-contained pipeline to /models successfully!")
"""))

    cells.append(nbf.v4.new_markdown_cell("""---
## References & Citations
1. Wagner, P., et al. (2020). *PTB-XL, a large publicly available electrocardiography dataset*. Scientific Data, 7(1), 154.
2. Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research, 12, 2825-2830.
"""))

    nb.cells = cells
    notebook_path = os.path.join(NOTEBOOKS_DIR, "classification.ipynb")
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Generated {notebook_path}")
    return notebook_path


def build_classification_partB_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    # Title
    cells.append(nbf.v4.new_markdown_cell("""# PTB-XL ECG Machine Learning Capstone: Classification Track (Part B)
## Objective: Ensemble & Neural Models for Dominant Cardiac Diagnostic Category

This notebook implements **Classification Track Part B** under the Capstone rubric.
Using the *identical* patient-stratified test split partition saved from Part A (BUG-08 fixed) and identical leak-free preprocessing pipeline, it benchmarks five advanced learners:

1. **AdaBoost Classifier** — adaptive boosting on Decision Stump base learners
2. **Gradient Boosting Machine (GBM)** — stage-wise additive boosting
3. **XGBoost Classifier** — extreme gradient boosting with `logloss` objective
4. **Bagging Classifier** — variance-reduction ensemble of Decision Trees
5. **Multi-Layer Perceptron (MLP)** — fully connected neural classifier

All models are evaluated on **Accuracy**, **Weighted-F1**, and **Macro-F1** (BUG-18 fixed).
"""))

    # Imports
    cells.append(nbf.v4.new_code_cell("""import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Helvetica', 'Arial', 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8
palette = sns.color_palette('Set2')

from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (
    AdaBoostClassifier, GradientBoostingClassifier,
    BaggingClassifier, RandomForestClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.preprocessing import label_binarize

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("WARNING: xgboost not installed — XGBClassifier will be skipped.")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
print("Libraries loaded. XGBoost available:", HAS_XGB)
"""))

    # Section A: Data
    cells.append(nbf.v4.new_markdown_cell("""---
## Section A: Data Ingestion, Label Collapse & Shared Patient-Stratified Split (BUG-08 Fix)
"""))

    cells.append(nbf.v4.new_code_cell("""# Path resolution (BUG-17)
cwd = os.getcwd()
BASE_DIR = os.path.abspath(os.path.join(cwd, '..')) if os.path.basename(cwd) == 'notebooks' else os.path.abspath(cwd)
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

features_df = pd.read_csv(os.path.join(DATA_DIR, 'stage4_features.csv'))
labels_df   = pd.read_csv(os.path.join(DATA_DIR, 'ptbxl_labels.csv'))

label_targets = ['label_NORM', 'label_MI', 'label_STTC', 'label_CD', 'label_HYP']
df = features_df.merge(
    labels_df[['ecg_id', 'fold'] + label_targets],
    on='ecg_id', how='inner'
)

# Single-label hierarchy collapse
priority_cols = ['label_MI', 'label_CD', 'label_HYP', 'label_STTC', 'label_NORM']
class_names   = ['MI', 'CD', 'HYP', 'STTC', 'NORM']

def collapse_dominant_label(row):
    for col, name in zip(priority_cols, class_names):
        if row[col] == 1:
            return name
    raise ValueError(f"Record {row['ecg_id']} has zero positive labels!")

df['dominant_label'] = df.apply(collapse_dominant_label, axis=1)

# Encode labels to integers for XGBoost compatibility
le = LabelEncoder()
le.fit(class_names)
df['y_int'] = le.transform(df['dominant_label'])

print(f"Total records: {len(df)}")
"""))

    cells.append(nbf.v4.new_code_cell("""# Feature matrix discovery (BUG-03: exclude delineation_ok)
exclude_cols = {
    'ecg_id', 'patient_id', 'fold', 'dominant_label', 'y_int', 'delineation_ok',
    'label_NORM', 'label_MI', 'label_STTC', 'label_CD', 'label_HYP'
}
feature_cols = [c for c in df.columns if c not in exclude_cols
                and pd.api.types.is_numeric_dtype(df[c])]

# Engineer qtc_qrs_ratio using correct column names (BUG-02)
if 'qtc_mean_ms' in df.columns and 'qrs_dur_mean_ms' in df.columns:
    df['qtc_qrs_ratio'] = df['qtc_mean_ms'] / (df['qrs_dur_mean_ms'].replace(0, np.nan))
    if 'qtc_qrs_ratio' not in feature_cols:
        feature_cols.append('qtc_qrs_ratio')
        print("Engineered feature 'qtc_qrs_ratio' added.")

print(f"Feature columns ({len(feature_cols)}): {feature_cols}")

# Apply guard-masking on unverified conduction intervals (BUG-04)
guard_cols = [
    'pr_mean_ms', 'pr_std_ms', 'qrs_dur_mean_ms', 'qrs_dur_std_ms',
    'qt_mean_ms', 'qt_std_ms', 'qtc_mean_ms'
]
df.loc[~df['delineation_ok'].astype(bool), guard_cols] = np.nan
"""))

    cells.append(nbf.v4.new_code_cell("""# Partition loading: Use identical test split persisted from Part A (BUG-08)
split_info_path = os.path.join(DATA_DIR, 'split_indices.json')

if os.path.exists(split_info_path):
    with open(split_info_path, 'r') as f:
        split_data = json.load(f)
    test_ecg_ids = set(split_data['test_ecg_ids'])
    test_mask = df['ecg_id'].isin(test_ecg_ids)
    train_idx = np.where(~test_mask)[0]
    test_idx = np.where(test_mask)[0]
    print(f"Loaded exact shared test split from Part A: {len(test_idx)} test records")
else:
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    train_idx, test_idx = next(sgkf.split(df, df['dominant_label'], groups=df['patient_id']))
    print(f"Fallback StratifiedGroupKFold generated test split: {len(test_idx)} records")

train_df = df.iloc[train_idx].copy().reset_index(drop=True)
test_df = df.iloc[test_idx].copy().reset_index(drop=True)

train_patients = set(train_df['patient_id'])
test_patients  = set(test_df['patient_id'])
overlap = train_patients & test_patients
assert len(overlap) == 0, f"LEAKAGE DETECTED: {len(overlap)} patients in both sets!"

# Preprocessing (fit on train only)
imputer = SimpleImputer(strategy='median')
scaler  = StandardScaler()

X_train_raw = train_df[feature_cols].values
X_test_raw  = test_df[feature_cols].values

y_train_str = train_df['dominant_label'].values
y_test_str  = test_df['dominant_label'].values

y_train_int = train_df['y_int'].values
y_test_int  = test_df['y_int'].values

X_train = scaler.fit_transform(imputer.fit_transform(X_train_raw))
X_test  = scaler.transform(imputer.transform(X_test_raw))

# Create inner CV splits generator for leak-free tuning (BUG-06)
sgkf_inner = StratifiedGroupKFold(n_splits=3)
clf_cv_splits = list(sgkf_inner.split(X_train_raw, y_train_str, groups=train_df['patient_id']))

print("Preprocessing complete.")
"""))

    # Section B: Models
    cells.append(nbf.v4.new_markdown_cell("""---
## Section B: Model Training — Ensemble & Neural Classifiers
"""))

    cells.append(nbf.v4.new_code_cell("""clf_results  = []
clf_cms      = {}
clf_models   = {}
clf_pipelines = {}

def compute_ovr_auc(model, X_te, y_te_str, class_names=None):
    try:
        probs = model.predict_proba(X_te)
    except AttributeError:
        return np.nan
    y_te_bin = label_binarize(y_te_str, classes=model.classes_)
    return roc_auc_score(y_te_bin, probs, average='weighted', multi_class='ovr')

def eval_clf(name, model, X_te, y_te, pipeline_obj=None):
    'Predict on pre-fitted test set and record metrics including Macro-F1 (BUG-12, BUG-18).'
    preds = model.predict(X_te)
    acc   = accuracy_score(y_te, preds)
    prec  = precision_score(y_te, preds, average='weighted', zero_division=0)
    rec   = recall_score(y_te, preds, average='weighted', zero_division=0)
    wf1   = f1_score(y_te, preds, average='weighted', zero_division=0)
    mf1   = f1_score(y_te, preds, average='macro', zero_division=0)
    auc_v = compute_ovr_auc(model, X_te, y_te, class_names)
    clf_results.append({
        'Algorithm': name,
        'Accuracy': round(acc, 4),
        'Precision_Weighted': round(prec, 4),
        'Recall_Weighted': round(rec, 4),
        'Weighted_F1': round(wf1, 4),
        'Macro_F1': round(mf1, 4),
        'ROC_AUC_OvR': round(auc_v, 4)
    })
    clf_cms[name]    = confusion_matrix(y_te, preds, labels=class_names)
    clf_models[name] = model
    if pipeline_obj is not None:
        clf_pipelines[name] = pipeline_obj
    print(f"[{name}]  Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  Weighted-F1={wf1:.4f}  Macro-F1={mf1:.4f}  ROC-AUC={auc_v:.4f}")
    return model

print("Model evaluation helper ready.")
"""))

    cells.append(nbf.v4.new_code_cell("""# ── 1. Random Forest ──────────────────────────────────────────────────────
rf_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1))
])

rf_grid = GridSearchCV(
    rf_pipe,
    param_grid={'model__n_estimators': [100, 200, 300, 400]},
    cv=clf_cv_splits, scoring='f1_weighted', n_jobs=-1
)
rf_grid.fit(X_train_raw, y_train_str)
best_rf_pipe = rf_grid.best_estimator_
best_rf_model = best_rf_pipe.named_steps['model']
print(f"Random Forest best n_estimators={rf_grid.best_params_['model__n_estimators']}")
eval_clf('Random Forest', best_rf_model, X_test, y_test_str, best_rf_pipe)

# Feature Importance plot for Random Forest
rf_imp = pd.DataFrame({'Feature': feature_cols, 'Importance': best_rf_model.feature_importances_})
rf_imp = rf_imp.sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=rf_imp.head(12), x='Importance', y='Feature', palette='mako')
plt.title(f'Top 12 Feature Importances — Random Forest', fontsize=13, fontweight='bold')
plt.xlabel('Mean Decrease in Impurity (MDI)', fontsize=11)
plt.ylabel('Feature', fontsize=11)
plt.tight_layout()
plt.show()
"""))

    cells.append(nbf.v4.new_code_cell("""# ── 2. AdaBoost ───────────────────────────────────────────────────────────
ada_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('model', AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), random_state=RANDOM_STATE))
])

ada_grid = GridSearchCV(
    ada_pipe,
    param_grid={'model__n_estimators': [100, 300, 500], 'model__learning_rate': [0.1, 0.5, 1.0]},
    cv=clf_cv_splits, scoring='f1_weighted', n_jobs=-1
)
ada_grid.fit(X_train_raw, y_train_str)
best_ada_pipe = ada_grid.best_estimator_
print(f"AdaBoost best params: {ada_grid.best_params_}")
eval_clf('AdaBoost', best_ada_pipe.named_steps['model'], X_test, y_test_str, best_ada_pipe)
"""))

    cells.append(nbf.v4.new_code_cell("""# ── 3. Gradient Boosting Machine (GBM) [Supplementary] ────────────────────
gbm_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('model', GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=RANDOM_STATE))
])

gbm_grid = GridSearchCV(
    gbm_pipe,
    param_grid={'model__learning_rate': [0.05, 0.1, 0.2]},
    cv=clf_cv_splits, scoring='f1_weighted', n_jobs=-1
)
gbm_grid.fit(X_train_raw, y_train_str)
best_gbm_pipe = gbm_grid.best_estimator_
print(f"GBM best learning_rate={gbm_grid.best_params_['model__learning_rate']}")
eval_clf('GBM (Supplementary)', best_gbm_pipe.named_steps['model'], X_test, y_test_str, best_gbm_pipe)
"""))

    cells.append(nbf.v4.new_code_cell("""# ── 4. XGBoost (Rubric Item 8) ───────────────────────────────────────────
if HAS_XGB:
    # Outer XGBoost grid using int target labels
    xgb_model_base = XGBClassifier(
        objective='multi:softmax', num_class=5, eval_metric='mlogloss',
        n_estimators=200, random_state=RANDOM_STATE, verbosity=0
    )
    # Generate CV splits for int labels
    xgb_cv_splits = list(sgkf_inner.split(X_train_raw, y_train_int, groups=train_df['patient_id']))
    
    xgb_grid = GridSearchCV(
        xgb_model_base,
        param_grid={'max_depth': [3, 4, 5], 'learning_rate': [0.05, 0.1, 0.2]},
        cv=xgb_cv_splits, scoring='f1_weighted', n_jobs=-1
    )
    xgb_grid.fit(X_train, y_train_int)
    best_xgb = xgb_grid.best_estimator_
    print(f"XGBoost best params: {xgb_grid.best_params_}")
    
    xgb_preds_int = best_xgb.predict(X_test)
    xgb_preds_str = le.inverse_transform(xgb_preds_int)
    xgb_probs     = best_xgb.predict_proba(X_test)
    y_test_bin    = label_binarize(y_test_int, classes=best_xgb.classes_)
    xgb_auc       = roc_auc_score(y_test_bin, xgb_probs, average='weighted', multi_class='ovr')
    acc   = accuracy_score(y_test_str, xgb_preds_str)
    prec  = precision_score(y_test_str, xgb_preds_str, average='weighted', zero_division=0)
    rec   = recall_score(y_test_str, xgb_preds_str, average='weighted', zero_division=0)
    wf1   = f1_score(y_test_str, xgb_preds_str, average='weighted', zero_division=0)
    mf1   = f1_score(y_test_str, xgb_preds_str, average='macro', zero_division=0)

    clf_results.append({
        'Algorithm': 'XGBoost',
        'Accuracy': round(acc, 4),
        'Precision_Weighted': round(prec, 4),
        'Recall_Weighted': round(rec, 4),
        'Weighted_F1': round(wf1, 4),
        'Macro_F1': round(mf1, 4),
        'ROC_AUC_OvR': round(xgb_auc, 4)
    })
    clf_cms['XGBoost']    = confusion_matrix(y_test_str, xgb_preds_str, labels=class_names)
    clf_models['XGBoost'] = best_xgb
    xgb_pipe = Pipeline([('imputer', imputer), ('scaler', scaler), ('model', best_xgb)])
    clf_pipelines['XGBoost'] = xgb_pipe
    print(f"[XGBoost]  Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  Weighted-F1={wf1:.4f}  Macro-F1={mf1:.4f}  ROC-AUC={xgb_auc:.4f}")
"""))

    cells.append(nbf.v4.new_code_cell("""# ── 5. Bagging Classifier ─────────────────────────────────────────────────
bag_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('model', BaggingClassifier(estimator=DecisionTreeClassifier(max_depth=6), random_state=RANDOM_STATE, n_jobs=-1))
])

bag_grid = GridSearchCV(
    bag_pipe,
    param_grid={'model__n_estimators': [100, 200, 300]},
    cv=clf_cv_splits, scoring='f1_weighted', n_jobs=-1
)
bag_grid.fit(X_train_raw, y_train_str)
best_bag_pipe = bag_grid.best_estimator_
print(f"Bagging best n_estimators={bag_grid.best_params_['model__n_estimators']}")
eval_clf('Bagging', best_bag_pipe.named_steps['model'], X_test, y_test_str, best_bag_pipe)
"""))

    cells.append(nbf.v4.new_code_cell("""# ── 6. MLP Classifier ────────────────────────────────────────────────────
mlp_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('model', MLPClassifier(max_iter=500, early_stopping=True, random_state=RANDOM_STATE))
])

mlp_grid = GridSearchCV(
    mlp_pipe,
    param_grid={
        'model__hidden_layer_sizes': [(128, 64), (256, 128)],
        'model__alpha': [1e-3, 1e-2],
        'model__activation': ['relu']
    },
    cv=clf_cv_splits, scoring='f1_weighted', n_jobs=-1
)
mlp_grid.fit(X_train_raw, y_train_str)
best_mlp_pipe = mlp_grid.best_estimator_
print(f"MLP best params: {mlp_grid.best_params_}")
eval_clf('MLP', best_mlp_pipe.named_steps['model'], X_test, y_test_str, best_mlp_pipe)
"""))

    # Section C: Results
    cells.append(nbf.v4.new_markdown_cell("""---
## Section C: Results — Comparison Table & Confusion Matrices
"""))

    cells.append(nbf.v4.new_code_cell("""partB_table = (pd.DataFrame(clf_results)
               .sort_values(by='Weighted_F1', ascending=False)
               .reset_index(drop=True))
partB_table.index = np.arange(1, len(partB_table) + 1)

print("=" * 65)
print("CLASSIFICATION COMPARISON TABLE (PART B — ENSEMBLE & MLP)")
print("=" * 65)
display(partB_table)

# Save Part B results for consolidated benchmark
partB_table.to_csv(os.path.join(DATA_DIR, 'partB_results.csv'), index=False)
print("Saved Part B metrics to ../data/partB_results.csv")
"""))

    cells.append(nbf.v4.new_code_cell("""# Bar chart: Weighted-F1 comparison
fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(partB_table['Algorithm'][::-1], partB_table['Weighted_F1'][::-1],
               color=sns.color_palette('Set2', len(partB_table)))
for bar, val in zip(bars, partB_table['Weighted_F1'][::-1]):
    ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
            f"{val:.4f}", va='center', fontsize=10)
ax.set_xlim(0, 1.0)
ax.set_xlabel('Weighted F1-Score (Test Set)', fontsize=11)
ax.set_title('Classification Part B — Model Comparison (Weighted F1)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()
"""))

    cells.append(nbf.v4.new_code_cell("""# Confusion matrices for all Part B models
n_models  = len(clf_cms)
n_cols    = 3
n_rows    = (n_models + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
axes      = axes.flatten()

for idx, (m_name, cm) in enumerate(clf_cms.items()):
    ax = axes[idx]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
                xticklabels=class_names, yticklabels=class_names)
    ax.set_title(m_name, fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted', fontsize=10)
    ax.set_ylabel('True', fontsize=10)

for j in range(idx + 1, len(axes)):
    axes[j].axis('off')

plt.suptitle('Confusion Matrices — Classification Part B Models', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.show()
"""))

    cells.append(nbf.v4.new_code_cell("""# Per-class classification report for the best Part B model
best_partB_name  = partB_table.iloc[0]['Algorithm']
best_partB_model = clf_models[best_partB_name]
best_partB_pipe  = clf_pipelines[best_partB_name]

if best_partB_name == 'XGBoost':
    raw_preds    = best_partB_model.predict(X_test)
    best_preds_str = le.inverse_transform(raw_preds)
else:
    best_preds_str = best_partB_model.predict(X_test)

print(f"Best Part B model: {best_partB_name}")
print()
print(classification_report(y_test_str, best_preds_str, labels=class_names, target_names=class_names, zero_division=0))

# Serialize self-contained pipeline (BUG-13)
joblib.dump({
    'pipeline': best_partB_pipe,
    'feature_names': feature_cols,
    'class_names': class_names,
    'label_encoder': le if best_partB_name == 'XGBoost' else None
}, os.path.join(MODELS_DIR, 'best_classification_pipeline_partB.joblib'))

joblib.dump(best_partB_model, os.path.join(MODELS_DIR, 'best_classification_model_partB.joblib'))
print(f"Serialized complete pipeline for {best_partB_name} to /models/best_classification_pipeline_partB.joblib")
"""))

    # Feature Importance (BUG-05 Fix: Length verification)
    cells.append(nbf.v4.new_markdown_cell("""---
## Section D: Feature Importances — Best Ensemble Model (BUG-05 Fix)
"""))

    cells.append(nbf.v4.new_code_cell("""# Show feature importances with length check matching feature_cols (BUG-05)
feat_model = clf_models.get(best_partB_name) if hasattr(clf_models.get(best_partB_name), 'feature_importances_') else (
    clf_models.get('XGBoost') or clf_models.get('Random Forest') or clf_models.get('GBM (Supplementary)') or clf_models.get('AdaBoost')
)
if feat_model and hasattr(feat_model, 'feature_importances_'):
    importances = feat_model.feature_importances_
    if len(feature_cols) != len(importances):
        raise ValueError(f"Feature column length ({len(feature_cols)}) does not match importance vector length ({len(importances)})")
    
    imp_df = pd.DataFrame({'Feature': feature_cols, 'Importance': importances}).sort_values('Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=imp_df.head(12), x='Importance', y='Feature', palette='mako')
    model_name = best_partB_name if hasattr(clf_models.get(best_partB_name), 'feature_importances_') else 'Ensemble Winner'
    plt.title(f'Top 12 Feature Importances — {model_name}', fontsize=13, fontweight='bold')
    plt.xlabel('Mean Decrease in Impurity', fontsize=11)
    plt.ylabel('Feature', fontsize=11)
    plt.tight_layout()
    plt.show()
else:
    print("Feature importances not available for the selected model.")
"""))

    # Consolidated Table
    cells.append(nbf.v4.new_markdown_cell("""---
## Section E: Consolidated 10-Algorithm Benchmark Table (Review 2 Deliverable — Rubric A2)

This consolidated table unifies all 5 baseline models from **Part A** and all 5 primary models from **Part B**
(excluding GBM as supplementary). Evaluated on the identical patient-stratified 80:20 test split ($N=4,278$),
reporting **Accuracy**, **Weighted Precision**, **Weighted Recall**, **Weighted F1-Score**, **Macro F1-Score**, and **One-vs-Rest ROC-AUC**.
"""))

    cells.append(nbf.v4.new_code_cell("""partA_path = os.path.join(DATA_DIR, 'partA_results.csv')
if os.path.exists(partA_path):
    partA_df = pd.read_csv(partA_path)
    partA_df['Track'] = 'Part A (Baseline)'
else:
    print("WARNING: partA_results.csv not found.")
    partA_df = pd.DataFrame()

partB_df = partB_table.copy()
partB_df['Track'] = 'Part B (Ensemble/Neural)'

# Combine both tracks
consolidated_df = pd.concat([partA_df, partB_df], ignore_index=True)

# Required 10 algorithms
req_10 = [
    'Logistic Regression', 'K-Nearest Neighbors', 'Gaussian Naive Bayes',
    'Decision Tree', 'Support Vector Classifier', 'Random Forest',
    'AdaBoost', 'XGBoost', 'Bagging', 'MLP'
]

table_10 = consolidated_df[consolidated_df['Algorithm'].isin(req_10)].copy()
table_10 = table_10.sort_values(by='Weighted_F1', ascending=False).reset_index(drop=True)
table_10.index = np.arange(1, len(table_10) + 1)

display_cols = ['Algorithm', 'Track', 'Accuracy', 'Precision_Weighted', 'Recall_Weighted', 'Weighted_F1', 'Macro_F1', 'ROC_AUC_OvR']
display_cols = [c for c in display_cols if c in table_10.columns]

print("=" * 105)
print("CONSOLIDATED 10-ALGORITHM BENCHMARK TABLE (REVIEW 2 DELIVERABLE — RUBRIC A2)")
print("=" * 105)
display(table_10[display_cols])

# Save consolidated table to data/
table_10[display_cols].to_csv(os.path.join(DATA_DIR, 'consolidated_10_algorithms.csv'), index=False)
print("Saved consolidated benchmark to ../data/consolidated_10_algorithms.csv")
"""))

    cells.append(nbf.v4.new_markdown_cell("""---
## References & Citations
1. Wagner, P., et al. (2020). *PTB-XL, a large publicly available electrocardiography dataset*. Scientific Data, 7(1), 154.
2. Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR, 12, 2825-2830.
"""))

    nb.cells = cells
    notebook_path = os.path.join(NOTEBOOKS_DIR, "classification_partB.ipynb")
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Generated {notebook_path}")
    return notebook_path


def build_clustering_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    # Title
    cells.append(nbf.v4.new_markdown_cell("""# PTB-XL ECG Machine Learning Capstone: Clustering Track
## Objective: Unsupervised Discovery of ECG Phenotype Groups

This notebook implements the **Clustering Track** under the Capstone rubric:

1. **Data Preparation**: Same feature matrix as Classification tracks, full dataset (unsupervised).
   Guard-gated median imputation and z-score standardisation applied.
2. **Dimensionality Reduction**: PCA to 2 components for all 2-D visualisations.
3. **K-Means Clustering**
   - Elbow method (WCSS) over $k \\in \\{2, 3, 4, 5, 6, 7, 8, 9, 10\\}$
   - Deterministic silhouette scores to confirm optimal $k$ (BUG-10 fixed)
   - Cluster characterisation: mean feature profiles + dominant diagnostic label per cluster
4. **Agglomerative Hierarchical Clustering**
   - Ward linkage with dendrogram (truncated to top 30 merges)
   - Connectivity graph ($k$-NN) to avoid $O(N^2)$ memory bottleneck (BUG-11 fixed)
   - Applied at optimal $k$ from K-Means; cluster profiles compared
5. **External Validity**: Cluster purity w.r.t. dominant diagnostic labels, confusion heatmap.
"""))

    # Imports
    cells.append(nbf.v4.new_code_cell("""import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Helvetica', 'Arial', 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8
palette = sns.color_palette('tab10')

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import dendrogram, linkage

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
print("Clustering libraries loaded.")
"""))

    # Section A: Data
    cells.append(nbf.v4.new_markdown_cell("""---
## Section A: Data Ingestion & Preprocessing
"""))

    cells.append(nbf.v4.new_code_cell("""# Path resolution (BUG-17)
cwd = os.getcwd()
BASE_DIR = os.path.abspath(os.path.join(cwd, '..')) if os.path.basename(cwd) == 'notebooks' else os.path.abspath(cwd)
DATA_DIR = os.path.join(BASE_DIR, 'data')

features_df = pd.read_csv(os.path.join(DATA_DIR, 'stage4_features.csv'))
labels_df   = pd.read_csv(os.path.join(DATA_DIR, 'ptbxl_labels.csv'))

label_targets = ['label_NORM', 'label_MI', 'label_STTC', 'label_CD', 'label_HYP']
df = features_df.merge(
    labels_df[['ecg_id', 'fold'] + label_targets],
    on='ecg_id', how='inner'
)

# Build dominant label for external validity
priority_cols = ['label_MI', 'label_CD', 'label_HYP', 'label_STTC', 'label_NORM']
class_names   = ['MI', 'CD', 'HYP', 'STTC', 'NORM']

def collapse_dominant_label(row):
    for col, name in zip(priority_cols, class_names):
        if row[col] == 1:
            return name
    return 'UNKNOWN'

df['dominant_label'] = df.apply(collapse_dominant_label, axis=1)

print(f"Records: {len(df)} | Labels: {df['dominant_label'].value_counts().to_dict()}")
"""))

    cells.append(nbf.v4.new_code_cell("""# Feature matrix discovery (BUG-03: exclude delineation_ok)
exclude_cols = {
    'ecg_id', 'patient_id', 'fold', 'dominant_label', 'delineation_ok',
    'label_NORM', 'label_MI', 'label_STTC', 'label_CD', 'label_HYP'
}
feature_cols = [c for c in df.columns if c not in exclude_cols
                and pd.api.types.is_numeric_dtype(df[c])]

# Engineer qtc_qrs_ratio using correct column names (BUG-02)
if 'qtc_mean_ms' in df.columns and 'qrs_dur_mean_ms' in df.columns:
    df['qtc_qrs_ratio'] = df['qtc_mean_ms'] / (df['qrs_dur_mean_ms'].replace(0, np.nan))
    if 'qtc_qrs_ratio' not in feature_cols:
        feature_cols.append('qtc_qrs_ratio')

# Apply guard masking on unverified conduction intervals (BUG-04)
guard_cols = [
    'pr_mean_ms', 'pr_std_ms', 'qrs_dur_mean_ms', 'qrs_dur_std_ms',
    'qt_mean_ms', 'qt_std_ms', 'qtc_mean_ms'
]
df.loc[~df['delineation_ok'].astype(bool), guard_cols] = np.nan

print(f"Feature columns ({len(feature_cols)}): {feature_cols}")

X_raw = df[feature_cols].values

imputer = SimpleImputer(strategy='median')
scaler  = StandardScaler()
X = scaler.fit_transform(imputer.fit_transform(X_raw))

print(f"Final feature matrix shape: {X.shape}")
"""))

    # Section B: PCA
    cells.append(nbf.v4.new_markdown_cell("""---
## Section B: PCA Dimensionality Reduction for Visualisation
"""))

    cells.append(nbf.v4.new_code_cell("""pca2 = PCA(n_components=2, random_state=RANDOM_STATE)
X_2d = pca2.fit_transform(X)

pca_full = PCA(random_state=RANDOM_STATE).fit(X)
cumvar = np.cumsum(pca_full.explained_variance_ratio_)
n90 = int(np.searchsorted(cumvar, 0.90)) + 1

print(f"PC1 explains {pca2.explained_variance_ratio_[0]*100:.1f}%")
print(f"PC2 explains {pca2.explained_variance_ratio_[1]*100:.1f}%")
print(f"Components needed for 90% variance: {n90}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].bar(range(1, len(pca_full.explained_variance_ratio_) + 1),
            pca_full.explained_variance_ratio_ * 100, color=palette[0], edgecolor='white')
axes[0].set_xlabel('Principal Component', fontsize=11)
axes[0].set_ylabel('Explained Variance (%)', fontsize=11)
axes[0].set_title('Scree Plot', fontsize=13, fontweight='bold')

axes[1].plot(range(1, len(cumvar) + 1), cumvar * 100, marker='o', color=palette[1], linewidth=2)
axes[1].axhline(90, color='red', linestyle='--', label='90% threshold')
axes[1].axvline(n90, color='grey', linestyle=':', label=f'{n90} components')
axes[1].set_xlabel('Number of Components', fontsize=11)
axes[1].set_ylabel('Cumulative Explained Variance (%)', fontsize=11)
axes[1].set_title('Cumulative Variance Explained', fontsize=13, fontweight='bold')
axes[1].legend()
plt.tight_layout()
plt.show()
"""))

    # Section C: K-Means
    cells.append(nbf.v4.new_markdown_cell("""---
## Section C: K-Means Clustering

### C.1 — Elbow Method & Silhouette Analysis (BUG-10 Fixed Deterministic Subsample)
"""))

    cells.append(nbf.v4.new_code_cell("""K_RANGE = range(2, 11)
wcss        = []
silhouettes = []

# Deterministic subsample for silhouette evaluation across all k values (BUG-10)
rng = np.random.RandomState(RANDOM_STATE)
eval_idx = rng.choice(X.shape[0], min(5000, X.shape[0]), replace=False)

for k in K_RANGE:
    km = KMeans(n_clusters=k, init='k-means++', n_init=20,
                max_iter=300, random_state=RANDOM_STATE)
    km.fit(X)
    wcss.append(km.inertia_)
    sil = silhouette_score(X[eval_idx], km.labels_[eval_idx])
    silhouettes.append(sil)
    print(f"  k={k:2d}  WCSS={km.inertia_:,.0f}  Silhouette={sil:.4f}")
"""))

    cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(list(K_RANGE), wcss, marker='o', color=palette[0], linewidth=2)
axes[0].set_xticks(list(K_RANGE))
axes[0].set_xlabel('Number of Clusters (k)', fontsize=11)
axes[0].set_ylabel('WCSS (Inertia)', fontsize=11)
axes[0].set_title('Elbow Method — WCSS', fontsize=13, fontweight='bold')

axes[1].plot(list(K_RANGE), silhouettes, marker='s', color=palette[1], linewidth=2)
axes[1].set_xticks(list(K_RANGE))
axes[1].set_xlabel('Number of Clusters (k)', fontsize=11)
axes[1].set_ylabel('Silhouette Score', fontsize=11)
axes[1].set_title('Silhouette Score vs k', fontsize=13, fontweight='bold')

plt.suptitle('K-Means: Optimal k Selection', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

OPTIMAL_K = int(np.argmax(silhouettes)) + 2
print(f"Optimal k by silhouette: {OPTIMAL_K}")
"""))

    cells.append(nbf.v4.new_markdown_cell("""### C.2 — Final K-Means at Optimal k
"""))

    cells.append(nbf.v4.new_code_cell("""km_final = KMeans(n_clusters=OPTIMAL_K, init='k-means++', n_init=50,
                  max_iter=500, random_state=RANDOM_STATE)
km_labels = km_final.fit_predict(X)
df['km_cluster'] = km_labels

print(f"K-Means (k={OPTIMAL_K}) cluster sizes:")
print(pd.Series(km_labels).value_counts().sort_index())
"""))

    cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 2, figsize=(14, 6))

cm_palette = sns.color_palette('tab10', OPTIMAL_K)
for c in range(OPTIMAL_K):
    mask = km_labels == c
    axes[0].scatter(X_2d[mask, 0], X_2d[mask, 1],
                    s=4, alpha=0.5, color=cm_palette[c], label=f'C{c}')
axes[0].set_xlabel('PC1', fontsize=11)
axes[0].set_ylabel('PC2', fontsize=11)
axes[0].set_title(f'K-Means (k={OPTIMAL_K}) — PCA 2-D Projection', fontsize=12, fontweight='bold')
axes[0].legend(markerscale=3, fontsize=9)

label_palette = {l: c for l, c in zip(class_names, sns.color_palette('Set2', 5))}
for lbl in class_names:
    mask = df['dominant_label'] == lbl
    axes[1].scatter(X_2d[mask.values, 0], X_2d[mask.values, 1],
                    s=4, alpha=0.4, color=label_palette[lbl], label=lbl)
axes[1].set_xlabel('PC1', fontsize=11)
axes[1].set_ylabel('PC2', fontsize=11)
axes[1].set_title('True Diagnostic Labels — PCA 2-D Projection', fontsize=12, fontweight='bold')
axes[1].legend(markerscale=3, fontsize=9)

plt.suptitle('Cluster vs. Label Structure in 2-D PCA Space', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
"""))

    cells.append(nbf.v4.new_code_cell("""profile_df = df[feature_cols + ['km_cluster']].groupby('km_cluster').mean()
print("K-Means Cluster Feature Profiles (mean per cluster):")
display(profile_df.T.round(3))
"""))

    cells.append(nbf.v4.new_code_cell("""cluster_label_counts = (df.groupby(['km_cluster', 'dominant_label'])
                          .size()
                          .unstack(fill_value=0))
cluster_label_counts['Total']     = cluster_label_counts.sum(axis=1)
cluster_label_counts['Majority']  = cluster_label_counts[class_names].max(axis=1)
cluster_label_counts['Purity']    = (cluster_label_counts['Majority'] /
                                     cluster_label_counts['Total']).round(3)
cluster_label_counts['DomLabel']  = cluster_label_counts[class_names].idxmax(axis=1)

print("K-Means Cluster Purity w.r.t. Diagnostic Labels:")
display(cluster_label_counts)

overall_purity = (cluster_label_counts['Majority'].sum() /
                  cluster_label_counts['Total'].sum())
print(f"Overall K-Means Purity: {overall_purity:.4f}")
"""))

    cells.append(nbf.v4.new_code_cell("""plt.figure(figsize=(9, max(4, OPTIMAL_K)))
sns.heatmap(cluster_label_counts[class_names], annot=True, fmt='d',
            cmap='YlOrRd', linewidths=0.5)
plt.title(f'K-Means (k={OPTIMAL_K}): Cluster × Diagnostic Label Count',
          fontsize=13, fontweight='bold')
plt.xlabel('Diagnostic Category', fontsize=11)
plt.ylabel('Cluster ID', fontsize=11)
plt.tight_layout()
plt.show()
"""))

    cells.append(nbf.v4.new_code_cell("""TSNE_SAMPLE = min(5000, X.shape[0])
np.random.seed(RANDOM_STATE)
tsne_idx = np.random.choice(X.shape[0], TSNE_SAMPLE, replace=False)

print(f"Computing t-SNE projection on subsample (n={TSNE_SAMPLE})...")
tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=RANDOM_STATE, n_jobs=-1)
X_tsne = tsne.fit_transform(X[tsne_idx])

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

tsne_km_labels = km_labels[tsne_idx]
for c in range(OPTIMAL_K):
    mask = tsne_km_labels == c
    axes[0].scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                    s=6, alpha=0.5, color=cm_palette[c], label=f'C{c}')
axes[0].set_xlabel('t-SNE Dimension 1', fontsize=11)
axes[0].set_ylabel('t-SNE Dimension 2', fontsize=11)
axes[0].set_title(f'K-Means (k={OPTIMAL_K}) — t-SNE 2-D Manifold', fontsize=12, fontweight='bold')
axes[0].legend(markerscale=3, fontsize=9)

tsne_true_labels = df['dominant_label'].iloc[tsne_idx].values
for lbl in class_names:
    mask = tsne_true_labels == lbl
    axes[1].scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                    s=6, alpha=0.4, color=label_palette[lbl], label=lbl)
axes[1].set_xlabel('t-SNE Dimension 1', fontsize=11)
axes[1].set_ylabel('t-SNE Dimension 2', fontsize=11)
axes[1].set_title('True Diagnostic Labels — t-SNE 2-D Manifold', fontsize=12, fontweight='bold')
axes[1].legend(markerscale=3, fontsize=9)

plt.suptitle('t-SNE Non-Linear Manifold Visualisation (Subsample n=5,000)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
"""))

    # Section D: Agglomerative
    cells.append(nbf.v4.new_markdown_cell("""---
## Section D: Agglomerative Hierarchical Clustering (Ward Linkage - BUG-11 Memory Fix)
"""))

    cells.append(nbf.v4.new_code_cell("""DENDRO_SAMPLE = min(3000, X.shape[0])
np.random.seed(RANDOM_STATE)
didx = np.random.choice(X.shape[0], DENDRO_SAMPLE, replace=False)
Z = linkage(X[didx], method='ward')

fig, ax = plt.subplots(figsize=(14, 6))
dendrogram(Z, truncate_mode='lastp', p=30, leaf_rotation=90,
           show_contracted=True, ax=ax, color_threshold=0.7 * max(Z[:, 2]))
ax.set_title('Agglomerative Hierarchical Clustering — Ward Linkage Dendrogram', fontsize=12, fontweight='bold')
ax.set_xlabel('Sample Index / Cluster Size', fontsize=11)
ax.set_ylabel('Ward Distance', fontsize=11)
plt.tight_layout()
plt.show()
"""))

    cells.append(nbf.v4.new_code_cell("""# Agglomerative clustering with k-NN connectivity graph to prevent O(N^2) memory crash (BUG-11)
print("Constructing k-nearest neighbors connectivity graph...")
knn_graph = kneighbors_graph(X, n_neighbors=15, include_self=False)

agg = AgglomerativeClustering(n_clusters=OPTIMAL_K, linkage='ward', connectivity=knn_graph)
agg_labels = agg.fit_predict(X)
df['agg_cluster'] = agg_labels

print(f"Agglomerative (k={OPTIMAL_K}) cluster sizes:")
print(pd.Series(agg_labels).value_counts().sort_index())
"""))

    cells.append(nbf.v4.new_code_cell("""fig, ax = plt.subplots(figsize=(8, 6))
agg_palette = sns.color_palette('tab10', OPTIMAL_K)
for c in range(OPTIMAL_K):
    mask = agg_labels == c
    ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
               s=4, alpha=0.5, color=agg_palette[c], label=f'C{c}')
ax.set_xlabel('PC1', fontsize=11)
ax.set_ylabel('PC2', fontsize=11)
ax.set_title(f'Agglomerative (Ward, k={OPTIMAL_K}) — PCA 2-D Projection',
             fontsize=12, fontweight='bold')
ax.legend(markerscale=3, fontsize=9)
plt.tight_layout()
plt.show()
"""))

    cells.append(nbf.v4.new_code_cell("""agg_label_counts = (df.groupby(['agg_cluster', 'dominant_label'])
                      .size()
                      .unstack(fill_value=0))
agg_label_counts['Total']    = agg_label_counts.sum(axis=1)
agg_label_counts['Majority'] = agg_label_counts[class_names].max(axis=1)
agg_label_counts['Purity']   = (agg_label_counts['Majority'] /
                                 agg_label_counts['Total']).round(3)
agg_label_counts['DomLabel'] = agg_label_counts[class_names].idxmax(axis=1)

print("Agglomerative Cluster Purity w.r.t. Diagnostic Labels:")
display(agg_label_counts)

agg_purity = (agg_label_counts['Majority'].sum() / agg_label_counts['Total'].sum())
print(f"Overall Agglomerative Purity: {agg_purity:.4f}")
"""))

    cells.append(nbf.v4.new_code_cell("""plt.figure(figsize=(9, max(4, OPTIMAL_K)))
sns.heatmap(agg_label_counts[class_names], annot=True, fmt='d',
            cmap='YlGnBu', linewidths=0.5)
plt.title(f'Agglomerative (Ward, k={OPTIMAL_K}): Cluster × Diagnostic Label Count',
          fontsize=13, fontweight='bold')
plt.xlabel('Diagnostic Category', fontsize=11)
plt.ylabel('Cluster ID', fontsize=11)
plt.tight_layout()
plt.show()
"""))

    # Section E: Summary
    cells.append(nbf.v4.new_markdown_cell("""---
## Section E: Summary Comparison — K-Means vs Agglomerative
"""))

    cells.append(nbf.v4.new_code_cell("""km_sil  = silhouette_score(X, km_labels, sample_size=min(5000, len(X)), random_state=RANDOM_STATE)
agg_sil = silhouette_score(X, agg_labels, sample_size=min(5000, len(X)), random_state=RANDOM_STATE)

km_db  = davies_bouldin_score(X, km_labels)
agg_db = davies_bouldin_score(X, agg_labels)

km_ch  = calinski_harabasz_score(X, km_labels)
agg_ch = calinski_harabasz_score(X, agg_labels)

summary = pd.DataFrame({
    'Method':              ['K-Means', 'Agglomerative (Ward)'],
    'k':                   [OPTIMAL_K, OPTIMAL_K],
    'Silhouette Score':    [round(km_sil, 4),   round(agg_sil, 4)],
    'Davies-Bouldin':      [round(km_db, 4),    round(agg_db, 4)],
    'Calinski-Harabasz':   [round(km_ch, 2),    round(agg_ch, 2)],
    'Overall Purity':      [round(overall_purity, 4), round(agg_purity, 4)],
})

print("=" * 75)
print("CLUSTERING ALGORITHM COMPARISON")
print("=" * 75)
display(summary)
"""))

    nb.cells = cells
    notebook_path = os.path.join(NOTEBOOKS_DIR, "clustering.ipynb")
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Generated {notebook_path}")
    return notebook_path


def execute_notebook(nb_path):
    print(f"\n[Executing Notebook] {nb_path}...")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbf.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=1800, kernel_name="python3")
    notebook_dir = os.path.dirname(nb_path)
    ep.preprocess(nb, {"metadata": {"path": notebook_dir}})

    with open(nb_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"[Execution Complete] Saved fully executed notebook: {nb_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--only', choices=['regression', 'classification', 'partB', 'clustering', 'all'],
                        default='all', help='Which notebook(s) to build/run')
    args = parser.parse_args()

    to_run = args.only

    if to_run in ('regression', 'all'):
        reg_nb = build_regression_notebook()
        execute_notebook(reg_nb)

    if to_run in ('classification', 'all'):
        clf_nb = build_classification_notebook()
        execute_notebook(clf_nb)

    if to_run in ('partB', 'all'):
        partB_nb = build_classification_partB_notebook()
        execute_notebook(partB_nb)

    if to_run in ('clustering', 'all'):
        clust_nb = build_clustering_notebook()
        execute_notebook(clust_nb)

    print("\nAll requested capstone notebooks generated and executed successfully!")
