"""
================================================================================
PTB-XL ECG Diagnostic & Prognostic Prediction Engine
Classification (Part A & Part B Ensembles) + Biological Age Regression
================================================================================

This module provides a unified inference and clinical decision-support interface
for predicting the primary cardiac pathology and estimated biological patient age
from 12-lead ECG features.

Supported Diagnostic Superclasses:
  - MI   : Myocardial Infarction (Ischemia / Necrosis)
  - CD   : Conduction Disturbance (Bundle Branch Blocks / AV Blocks)
  - HYP  : Ventricular / Atrial Hypertrophy
  - STTC : ST-T Wave Changes (Repolarization Abnormality)
  - NORM : Normal Sinus ECG

Supported Regression Target:
  - Biological Age (years) derived from 12-lead electrophysiological features

Usage Examples:
  CLI - Clinical Demo with representative test cases:
    python prediction.py --demo

  CLI - Predict for a specific record in the dataset:
    python prediction.py --ecg-id 14314

  CLI - Predict from custom physiological parameters:
    python prediction.py --hr 72 --pr 160 --qrs 95 --qtc 410 --st -0.02 --t-area 0.008

  CLI - Batch predict from a CSV file:
    python prediction.py --csv path/to/features.csv --output results.csv

  Python API:
    from prediction import ECGPredictor
    predictor = ECGPredictor()
    result = predictor.predict({'hr_bpm': 75.0, 'qrs_dur_mean_ms': 95.0, ...})
    print(result['predicted_class'], result['confidence'], result['estimated_age'])
================================================================================
"""

import os
import sys
import argparse
import warnings

# Suppress warnings for clean CLI and module usage
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

DEFAULT_MODEL_PATH = os.path.join(MODELS_DIR, "best_classification_pipeline_partB.joblib")
FALLBACK_MODEL_PATH = os.path.join(MODELS_DIR, "best_classification_pipeline_partA.joblib")
REGRESSION_MODEL_PATH = os.path.join(MODELS_DIR, "best_regression_pipeline.joblib")

CLASS_NAMES = ["MI", "CD", "HYP", "STTC", "NORM"]

CLASS_DESCRIPTIONS = {
    "MI": "Myocardial Infarction (Acute/Prior Coronary Occlusion or Ischemia)",
    "CD": "Conduction Disturbance (Bundle Branch Block or AV Delay)",
    "HYP": "Ventricular or Atrial Hypertrophy (Chamber Enlargement)",
    "STTC": "Non-Specific ST-T Segment Changes (Repolarization Anomaly)",
    "NORM": "Normal Electrocardiogram (No Significant Diagnostic Pathology)"
}

GUARD_COLS = [
    "pr_mean_ms", "pr_std_ms", "qrs_dur_mean_ms", "qrs_dur_std_ms",
    "qt_mean_ms", "qt_std_ms", "qtc_mean_ms"
]

# Canonical representative test-fold records for clinical demonstration
REPRESENTATIVE_TEST_CASES = {
    "MI": 14314,   # Acute anterior/inferior MI with ST elevation
    "CD": 18499,   # Bundle branch block with severe QRS prolongation
    "HYP": 2417,   # Left ventricular hypertrophy with strain pattern
    "STTC": 6735,  # Non-specific repolarization anomaly with tachycardia
    "NORM": 9010   # Healthy normal sinus electrocardiogram
}


class ECGPredictor:
    """
    Unified inference engine that encapsulates pipeline loading, feature validation,
    ratio engineering, guard handling, diagnostic classification, and biological age regression.
    """

    def __init__(self, model_path=None, model_type="partB", reg_model_path=None):
        self.model_type = model_type
        if model_path is None:
            if model_type.lower() == "parta" and os.path.exists(FALLBACK_MODEL_PATH):
                model_path = FALLBACK_MODEL_PATH
            elif os.path.exists(DEFAULT_MODEL_PATH):
                model_path = DEFAULT_MODEL_PATH
            elif os.path.exists(FALLBACK_MODEL_PATH):
                model_path = FALLBACK_MODEL_PATH
            else:
                raise FileNotFoundError(
                    f"No serialized classification model found at {DEFAULT_MODEL_PATH} or {FALLBACK_MODEL_PATH}. "
                    "Please ensure the models/ directory contains the trained pipeline."
                )

        self.model_path = model_path
        self.reg_model_path = reg_model_path or REGRESSION_MODEL_PATH
        self._load_artifacts()

    def _load_artifacts(self):
        # 1. Load Classification Pipeline
        bundle = joblib.load(self.model_path)
        if isinstance(bundle, dict):
            self.pipeline = bundle.get("pipeline")
            self.feature_names = bundle.get("feature_names", [])
            self.class_names = bundle.get("class_names", CLASS_NAMES)
            self.label_encoder = bundle.get("label_encoder", None)
        else:
            self.pipeline = bundle
            self.feature_names = []
            self.class_names = CLASS_NAMES
            self.label_encoder = None

        if self.pipeline is None:
            raise ValueError(f"Failed to load classification pipeline from {self.model_path}")

        # 2. Load Regression Pipeline (Biological Age)
        self.reg_pipeline = None
        self.reg_feature_names = []
        if os.path.exists(self.reg_model_path):
            try:
                reg_bundle = joblib.load(self.reg_model_path)
                if isinstance(reg_bundle, dict):
                    self.reg_pipeline = reg_bundle.get("pipeline")
                    self.reg_feature_names = reg_bundle.get("feature_names", self.feature_names)
                else:
                    self.reg_pipeline = reg_bundle
                    self.reg_feature_names = self.feature_names
            except Exception:
                self.reg_pipeline = None

    def prepare_features(self, input_df, target_features=None):
        """
        Ensures all expected features are present, calculates the physiological
        qtc_qrs_ratio, and applies delineation guard masking.
        Missing values are populated with np.nan so the trained pipeline's
        SimpleImputer(strategy='median') accurately populates population medians.
        """
        df = input_df.copy()
        feat_list = target_features if target_features is not None else self.feature_names

        # 1. Derive repolarization/depolarization ratio
        if "qtc_qrs_ratio" not in df.columns or df["qtc_qrs_ratio"].isna().all():
            if "qtc_mean_ms" in df.columns and "qrs_dur_mean_ms" in df.columns:
                qrs_safe = df["qrs_dur_mean_ms"].replace(0, np.nan)
                df["qtc_qrs_ratio"] = df["qtc_mean_ms"] / qrs_safe
            else:
                df["qtc_qrs_ratio"] = np.nan

        # 2. Derive RR interval if heart rate is provided
        if "rr_mean_ms" not in df.columns or df["rr_mean_ms"].isna().all():
            if "hr_bpm" in df.columns:
                df["rr_mean_ms"] = 60000.0 / df["hr_bpm"].replace(0, np.nan)

        # 3. Guard masking on unverified beats
        if "delineation_ok" in df.columns:
            unreliable = ~df["delineation_ok"].astype(bool)
            for gc in GUARD_COLS:
                if gc in df.columns:
                    df.loc[unreliable, gc] = np.nan

        # 4. Fill missing expected columns with NaN (the pipeline imputer will handle them)
        for col in feat_list:
            if col not in df.columns:
                df[col] = np.nan

        # Order columns exactly as expected by the trained pipeline
        return df[feat_list]

    def predict(self, data):
        """
        Accepts dict, Series, or DataFrame.
        Returns prediction dictionary (or list of dicts) with:
          - predicted_class: Primary diagnostic pathology (MI, CD, HYP, STTC, NORM)
          - description: Clinical description of the condition
          - confidence: Model prediction probability
          - probabilities: Full class distribution
          - estimated_age: Predicted biological patient age (years)
          - clinical_flags: Actionable electrophysiological alerts
        """
        if isinstance(data, dict):
            df_in = pd.DataFrame([data])
            single = True
        elif isinstance(data, pd.Series):
            df_in = pd.DataFrame([data.to_dict()])
            single = True
        elif isinstance(data, pd.DataFrame):
            df_in = data
            single = False
        else:
            raise TypeError("Input data must be a dict, pd.Series, or pd.DataFrame")

        X_prep = self.prepare_features(df_in, target_features=self.feature_names)
        X_arr = X_prep.values

        # Determine exact class label order from the underlying estimator
        model_step = getattr(self.pipeline, "named_steps", {}).get("model", self.pipeline)
        if hasattr(model_step, "classes_"):
            raw_classes = list(model_step.classes_)
            if self.label_encoder is not None:
                model_class_labels = list(self.label_encoder.inverse_transform(raw_classes))
            else:
                model_class_labels = [str(c) for c in raw_classes]
        else:
            model_class_labels = self.class_names

        # Classification inference
        if hasattr(self.pipeline, "predict_proba"):
            probas = self.pipeline.predict_proba(X_arr)
        else:
            probas = None

        preds_raw = self.pipeline.predict(X_arr)
        if self.label_encoder is not None:
            preds_str = self.label_encoder.inverse_transform(preds_raw)
        else:
            preds_str = [str(p) for p in preds_raw]

        # Biological Age Regression inference
        if self.reg_pipeline is not None:
            reg_feat_list = self.reg_feature_names if self.reg_feature_names else self.feature_names
            X_reg_prep = self.prepare_features(df_in, target_features=reg_feat_list)
            try:
                age_preds = self.reg_pipeline.predict(X_reg_prep.values)
            except Exception:
                age_preds = [np.nan] * len(df_in)
        else:
            age_preds = [np.nan] * len(df_in)

        results = []
        for idx in range(len(df_in)):
            pred_class = str(preds_str[idx])
            prob_dict = {}

            if probas is not None:
                p_row = probas[idx]
                for c_idx, c_name in enumerate(model_class_labels):
                    prob_dict[c_name] = float(p_row[c_idx])
                confidence = prob_dict.get(pred_class, float(np.max(p_row)))
            else:
                confidence = None

            est_age = float(age_preds[idx]) if not np.isnan(age_preds[idx]) else None

            row_record = df_in.iloc[idx].to_dict()
            clinical_flags = self._generate_clinical_flags(row_record, pred_class)

            results.append({
                "predicted_class": pred_class,
                "description": CLASS_DESCRIPTIONS.get(pred_class, "Cardiac condition"),
                "confidence": confidence,
                "probabilities": prob_dict,
                "estimated_age": est_age,
                "clinical_flags": clinical_flags
            })

        return results[0] if single else results

    def _generate_clinical_flags(self, record, predicted_class):
        """
        Evaluates physiological thresholds and produces actionable clinical warnings.
        """
        flags = []

        st_level = record.get("st_level_median_mv")
        qrs_dur = record.get("qrs_dur_mean_ms")
        qtc = record.get("qtc_mean_ms")
        hr = record.get("hr_bpm")
        ratio = record.get("qtc_qrs_ratio")
        if ratio is None or pd.isna(ratio):
            if qtc and qrs_dur and qrs_dur > 0:
                ratio = qtc / qrs_dur

        # ST elevation / depression
        if st_level is not None and not pd.isna(st_level):
            if st_level > 0.10:
                flags.append(f"ST-segment elevation ({st_level:+.2f} mV): Acute myocardial injury / STEMI alert")
            elif st_level < -0.10:
                flags.append(f"ST-segment depression ({st_level:+.2f} mV): Subendocardial ischemia alert")

        # QRS prolongation
        if qrs_dur is not None and not pd.isna(qrs_dur):
            if qrs_dur > 120.0:
                flags.append(f"Prolonged QRS duration ({qrs_dur:.1f} ms): Bundle branch block / intraventricular conduction delay")

        # QTc prolongation
        if qtc is not None and not pd.isna(qtc):
            if qtc > 450.0:
                flags.append(f"Prolonged QTc interval ({qtc:.1f} ms): Increased risk of ventricular arrhythmia (Torsades de Pointes)")
            elif qtc < 350.0:
                flags.append(f"Shortened QTc interval ({qtc:.1f} ms): Short QT syndrome risk")

        # Repolarization / Depolarization Ratio
        if ratio is not None and not pd.isna(ratio) and ratio > 4.5:
            flags.append(f"High QTc/QRS ratio ({ratio:.2f}): Severe repolarization delay relative to depolarization speed")

        # Heart rate
        if hr is not None and not pd.isna(hr):
            if hr > 100.0:
                flags.append(f"Sinus tachycardia ({hr:.0f} bpm)")
            elif hr < 60.0:
                flags.append(f"Sinus bradycardia ({hr:.0f} bpm)")

        if not flags:
            flags.append("Physiological conduction and morphology metrics within standard reference ranges")

        return flags


def print_prediction_card(res, ecg_id=None, true_label=None, true_age=None):
    """
    Renders an ASCII clinical report card to the terminal.
    """
    pred = res["predicted_class"]
    conf = res["confidence"]
    probs = res["probabilities"]
    age = res.get("estimated_age")
    flags = res["clinical_flags"]

    print("=" * 75)
    header = "ECG DIAGNOSTIC PREDICTION REPORT"
    if ecg_id is not None:
        header += f" (Record ID: {ecg_id})"
    print(header.center(75))
    print("=" * 75)

    if true_label:
        match_str = "[CORRECT MATCH]" if pred == true_label else f"[TRUE LABEL: {true_label}]"
        print(f"  Diagnosis Result  : >>> {pred} <<< ({CLASS_DESCRIPTIONS.get(pred)})  {match_str}")
    else:
        print(f"  Diagnosis Result  : >>> {pred} <<< ({CLASS_DESCRIPTIONS.get(pred)})")

    if conf is not None:
        print(f"  Confidence        : {conf * 100:.1f}%")

    if age is not None:
        if true_age is not None and not pd.isna(true_age):
            delta = age - true_age
            print(f"  Estimated Bio Age : {age:.1f} years  (Chronological: {true_age:.0f} yrs, Delta: {delta:+.1f} yrs)")
        else:
            print(f"  Estimated Bio Age : {age:.1f} years")

    print("\n  Class Probability Distribution:")
    print("  " + "-" * 55)
    for cls in CLASS_NAMES:
        p = probs.get(cls, 0.0)
        bar_len = int(round(p * 30))
        bar = "#" * bar_len + "-" * (30 - bar_len)
        marker = " <== PREDICTED" if cls == pred else ""
        print(f"    {cls:<5} : [{bar}] {p * 100:5.1f}%{marker}")
    print("  " + "-" * 55)

    print("\n  Clinical Interpretations & Alerts:")
    for f in flags:
        print(f"    * {f}")
    print("=" * 75)
    print()


def run_demo(predictor):
    """
    Demonstrates model predictions across representative clinical benchmark cases
    from the unseen test fold (Fold 10).
    """
    feat_file = os.path.join(DATA_DIR, "stage4_features.csv")
    label_file = os.path.join(DATA_DIR, "ptbxl_labels.csv")
    meta_file = os.path.join(DATA_DIR, "ptbxl_metadata.csv")

    if not (os.path.exists(feat_file) and os.path.exists(label_file)):
        print("Dataset files not found for demo. Predicting on synthetic clinical prototypes instead:\n")
        prototypes = [
            {"name": "Acute MI Prototype", "data": {"hr_bpm": 85, "pr_mean_ms": 165, "qrs_dur_mean_ms": 95, "qtc_mean_ms": 420, "st_level_median_mv": 0.25, "t_area_median": -0.015}},
            {"name": "Conduction Delay Prototype", "data": {"hr_bpm": 68, "pr_mean_ms": 220, "qrs_dur_mean_ms": 155, "qtc_mean_ms": 460, "st_level_median_mv": -0.02, "rs_ratio_median": 0.35, "t_area_median": 0.005}},
            {"name": "Normal Sinus Prototype", "data": {"hr_bpm": 70, "pr_mean_ms": 150, "qrs_dur_mean_ms": 90, "qtc_mean_ms": 405, "st_level_median_mv": 0.01, "t_area_median": 0.008}}
        ]
        for p in prototypes:
            print(f"--- Prototype: {p['name']} ---")
            res = predictor.predict(p["data"])
            print_prediction_card(res)
        return

    f_df = pd.read_csv(feat_file)
    l_df = pd.read_csv(label_file)
    m_df = pd.read_csv(meta_file) if os.path.exists(meta_file) else None

    priority_cols = ['label_MI', 'label_CD', 'label_HYP', 'label_STTC', 'label_NORM']
    class_names = ['MI', 'CD', 'HYP', 'STTC', 'NORM']

    def get_dom(row):
        for col, name in zip(priority_cols, class_names):
            if row[col] == 1:
                return name
        return 'UNKNOWN'

    l_df['dom'] = l_df.apply(get_dom, axis=1)
    merged = f_df.merge(l_df[['ecg_id', 'dom', 'fold']], on='ecg_id', how='inner')
    if m_df is not None:
        merged = merged.merge(m_df[['ecg_id', 'age', 'sex']], on='ecg_id', how='left')

    test_cases = merged[merged['fold'] == 'test']
    print("Demonstration: Evaluating 5 Clinical Benchmark Cases from Unseen Test Fold (Fold 10)...\n")

    for target in CLASS_NAMES:
        eid = REPRESENTATIVE_TEST_CASES.get(target)
        match = test_cases[test_cases['ecg_id'] == eid]
        if match.empty:
            match = test_cases[test_cases['dom'] == target]
        if not match.empty:
            case = match.iloc[0]
            true_label = case.get('dom', target)
            true_age = case.get('age') if 'age' in case else None
            res = predictor.predict(case)
            print_prediction_card(res, ecg_id=int(case['ecg_id']), true_label=true_label, true_age=true_age)


def main():
    parser = argparse.ArgumentParser(description="PTB-XL ECG Diagnostic & Prognostic Prediction Engine")
    parser.add_argument("--demo", action="store_true", help="Run demonstration on 5 representative clinical cases")
    parser.add_argument("--ecg-id", type=int, default=None, help="Look up record by ECG ID from dataset and predict")
    parser.add_argument("--csv", type=str, default=None, help="Path to input CSV containing ECG features for batch prediction")
    parser.add_argument("--output", type=str, default="predictions_output.csv", help="Output path for batch predictions")
    parser.add_argument("--model", type=str, default="partB", choices=["partA", "partB"], help="Model pipeline to use (default: partB)")

    # Physiological parameter flags for ad-hoc queries
    parser.add_argument("--hr", type=float, default=None, help="Heart rate in beats per minute (e.g. 75.0)")
    parser.add_argument("--pr", type=float, default=None, help="PR interval duration in ms (e.g. 160.0)")
    parser.add_argument("--qrs", type=float, default=None, help="QRS complex duration in ms (e.g. 95.0)")
    parser.add_argument("--qtc", type=float, default=None, help="Bazett-corrected QTc interval in ms (e.g. 415.0)")
    parser.add_argument("--st", type=float, default=None, help="ST-segment amplitude level in mV (e.g. -0.02)")
    parser.add_argument("--t-area", type=float, default=None, help="T-wave integrated area in mV·s (e.g. 0.008)")

    args = parser.parse_args()

    predictor = ECGPredictor(model_type=args.model)

    # 1. Demo Mode
    if args.demo:
        run_demo(predictor)
        return

    # 2. Lookup by ECG ID
    if args.ecg_id is not None:
        feat_path = os.path.join(DATA_DIR, "stage4_features.csv")
        labels_path = os.path.join(DATA_DIR, "ptbxl_labels.csv")
        meta_path = os.path.join(DATA_DIR, "ptbxl_metadata.csv")

        if not os.path.exists(feat_path):
            print(f"Error: Feature dataset not found at {feat_path}")
            sys.exit(1)

        f_df = pd.read_csv(feat_path)
        record = f_df[f_df["ecg_id"] == args.ecg_id]

        if record.empty:
            print(f"Error: Record ecg_id={args.ecg_id} not found in {feat_path}")
            sys.exit(1)

        true_label = None
        if os.path.exists(labels_path):
            l_df = pd.read_csv(labels_path)
            l_row = l_df[l_df["ecg_id"] == args.ecg_id]
            if not l_row.empty:
                priority_cols = ['label_MI', 'label_CD', 'label_HYP', 'label_STTC', 'label_NORM']
                class_names = ['MI', 'CD', 'HYP', 'STTC', 'NORM']
                for col, name in zip(priority_cols, class_names):
                    if col in l_row and l_row.iloc[0][col] == 1:
                        true_label = name
                        break

        true_age = None
        if os.path.exists(meta_path):
            m_df = pd.read_csv(meta_path)
            m_row = m_df[m_df["ecg_id"] == args.ecg_id]
            if not m_row.empty and 'age' in m_row:
                age_val = m_row.iloc[0]['age']
                if not pd.isna(age_val) and age_val < 300:
                    true_age = float(age_val)

        res = predictor.predict(record.iloc[0])
        print_prediction_card(res, ecg_id=args.ecg_id, true_label=true_label, true_age=true_age)
        return

    # 3. Batch prediction on CSV
    if args.csv is not None:
        if not os.path.exists(args.csv):
            print(f"Error: Input file {args.csv} does not exist.")
            sys.exit(1)

        input_df = pd.read_csv(args.csv)
        print(f"Loaded {len(input_df)} records from {args.csv}. Running predictions...")

        results = predictor.predict(input_df)

        output_df = input_df.copy()
        output_df["predicted_diagnosis"] = [r["predicted_class"] for r in results]
        output_df["confidence"] = [r["confidence"] for r in results]
        output_df["estimated_age"] = [r.get("estimated_age") for r in results]

        for cls in CLASS_NAMES:
            output_df[f"prob_{cls}"] = [r["probabilities"].get(cls, np.nan) for r in results]

        output_df["clinical_flags"] = ["; ".join(r["clinical_flags"]) for r in results]
        output_df.to_csv(args.output, index=False)

        print(f"[SUCCESS] Predictions written to: {args.output}")
        print("\nSummary of Predicted Classes:")
        print(output_df["predicted_diagnosis"].value_counts().to_string())
        return

    # 4. Ad-hoc physiological parameters from command line
    if any(param is not None for param in [args.hr, args.pr, args.qrs, args.qtc, args.st, args.t_area]):
        custom_params = {}
        if args.hr is not None: custom_params["hr_bpm"] = args.hr
        if args.pr is not None: custom_params["pr_mean_ms"] = args.pr
        if args.qrs is not None: custom_params["qrs_dur_mean_ms"] = args.qrs
        if args.qtc is not None: custom_params["qtc_mean_ms"] = args.qtc
        if args.st is not None: custom_params["st_level_median_mv"] = args.st
        if args.t_area is not None: custom_params["t_area_median"] = args.t_area

        res = predictor.predict(custom_params)
        print_prediction_card(res)
        return

    # If no flags passed, run demo by default
    print("No arguments provided. Running default clinical demonstration mode:\n")
    run_demo(predictor)


if __name__ == "__main__":
    main()
