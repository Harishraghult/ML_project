"""
================================================================================
PTB-XL ECG Diagnostic Intelligence & Biological Age Deployment Platform
Interactive Streamlit Application for 12-Lead ECG Classification & Age Prediction
================================================================================
"""

import os
import sys
import warnings

# Suppress unneeded warnings for clean clinical UI
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import joblib

# Page Configuration
st.set_page_config(
    page_title="CardioPulse AI | 12-Lead ECG Diagnostic Platform",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Clinical Dashboard Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Outfit:wght@400;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .main-hero {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #0F2847 100%);
        padding: 2rem 2.5rem;
        border-radius: 1rem;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.2), 0 8px 10px -6px rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .hero-title {
        font-family: 'Outfit', sans-serif;
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 0.4rem;
        background: linear-gradient(90deg, #60A5FA, #A78BFA, #F43F5E);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        color: #94A3B8;
        font-weight: 400;
        max-width: 850px;
        line-height: 1.5;
    }

    .card-container {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 1rem;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.05);
        margin-bottom: 1.5rem;
        height: 100%;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .card-header {
        font-family: 'Outfit', sans-serif;
        font-size: 1.25rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 1.2rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        border-bottom: 1px solid #F1F5F9;
        padding-bottom: 0.75rem;
    }

    .diagnosis-badge {
        font-family: 'Outfit', sans-serif;
        font-size: 1.6rem;
        font-weight: 800;
        padding: 0.6rem 1.4rem;
        border-radius: 0.6rem;
        display: inline-block;
        color: white;
        letter-spacing: 0.03em;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    .badge-mi   { background: linear-gradient(135deg, #EF4444 0%, #B91C1C 100%); }
    .badge-cd   { background: linear-gradient(135deg, #F59E0B 0%, #D97706 100%); }
    .badge-hyp  { background: linear-gradient(135deg, #8B5CF6 0%, #6D28D9 100%); }
    .badge-sttc { background: linear-gradient(135deg, #3B82F6 0%, #1D4ED8 100%); }
    .badge-norm { background: linear-gradient(135deg, #10B981 0%, #047857 100%); }

    .prob-bar-wrapper {
        margin-bottom: 0.85rem;
    }
    .prob-bar-label {
        display: flex;
        justify-content: space-between;
        font-size: 0.9rem;
        font-weight: 600;
        color: #334155;
        margin-bottom: 0.25rem;
    }
    .prob-bar-outer {
        background-color: #F1F5F9;
        border-radius: 9999px;
        height: 10px;
        overflow: hidden;
    }
    .prob-bar-inner {
        height: 100%;
        border-radius: 9999px;
        transition: width 0.4s ease;
    }

    .metric-box {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 0.75rem;
        padding: 1rem 1.25rem;
        margin-bottom: 1rem;
    }
    .metric-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-val {
        font-family: 'Outfit', sans-serif;
        font-size: 2rem;
        font-weight: 800;
        color: #0F172A;
    }
    .metric-sub {
        font-size: 0.85rem;
        font-weight: 500;
        margin-top: 0.2rem;
    }

    .alert-card {
        padding: 0.75rem 1rem;
        border-radius: 0.5rem;
        font-size: 0.9rem;
        font-weight: 500;
        margin-bottom: 0.6rem;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }
    .alert-danger  { background: #FEF2F2; color: #991B1B; border-left: 4px solid #EF4444; }
    .alert-warning { background: #FFFBEB; color: #92400E; border-left: 4px solid #F59E0B; }
    .alert-info    { background: #EFF6FF; color: #1E40AF; border-left: 4px solid #3B82F6; }
    .alert-success { background: #ECFDF5; color: #065F46; border-left: 4px solid #10B981; }

    .tag-chip {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        background: #E2E8F0;
        color: #475569;
        margin-right: 0.4rem;
    }
</style>
""", unsafe_allow_html=True)

# Helper Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

CLASS_DESCRIPTIONS = {
    "MI": "Myocardial Infarction (Acute/Prior Coronary Occlusion or Ischemia)",
    "CD": "Conduction Disturbance (Bundle Branch Block or AV Delay)",
    "HYP": "Ventricular or Atrial Hypertrophy (Chamber Enlargement)",
    "STTC": "Non-Specific ST-T Segment Changes (Repolarization Anomaly)",
    "NORM": "Normal Electrocardiogram (No Significant Diagnostic Pathology)"
}

CLASS_COLORS = {
    "MI": "#EF4444",
    "CD": "#F59E0B",
    "HYP": "#8B5CF6",
    "STTC": "#3B82F6",
    "NORM": "#10B981"
}


@st.cache_resource
def load_models():
    """Loads classification models (Part B and Part A) and regression pipeline."""
    models_dict = {}

    clf_path_b = os.path.join(MODELS_DIR, "best_classification_pipeline_partB.joblib")
    clf_path_a = os.path.join(MODELS_DIR, "best_classification_pipeline_partA.joblib")
    reg_path = os.path.join(MODELS_DIR, "best_regression_pipeline.joblib")

    if os.path.exists(clf_path_b):
        models_dict["partB"] = joblib.load(clf_path_b)
    if os.path.exists(clf_path_a):
        models_dict["partA"] = joblib.load(clf_path_a)
    if os.path.exists(reg_path):
        models_dict["regression"] = joblib.load(reg_path)

    return models_dict


@st.cache_data
def load_sample_dataset():
    """Loads and merges feature matrices, diagnostic labels, and patient demographics."""
    feat_path = os.path.join(DATA_DIR, "stage4_features.csv")
    labels_path = os.path.join(DATA_DIR, "ptbxl_labels.csv")
    meta_path = os.path.join(DATA_DIR, "ptbxl_metadata.csv")

    if os.path.exists(feat_path) and os.path.exists(labels_path) and os.path.exists(meta_path):
        f_df = pd.read_csv(feat_path)
        l_df = pd.read_csv(labels_path)
        m_df = pd.read_csv(meta_path)

        priority_cols = ['label_MI', 'label_CD', 'label_HYP', 'label_STTC', 'label_NORM']
        class_names = ['MI', 'CD', 'HYP', 'STTC', 'NORM']

        def get_dom(row):
            for col, name in zip(priority_cols, class_names):
                if row[col] == 1:
                    return name
            return 'NORM'

        l_df['dominant_pathology'] = l_df.apply(get_dom, axis=1)

        merged = f_df.merge(l_df[['ecg_id', 'dominant_pathology', 'fold']], on='ecg_id', how='inner')
        merged = merged.merge(m_df[['ecg_id', 'age', 'sex']], on='ecg_id', how='inner')
        return merged
    return None


def main():
    # Hero Header Banner
    st.markdown("""
    <div class="main-hero">
        <div class="hero-title">
            <span>🫀 CardioPulse AI Diagnostic Intelligence</span>
        </div>
        <div class="hero-subtitle">
            Clinical Decision Support System for Automated 12-Lead Electrocardiography:
            Real-time Primary Diagnostic Classification across 5 Pathological Superclasses
            coupled with Electrophysiological Biological Age Estimation.
        </div>
    </div>
    """, unsafe_allow_html=True)

    models_dict = load_models()
    sample_df = load_sample_dataset()

    if not models_dict:
        st.error("⚠️ Serialized model pipelines not found in `/models`. Please ensure trained models are present.")
        st.stop()

    # Sidebar: Model Architecture Selection
    st.sidebar.markdown("### ⚙️ Diagnostic Model Engine")
    model_choices = {}
    if "partB" in models_dict:
        model_choices["Ensemble Neural Network (Part B - MLP)"] = "partB"
    if "partA" in models_dict:
        model_choices["Support Vector Classifier (Part A - SVC)"] = "partA"

    chosen_model_label = st.sidebar.selectbox("Classifier Architecture:", list(model_choices.keys()))
    active_clf_key = model_choices[chosen_model_label]
    clf_data = models_dict.get(active_clf_key)
    reg_data = models_dict.get("regression")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 Input Data Stream")
    mode = st.sidebar.radio("Input Source Mode:", ["Clinical Sample Case Loader", "Manual Physiological Parameter Input"])

    input_data = {}
    selected_sample = None

    if mode == "Clinical Sample Case Loader" and sample_df is not None:
        st.sidebar.markdown("#### Patient Cohort Explorer")
        filter_class = st.sidebar.selectbox(
            "Filter Cases by Clinical Ground Truth:",
            ["All Diagnoses", "MI (Myocardial Infarction)", "CD (Conduction Disturbance)", "HYP (Hypertrophy)", "STTC (ST-T Changes)", "NORM (Normal)"]
        )

        filtered_df = sample_df
        if filter_class != "All Diagnoses":
            code = filter_class.split()[0]
            filtered_df = sample_df[sample_df['dominant_pathology'] == code]

        # Representative benchmark cases prioritized at top
        benchmark_ids = [14314, 18499, 2417, 6735, 9010]
        case_options = [eid for eid in benchmark_ids if eid in filtered_df['ecg_id'].values]
        remaining_options = [eid for eid in filtered_df['ecg_id'].tolist() if eid not in case_options]
        all_options = case_options + remaining_options[:150]

        sample_id = st.sidebar.selectbox("Select Patient Record (ECG ID):", all_options)
        selected_sample = sample_df[sample_df['ecg_id'] == sample_id].iloc[0]

        # Patient Info Card in Sidebar
        sex_str = "Male" if selected_sample['sex'] == 0 else "Female"
        true_dx = selected_sample['dominant_pathology']
        true_age = int(selected_sample['age']) if selected_sample['age'] < 300 else "Unknown"

        st.sidebar.markdown(f"""
        <div style="background: rgba(241, 245, 249, 0.8); padding: 0.75rem 1rem; border-radius: 0.5rem; margin-top: 0.5rem; font-size: 0.85rem; border-left: 4px solid #3B82F6;">
            <div><strong>Patient ID:</strong> {int(selected_sample['patient_id'])}</div>
            <div><strong>Demographics:</strong> {true_age} yrs, {sex_str}</div>
            <div><strong>Ground Truth Dx:</strong> <span class="tag-chip" style="background:#E0E7FF; color:#3730A3; font-weight:700;">{true_dx}</span></div>
            <div><strong>Partition:</strong> {str(selected_sample['fold']).upper()}</div>
        </div>
        """, unsafe_allow_html=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🫀 Key 12-Lead Parameters")

    def get_val(col, default_val):
        if selected_sample is not None and col in selected_sample and not pd.isna(selected_sample[col]):
            return float(selected_sample[col])
        return default_val

    hr_bpm = st.sidebar.slider("Heart Rate (bpm)", 30.0, 180.0, get_val('hr_bpm', 72.0), step=1.0)
    pr_mean_ms = st.sidebar.slider("PR Interval (ms)", 80.0, 320.0, get_val('pr_mean_ms', 160.0), step=1.0)
    qrs_dur_mean_ms = st.sidebar.slider("QRS Duration (ms)", 50.0, 220.0, get_val('qrs_dur_mean_ms', 98.0), step=1.0)
    qtc_mean_ms = st.sidebar.slider("QTc Interval (ms)", 300.0, 600.0, get_val('qtc_mean_ms', 410.0), step=1.0)
    st_level_median_mv = st.sidebar.slider("ST-Segment Level (mV)", -0.50, 0.60, get_val('st_level_median_mv', -0.02), step=0.01)
    t_area_median = st.sidebar.slider("T-Wave Area (mV·s)", -0.100, 0.100, get_val('t_area_median', 0.008), step=0.002, format="%.3f")

    # Ratio Calculation
    qtc_qrs_ratio = qtc_mean_ms / (qrs_dur_mean_ms if qrs_dur_mean_ms > 0 else 1.0)

    # Populate feature vector
    feature_names = clf_data.get('feature_names') if clf_data else (reg_data.get('feature_names') if reg_data else [])

    for feat in feature_names:
        if feat == 'hr_bpm':
            input_data[feat] = hr_bpm
        elif feat == 'pr_mean_ms':
            input_data[feat] = pr_mean_ms
        elif feat == 'qrs_dur_mean_ms':
            input_data[feat] = qrs_dur_mean_ms
        elif feat == 'qtc_mean_ms':
            input_data[feat] = qtc_mean_ms
        elif feat == 'st_level_median_mv':
            input_data[feat] = st_level_median_mv
        elif feat == 't_area_median':
            input_data[feat] = t_area_median
        elif feat == 'qtc_qrs_ratio':
            input_data[feat] = qtc_qrs_ratio
        elif feat == 'rr_mean_ms':
            input_data[feat] = 60000.0 / hr_bpm
        elif selected_sample is not None and feat in selected_sample and not pd.isna(selected_sample[feat]):
            input_data[feat] = float(selected_sample[feat])
        else:
            input_data[feat] = np.nan  # SimpleImputer replaces NaN with population medians

    input_df = pd.DataFrame([input_data])

    # Inference Execution
    pipeline = clf_data['pipeline']
    le = clf_data.get('label_encoder')
    model_step = getattr(pipeline, 'named_steps', {}).get('model', pipeline)

    if hasattr(model_step, 'classes_'):
        raw_classes = list(model_step.classes_)
        if le is not None:
            model_classes = list(le.inverse_transform(raw_classes))
        else:
            model_classes = [str(c) for c in raw_classes]
    else:
        model_classes = clf_data.get('class_names', ['MI', 'CD', 'HYP', 'STTC', 'NORM'])

    X_val = input_df[feature_names].values
    probs = pipeline.predict_proba(X_val)[0] if hasattr(pipeline, 'predict_proba') else None

    if le is not None:
        preds_int = pipeline.predict(X_val)[0]
        pred_label = str(le.inverse_transform([preds_int])[0])
    else:
        pred_label = str(pipeline.predict(X_val)[0])

    prob_map = {c: float(p) for c, p in zip(model_classes, probs)} if probs is not None else {}
    confidence = prob_map.get(pred_label, float(np.max(probs))) if probs is not None else 1.0

    # Age Regression Execution
    predicted_age = None
    if reg_data is not None:
        reg_pipe = reg_data['pipeline']
        reg_fn = reg_data.get('feature_names', feature_names)
        X_reg = input_df[reg_fn].values
        predicted_age = float(reg_pipe.predict(X_reg)[0])

    # Clinical Alert Generation
    clinical_alerts = []
    if st_level_median_mv > 0.10:
        clinical_alerts.append(("danger", f"🚨 ST-Segment Elevation (+{st_level_median_mv:.2f} mV): Acute myocardial injury / STEMI protocol triggered."))
    elif st_level_median_mv < -0.10:
        clinical_alerts.append(("warning", f"⚠️ ST-Segment Depression ({st_level_median_mv:.2f} mV): Subendocardial ischemia or reciprocal strain anomaly."))

    if qrs_dur_mean_ms > 120.0:
        clinical_alerts.append(("warning", f"⚠️ Prolonged QRS Duration ({qrs_dur_mean_ms:.1f} ms): Bundle Branch Block or intraventricular conduction defect."))

    if qtc_mean_ms > 450.0:
        clinical_alerts.append(("danger", f"🚨 Prolonged QTc Interval ({qtc_mean_ms:.1f} ms): Elevated risk of polymorphic VT / Torsades de Pointes."))
    elif qtc_mean_ms < 350.0:
        clinical_alerts.append(("info", f"ℹ️ Shortened QTc Interval ({qtc_mean_ms:.1f} ms): Short QT syndrome consideration."))

    if qtc_qrs_ratio > 4.5:
        clinical_alerts.append(("warning", f"⚠️ Elevated QTc/QRS Ratio ({qtc_qrs_ratio:.2f}): Severe repolarization delay relative to ventricular activation speed."))

    if hr_bpm > 100.0:
        clinical_alerts.append(("info", f"ℹ️ Sinus Tachycardia ({hr_bpm:.0f} bpm)."))
    elif hr_bpm < 60.0:
        clinical_alerts.append(("info", f"ℹ️ Sinus Bradycardia ({hr_bpm:.0f} bpm)."))

    if not clinical_alerts:
        clinical_alerts.append(("success", "✅ All electrophysiological metrics and intervals remain within reference clinical limits."))

    # Render Two-Column Dashboard Layout
    col_clf, col_reg = st.columns([1.1, 0.9], gap="large")

    with col_clf:
        st.markdown("""
        <div class="card-container">
            <div class="card-header">
                🩺 <span>Primary Diagnostic Classification</span>
            </div>
        """, unsafe_allow_html=True)

        badge_class = f"badge-{pred_label.lower()}"
        st.markdown(f"""
        <div style="margin-bottom: 1.5rem;">
            <div style="font-size: 0.85rem; font-weight:600; color:#64748B; text-transform:uppercase; margin-bottom:0.4rem;">
                Predicted Dominant Pathology
            </div>
            <div style="display:flex; align-items:center; gap:1rem; flex-wrap:wrap;">
                <span class="diagnosis-badge {badge_class}">{pred_label}</span>
                <div>
                    <div style="font-size: 1.15rem; font-weight:700; color:#1E293B;">
                        {CLASS_DESCRIPTIONS.get(pred_label)}
                    </div>
                    <div style="font-size: 0.9rem; color:#64748B;">
                        Diagnostic Model Confidence: <strong>{confidence * 100:.1f}%</strong>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if selected_sample is not None:
            match = (pred_label == selected_sample['dominant_pathology'])
            match_badge = '<span class="tag-chip" style="background:#D1FAE5; color:#065F46; font-weight:700;">✅ MATCH WITH GROUND TRUTH</span>' if match else f'<span class="tag-chip" style="background:#FEE2E2; color:#991B1B; font-weight:700;">⚠️ GROUND TRUTH: {selected_sample["dominant_pathology"]}</span>'
            st.markdown(f"<div style='margin-bottom: 1.2rem;'><strong>Validation Status:</strong> {match_badge}</div>", unsafe_allow_html=True)

        st.markdown("<div style='font-size: 0.95rem; font-weight: 700; color:#1E293B; margin-bottom: 0.8rem;'>Pathology Probability Spectrum:</div>", unsafe_allow_html=True)

        display_order = ['MI', 'CD', 'HYP', 'STTC', 'NORM']
        for c in display_order:
            p = prob_map.get(c, 0.0)
            p_pct = p * 100
            bar_color = CLASS_COLORS.get(c, "#2563EB")
            is_pred = (c == pred_label)
            weight = "800" if is_pred else "500"
            marker = " ◀ PREDICTED" if is_pred else ""

            st.markdown(f"""
            <div class="prob-bar-wrapper">
                <div class="prob-bar-label" style="font-weight:{weight};">
                    <span>{c} — {CLASS_DESCRIPTIONS.get(c).split('(')[0].strip()}{marker}</span>
                    <span>{p_pct:.1f}%</span>
                </div>
                <div class="prob-bar-outer">
                    <div class="prob-bar-inner" style="width: {p_pct}%; background-color: {bar_color};"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    with col_reg:
        st.markdown("""
        <div class="card-container">
            <div class="card-header">
                ⏳ <span>Biological Age & Conduction Insights</span>
            </div>
        """, unsafe_allow_html=True)

        if predicted_age is not None:
            true_age_val = selected_sample['age'] if selected_sample is not None else None
            has_true_age = (true_age_val is not None and not pd.isna(true_age_val) and true_age_val < 300)

            delta_str = ""
            sub_text = "Estimated from 25 resting 12-lead electrophysiological features"
            if has_true_age:
                diff = predicted_age - true_age_val
                sign = "+" if diff > 0 else ""
                delta_color = "#DC2626" if diff > 5 else ("#16A34A" if diff < -5 else "#2563EB")
                delta_str = f"<span style='color:{delta_color}; font-weight:700;'>({sign}{diff:.1f} yrs vs Chronological {true_age_val:.0f})</span>"
                if diff > 5:
                    sub_text = "⚠️ Accelerated cardiovascular electrical aging phenotype"
                elif diff < -5:
                    sub_text = "✨ Preserved electrophysiological profile (Younger than chronological)"
                else:
                    sub_text = "✅ Electrical age aligned with chronological age"

            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-title">Estimated Patient Biological Age</div>
                <div class="metric-val">{predicted_age:.1f} <span style="font-size:1.1rem; font-weight:600; color:#64748B;">years</span> {delta_str}</div>
                <div class="metric-sub" style="color:#64748B;">{sub_text}</div>
            </div>
            """, unsafe_allow_html=True)

        # Repolarization / Depolarization Ratio
        ratio_status = "Optimal Coupling" if qtc_qrs_ratio <= 4.5 else "Impaired Ratio (>4.5)"
        ratio_color = "#10B981" if qtc_qrs_ratio <= 4.5 else "#F59E0B"

        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-title">QTc / QRS Repolarization Coupling Ratio</div>
            <div class="metric-val" style="color: {ratio_color};">{qtc_qrs_ratio:.2f}</div>
            <div class="metric-sub" style="color:#64748B;">Clinical Status: <strong>{ratio_status}</strong> (Normal threshold: ≤ 4.50)</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='font-size: 0.95rem; font-weight: 700; color:#1E293B; margin-bottom: 0.6rem;'>Clinical Diagnostic Notifications:</div>", unsafe_allow_html=True)
        for level, msg in clinical_alerts:
            st.markdown(f"""
            <div class="alert-card alert-{level}">
                {msg}
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
