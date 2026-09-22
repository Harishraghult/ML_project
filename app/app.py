"""
================================================================================
PTB-XL ECG Analytics & Machine Learning Deployment App
Interactive Streamlit Application for Clinical ECG Classification & Age Prediction
================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import joblib

# Page Configuration
st.set_page_config(
    page_title="PTB-XL ECG Diagnostic Intelligence",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1.2rem;
        border-radius: 0.8rem;
        border-left: 5px solid #2563EB;
        margin-bottom: 1rem;
    }
    .diagnosis-badge {
        font-size: 1.5rem;
        font-weight: 700;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        display: inline-block;
        color: white;
    }
    .badge-mi { background-color: #DC2626; }
    .badge-cd { background-color: #D97706; }
    .badge-hyp { background-color: #7C3AED; }
    .badge-sttc { background-color: #2563EB; }
    .badge-norm { background-color: #059669; }
</style>
""", unsafe_allow_html=True)

# Helper Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

@st.cache_resource
def load_models():
    clf_pipeline_data = None
    reg_pipeline_data = None

    # Load Classification Model
    clf_path_b = os.path.join(MODELS_DIR, "best_classification_pipeline_partB.joblib")
    clf_path_a = os.path.join(MODELS_DIR, "best_classification_pipeline_partA.joblib")
    
    if os.path.exists(clf_path_b):
        clf_pipeline_data = joblib.load(clf_path_b)
    elif os.path.exists(clf_path_a):
        clf_pipeline_data = joblib.load(clf_path_a)

    # Load Regression Model
    reg_path = os.path.join(MODELS_DIR, "best_regression_pipeline.joblib")
    if os.path.exists(reg_path):
        reg_pipeline_data = joblib.load(reg_path)

    return clf_pipeline_data, reg_pipeline_data

@st.cache_data
def load_sample_dataset():
    feat_path = os.path.join(DATA_DIR, "stage4_features.csv")
    labels_path = os.path.join(DATA_DIR, "ptbxl_labels.csv")
    meta_path = os.path.join(DATA_DIR, "ptbxl_metadata.csv")

    if os.path.exists(feat_path) and os.path.exists(labels_path) and os.path.exists(meta_path):
        f_df = pd.read_csv(feat_path)
        l_df = pd.read_csv(labels_path)
        m_df = pd.read_csv(meta_path)
        merged = f_df.merge(l_df[['ecg_id', 'label_NORM', 'label_MI', 'label_STTC', 'label_CD', 'label_HYP']], on='ecg_id', how='inner')
        merged = merged.merge(m_df[['ecg_id', 'age', 'sex']], on='ecg_id', how='inner')
        return merged
    return None

def main():
    st.markdown('<div class="main-header">🫀 PTB-XL ECG Diagnostic Intelligence Platform</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Real-time 12-Lead Electrocardiography Diagnostic Classification & Biological Age Estimation</div>', unsafe_allow_html=True)

    clf_data, reg_data = load_models()
    sample_df = load_sample_dataset()

    if clf_data is None and reg_data is None:
        st.warning("⚠️ Serialized model pipelines not found in `/models`. Please run `python build_notebooks.py` to generate and serialize trained models first.")
        st.stop()

    st.sidebar.header("📋 Input Mode Selection")
    mode = st.sidebar.radio("Choose Input Source:", ["Clinical Sample Case Loader", "Manual Physiological Parameter Input"])

    input_data = {}
    selected_sample = None

    if mode == "Clinical Sample Case Loader" and sample_df is not None:
        st.sidebar.subheader("Select Dataset Case")
        sample_id = st.sidebar.selectbox("Choose Record ECG ID:", sample_df['ecg_id'].head(100).tolist())
        selected_sample = sample_df[sample_df['ecg_id'] == sample_id].iloc[0]
        
        st.sidebar.markdown(f"**Known Patient Age:** {selected_sample['age']} yrs")
        st.sidebar.markdown(f"**Patient Sex:** {'Male' if selected_sample['sex'] == 0 else 'Female'}")
    
    st.sidebar.subheader("12-Lead ECG Features")

    # Feature input sliders / values
    def get_val(col, default_val, min_v, max_v, step_v):
        if selected_sample is not None and col in selected_sample and not pd.isna(selected_sample[col]):
            return float(selected_sample[col])
        return default_val

    hr_bpm = st.sidebar.slider("Heart Rate (bpm)", 30.0, 180.0, get_val('hr_bpm', 72.0, 30.0, 180.0, 1.0))
    pr_mean_ms = st.sidebar.slider("PR Interval (ms)", 80.0, 320.0, get_val('pr_mean_ms', 160.0, 80.0, 320.0, 1.0))
    qrs_dur_mean_ms = st.sidebar.slider("QRS Duration (ms)", 50.0, 220.0, get_val('qrs_dur_mean_ms', 98.0, 50.0, 220.0, 1.0))
    qtc_mean_ms = st.sidebar.slider("QTc Interval (ms)", 300.0, 600.0, get_val('qtc_mean_ms', 410.0, 300.0, 600.0, 1.0))
    st_level_median_mv = st.sidebar.slider("ST-Segment Level (mV)", -0.5, 0.6, get_val('st_level_median_mv', 0.02, -0.5, 0.6, 0.01))
    t_area_median = st.sidebar.slider("T-Wave Area (mV·ms)", -20.0, 50.0, get_val('t_area_median', 8.5, -20.0, 50.0, 0.5))
    
    # Calculate physiological ratio
    qtc_qrs_ratio = qtc_mean_ms / (qrs_dur_mean_ms if qrs_dur_mean_ms != 0 else 1.0)

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
            input_data[feat] = 0.0

    input_df = pd.DataFrame([input_data])

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🩺 Diagnostic Classification Analysis")
        if clf_data:
            pipeline = clf_data['pipeline']
            class_names = clf_data.get('class_names', ['MI', 'CD', 'HYP', 'STTC', 'NORM'])
            le = clf_data.get('label_encoder')

            # Prediction
            if le is not None:
                preds_int = pipeline.predict(input_df)[0]
                pred_label = le.inverse_transform([preds_int])[0]
                probs = pipeline.predict_proba(input_df)[0]
            else:
                pred_label = pipeline.predict(input_df)[0]
                probs = pipeline.predict_proba(input_df)[0]

            badge_class = f"badge-{pred_label.lower()}"
            st.markdown(f"**Predicted Primary Pathology:** <span class='diagnosis-badge {badge_class}'>{pred_label}</span>", unsafe_allow_html=True)
            st.write("")

            st.write("### Diagnostic Probability Distribution:")
            for cls_name, prob in zip(class_names, probs):
                st.write(f"**{cls_name}:** {prob * 100:.1f}%")
                st.progress(float(prob))
        else:
            st.info("Classification pipeline unavailable.")

    with col2:
        st.subheader("⏳ Biological Age Regression Analysis")
        if reg_data:
            reg_pipeline = reg_data['pipeline']
            predicted_age = reg_pipeline.predict(input_df)[0]
            
            st.metric("Estimated Patient Biological Age", f"{predicted_age:.1f} years", delta=f"{predicted_age - selected_sample['age']:.1f} vs true age" if selected_sample is not None else None)

            st.write("### Electrophysiological Ratio Insights:")
            st.metric("QTc / QRS Repolarization Ratio", f"{qtc_qrs_ratio:.2f}", help="Repolarization duration relative to intraventricular depolarization speed")
            
            if qtc_qrs_ratio > 4.5:
                st.warning("⚠️ High Repolarization Ratio (> 4.5): Indicates prolonged repolarization relative to depolarization, common in drug-induced QTc prolongation or Long-QT Syndrome.")
            elif qrs_dur_mean_ms > 120:
                st.warning("⚠️ Prolonged QRS Duration (> 120ms): Indicates intraventricular conduction delay or bundle branch block (CD).")
            elif st_level_median_mv > 0.1:
                st.error("🚨 ST-Segment Elevation (> 0.1mV): Strong physiological indicator of acute myocardial injury (MI).")
            else:
                st.success("✅ Conduction intervals within physiological baseline thresholds.")
        else:
            st.info("Regression pipeline unavailable.")

if __name__ == "__main__":
    main()
