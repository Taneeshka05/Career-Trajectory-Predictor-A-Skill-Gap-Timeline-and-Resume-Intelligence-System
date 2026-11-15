import sys
from pathlib import Path
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "karta hai") not in sys.path:
    sys.path.insert(0, str(ROOT / "karta hai"))

from core import config as cfg
from core import charts
from core.logger_setup import tail_log
from core.models import LEVEL_ORDER
from core.ml_pipeline import DataValidator, MLWorkflowEngine
from core.ui import setup

engine = setup("Model Lab", "🧪")
b = engine.b
rel = b["reliability"]
st.title("🧪 Interactive Machine Learning Pipeline & Model Lab")
st.caption("Demonstrates genuine end-to-end ML: Data validation, exploratory analysis, leakage-free training, unseen test evaluation, and live inference.")

mode = st.radio(
    "Select Lab Mode",
    ["Dynamic ML Training Pipeline (Upload or Bundled Data)", "Core System Model Benchmarks & Stress Tests"],
    horizontal=True
)

if mode == "Dynamic ML Training Pipeline (Upload or Bundled Data)":
    st.subheader("Real-Time Machine Learning Pipeline")
    st.caption("DATA INPUT &rarr; VALIDATION &rarr; EDA &rarr; PREPROCESSING &rarr; MODEL TRAINING &rarr; UNSEEN TEST EVALUATION &rarr; LIVE PREDICTION")

    data_source = st.radio(
        "Choose Data Source",
        ["Use Bundled Career Dataset (650 records)", "Upload Custom CSV or Excel File"],
        horizontal=True
    )

    df_loaded = None
    if data_source == "Use Bundled Career Dataset (650 records)":
        sample_path = cfg.DATA_DIR / "career_ml_sample_dataset.csv"
        if sample_path.exists():
            df_loaded = pd.read_csv(sample_path)
            st.success("Loaded bundled dataset: `career_ml_sample_dataset.csv` (650 profiles with regression and classification targets).")
        else:
            st.error("Sample dataset file not found.")
    else:
        uploaded_file = st.file_uploader("Upload tabular dataset (.csv, .xlsx)", type=["csv", "xlsx", "xls"])
        if uploaded_file is not None:
            try:
                df_loaded = DataValidator.load_dataset(uploaded_file, uploaded_file.name)
                st.success(f"Successfully loaded '{uploaded_file.name}'.")
            except Exception as e:
                st.error(f"Error loading dataset: {e}")

    if df_loaded is not None and not df_loaded.empty:
        # STEP 1: VALIDATION
        st.markdown("### Step 1: Data Input & Quality Audit")
        audit = DataValidator.audit(df_loaded)
        if not audit["valid"]:
            st.error(f"Data Validation Failed: {audit['error']}")
            st.stop()

        v1, v2, v3, v4 = st.columns(4)
        v1.metric("Total Records", f"{audit['rows']:,}")
        v2.metric("Total Columns", audit["cols"])
        v3.metric("Missing Value Rate", f"{audit['missing_rate_pct']}%")
        v4.metric("Data Quality Health", f"{audit['health_score']}/100")

        if audit["duplicates"] > 0:
            st.warning(f"Detected {audit['duplicates']} duplicate rows in the dataset.")

        st.dataframe(df_loaded.head(6), width="stretch", hide_index=True)

        # STEP 2: EDA
        st.markdown("### Step 2: Exploratory Data Analysis (EDA)")
        eda_tab1, eda_tab2, eda_tab3 = st.tabs(["Summary Statistics", "Missing Values", "Correlation Matrix"])
        with eda_tab1:
            st.dataframe(df_loaded.describe().round(2), width="stretch")
        with eda_tab2:
            miss_series = df_loaded.isnull().sum()
            if miss_series.sum() > 0:
                st.plotly_chart(charts.eda_missing_bar(miss_series), width="stretch")
            else:
                st.info("No missing values found in the dataset.")
        with eda_tab3:
            num_df = df_loaded.select_dtypes(include=["number"])
            if len(num_df.columns) >= 2:
                st.plotly_chart(charts.eda_corr_heatmap(num_df), width="stretch")
            else:
                st.info("Not enough numeric columns for a correlation heatmap.")

        # STEP 3: CONFIGURE PIPELINE
        st.markdown("### Step 3: Model Configuration")
        cols = df_loaded.columns.tolist()
        c_left, c_right = st.columns([1, 2])
        with c_left:
            default_target_idx = cols.index("preparation_weeks") if "preparation_weeks" in cols else (
                cols.index("readiness_tier") if "readiness_tier" in cols else len(cols) - 1
            )
            target_col = st.selectbox("Select Target Variable (Y)", cols, index=default_target_idx)
            task_type = st.selectbox("Task Type", ["Auto", "Regression", "Classification"])
            test_split = st.slider("Unseen Test Split Ratio", 0.1, 0.4, 0.2, step=0.05)
        with c_right:
            potential_features = [c for c in cols if c != target_col]
            feature_cols = st.multiselect(
                "Select Feature Predictors (X)",
                potential_features,
                default=potential_features[:min(8, len(potential_features))]
            )

        if st.button("Train, Validate & Compare Models", type="primary"):
            if not feature_cols:
                st.error("Please select at least one feature column for training.")
            else:
                with st.spinner("Executing pipeline: preprocessing, fitting models, and evaluating on unseen test data..."):
                    engine_ml = MLWorkflowEngine(df_loaded, target_col, feature_cols, task_type)
                    report = engine_ml.run_pipeline(test_size=test_split)
                    st.session_state["active_ml_engine"] = engine_ml

        active_ml = st.session_state.get("active_ml_engine")
        if active_ml and active_ml.evaluation_results is not None:
            # STEP 4: EVALUATION
            st.markdown("### Step 4: Model Evaluation on Unseen Test Data")
            res_df = active_ml.evaluation_results
            st.dataframe(res_df, width="stretch", hide_index=True)
            st.success(f"🏆 Best Performing Model: **{active_ml.best_model_name}**")

            m_col1, m_col2 = st.columns(2)
            with m_col1:
                metric_name = "R² Score" if active_ml.task_type == "Regression" else "F1-Score"
                st.plotly_chart(charts.metric_bars(res_df, metric_name, f"Model Comparison ({metric_name})"), width="stretch")
            with m_col2:
                if active_ml.task_type == "Classification" and active_ml.best_model_name in active_ml.confusion_matrices:
                    classes = sorted(df_loaded[target_col].dropna().unique())
                    cm = active_ml.confusion_matrices[active_ml.best_model_name]
                    st.plotly_chart(charts.confusion(cm, classes, f"Confusion Matrix: {active_ml.best_model_name}"), width="stretch")
                elif active_ml.task_type == "Regression":
                    scatter_df = pd.DataFrame({
                        "actual": active_ml.y_test,
                        "predicted": active_ml.test_predictions[active_ml.best_model_name]
                    })
                    st.plotly_chart(charts.reg_scatter(scatter_df), width="stretch")

            if active_ml.importances is not None:
                st.markdown("### Feature Importance")
                st.plotly_chart(charts.importance_bar(active_ml.importances.head(10)), width="stretch")

            # STEP 5: LIVE PREDICTION SANDBOX
            st.markdown("### Step 5: Live Inference Sandbox")
            st.caption("Supply custom inputs to receive real-time predictions from your trained pipeline:")
            with st.form("live_predict_form"):
                inputs = {}
                input_cols = st.columns(min(3, len(feature_cols)))
                for i, col_name in enumerate(feature_cols):
                    with input_cols[i % len(input_cols)]:
                        if np.issubdtype(df_loaded[col_name].dtype, np.number):
                            default_val = float(df_loaded[col_name].median())
                            min_val = float(df_loaded[col_name].min())
                            max_val = float(df_loaded[col_name].max())
                            inputs[col_name] = st.number_input(col_name, value=default_val, min_value=min_val, max_value=max_val)
                        else:
                            unique_vals = df_loaded[col_name].dropna().unique().tolist()
                            inputs[col_name] = st.selectbox(col_name, unique_vals)
                submit_pred = st.form_submit_button("Generate Prediction", type="primary")
                if submit_pred:
                    pred_val = active_ml.predict_new(inputs)
                    if active_ml.task_type == "Regression":
                        st.success(f"🎯 Model Predicted `{target_col}`: **{pred_val:.2f}**")
                    else:
                        st.success(f"🎯 Model Predicted Class for `{target_col}`: **{pred_val}**")

else:
    # CORE SYSTEM BENCHMARKS
    st.subheader("Core System Model Benchmarks")
    tabs = st.tabs(["Data Stats", "Regression Models", "Classifiers", "Neural Network", "Reliability & Stress Tests", "Normalization", "Logs"])

    with tabs[0]:
        s = b["stats"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Resumes", f"{s['n_resumes']:,}")
        c2.metric("Job Postings", s["n_jobs"])
        c3.metric("Skills in Reference Taxonomy", s["n_skills"])
        st.write("Data sources breakdown:", s["source_counts"])
        st.markdown("**Readiness Level Distribution**")
        st.bar_chart(pd.Series(s["level_counts"]))
        st.markdown("**Profiles per Domain**")
        st.bar_chart(pd.Series(s["domain_counts"]))

    with tabs[1]:
        st.dataframe(b["reg"]["results"].round(3), width="stretch", hide_index=True)
        st.success(f"Best Model: {b['reg']['best_name']} (L2 Ridge Alpha = {b['reg']['models']['Ridge Regression'].alpha})")
        st.plotly_chart(charts.reg_scatter(b["reg_scatter"]), width="stretch")
        coef = pd.DataFrame({
            "feature": cfg.REG_FEATURES,
            "Linear Regression Coef": b["reg"]["models"]["Linear Regression"].coef_,
            "Ridge Regression Coef": b["reg"]["models"]["Ridge Regression"].coef_
        })
        st.markdown("**Model Coefficients (on Normalized Features)**")
        st.dataframe(coef.round(3), width="stretch", hide_index=True)

    with tabs[2]:
        res = b["clf"]["results"]
        st.dataframe(res.round(3), width="stretch", hide_index=True)
        c1, c2 = st.columns(2)
        c1.plotly_chart(charts.metric_bars(res, "F1", "F1-Score by Classifier"), width="stretch")
        c2.plotly_chart(charts.confusion(b["clf"]["cm"], LEVEL_ORDER, f"Confusion Matrix: {b['clf']['best_name']}"), width="stretch")

    with tabs[3]:
        st.dataframe(b["nn_comparison"].round(3), width="stretch", hide_index=True)
        c1, c2 = st.columns(2)
        c1.plotly_chart(charts.curves(b["nn_history"]), width="stretch")
        c2.plotly_chart(charts.confusion(b["nn_cm"], ["Weak", "Strong"], "Language Model Confusion Matrix"), width="stretch")

    with tabs[4]:
        st.markdown("**5-Fold Stratified Cross-Validation Stability**")
        st.dataframe(rel["cv"].round(3), width="stretch", hide_index=True)
        c1, c2 = st.columns(2)
        c1.plotly_chart(charts.line_chart(rel["curve"], "training samples", "validation accuracy", "Learning Curve (Best Classifier)"), width="stretch")
        c2.plotly_chart(charts.line_chart(rel["noise"], "label noise %", "test accuracy", "Stress Test: Label Noise Degradation"), width="stretch")
        st.plotly_chart(charts.importance_bar(rel["importance"]), width="stretch")

    with tabs[5]:
        st.dataframe(b["scaler_table"].round(3), width="stretch", hide_index=True)

    with tabs[6]:
        st.code(tail_log(60), language="text")
