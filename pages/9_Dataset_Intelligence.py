import sys
from pathlib import Path
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "karta hai") not in sys.path:
    sys.path.insert(0, str(ROOT / "karta hai"))

from core import config as cfg
from core import charts
from core.ml_pipeline import DataValidator
from core.ui import setup

engine = setup("Dataset Intelligence", "📊")
st.title("📊 Dataset & Data Source Intelligence Panel")
st.caption("Deep profiling, schema validation, quality health metrics, and transparency disclosures across all system datasets.")

datasets = {
    "Resumes Dataset (Synthetic & Real Corpus)": cfg.DATA_DIR / "resumes_dataset.csv",
    "Job Postings Dataset (7 Domains, 420 Positions)": cfg.DATA_DIR / "jobs_dataset.csv",
    "Skill Taxonomy & Reference Table": cfg.DATA_DIR / "skill_reference.csv",
    "Bundled ML Pipeline Benchmark Dataset": cfg.DATA_DIR / "career_ml_sample_dataset.csv",
}

selected_name = st.selectbox("Select Dataset to Profile", list(datasets.keys()))
csv_path = datasets[selected_name]

if csv_path.exists():
    df = pd.read_csv(csv_path)
    audit = DataValidator.audit(df)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Records (Rows)", f"{audit['rows']:,}")
    m2.metric("Features (Columns)", audit["cols"])
    m3.metric("Missing Value Rate", f"{audit['missing_rate_pct']}%")
    m4.metric("Data Quality Health", f"{audit['health_score']}/100")

    tab1, tab2, tab3, tab4 = st.tabs(["Data Sample & Schema", "Descriptive Statistics", "Data Quality Audit", "Academic Limitations & Source Info"])

    with tab1:
        st.markdown("#### Sample Records (Top 8 Rows)")
        st.dataframe(df.head(8), width="stretch", hide_index=True)
        st.markdown("#### Schema & Data Types")
        schema_df = pd.DataFrame({
            "Column Name": df.columns,
            "Inferred Data Type": [str(t) for t in df.dtypes],
            "Non-Null Count": df.notnull().sum().values,
            "Unique Values": df.nunique().values
        })
        st.dataframe(schema_df, width="stretch", hide_index=True)

    with tab2:
        st.markdown("#### Numerical Summary Statistics")
        num_cols = df.select_dtypes(include=["number"])
        if not num_cols.empty:
            st.dataframe(num_cols.describe().round(2), width="stretch")
            if len(num_cols.columns) >= 2:
                st.plotly_chart(charts.eda_corr_heatmap(num_cols), width="stretch")
        else:
            st.info("No numerical columns present in this table.")

        cat_cols = df.select_dtypes(exclude=["number"])
        if not cat_cols.empty:
            st.markdown("#### Categorical Distributions")
            for c in cat_cols.columns[:3]:
                if df[c].nunique() < 30:
                    st.write(f"**Value distribution for `{c}`:**")
                    st.bar_chart(df[c].value_counts())

    with tab3:
        st.markdown("#### Missing & Duplicate Record Audit")
        c_left, c_right = st.columns(2)
        with c_left:
            st.write(f"- **Duplicate Rows:** {audit['duplicates']}")
            st.write(f"- **Total Missing Cells:** {audit['missing_total']}")
            st.write(f"- **Missing Rate:** {audit['missing_rate_pct']}%")
        with c_right:
            miss = df.isnull().sum()
            if miss.sum() > 0:
                st.plotly_chart(charts.eda_missing_bar(miss), width="stretch")
            else:
                st.success("Clean Dataset: Zero missing values detected across all columns.")

    with tab4:
        st.markdown("#### Data Provenance, Pipeline Transformations & Honest Limitations")
        if "resumes_dataset" in str(csv_path):
            st.markdown(
                "- **Data Origin:** Synthetically generated resume corpus (5,500 profiles) combined with verified text patterns.\n"
                "- **Target Variables:** `language_label` (Binary active language score), `years_exp` (Experience duration).\n"
                "- **Preprocessing Performed:** Lower-casing, regex tokenization, stopword removal, TF-IDF unigram+bigram vectorization, L2 vector normalization.\n"
                "- **Academic Limitations:** Synthetic resumes follow structured generation templates. Real candidate resumes exhibit higher formatting variety, OCR artifacts, and non-linear career timelines."
            )
        elif "jobs_dataset" in str(csv_path):
            st.markdown(
                "- **Data Origin:** Curated job postings spanning 7 IT and engineering domains (60 postings per domain, 420 total).\n"
                "- **Target Variables:** `domain`, `title`, `job_skills`.\n"
                "- **Preprocessing Performed:** Text normalization, stopword filtering, canonical skill extraction using taxonomy regex patterns.\n"
                "- **Academic Limitations:** Postings represent prototypical market job specifications. Live labor market listings frequently vary in terminology."
            )
        elif "career_ml_sample" in str(csv_path):
            st.markdown(
                "- **Data Origin:** Bundled tabular dataset (650 records) designed for real supervised ML training, validation, and benchmarking.\n"
                "- **Target Variables:** `preparation_weeks` (Regression) and `readiness_tier` (Classification: Junior, Mid-Level, Senior).\n"
                "- **Preprocessing Performed:** Column-wise SimpleImputer, One-Hot Encoding for categorical domains, StandardScaler for numerical predictors.\n"
                "- **Academic Limitations:** Designed for academic validation and algorithmic demonstration."
            )
        else:
            st.markdown(
                "- **Data Origin:** Domain reference table with 52 standardized skills, difficulty indices (1-5), and estimated baseline learning weeks.\n"
                "- **Academic Limitations:** Learning durations assume 10 hours/week of dedicated study with foundation prerequisites satisfied."
            )
else:
    st.error(f"Dataset file `{csv_path}` does not exist on disk.")
