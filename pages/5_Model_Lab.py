import pandas as pd
import streamlit as st
from core import config as cfg
from core import charts
from core.logger_setup import tail_log
from core.models import LEVEL_ORDER
from core.ui import setup

engine = setup("Model Lab", "🧪")
b = engine.b
rel = b["reliability"]
st.title("🧪 Model Lab")
st.caption("Everything behind the predictions: data, models, metrics, reliability, normalization and logs.")

tabs = st.tabs(["Data", "Regression", "Classifiers", "Neural Network", "Reliability", "Normalization", "Logs"])

with tabs[0]:
    s = b["stats"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Resumes", f"{s['n_resumes']:,}")
    c2.metric("Job postings", s["n_jobs"])
    c3.metric("Skills in reference table", s["n_skills"])
    st.write("Data sources:", s["source_counts"])
    st.warning("Synthetic resumes and all job postings are generated. Labels (readiness level, time to ready, "
               "language quality) come from documented rules. High scores show the models recovered those "
               "rules, not real hiring outcomes. See the Reliability tab for stress tests.")
    st.markdown("**Readiness level distribution**")
    st.bar_chart(pd.Series(s["level_counts"]))
    st.markdown("**Profiles per domain**")
    st.bar_chart(pd.Series(s["domain_counts"]))
    st.markdown("**Sample resumes**")
    st.dataframe(s["sample_resumes"], width="stretch", hide_index=True)

with tabs[1]:
    st.dataframe(b["reg"]["results"].round(3), width="stretch", hide_index=True)
    st.success(f"Best model: {b['reg']['best_name']} (Ridge penalty alpha = "
               f"{b['reg']['models']['Ridge Regression'].alpha})")
    st.plotly_chart(charts.reg_scatter(b["reg_scatter"]), width="stretch")
    coef = pd.DataFrame({"feature": cfg.REG_FEATURES,
                         "linear": b["reg"]["models"]["Linear Regression"].coef_,
                         "ridge": b["reg"]["models"]["Ridge Regression"].coef_})
    st.markdown("**Coefficients (on normalized features)**")
    st.dataframe(coef.round(3), width="stretch", hide_index=True)

with tabs[2]:
    res = b["clf"]["results"]
    st.dataframe(res.round(3), width="stretch", hide_index=True)
    c1, c2 = st.columns(2)
    c1.plotly_chart(charts.metric_bars(res, "F1", "F1 score by model"), width="stretch")
    c2.plotly_chart(charts.confusion(b["clf"]["cm"], LEVEL_ORDER, f"Confusion matrix: {b['clf']['best_name']}"),
                    width="stretch")
    st.markdown("**PCA and K-Means:** PCA explained variance "
                f"{[round(float(v), 3) for v in b['pca']['pca'].explained_variance_ratio_]}")
    st.dataframe(b["pca"]["crosstab"], width="stretch")

with tabs[3]:
    st.dataframe(b["nn_comparison"].round(3), width="stretch", hide_index=True)
    c1, c2 = st.columns(2)
    c1.plotly_chart(charts.curves(b["nn_history"]), width="stretch")
    c2.plotly_chart(charts.confusion(b["nn_cm"], ["Weak", "Strong"], "Language model confusion matrix"),
                    width="stretch")
    st.code("Input(300 TF-IDF features) -> Dense(64, relu) -> BatchNormalization -> Dropout(0.3)\n"
            "-> Dense(32, relu) -> Dropout(0.3) -> Dense(1, sigmoid)\n"
            "Optimizer: Adam | Loss: binary cross-entropy | EarlyStopping(patience=5)")

with tabs[4]:
    st.markdown("**5-fold cross-validation** (accuracy on 5 different train/test splits)")
    st.dataframe(rel["cv"].round(3), width="stretch", hide_index=True)
    c1, c2 = st.columns(2)
    c1.plotly_chart(charts.line_chart(rel["curve"], "training samples", "validation accuracy",
                                      "Learning curve of the best classifier"), width="stretch")
    c2.plotly_chart(charts.line_chart(rel["noise"], "label noise %", "test accuracy",
                                      "Accuracy when training labels are corrupted"), width="stretch")
    st.plotly_chart(charts.importance_bar(rel["importance"]), width="stretch")
    st.markdown("**Accuracy by data source**")
    st.dataframe(rel["by_source"].round(3), width="stretch", hide_index=True)
    st.info("Reading this honestly: cross-validation shows the scores are stable, the learning curve shows "
            "how much data the model needs, and the noise test shows accuracy falls when labels are wrong. "
            "Labels here are rule based, so accuracy measures how well the models learned those rules.")

with tabs[5]:
    st.markdown("Numeric features are z-score normalized with StandardScaler (fitted on training data only). "
                "Text is lower-cased, stripped of punctuation and stop words, and TF-IDF vectors are L2 "
                "normalized so cosine similarity is a simple dot product.")
    st.dataframe(b["scaler_table"].round(3), width="stretch", hide_index=True)

with tabs[6]:
    st.caption("Last 60 lines of logs/app.log")
    st.code(tail_log(60), language="text")