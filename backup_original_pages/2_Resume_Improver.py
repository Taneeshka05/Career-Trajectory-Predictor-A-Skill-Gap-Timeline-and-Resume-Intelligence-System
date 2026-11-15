import pandas as pd
import streamlit as st
from core import charts
from core import nlp_feedback as nlp
from core.ui import setup, resume_panel

engine = setup("Resume Improver", "✍️")
st.title("✍️ Resume Improver")
st.caption("Replaces weak phrases with action verbs and shows the language score before and after.")

text = resume_panel("resume_improve")
if st.button("Improve my resume", type="primary"):
    improved, changes = nlp.auto_improve(text)
    st.session_state["improved"] = {
        "original": text, "improved": improved, "changes": changes,
        "before": engine.language_score(text), "after": engine.language_score(improved),
        "n_before": nlp.analyze_text(text), "n_after": nlp.analyze_text(improved),
    }

res = st.session_state.get("improved")
if res:
    st.divider()
    g1, g2, g3 = st.columns(3)
    g1.plotly_chart(charts.gauge(res["before"] * 100, "Before"), width="stretch")
    g2.plotly_chart(charts.gauge(res["after"] * 100, "After"), width="stretch")
    g3.metric("Language score change", f"{(res['after'] - res['before']) * 100:+.1f} points")
    g3.metric("Weak phrases fixed", len(res["changes"]))

    a, b = st.columns(2)
    a.markdown("**Original**")
    a.markdown(f"<div class='resume-box'>{nlp.highlight_html(res['original'])}</div>", unsafe_allow_html=True)
    b.markdown("**Improved**")
    b.markdown(f"<div class='resume-box'>{nlp.highlight_html(res['improved'])}</div>", unsafe_allow_html=True)

    if res["changes"]:
        st.markdown("**Changes made**")
        st.dataframe(pd.DataFrame(res["changes"]), width="stretch", hide_index=True)
    else:
        st.success("No weak phrases found. Your language is already action oriented.")

    st.markdown("**Still to do manually**")
    for tip in res["n_after"]["tips"]:
        st.write("- " + tip)
    st.download_button("Download improved resume (.txt)", res["improved"], "improved_resume.txt")