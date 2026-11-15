import sys
from pathlib import Path
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "karta hai") not in sys.path:
    sys.path.insert(0, str(ROOT / "karta hai"))

from core import charts
from core import nlp_feedback as nlp
from core.preprocess import clean_text, extract_skills
from core.ui import setup, resume_panel, valid_resume_or_dialog

engine = setup("Resume Improver", "✍️")
st.title("✍️ NLP Resume Improver & Quality Audit")
st.caption("Transforms weak passive phrases into high-impact action verbs, analyzes linguistic metrics, and suggests quantified STAR bullets.")

text = resume_panel("resume_improve")
if st.button("Audit & Improve Resume Content", type="primary"):
    if valid_resume_or_dialog(text):
        improved, changes = nlp.auto_improve(text)
        clean = clean_text(text)
        cand_skills = extract_skills(clean)
        st.session_state["improved"] = {
            "original": text,
            "improved": improved,
            "changes": changes,
            "before": engine.language_score(text),
            "after": engine.language_score(improved),
            "n_before": nlp.analyze_text(text, found_skills=cand_skills),
            "n_after": nlp.analyze_text(improved, found_skills=cand_skills),
            "skills": cand_skills,
        }

res = st.session_state.get("improved")
if res:
    st.divider()
    g1, g2, g3 = st.columns(3)
    g1.plotly_chart(charts.gauge(res["before"] * 100, "Initial Language Score"), width="stretch")
    g2.plotly_chart(charts.gauge(res["after"] * 100, "Improved Language Score"), width="stretch")
    g3.metric("Neural Quality Delta", f"{(res['after'] - res['before']) * 100:+.1f} points")
    g3.metric("Weak / Passive Phrases Fixed", len(res["changes"]))

    st.markdown("### Side-by-Side Comparison")
    a, b = st.columns(2)
    with a:
        st.markdown("**Original Text (Red = Weak/Passive Phrases)**")
        st.markdown(f"<div class='resume-box'>{nlp.highlight_html(res['original'])}</div>", unsafe_allow_html=True)
    with b:
        st.markdown("**Auto-Improved Text (Green = Active Verbs)**")
        st.markdown(f"<div class='resume-box'>{nlp.highlight_html(res['improved'])}</div>", unsafe_allow_html=True)

    if res["changes"]:
        st.markdown("### Replaced Passive Verbs")
        st.dataframe(pd.DataFrame(res["changes"]), width="stretch", hide_index=True)
    else:
        st.success("No passive verbs detected. Language is already active and direct.")

    # Vocabulary & Quality Audit
    st.markdown("### Linguistic Quality Audit")
    n_after = res["n_after"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Word Count", n_after["n_words"])
    m2.metric("Avg Sentence Length", f"{n_after['avg_sentence_len']} words")
    m3.metric("Vocabulary Richness (TTR)", f"{n_after['vocabulary']['ttr'] * 100:.0f}%")
    m4.metric("Quantified Metrics Found", n_after["quantified"])

    if n_after.get("rubric"):
        st.markdown("#### Quality Rubric Breakdown")
        rub_cols = st.columns(4)
        for col, (rub_name, val) in zip(rub_cols, n_after["rubric"].items()):
            col.metric(rub_name, f"{val}/100")

    # STAR bullet suggestions
    st.markdown("### 🎯 Quantified STAR Bullet Point Generator")
    st.caption("Crafted for your detected skills to replace generic task listings with measurable achievements:")
    star_bullets = nlp.generate_star_bullet_suggestions(res["skills"])
    for b in star_bullets:
        st.markdown(
            f"<div class='card-dark' style='margin-bottom:8px;'>"
            f"<b>{b['category']} ({b['skill']}):</b><br>"
            f"<code>&bull; {b['template']}</code><br>"
            f"<span style='font-size:0.8rem;color:#a5b4fc;'>Structure: {b['formula']}</span></div>",
            unsafe_allow_html=True
        )

    st.markdown("### Manual Polish Recommendations")
    for tip in n_after["tips"]:
        st.write("- " + tip)

    st.download_button("Download Improved Resume (.txt)", res["improved"], "improved_resume.txt", type="primary")
