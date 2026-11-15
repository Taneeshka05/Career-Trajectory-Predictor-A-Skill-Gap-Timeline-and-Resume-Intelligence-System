import sys
from pathlib import Path
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "karta hai") not in sys.path:
    sys.path.insert(0, str(ROOT / "karta hai"))

from core import config as cfg
from core import charts
from core.preprocess import clean_text, extract_skills
from core.taxonomy import find_transferable_credits
from core.ui import setup, chips, resume_panel, resume_years, valid_resume_or_dialog

engine = setup("Career Map", "🗺️")
st.title("🗺️ Career Space & Multi-Path Comparison System")
st.caption("Visualizes where your skillset clusters in the 5,500-profile PCA skill space and compares alternate career paths side-by-side.")

left, right = st.columns([3, 2])
with left:
    text = resume_panel("resume_map", height=200)
with right:
    years = resume_years(text)
    go = st.button("Project My Skill Space & Compare", type="primary", width="stretch")

if go:
    if valid_resume_or_dialog(text):
        clean = clean_text(text)
        skills = extract_skills(clean)
        st.session_state["map"] = {
            "skills": skills,
            "point": engine.candidate_point(skills),
            "alts": engine.alternate_careers(clean, skills, years),
            "coverage": engine.domain_coverage(skills),
        }

m = st.session_state.get("map")
if m:
    st.divider()
    st.markdown("**Detected Candidate Skills:** " + chips(m["skills"], "info"), unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["Skill Space Embedding", "Side-by-Side Career Comparison"])

    with tab1:
        c1, c2 = st.columns([3, 2])
        with c1:
            st.plotly_chart(charts.career_map(engine.b["pca"]["map_df"], m["point"]), width="stretch")
        with c2:
            st.plotly_chart(charts.radar(m["coverage"]), width="stretch")

        st.subheader("Closest Alternative Career Paths")
        cols = st.columns(len(m["alts"]))
        for col, a in zip(cols, m["alts"]):
            with col:
                st.markdown(
                    f"<div class='card'><b>{a['domain']}</b><br>"
                    f"Fit Index: <b>{a['fit']:.0f}%</b><br>"
                    f"Target Position: {a['job']}<br>"
                    f"Readiness: <b>{a['level']}</b><br>"
                    f"Est. Prep: <b>{a['weeks']:.1f} weeks</b><br>"
                    f"<span style='font-size:0.8rem;color:#475569;'>{a.get('rationale', '')}</span></div>",
                    unsafe_allow_html=True
                )

    with tab2:
        st.subheader("Compare Target Career Paths Side-by-Side")
        st.caption("Select two or three career domains to evaluate differences in skill overlap, gaps, and preparation time:")
        selected_domains = st.multiselect(
            "Select Career Domains to Compare",
            cfg.DOMAIN_NAMES,
            default=cfg.DOMAIN_NAMES[:2],
            max_selections=3
        )
        if len(selected_domains) >= 2:
            st.plotly_chart(charts.career_comparison_radar(m["coverage"], selected_domains), width="stretch")
            
            comp_rows = []
            cand_skill_set = set(m["skills"])
            for d in selected_domains:
                d_skills = set(cfg.DOMAIN_SKILLS[d])
                overlap = cand_skill_set & d_skills
                missing = d_skills - cand_skill_set
                trans = find_transferable_credits(cand_skill_set, missing)
                cov = m["coverage"].get(d, 0.0) * 100
                comp_rows.append({
                    "Career Domain": d,
                    "Direct Skill Match": f"{cov:.1f}% ({len(overlap)}/{len(d_skills)})",
                    "Missing Requirements": ", ".join(sorted(missing)) or "None",
                    "Transferable Bridges": f"{len(trans)} bridges identified",
                    "Transition Difficulty": "Low" if cov > 65 else ("Moderate" if cov > 35 else "High")
                })
            st.dataframe(pd.DataFrame(comp_rows), width="stretch", hide_index=True)
        else:
            st.info("Please select at least 2 domains to display side-by-side comparison.")
