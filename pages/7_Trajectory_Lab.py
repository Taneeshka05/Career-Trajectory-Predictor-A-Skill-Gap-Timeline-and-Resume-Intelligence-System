import sys
from pathlib import Path
from datetime import date, timedelta
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "karta hai") not in sys.path:
    sys.path.insert(0, str(ROOT / "karta hai"))

from core import config as cfg
from core.trajectory import what_if
from core.ui import setup, chips, resume_panel, resume_years, valid_resume_or_dialog

engine = setup("Trajectory Lab", "🚀")
st.title("🚀 Career What-If Simulator & Timeline Lab")
st.caption("Simulate hypothetical skill acquisitions to see real-time impacts on your match percentage, readiness tier, and remaining study timeline.")

r = st.session_state.get("analysis")

if not r:
    st.info("💡 You can either analyze your resume right here, or open the detailed Analyze Resume page first.")
    with st.expander("Quick Analyze Your Resume Here", expanded=True):
        col_t, col_s = st.columns([3, 2])
        with col_t:
            text = resume_panel("trajectory_resume")
        with col_s:
            years = resume_years(text)
            domain = st.selectbox("Target domain", cfg.DOMAIN_NAMES, key="trajectory_domain")
            ids = [i for i, d in enumerate(engine.job_domains) if d == domain]
            job_id = st.selectbox("Target job", ids, key="trajectory_job", format_func=lambda j: engine.job_titles[j])
            if st.button("Analyze for Simulator", type="primary", width="stretch"):
                if valid_resume_or_dialog(text):
                    st.session_state["analysis"] = engine.analyze(text, years, job_id)
                    st.rerun()
    st.stop()

if not r["missing"]:
    st.success(f"🎉 Perfect Alignment: You already possess every skill required for **{r['job_title']}**!")
    st.stop()

st.subheader(f"Target Role: {r['job_title']} ({r['domain']})")

st.markdown("### Hypothetical Skill Acquisition Simulation")
st.caption("Select one or more missing skills to calculate the exact readiness progression using our trained ML models:")

c1, c2 = st.columns([3, 2])
chosen = c1.multiselect("Hypothetical Skills to Master", r["missing"], default=r["missing"][:min(2, len(r["missing"]))])
hours = c2.slider("Weekly Study Hours", 3, 40, cfg.HOURS_BASELINE, key="hours_lab")

if chosen:
    w = what_if(engine, r, chosen)
    b, a = w["before"], w["after"]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Skill Match", f"{a['skill_match_pct']:.1f}%", f"{a['skill_match_pct'] - b['skill_match_pct']:+.1f}%")
    m2.metric("Text Match", f"{a['match_pct']:.1f}%", f"{a['match_pct'] - b['match_pct']:+.1f}%")
    m3.metric("Predicted Readiness", a["level"], "Elevated!" if a["level"] != b["level"] else "Same Tier")
    m4.metric("Weeks Still Needed", f"{a['weeks_left']:.1f} wks", f"{a['weeks_left'] - b['weeks_left']:+.1f} wks")

    weeks = engine.learning_weeks(w["order"], r, hours)
    finish_date = (date.today() + timedelta(weeks=weeks)).strftime("%d %B %Y")
    
    st.markdown(
        f"<div class='card-dark'>"
        f"📅 <b>Target Milestones:</b> Acquiring <b>{' &rarr; '.join(w['order'])}</b> will take approximately "
        f"<b>{weeks:.1f} weeks</b> at {hours} hrs/week (Projected completion: <b>{finish_date}</b>).</div>",
        unsafe_allow_html=True
    )

    if w["foundations"]:
        st.info("Prerequisite Foundations Automatically Included: " + chips(w["foundations"], "info"))
    if a["level"] != b["level"]:
        st.balloons()
        st.success(f"🎉 Career Tier Promotion: This learning plan elevates your evaluated readiness from **{b['level']}** to **{a['level']}**!")

st.markdown("### Comparative Study Schedules")
st.caption("See how your completion timeline scales across different weekly learning commitments:")
rows = []
for h in [5, 10, 15, 20, 30, 40]:
    tl = engine.timeline(r["missing"], r["cand_skills"], r["weeks"], h)
    total = float(tl["end"].max()) if not tl.empty else 0.0
    ready = date.today() + timedelta(weeks=total)
    rows.append({
        "Study Hours / Week": h,
        "Total Duration (Weeks)": round(total, 1),
        "Projected Ready Date": ready.strftime("%d %B %Y")
    })
plan_df = pd.DataFrame(rows)
st.bar_chart(plan_df.set_index("Study Hours / Week")["Total Duration (Weeks)"])
st.dataframe(plan_df, width="stretch", hide_index=True)
