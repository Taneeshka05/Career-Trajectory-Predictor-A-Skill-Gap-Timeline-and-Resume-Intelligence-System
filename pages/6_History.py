import sys
from pathlib import Path
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "karta hai") not in sys.path:
    sys.path.insert(0, str(ROOT / "karta hai"))

from core import charts
from core import database as db
from core.ui import setup, profile

engine = setup("History", "📚")
st.title("📚 History & Milestone Progress Tracking")
if not db.history_enabled():
    st.info("🔒 History storage is currently disabled (Privacy Mode). Run locally with `CAREERPATH_ENABLE_HISTORY=1` to persist user analyses.")
    st.stop()

st.caption(f"Saved analysis records for profile: **{profile()}**")

rows = db.list_analyses(profile())
if not rows:
    st.info("No saved records found for this profile. Run an analysis on the Analyze Resume page and click 'Save this analysis'.")
    st.stop()

df = pd.DataFrame(rows)[["id", "created_at", "job_title", "match_pct", "skill_match_pct", "level",
                         "weeks", "strength"]]
disp_df = df.round(1).copy()
disp_df.columns = ["Record ID", "Date", "Target Role", "Text Match (%)", "Skill Match (%)", "Readiness Level", "Weeks to Ready", "Content Score"]
st.dataframe(disp_df, width="stretch", hide_index=True)
st.download_button("Export History (.csv)", disp_df.to_csv(index=False), "career_history.csv")

if len(df) > 1:
    st.plotly_chart(charts.trend(df.sort_values("id")), width="stretch")

st.divider()
st.subheader("Interactive Roadmap Progress Checklist")
chosen = st.selectbox(
    "Select Saved Analysis Record",
    df["id"].tolist(),
    format_func=lambda i: f"#{i} | {df.loc[df['id'] == i, 'job_title'].iloc[0]}"
)
a = db.get_analysis(chosen)
progress = db.get_progress(chosen)

if not a["plan"]:
    st.success("This role plan had zero skill gaps.")
else:
    done_weeks = 0.0
    total_weeks = sum(p["weeks"] for p in a["plan"])
    for p in a["plan"]:
        checked = st.checkbox(
            f"{p['skill'].title()} ({p['weeks']:.1f} weeks)",
            value=progress.get(p["skill"], False),
            key=f"chk_{chosen}_{p['skill']}"
        )
        if checked != progress.get(p["skill"], False):
            db.set_progress(chosen, p["skill"], checked)
        if checked:
            done_weeks += p["weeks"]
    pct = done_weeks / total_weeks if total_weeks else 1.0
    st.progress(pct, text=f"{pct * 100:.0f}% of Learning Roadmap Completed")
    m1, m2 = st.columns(2)
    m1.metric("Weeks Completed", f"{done_weeks:.1f} wks")
    m2.metric("Remaining Timeline", f"{total_weeks - done_weeks:.1f} wks")

if st.button("Delete This Saved Record", type="secondary"):
    db.delete_analysis(chosen)
    st.rerun()
