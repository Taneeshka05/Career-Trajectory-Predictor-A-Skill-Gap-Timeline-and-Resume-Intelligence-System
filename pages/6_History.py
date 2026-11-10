import pandas as pd
import streamlit as st
from core import charts
from core import database as db
from core.ui import setup, profile

engine = setup("History", "📚")
st.title("📚 History and Progress")
if not db.history_enabled():
    st.info("History is disabled in public demo mode. Your resumes and analysis results are not saved.")
    st.stop()
st.caption(f"Saved analyses for profile: {profile()}")

rows = db.list_analyses(profile())
if not rows:
    st.info("Nothing saved yet. Run an analysis and press Save this analysis.")
    st.stop()

df = pd.DataFrame(rows)[["id", "created_at", "job_title", "match_pct", "skill_match_pct", "level",
                         "weeks", "strength"]]
st.dataframe(df.round(1), width="stretch", hide_index=True)
st.download_button("Export history (.csv)", df.to_csv(index=False), "history.csv")
if len(df) > 1:
    st.plotly_chart(charts.trend(df.sort_values("id")), width="stretch")

st.divider()
st.subheader("Roadmap tracker")
chosen = st.selectbox("Open analysis", df["id"].tolist(),
                      format_func=lambda i: f"#{i} | {df.loc[df['id'] == i, 'job_title'].iloc[0]}")
a = db.get_analysis(chosen)
progress = db.get_progress(chosen)

if not a["plan"]:
    st.success("This analysis had no missing skills.")
else:
    done_weeks = 0.0
    total_weeks = sum(p["weeks"] for p in a["plan"])
    for p in a["plan"]:
        checked = st.checkbox(f"{p['skill']} ({p['weeks']:.1f} weeks)", value=progress.get(p["skill"], False),
                              key=f"chk_{chosen}_{p['skill']}")
        if checked != progress.get(p["skill"], False):
            db.set_progress(chosen, p["skill"], checked)
        if checked:
            done_weeks += p["weeks"]
    pct = done_weeks / total_weeks if total_weeks else 1.0
    st.progress(pct, text=f"{pct * 100:.0f}% of the roadmap completed")
    m1, m2 = st.columns(2)
    m1.metric("Weeks completed", f"{done_weeks:.1f}")
    m2.metric("Weeks remaining", f"{total_weeks - done_weeks:.1f}")

if st.button("Delete this analysis"):
    db.delete_analysis(chosen)
    st.rerun()