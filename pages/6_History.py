import sys
import os
from pathlib import Path
from datetime import datetime
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "karta hai") not in sys.path:
    sys.path.insert(0, str(ROOT / "karta hai"))

from core import charts
from core import config as cfg
from core import database as db
from core.ui import setup, profile, chips

# Ensure history is initialized and enabled for local session
if not db.history_enabled():
    os.environ[db.HISTORY_ENV] = "1"
    db.init_db()

engine = setup("History", "📚")
st.title("📚 Document Upload History & Career Trajectory Tracking")
st.caption("Audit all individual resumes and documents analyzed, tracking qualification match, readiness metrics, and learning milestone progress.")

current_profile = profile()
rows = db.list_analyses(current_profile)

# If no records for current profile, also check if any other profiles exist or allow demo loading
if not rows:
    st.info(f"ℹ️ No saved resume analyses found for profile **'{current_profile}'** yet.")
    st.markdown(
        """
        You can analyze a resume on the **Analyze Resume** page to automatically record it here, 
        or load the benchmark demonstration resume to test the history tracking system right now:
        """
    )
    if st.button("🚀 Load Demonstration Resume to Test History", type="primary"):
        demo_analysis = engine.analyze(cfg.DEMO_RESUME, 3, 0)
        plan = engine.plan_for_save(demo_analysis)
        db.save_analysis(current_profile, demo_analysis, plan, filename="Demonstration_Resume.pdf")
        st.success("Demonstration resume saved to history!")
        st.rerun()
    st.stop()

# 1. SUMMARY KPI CARDS
st_data = db.stats(current_profile)
total_docs = len(rows)

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
kpi1.metric("📄 Resumes Uploaded", f"{total_docs} {'File' if total_docs == 1 else 'Files'}")
kpi2.metric("🎯 Avg Job Match", f"{st_data['match']:.1f}%")
kpi3.metric("💡 Avg Skill Match", f"{st_data.get('skill_match', 0.0):.1f}%")
kpi4.metric("⏳ Avg Timeline", f"{st_data['weeks']:.1f} wks")
kpi5.metric("⭐ Avg Content Score", f"{st_data['strength']:.1f}/100")

st.divider()

# 2. DOCUMENT-BY-DOCUMENT SUMMARY TABLE
st.subheader("📑 All Uploaded Documents History")
st.caption("Detailed overview of every resume uploaded and analyzed by this user:")

table_data = []
for r in rows:
    created_dt = r.get("created_at", "")
    try:
        formatted_date = datetime.fromisoformat(created_dt).strftime("%Y-%m-%d %H:%M")
    except Exception:
        formatted_date = str(created_dt)

    table_data.append({
        "Record ID": r["id"],
        "Document Name": r.get("filename", f"Resume Document #{r['id']}"),
        "Uploaded Date": formatted_date,
        "Target Role": r["job_title"],
        "Domain": r["domain"],
        "Experience": f"{r['years']} yrs",
        "Text Match (%)": round(r["match_pct"], 1),
        "Skill Match (%)": round(r["skill_match_pct"], 1),
        "Readiness": r["level"],
        "Timeline (wks)": round(r["weeks"], 1),
        "Content Score": round(r["strength"], 1),
    })

disp_df = pd.DataFrame(table_data)

# Filter by document name or target role
search_q = st.text_input("🔍 Search document history by name or role:", placeholder="e.g. data analyst, python, resume.pdf")
if search_q.strip():
    q = search_q.strip().lower()
    disp_df = disp_df[
        disp_df["Document Name"].str.lower().str.contains(q)
        | disp_df["Target Role"].str.lower().str.contains(q)
        | disp_df["Domain"].str.lower().str.contains(q)
    ]

st.dataframe(disp_df, width="stretch", hide_index=True)

col_dl, col_clr = st.columns([3, 1])
with col_dl:
    st.download_button("📥 Export Full History (.csv)", disp_df.to_csv(index=False), "uploaded_resumes_history.csv")
with col_clr:
    if st.button("🗑️ Clear All History", type="secondary"):
        db.clear_all_analyses(current_profile)
        st.success("All history records cleared.")
        st.rerun()

# 3. HISTORICAL PERFORMANCE TRENDS
if len(rows) > 1:
    st.markdown("### 📈 Historical Progression Across Uploads")
    st.caption("Visualizing compatibility and content score improvements over time:")
    trend_df = pd.DataFrame(rows)[["id", "match_pct", "strength"]].sort_values("id")
    st.plotly_chart(charts.trend(trend_df), width="stretch")

st.divider()

# 4. INDIVIDUAL DOCUMENT AUDIT & MILESTONE PROGRESS CHECKLIST
st.subheader("🔍 Inspect Individual Document & Track Learning Roadmap")
record_map = {r["id"]: r for r in rows}

chosen = st.selectbox(
    "Select Uploaded Document to Inspect:",
    options=[r["id"] for r in rows],
    format_func=lambda i: f"#{i} | 📄 {record_map[i].get('filename', 'Resume')} ➔ {record_map[i]['job_title']} ({record_map[i].get('created_at', '')[:10]})"
)

a = db.get_analysis(chosen)
if a:
    doc_title = a.get("filename", f"Resume Document #{chosen}")
    st.markdown(
        f"<div class='card-dark'>"
        f"<h4>📄 {doc_title}</h4>"
        f"<b>Target Role:</b> {a['job_title']} ({a['domain']}) &nbsp;|&nbsp; "
        f"<b>Candidate Experience:</b> {a['years']} years &nbsp;|&nbsp; "
        f"<b>Predicted Readiness Level:</b> {a['level']} &nbsp;|&nbsp; "
        f"<b>Compatibility Score:</b> {a['match_pct']:.1f}%"
        f"</div>",
        unsafe_allow_html=True
    )

    col_meta1, col_meta2 = st.columns(2)
    with col_meta1:
        st.markdown(f"**Missing Target Skills ({len(a['missing'])})**")
        if a["missing"]:
            st.markdown(chips(a["missing"], "miss"), unsafe_allow_html=True)
        else:
            st.success("Zero missing skills detected for this position!")
    with col_meta2:
        st.markdown("**Learning Roadmap Status**")
        progress = db.get_progress(chosen)
        if not a["plan"]:
            st.success("Candidate fully meets all required technical competencies.")
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
                    st.rerun()
                if checked:
                    done_weeks += p["weeks"]

            pct = done_weeks / total_weeks if total_weeks else 1.0
            st.progress(pct, text=f"{pct * 100:.0f}% of Learning Roadmap Completed")
            m1, m2 = st.columns(2)
            m1.metric("Weeks Completed", f"{done_weeks:.1f} wks")
            m2.metric("Remaining Timeline", f"{total_weeks - done_weeks:.1f} wks")

    with st.expander("📄 View Extracted Text for this Resume Document"):
        st.text_area("Extracted Resume Text", value=a.get("resume_text", ""), height=220, disabled=True, key=f"text_view_{chosen}")

    if st.button("Delete This Document Record", type="secondary", key=f"del_{chosen}"):
        db.delete_analysis(chosen)
        st.success(f"Record #{chosen} deleted.")
        st.rerun()
