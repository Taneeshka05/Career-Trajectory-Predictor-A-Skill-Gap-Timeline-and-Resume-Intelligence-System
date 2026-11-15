from datetime import date
from core import charts

STYLE = (
    "body{font-family:'Segoe UI',Roboto,Helvetica,Arial,sans-serif;max-width:960px;margin:30px auto;"
    "color:#1f2937;background:#f8fafc;padding:24px}h1{color:#4338ca;margin-bottom:4px}"
    "h2{margin-top:32px;border-bottom:2px solid #e0e7ff;padding-bottom:6px;color:#1e1b4b}"
    "table{border-collapse:collapse;width:100%;margin-top:10px;background:#ffffff;border-radius:8px;"
    "overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,0.05)}th,td{border:1px solid #e2e8f0;padding:8px 12px;"
    "text-align:left}th{background:#f1f5f9;color:#334155;font-weight:600}.chip{display:inline-block;"
    "padding:3px 10px;margin:2px;border-radius:999px;font-size:12px;font-weight:500}"
    ".ok{background:#dcfce7;color:#166534}.miss{background:#fee2e2;color:#991b1b}.info{background:#e0e7ff;color:#3730a3}"
    ".badge{padding:4px 8px;border-radius:4px;font-weight:bold;font-size:12px}"
    ".card{background:#ffffff;border:1px solid #e2e8f0;border-radius:12px;padding:16px;margin:12px 0}"
)


def chips(items, kind):
    return "".join(f"<span class='chip {kind}'>{i}</span>" for i in items) or "None"


def build_html_report(result, timeline_df, traj_df, roi_df):
    compat = result.get("composite_compat", result["match_pct"])
    transfer_pct = result.get("transferable_match_pct", result["skill_match_pct"])
    meta = result.get("metadata", {})
    degrees = ", ".join(meta.get("degrees", [])) or "None detected"
    certs = ", ".join(meta.get("certifications", [])) or "None detected"

    parts = [
        f"<html><head><meta charset='utf-8'><title>CareerPath AI Report: {result['job_title']}</title>",
        f"<style>{STYLE}</style></head><body>",
        "<h1>CareerPath AI: Career Intelligence & Trajectory Report</h1>",
        f"<p style='color:#64748b;'>Generated on {date.today().strftime('%d %B %Y')} for target role: <b>{result['job_title']}</b> ({result['domain']})</p>",
        
        "<h2>Executive Readiness Summary</h2><table>",
        f"<tr><th>Overall Compatibility Score</th><td><b>{compat:.1f} / 100</b> (Multi-dimensional weighted fit)</td></tr>",
        f"<tr><th>Direct Skill Match</th><td>{result['skill_match_pct']:.1f}%</td></tr>",
        f"<tr><th>Transferable Skill Match</th><td>{transfer_pct:.1f}% (Including cross-skill credits)</td></tr>",
        f"<tr><th>Textual Semantic Match</th><td>{result['match_pct']:.1f}% (TF-IDF cosine similarity)</td></tr>",
        f"<tr><th>Assessed Readiness Level</th><td><b>{result['level']}</b></td></tr>",
        f"<tr><th>Estimated Preparation Time</th><td><b>{result['weeks']:.1f} weeks</b> (@ 10 hours/week)</td></tr>",
        f"<tr><th>Resume Language Strength</th><td>{result['strength']} / 100</td></tr>",
        f"<tr><th>Education & Certifications</th><td>Degrees: {degrees} | Certifications: {certs}</td></tr>",
        "</table>",

        "<h2>Skill Gap Analysis</h2>",
        f"<p><b>Matched Core Skills ({len(result['matched'])}):</b> {chips(result['matched'], 'ok')}</p>",
        f"<p><b>Missing Target Skills ({len(result['missing'])}):</b> {chips(result['missing'], 'miss')}</p>"
    ]

    # Transferable skill bridges if any
    if result.get("transferable"):
        parts.append("<h3>Transferable Skill Bridges</h3><table>")
        parts.append("<tr><th>Target Missing Skill</th><th>Your Transferable Skill</th><th>Transferability Credit</th><th>Explanation</th></tr>")
        for t in result["transferable"]:
            parts.append(f"<tr><td><b>{t['target_skill']}</b></td><td>{t['source_skill']}</td><td>{t['percentage']}%</td><td>{t['reason']}</td></tr>")
        parts.append("</table>")

    # Detailed Gap Explanations
    if result.get("gap_explanations"):
        parts.append("<h3>Gap Explanations & Learning Hours</h3><table>")
        parts.append("<tr><th>Skill</th><th>Category</th><th>Difficulty</th><th>Priority</th><th>Estimated Hours</th><th>Purpose in Role</th></tr>")
        for g in result["gap_explanations"]:
            parts.append(f"<tr><td><b>{g['skill']}</b></td><td>{g['category']}</td><td>Level {g['difficulty']}/5</td><td>{g['priority']}</td><td>{g['hours']} hrs</td><td>{g['explanation']}</td></tr>")
        parts.append("</table>")

    if not timeline_df.empty:
        parts.append("<h2>Personalized Learning Roadmap</h2>")
        parts.append(charts.gantt(timeline_df).to_html(full_html=False, include_plotlyjs="cdn"))
        parts.append(timeline_df.round(1).to_html(index=False))
        parts.append("<h2>Career Trajectory Progression</h2>")
        parts.append(charts.trajectory_chart(traj_df).to_html(full_html=False, include_plotlyjs=False))
        parts.append(traj_df.round(1).to_html(index=False))

    if not roi_df.empty:
        parts.append("<h2>Recommended Learning Priority (ROI)</h2>")
        parts.append(roi_df.round(2).to_html(index=False))

    parts.append("<h2>NLP Resume Feedback & Improvement Opportunities</h2><ul>")
    parts += [f"<li>{t}</li>" for t in result["nlp"]["tips"]]
    parts.append("</ul>")

    parts.append("<h2>Alternative Career Recommendations</h2><table><tr><th>Career Domain</th><th>Fit %</th>"
                 "<th>Best Matching Role</th><th>Assessed Level</th><th>Weeks to Ready</th><th>Strategic Alignment</th></tr>")
    for a in result["alternates"]:
        rationale = a.get("rationale", "Skill overlap detected.")
        parts.append(f"<tr><td><b>{a['domain']}</b></td><td>{a['fit']:.0f}%</td><td>{a['job']}</td>"
                     f"<td>{a['level']}</td><td>{a['weeks']:.1f}</td><td>{rationale}</td></tr>")
    parts.append("</table>")

    parts.append(
        "<div style='margin-top:30px;padding:12px;background:#f1f5f9;border-left:4px solid #6366f1;font-size:12px;color:#475569;'>"
        "<b>Academic & Career Disclaimer:</b> Compatibility scores, readiness classifications, and timeline predictions "
        "are analytical estimates based on machine learning models and skill taxonomies. They represent structured guidance "
        "rather than guaranteed hiring outcomes.</div></body></html>"
    )
    return "".join(parts)
