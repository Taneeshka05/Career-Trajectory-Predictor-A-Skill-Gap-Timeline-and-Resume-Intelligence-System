from datetime import date
from core import charts

STYLE = ("body{font-family:Arial,sans-serif;max-width:900px;margin:30px auto;color:#1f2937}"
         "h1{color:#4f46e5}h2{margin-top:28px;border-bottom:2px solid #e0e7ff;padding-bottom:4px}"
         "table{border-collapse:collapse;width:100%}th,td{border:1px solid #e5e7eb;padding:6px 10px;"
         "text-align:left}th{background:#f3f4f9}.chip{display:inline-block;padding:2px 10px;margin:2px;"
         "border-radius:999px;font-size:13px}.ok{background:#dcfce7}.miss{background:#fee2e2}")


def chips(items, kind):
    return "".join(f"<span class='chip {kind}'>{i}</span>" for i in items) or "None"


def build_html_report(result, timeline_df, traj_df, roi_df):
    parts = [f"<html><head><meta charset='utf-8'><style>{STYLE}</style></head><body>",
             "<h1>CareerPath AI Report</h1>",
             f"<p>Generated on {date.today().isoformat()} for <b>{result['job_title']}</b></p>",
             "<h2>Summary</h2><table>",
             f"<tr><th>Text match</th><td>{result['match_pct']:.1f}%</td></tr>",
             f"<tr><th>Skill match</th><td>{result['skill_match_pct']:.1f}%</td></tr>",
             f"<tr><th>Readiness level</th><td>{result['level']}</td></tr>",
             f"<tr><th>Estimated time to ready</th><td>{result['weeks']:.1f} weeks</td></tr>",
             f"<tr><th>Resume strength</th><td>{result['strength']} / 100</td></tr></table>",
             "<h2>Skills</h2>",
             f"<p>Matched: {chips(result['matched'], 'ok')}</p>",
             f"<p>Missing: {chips(result['missing'], 'miss')}</p>"]
    if not timeline_df.empty:
        parts.append("<h2>Learning roadmap</h2>")
        parts.append(charts.gantt(timeline_df).to_html(full_html=False, include_plotlyjs="cdn"))
        parts.append(timeline_df.round(1).to_html(index=False))
        parts.append("<h2>Career trajectory</h2>")
        parts.append(charts.trajectory_chart(traj_df).to_html(full_html=False, include_plotlyjs=False))
        parts.append(traj_df.round(1).to_html(index=False))
    if not roi_df.empty:
        parts.append("<h2>Which skill to learn first</h2>")
        parts.append(roi_df.round(2).to_html(index=False))
    parts.append("<h2>Resume feedback</h2><ul>")
    parts += [f"<li>{t}</li>" for t in result["nlp"]["tips"]]
    parts.append("</ul><h2>Alternate careers</h2><table><tr><th>Career</th><th>Fit %</th>"
                 "<th>Best job</th><th>Level</th><th>Weeks</th></tr>")
    for a in result["alternates"]:
        parts.append(f"<tr><td>{a['domain']}</td><td>{a['fit']:.0f}</td><td>{a['job']}</td>"
                     f"<td>{a['level']}</td><td>{a['weeks']:.1f}</td></tr>")
    parts.append("</table><p><i>Predictions come from models trained on synthetic data with rule-based "
                 "labels. Use as guidance, not as a hiring decision.</i></p></body></html>")
    return "".join(parts)