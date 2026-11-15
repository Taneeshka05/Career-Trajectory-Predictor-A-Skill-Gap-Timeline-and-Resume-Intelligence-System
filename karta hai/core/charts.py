import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

INDIGO = "#4f46e5"
PURPLE = "#7c3aed"
EMERALD = "#10b981"
AMBER = "#f59e0b"
ROSE = "#ef4444"
SKY = "#0ea5e9"


def gauge(value, title):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=float(value), title={"text": title},
        number={"suffix": "%" if "Score" not in title else ""},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": INDIGO},
               "steps": [{"range": [0, 40], "color": "#fee2e2"},
                         {"range": [40, 70], "color": "#fef3c7"},
                         {"range": [70, 100], "color": "#dcfce7"}]}))
    fig.update_layout(height=230, margin=dict(l=20, r=20, t=60, b=10))
    return fig


def skill_gap_bar(missing, weeks_by_skill):
    fig = go.Figure(go.Bar(
        x=[weeks_by_skill.get(s, 5) for s in missing],
        y=missing,
        orientation="h",
        marker_color=ROSE,
        text=[f"{weeks_by_skill.get(s, 5):.1f} wks" for s in missing],
        textposition="auto"
    ))
    fig.update_layout(
        title="Missing Target Skills (Base Learning Duration)",
        height=max(250, 40 * len(missing) + 100),
        xaxis_title="Estimated Weeks",
        yaxis=dict(autorange="reversed"),
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig


def gantt(df):
    fig = go.Figure()
    for r in df.itertuples():
        color = AMBER if r.type == "foundation" else px.colors.sequential.Purples[min(getattr(r, "difficulty", 2) + 2, 8)]
        fig.add_trace(go.Bar(
            y=[r.skill],
            x=[r.weeks],
            base=[r.start],
            orientation="h",
            marker_color=color,
            hovertemplate=f"{r.skill} ({r.type})<br>Week {r.start:.1f} to {r.end:.1f}<extra></extra>",
            showlegend=False
        ))
    fig.update_layout(
        title="Personalized Learning Roadmap (Amber = Foundation prerequisite, Purple = Target skill)",
        xaxis_title="Week Timeline",
        yaxis=dict(autorange="reversed"),
        height=max(260, 45 * len(df) + 110),
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig


def radar(coverage):
    names = list(coverage.keys())
    values = [coverage[n] * 100 for n in names]
    fig = go.Figure(go.Scatterpolar(
        r=values + values[:1],
        theta=names + names[:1],
        fill="toself",
        fillcolor="rgba(79, 70, 229, 0.25)",
        line_color=INDIGO,
        name="Candidate Skill Coverage"
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(range=[0, 100], visible=True)),
        showlegend=False,
        height=380,
        title="Skill Coverage Across Career Domains (%)",
        margin=dict(l=40, r=40, t=60, b=20)
    )
    return fig


def compatibility_breakdown_radar(scores_dict: dict):
    """
    Renders multi-dimensional alignment radar for the candidate.
    scores_dict: {'Technical Skills': 75, 'Transferable': 60, 'Text Similarity': 70, 'Soft Skills': 85, 'Experience Fit': 80}
    """
    categories = list(scores_dict.keys())
    values = list(scores_dict.values())
    fig = go.Figure(go.Scatterpolar(
        r=values + values[:1],
        theta=categories + categories[:1],
        fill="toself",
        fillcolor="rgba(16, 185, 129, 0.25)",
        line_color=EMERALD,
        name="Fit Dimensions"
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(range=[0, 100], visible=True)),
        showlegend=False,
        height=340,
        title="Multi-Dimensional Career Alignment (%)",
        margin=dict(l=30, r=30, t=50, b=20)
    )
    return fig


def career_comparison_radar(domain_coverage_dict: dict, selected_domains: list):
    """
    Compares candidate coverage across 2 or 3 selected career domains.
    """
    fig = go.Figure()
    colors = [INDIGO, EMERALD, AMBER]
    for i, d in enumerate(selected_domains):
        cov = domain_coverage_dict.get(d, 0.0) * 100
        fig.add_trace(go.Bar(
            name=d,
            x=[d],
            y=[cov],
            marker_color=colors[i % len(colors)],
            text=[f"{cov:.1f}%"],
            textposition="auto"
        ))
    fig.update_layout(
        title="Direct Domain Skill Match Comparison",
        yaxis=dict(range=[0, 105], title="Match %"),
        height=340,
        margin=dict(l=30, r=30, t=50, b=20)
    )
    return fig


def career_map(map_df, point):
    sample = map_df.sample(min(1500, len(map_df)), random_state=1).copy()
    rng = np.random.default_rng(3)
    sample["x"] += rng.normal(0, 0.03, len(sample))
    sample["y"] += rng.normal(0, 0.03, len(sample))
    fig = px.scatter(
        sample, x="x", y="y", color="domain", opacity=0.45,
        labels={"x": "Principal Component 1 (Core Specialization)", "y": "Principal Component 2 (Tooling & Stack)"}
    )
    fig.add_trace(go.Scatter(
        x=[point[0]], y=[point[1]], mode="markers+text", text=["You (Candidate)"],
        textposition="top center", name="You",
        marker=dict(size=18, symbol="star", color="#f59e0b", line=dict(color="#ffffff", width=2))
    ))
    fig.update_layout(height=520, title="Skill Space Embedding (PCA of Profile Clusters)")
    return fig


def confusion(cm, labels, title):
    fig = px.imshow(
        cm, text_auto=True, x=labels, y=labels, color_continuous_scale="Blues",
        labels=dict(x="Predicted Class", y="Actual Ground Truth")
    )
    fig.update_layout(title=title, height=380)
    return fig


def curves(history):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=history["loss"], name="Train Loss", line=dict(color=ROSE)))
    fig.add_trace(go.Scatter(y=history["val_loss"], name="Validation Loss", line=dict(color=AMBER, dash="dash")))
    fig.add_trace(go.Scatter(y=history["accuracy"], name="Train Accuracy", yaxis="y2", line=dict(color=EMERALD, dash="dot")))
    fig.add_trace(go.Scatter(y=history["val_accuracy"], name="Validation Accuracy", yaxis="y2", line=dict(color=INDIGO, dash="dot")))
    fig.update_layout(
        title="Neural Network Training Curves (Loss & Accuracy)",
        xaxis_title="Epoch",
        height=380,
        yaxis=dict(title="Loss"),
        yaxis2=dict(title="Accuracy", overlaying="y", side="right", range=[0, 1.05])
    )
    return fig


def metric_bars(df, metric, title):
    fig = px.bar(
        df, x="Model", y=metric, text=df[metric].round(3), color="Model",
        color_discrete_sequence=px.colors.qualitative.Prism
    )
    fig.update_layout(title=title, showlegend=False, height=350, yaxis=dict(range=[0, 1.08]))
    return fig


def reg_scatter(df):
    fig = px.scatter(
        df, x="actual", y="predicted", opacity=0.5,
        labels={"actual": "Actual Target (Weeks)", "predicted": "Model Predicted (Weeks)"}
    )
    top = float(max(df["actual"].max(), df["predicted"].max()))
    fig.add_trace(go.Scatter(x=[0, top], y=[0, top], mode="lines", line=dict(color=ROSE, dash="dash"), name="Ideal 1:1"))
    fig.update_layout(title="Time-to-Ready Regression: Actual vs. Predicted (Unseen Test Data)", height=380)
    return fig


def rank_bar(df):
    fig = px.bar(
        df, x="fit_score", y="title", orientation="h", color="skill_match_pct",
        color_continuous_scale="Viridis", hover_data=["level", "weeks_to_ready"]
    )
    fig.update_layout(
        title="Top Matching Job Roles Ranked by Compatibility",
        height=max(300, 38 * len(df) + 120),
        xaxis_title="Overall Fit Score (%)",
        yaxis=dict(autorange="reversed")
    )
    return fig


def trend(df):
    fig = px.line(
        df, x="created_at", y=["match_pct", "strength"], markers=True,
        labels={"value": "Score / Index", "created_at": "Analysis Date", "variable": "Metric"}
    )
    fig.update_layout(title="Candidate Progress Over Time", height=340)
    return fig


LEVEL_COLORS = {"Junior": ROSE, "Mid": AMBER, "Senior": EMERALD}


def trajectory_chart(traj):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=traj["week"], y=traj["skill_match_pct"], mode="lines", name="Skill Match %",
        line=dict(color=INDIGO, width=2.5)
    ))
    fig.add_trace(go.Scatter(
        x=traj["week"], y=traj["match_pct"], mode="lines", name="Text Match %",
        line=dict(color="#94a3b8", dash="dot")
    ))
    fig.add_trace(go.Scatter(
        x=traj["week"], y=traj["skill_match_pct"], mode="markers+text",
        text=traj["skill"], textposition="top center", showlegend=False,
        marker=dict(size=14, color=[LEVEL_COLORS.get(l, INDIGO) for l in traj["level"]],
                    line=dict(width=1.5, color="white")),
        hovertext=[f"Role Level: {l}" for l in traj["level"]]
    ))
    fig.update_layout(
        title="Career Trajectory Projection (Dot Color = Assessed Level: Red Junior, Amber Mid, Green Senior)",
        xaxis_title="Elapsed Learning Weeks",
        yaxis_title="Match & Readiness Percentage (%)",
        height=420,
        yaxis=dict(range=[0, 118])
    )
    return fig


def roi_bar(df):
    fig = px.bar(
        df, x="roi", y="skill", orientation="h", color="match_gain_pct",
        color_continuous_scale="Teal", hover_data=["cost_weeks", "weeks_saved", "new_level"],
        text=df["roi"].round(2)
    )
    fig.update_layout(
        title="Skill Acquisition ROI (Match % Gain per Learning Week)",
        xaxis_title="Match % Gained per Week of Study",
        height=max(260, 42 * len(df) + 110),
        yaxis=dict(autorange="reversed")
    )
    return fig


def contrib_bar(df, title, unit):
    colors = [ROSE if c > 0 else EMERALD for c in df["contribution"]]
    fig = go.Figure(go.Bar(
        x=df["contribution"], y=df["factor"], orientation="h", marker_color=colors,
        text=df["contribution"].round(2), textposition="auto"
    ))
    fig.update_layout(
        title=title, xaxis_title=unit, yaxis=dict(autorange="reversed"),
        height=max(260, 40 * len(df) + 110), margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig


def line_chart(df, x, y, title):
    fig = px.line(df, x=x, y=y, markers=True)
    fig.update_layout(title=title, height=340, yaxis=dict(range=[0, 1.05]))
    return fig


def importance_bar(df):
    fig = px.bar(
        df, x="importance", y="feature", orientation="h",
        color="importance", color_continuous_scale="Purples"
    )
    fig.update_layout(
        title="Permutation Feature Importance (Validation Accuracy Degradation)",
        yaxis=dict(autorange="reversed"), height=360
    )
    return fig


def eda_corr_heatmap(df_numeric):
    corr = df_numeric.corr()
    fig = px.imshow(
        corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        title="Feature Correlation Matrix (EDA)"
    )
    fig.update_layout(height=420)
    return fig


def eda_missing_bar(missing_series):
    fig = px.bar(
        x=missing_series.index, y=missing_series.values,
        labels={"x": "Feature", "y": "Missing Count"},
        title="Missing Values per Column"
    )
    fig.update_layout(height=320)
    return fig
